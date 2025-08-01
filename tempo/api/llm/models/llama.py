from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

from tempo.api import RecurrentTensor, nn
from tempo.api.llm.tokenizer.tokenizer import RuntimeTokenizer
from tempo.api.nn.rope import RopeScalingParams
from tempo.api.nn.state_loader import StateDictLoader
from tempo.core.configs import ExecutionConfig
from tempo.core.domain import DomainLike
from tempo.core.dtype import DataType, dtypes
from tempo.core.index_expr import IndexAtom, Symbol
from tempo.utils.logger import get_logger

log = get_logger(__name__)

"""
    Llama 2/3.2 models share the same state dict structure, so we can use the same code for both.

    ====   Llama2  ====
    Llama2 state dict structure:
    'layers.<i>.attention.wk.weight': torch.bfloat16 torch.Size([4096, 4096])
    'layers.<i>.attention.wo.weight': torch.bfloat16 torch.Size([4096, 4096])
    'layers.<i>.attention.wq.weight': torch.bfloat16 torch.Size([4096, 4096])
    'layers.<i>.attention.wv.weight': torch.bfloat16 torch.Size([4096, 4096])
    'layers.<i>.attention_norm.weight': torch.bfloat16 torch.Size([4096])
    'layers.<i>.feed_forward.w1.weight': torch.bfloat16 torch.Size([11008, 4096])
    'layers.<i>.feed_forward.w2.weight': torch.bfloat16 torch.Size([4096, 11008])
    'layers.<i>.feed_forward.w3.weight': torch.bfloat16 torch.Size([11008, 4096])
    'layers.<i>.ffn_norm.weight': torch.bfloat16 torch.Size([4096])
    'norm.weight': torch.bfloat16 torch.Size([4096])
    'output.weight': torch.bfloat16 torch.Size([32000, 4096])
    'tok_embeddings.weight': torch.bfloat16 torch.Size([32000, 4096])
    'rope.freqs': torch.bfloat16 torch.Size([64]) #NOTE: we ignore this. Just compute it ourselves.

    ====   Llama3.2 (1B)  ====
    Llama3.2 (1B) state dict structure:
    layers.<l>.attention.wk.weight: torch.bfloat16 torch.Size([512, 2048])
    layers.<l>.attention.wo.weight: torch.bfloat16 torch.Size([2048, 2048])
    layers.<l>.attention.wq.weight: torch.bfloat16 torch.Size([2048, 2048])
    layers.<l>.attention.wv.weight: torch.bfloat16 torch.Size([512, 2048])
    layers.<l>.attention_norm.weight: torch.bfloat16 torch.Size([2048])
    layers.<l>.feed_forward.w1.weight: torch.bfloat16 torch.Size([8192, 2048])
    layers.<l>.feed_forward.w2.weight: torch.bfloat16 torch.Size([2048, 8192])
    layers.<l>.feed_forward.w3.weight: torch.bfloat16 torch.Size([8192, 2048])
    layers.<l>.ffn_norm.weight: torch.bfloat16 torch.Size([2048])
    norm.weight: torch.bfloat16 torch.Size([2048])
    output.weight: torch.bfloat16 torch.Size([128256, 2048])
    tok_embeddings.weight: torch.bfloat16 torch.Size([128256, 2048])

    Llama3.2 (1B) params:
    {
      "dim": 2048,
      "ffn_dim_multiplier": 1.5,
      "multiple_of": 256,
      "n_heads": 32,
      "n_kv_heads": 8,
      "n_layers": 16,
      "norm_eps": 1e-05,
      "rope_theta": 500000.0,
      "use_scaled_rope": true,
      "vocab_size": 128256
    }

    ====   Llama3.2 (3B)  ====
    Llama3.2 (3B) state dict structure:
    layers.<l>.attention.wk.weight: torch.bfloat16 torch.Size([1024, 3072])
    layers.<l>.attention.wo.weight: torch.bfloat16 torch.Size([3072, 3072])
    layers.<l>.attention.wq.weight: torch.bfloat16 torch.Size([3072, 3072])
    layers.<l>.attention.wv.weight: torch.bfloat16 torch.Size([1024, 3072])
    layers.<l>.attention_norm.weight: torch.bfloat16 torch.Size([3072])
    layers.<l>.feed_forward.w1.weight: torch.bfloat16 torch.Size([8192, 3072])
    layers.<l>.feed_forward.w2.weight: torch.bfloat16 torch.Size([3072, 8192])
    layers.<l>.feed_forward.w3.weight: torch.bfloat16 torch.Size([8192, 3072])
    layers.<l>.ffn_norm.weight: torch.bfloat16 torch.Size([3072])
    norm.weight: torch.bfloat16 torch.Size([3072])
    output.weight: torch.bfloat16 torch.Size([128256, 3072])
    tok_embeddings.weight: torch.bfloat16 torch.Size([128256, 3072])

    Llama3.2 (3B) params:
    {
      "dim": 3072,
      "ffn_dim_multiplier": 1.0,
      "multiple_of": 256,
      "n_heads": 24,
      "n_kv_heads": 8,
      "n_layers": 28,
      "norm_eps": 1e-05,
      "rope_theta": 500000.0,
      "use_scaled_rope": true,
      "vocab_size": 128256
    }

"""


@dataclass
class LlamaArgs:
    dim: int = 4096
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    n_heads: int = 32
    n_kv_heads: Optional[int] = None  # For GQA
    n_layers: int = 32
    norm_eps: float = 1e-5
    ffn_dim_multiplier: Optional[float] = None
    rope_theta: float = 10_000.0  # default for llama2
    # NOTE: the official llama3.2 code doenst use this, so we do not either.
    use_scaled_rope: bool = False  # default for llama2
    vocab_size: int = -1  # defined later by tokenizer
    tie_embeddings_and_output: bool = True
    rope_scaling_params: Optional[RopeScalingParams] = None


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float] = None,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
        state_loader: Optional[StateDictLoader] = None,
    ) -> None:
        super().__init__(domain, independent_domain)
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        if state_loader is None:
            state_loader = StateDictLoader.empty()

        self.w1 = nn.Linear(
            in_features=dim,
            out_features=hidden_dim,
            bias=False,
            domain=domain,
            independent_domain=independent_domain,
            w_init_fun=state_loader.load_tensor("w1.weight"),
        )
        self.w2 = nn.Linear(
            in_features=hidden_dim,
            out_features=dim,
            bias=False,
            domain=domain,
            independent_domain=independent_domain,
            w_init_fun=state_loader.load_tensor("w2.weight"),
        )
        self.w3 = nn.Linear(
            in_features=dim,
            out_features=hidden_dim,
            bias=False,
            domain=domain,
            independent_domain=independent_domain,
            w_init_fun=state_loader.load_tensor("w3.weight"),
        )

    def forward(self, x: RecurrentTensor) -> RecurrentTensor:
        return self.w2(self.w1(x).swish() * self.w3(x))  # type:ignore


class TransformerBlock(nn.Module):
    def __init__(
        self,
        params: LlamaArgs,
        seq_len: Symbol,
        layer_idx: Symbol,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
        state_loader: Optional[StateDictLoader] = None,
    ) -> None:
        super().__init__(domain, independent_domain)

        if state_loader is None:
            state_loader = StateDictLoader.empty()

        block_loader = state_loader.append("layers").append(layer_idx)

        attention_state_loader = block_loader.append("attention")
        self.attention = nn.MultiHeadAttention(
            params.dim,
            params.n_heads,
            seq_len,
            domain=domain,
            independent_domain=independent_domain,
            w_init_funs=[
                attention_state_loader.load_tensor("wq.weight"),
                attention_state_loader.load_tensor("wk.weight"),
                attention_state_loader.load_tensor("wv.weight"),
                attention_state_loader.load_tensor("wo.weight"),
            ],
            apply_rope=True,
            rope_theta=params.rope_theta,
            rope_scaling_params=params.rope_scaling_params,
            num_kv_heads=params.n_kv_heads,
        )

        self.attention_norm = nn.RMSNorm(
            params.dim,
            domain=domain,
            independent_domain=independent_domain,
            w_init_fun=block_loader.load_tensor("attention_norm.weight"),
        )

        self.ffn = FeedForward(
            params.dim,
            4 * params.dim,
            ffn_dim_multiplier=params.ffn_dim_multiplier,
            multiple_of=params.multiple_of,
            domain=domain,
            independent_domain=independent_domain,
            state_loader=block_loader.append("feed_forward"),
        )

        self.ffn_norm = nn.RMSNorm(
            params.dim,
            domain=domain,
            independent_domain=independent_domain,
            w_init_fun=block_loader.load_tensor("ffn_norm.weight"),
        )

    def forward(self, x: RecurrentTensor, pattern: Optional[IndexAtom] = None) -> RecurrentTensor:
        h = x + self.attention(self.attention_norm(x), pattern)
        out: RecurrentTensor = h + self.ffn(self.ffn_norm(h))

        return out


class Llama(nn.Module):
    def __init__(
        self,
        params: LlamaArgs,
        temporal_dim: Symbol,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
        state_loader: Optional[StateDictLoader] = None,
    ) -> None:
        super().__init__(domain, independent_domain)
        self.params = params
        self.vocab_size = params.vocab_size
        self.t = temporal_dim

        if state_loader is None:
            state_loader = StateDictLoader.empty()

        from tempo.api.tempo_context_manager import get_active_ctx_manager

        # NOTE: Create a new temporal dimension to represent the layers
        # This enables compilation to scale independently of number of layers by hinting at the
        # repetitive structure of the model.
        self.l, self.l_ub = get_active_ctx_manager().new_perm_var(bound=params.n_layers)

        self.tok_embeddings = nn.Embedding(
            self.params.vocab_size,
            self.params.dim,
            domain=domain,
            independent_domain=independent_domain,
            w_init_fun=state_loader.load_tensor("tok_embeddings.weight"),
        )

        self.transformer_block = TransformerBlock(
            params,
            temporal_dim,
            layer_idx=self.l,
            domain=domain,
            independent_domain=self.indep_dom.append_dim(self.l, self.l_ub),
            state_loader=state_loader,
        )
        self.norm = nn.RMSNorm(
            params.dim,
            eps=params.norm_eps,
            domain=domain,
            independent_domain=self.indep_dom,
            w_init_fun=state_loader.load_tensor("norm.weight"),
        )

        assert params.tie_embeddings_and_output, (
            "Tied embeddings and output is required for Llama3.2"
        )
        self.output = nn.Linear(
            self.params.dim,
            self.vocab_size,
            bias=False,
            domain=domain,
            independent_domain=self.indep_dom,
            w_init_fun=state_loader.load_tensor("output.weight")
            if not params.tie_embeddings_and_output
            else partial(
                RecurrentTensor.init_from_existing_tensor, self.tok_embeddings.embed_table
            ),
        )

    @staticmethod
    def sp_tokenizer_from_checkpoint(checkpoint_dir: str) -> RuntimeTokenizer:
        from tempo.api.llm.tokenizer.sp_tokenizer import SPTokenizer

        path = Path(checkpoint_dir) / "tokenizer.model"
        log.info("Loading tokenizer from %s", path)
        return SPTokenizer(path=path)

    @staticmethod
    def tiktoken_tokenizer_from_checkpoint(checkpoint_dir: str) -> RuntimeTokenizer:
        from pathlib import Path

        from tempo.api.llm.tokenizer.tiktoken_tokenizer import TiktokenTokenizer

        path_tiktoken = Path(checkpoint_dir) / "tokenizer.tiktoken"
        if path_tiktoken.is_file():
            path = path_tiktoken
        elif (Path(checkpoint_dir) / "original" / "tokenizer.tiktoken").is_file():
            path = Path(checkpoint_dir) / "original" / "tokenizer.tiktoken"
        elif (Path(checkpoint_dir) / "original" / "tokenizer.model").is_file():
            path = Path(checkpoint_dir) / "original" / "tokenizer.model"
        else:
            raise FileNotFoundError(f"No tiktoken tokenizer file found in {checkpoint_dir}")

        tokenizer_json_path = Path(checkpoint_dir) / "tokenizer.json"
        if not tokenizer_json_path.is_file():
            tokenizer_json_path = Path(checkpoint_dir) / "original" / "tokenizer.json"

        log.info("Loading tiktoken tokenizer from %s", path)
        return TiktokenTokenizer(path)

    @staticmethod
    def auto_tokenizer_from_checkpoint(checkpoint_dir: str) -> RuntimeTokenizer:
        if "Llama3.2" in checkpoint_dir:
            return Llama.tiktoken_tokenizer_from_checkpoint(checkpoint_dir)
        elif "Llama-2" in checkpoint_dir:
            return Llama.sp_tokenizer_from_checkpoint(checkpoint_dir)
        else:
            raise ValueError(f"Unknown model type in {checkpoint_dir}")

    @staticmethod
    def from_checkpoint(
        checkpoint_dir: str,
        temporal_dim: Symbol,
        n_words: int,
        dtype: DataType,
        exec_cfg: ExecutionConfig,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
    ) -> Llama:
        import json

        params_path = Path(checkpoint_dir) / "params.json"
        if not params_path.exists():
            params_path = Path(checkpoint_dir) / "original" / "params.json"

        log.info("Loading params from %s", params_path)
        params = dict(json.load(open(params_path)))
        params_dataclass = LlamaArgs(**params)
        params_dataclass.vocab_size = n_words
        log.info("Llama params: %s", params_dataclass)
        if params_dataclass.use_scaled_rope:
            log.info("Loading rope scaling params from config.json")
            config_path = Path(checkpoint_dir) / "config.json"
            config = json.load(open(config_path))
            rope_params = RopeScalingParams(**config["rope_scaling"])
            params_dataclass.rope_scaling_params = rope_params
        else:
            raise ValueError("Llama3.2 requires scaled rope")

        checkpoint_path = Path(checkpoint_dir) / "consolidated.00.pth"
        if not checkpoint_path.exists():
            checkpoint_path = Path(checkpoint_dir) / "original" / "consolidated.00.pth"

        state_loader = StateDictLoader.from_torch_checkpoint(
            checkpoint_path,
            exec_cfg=exec_cfg,
            cast_to_dtype=dtype,
        )

        return Llama(
            params_dataclass,
            temporal_dim,
            domain=domain,
            independent_domain=independent_domain,
            state_loader=state_loader,
        )

    def forward(
        self, x: RecurrentTensor, attention_pattern: Optional[IndexAtom] = None
    ) -> RecurrentTensor:
        l, L = self.l, self.l_ub

        # assert x.dtype == dtypes.int16, "x.dtype is not int16"

        x_emb = self.tok_embeddings(x)

        # assert x.dtype == dtypes.float16, "x.dtype is not float16"

        dom_ = x_emb.domain.append_dim(l, L)

        residual_stream = RecurrentTensor.placeholder(
            shape=(self.params.dim,),
            dtype=dtypes.default_float,
            domain=dom_,
            requires_grad=False,
        )

        # assert residual_stream.dtype == dtypes.float16, "residual_stream.dtype is not float16"

        l_idx = dom_.find_variable_index(l)
        dom_basis = dom_.basis_expr

        residual_stream[dom_basis.replace_idx(l_idx, 0)] = x_emb
        block_out = self.transformer_block(residual_stream, attention_pattern).cast(x_emb.dtype)
        # assert block_out.dtype == dtypes.float16, "block_out.dtype is not float16"
        residual_stream[True] = block_out[dom_basis.replace_idx(l_idx, l - 1)]

        normalized_out = self.norm(block_out[dom_basis.replace_idx(l_idx, L - 1)])
        result: RecurrentTensor = self.output(normalized_out)

        # assert result.dtype == dtypes.float16, "result.dtype is not float16"
        return result

    def sample(
        self,
        logits: RecurrentTensor,
        temperature: float = 0.6,
        top_p: float = 0.9,
        greedy: bool = False,
    ) -> RecurrentTensor:
        # Apply temperature scaling

        # TODO: scalable Sample top-p (nucleus sampling):
        # We have the function, but compiling it is horrible as it generates enormous sdgs.
        # We need to either make compilation considerably more scalable, or
        # encode it using symbolic dimensions.
        # One more alternative, is to make sorting a primitive op. Tbh, this is probably the best
        # option.
        # https://github.com/meta-llama/llama3/blob/main/llama/generation.py#L343

        # assert logits.dtype == dtypes.float16, "logits.dtype is not float16"

        if greedy:
            sampled = logits.argmax(dim=-1).cast(dtypes.default_int)
        else:
            # TODO: allow temperature to be a recurrent tensor?
            assert temperature >= 0.0, "Temperature must be positive"
            assert temperature <= 1.0, "Temperature must be less than 1.0"
            logits = logits / temperature
            probs = logits.cast(dtypes.float32).softmax(dim=-1)  # .cast(logits.dtype)
            sampled = probs.multinomial(num_samples=1).squeeze(-1).cast(dtypes.default_int)

            # probs = RecurrentTensor.softmax(logits / temperature, dim=-1)
            ## return probs.sample_top_p(top_p).squeeze(-1)
            # sampled = probs.sample_likely(above=0.05).squeeze(-1)

        # sampled[0 : b.as_bound()].sink_udf(lambda x_: print(f"Sampled tokens: {x_}"))
        return sampled
