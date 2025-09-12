from __future__ import annotations

import numpy as np

from repro.sec7_2_llama32_decode.shared import get_prompts
from tempo.api.llm.models.llama import Llama
from tempo.api.llm.tokenizer.tokenizer import RuntimeTokenizer
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.api.tempo_context_manager import TempoContext
from tempo.core import index_expr as ie
from tempo.core.datatypes import BackendTensorT
from tempo.core.dl_backend import DLBackend
from tempo.core.dtype import dtypes
from tempo.utils.jax_profiler import JaxProfiler
from tempo.utils.make_sink import make_step_by_step_sink

dtypes.default_float = dtypes.float16

DEFAULT_STATIFY_BLOCK_SIZE = 1024


def get_tempo_llama_config(**kwargs):
    from tempo.core.configs import ExecutionConfig

    cfg = ExecutionConfig.default()
    # NOTE: In this experiment we only use the JAX backend
    cfg.backend = kwargs.get("backend", "jax")
    cfg.dev = kwargs.get("dev", "gpu")
    cfg.path = str(kwargs["results_path"])

    cfg.validate_pipeline_stages = kwargs.get("validate", False)
    cfg.visualize_pipeline_stages = kwargs.get("visualize", False)
    cfg.render_schedule = kwargs.get("render_schedule", False)

    cfg.inc_statify_block_size = kwargs.get("statify_block_size", DEFAULT_STATIFY_BLOCK_SIZE)

    # NOTE: We explicitly want to see tempo use its custom tensor storages even if they are worse.
    cfg.enable_point_store_fallback = False
    cfg.torch_pinned_memory_enabled = False
    cfg.enable_non_trivial_vectorization = False
    cfg.enable_swap = False

    if kwargs.get("debug", False):
        cfg.executor_debug_mode = True
        cfg.visualize_pipeline_stages = True
        cfg.render_schedule = True
        cfg.validate_pipeline_stages = True

    if kwargs.get("compile_only", False):
        # NOTE: Disable all optional things that "incorrectly"
        # add to compile time without being interesting.
        cfg.render_schedule = False
        cfg.validate_pipeline_stages = False
        cfg.visualize_pipeline_stages = False
        cfg.visualize_debug_stages = False
        cfg.enable_codegen_thunk_warmup = False
        cfg.enable_constant_folding = False
        cfg.enable_non_trivial_vectorization = False
        cfg.enable_symbol_prealloc_store = False

    for key in kwargs.get("disable_cfg_keys", []):
        setattr(cfg, key, False)

    return cfg


class TempoLlama32InferenceRunner:
    """Reusable Tempo‑LLaMA inference & profiling wrapper."""

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        window_size: int,
        attn_type: str,
        temperature: float,
        checkpoint_dir: str,
        prompts: list[str],
        **kwargs,
    ) -> None:
        self.tokenizer: RuntimeTokenizer = Llama.auto_tokenizer_from_checkpoint(checkpoint_dir)

        self.checkpoint_dir = checkpoint_dir
        self.prompts = prompts
        self.seq_len = seq_len
        self.window_size = window_size
        self.attn_type = attn_type
        self.temperature = temperature

        self.batch_size = batch_size
        self.kwargs = kwargs
        self._sink_data: dict[str, BackendTensorT] = {}  # type: ignore

    def compile(self) -> None:
        cfg = get_tempo_llama_config(**self.kwargs)
        self.exec_cfg = cfg

        # TODO: make max_prompt_len a symbol and inject batch using a udf
        tokenized_prompts = [self.tokenizer.encode(p, bos=True, eos=False) for p in self.prompts]
        self.max_prompt_len = max(
            list(map(len, tokenized_prompts))
            # + [cfg.inc_statify_block_size if cfg.enable_statifying_incrementalization else 0]
        )

        ctx = TempoContext(cfg, num_dims=2)
        self.ctx = ctx

        b, t = ctx.variables
        B, T = ctx.upper_bounds

        with ctx:
            with ctx.tag_region("load_left_padded_prompts"):
                prompt_token_batch = self.tokenizer.left_pad_token_batch(
                    tokenized_prompts, self.max_prompt_len
                )

            with ctx.tag_region("model_def"):
                if self.kwargs.get("num_layers", None):
                    num_layers = self.kwargs.get("num_layers", None)
                    args = Llama.get_args_from_checkpoint(
                        self.checkpoint_dir, self.tokenizer.n_words, enable_rope_scaling=False
                    )
                    args.n_layers = num_layers
                    model = Llama(
                        params=args,
                        temporal_dim=t,
                    )
                else:
                    model = Llama.from_checkpoint(
                        checkpoint_dir=self.checkpoint_dir,
                        temporal_dim=t,
                        n_words=self.tokenizer.n_words,
                        dtype=dtypes.float16,
                        exec_cfg=self.exec_cfg,
                        # NOTE: We disable rope scaling because meta's
                        # torch impl does not support it.
                        enable_rope_scaling=False,
                        # force_rope_base=10000,
                    )

            # Ensures model and batch is loaded before we move on.
            # RecurrentTensor.barrier("model_loaded")

            with ctx.tag_region("tokens"):
                token_ids = RecurrentTensor.placeholder(
                    shape=(), dtype=dtypes.default_int, domain=(b, t), requires_grad=False
                )

                # TODO: make max_prompt_len a symbol
                # TODO: logits[s < p] = model.prefill(embedded_tokens[0:B, 0:p])
                # .index(0, b).index(0, t)
                token_ids[t < self.max_prompt_len] = prompt_token_batch.index(dim=0, index=b).index(
                    dim=0, index=t
                )

            with ctx.tag_region("model_fwd"):
                w = self.window_size
                attn_pat = (
                    ie.slice_(ie.max(0, t - w + 1), t + 1) if self.attn_type == "window" else None
                )
                logits = model.forward(token_ids, attn_pat)
                assert logits.dtype == dtypes.float16
            with ctx.tag_region("model_sample"):
                if self.temperature <= 0.0:
                    nxt = model.sample(logits, greedy=True)
                else:
                    nxt = model.sample(logits, greedy=False, temperature=self.temperature)

            with ctx.tag_region("tokens"):
                token_ids[True] = nxt[b, t - 1]

            with ctx.tag_region("sink_tokens"):
                RecurrentTensor.sink_udf(
                    # token_ids[0:B, 0:T], make_sink("tokens", self._sink_data)
                    token_ids[0:B, t],
                    make_step_by_step_sink("tokens", self._sink_data),
                )
        self.exec = self.ctx.compile({B: self.batch_size, T: self.seq_len})

    def reset(self):
        self._sink_data: dict[str, BackendTensorT] = {}  # type: ignore
        self.exec.reset()

    def warmup(self):
        self.run()

    def run(self):
        profiler = JaxProfiler.get(self.kwargs.get("profile", False), self.kwargs["results_path"])
        with profiler:
            self.exec.execute()
            self.exec.backend.sync()

    def get_decoded_outputs(self) -> list[str]:
        """Return decoded strings for each batch element."""
        bend = DLBackend.get_backend(self.exec_cfg.backend)
        toks = self._sink_data["tokens"]
        toks_np = [bend.to_numpy(tok) for tok in toks]

        tok = np.stack(toks_np, axis=1)

        tok_rows = [row[row != self.tokenizer.pad_id].tolist() for row in tok]
        # tok_rows = [row.tolist() for row in tok]

        stop_idx = []
        for row in tok_rows:
            idx = -1
            for stop_token in self.tokenizer.stop_tokens:
                if stop_token in row:
                    idx = row.index(stop_token)
                    break
            if idx == -1:
                idx = len(row)
            stop_idx.append(idx)

        print(f"Stop indices: {stop_idx}")

        if any(stop_idx[i] < len(tok_rows[i]) for i in range(len(tok_rows))):
            print("WARNING: Some tokens after EOS were filtered out.")

        tok_rows_filtered = [row[: stop_idx[i]] for i, row in enumerate(tok_rows)]

        responses = [self.tokenizer.decode(row) for row in tok_rows_filtered]
        for r in responses:
            print(r)


if __name__ == "__main__":
    # model = "Llama-2-7b"
    model = "Llama3.2-1B"
    # model = "Llama3.2-3B"

    base_params = {
        "backend": "torch",
        "dev": "fake-gpu",
        "results_path": "./results/tempo_llama32_decode/",
        "profile": False,
        "statify_block_size": 32,
        "visualize": True,
        "validate": False,
        "render_schedule": True,
        "batch_size": 2,
        "seq_len": 128,
        "window_size": 0,
        "attn_type": "causal",
        "max_prompt_len": 16,
        "temperature": 0.6,  # NOTE: 0.0 for greedy
        "checkpoint_dir": f"~/.llama/checkpoints/{model}/",
        "debug": False,
        # "validate": True,
    }
    prompts = get_prompts(**base_params)
    # prompts = ["Hello, how are you?", "What is the capital of France?"]
    # prompts = [
    #    "What is the most beautiful place in London?",
    #    "What is the best tube station in London?",
    # ]

    runner = TempoLlama32InferenceRunner(
        **base_params,
        prompts=prompts,
    )
    runner.compile()
    runner.run()
    print(runner.get_decoded_outputs())
