from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from tempo.api.llm.models.llama import Llama
from tempo.api.llm.tokenizer.tokenizer import RuntimeTokenizer
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.api.tempo_context_manager import TempoContext
from tempo.core import index_expr as ie
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT
from tempo.core.dtype import dtypes
from tempo.runtime.backends.backend import DLBackend
from tempo.utils.jax_profiler import JaxProfiler
from tempo.utils.make_sink import make_sink

dtypes.default_float = dtypes.float16


@dataclass
class InferenceStats:
    compile_s: float
    exec_s: float
    total_s: float
    # peak_gpu_mb: float
    mean_gpu_util: float
    peak_gpu_util: float


@dataclass
class ProfileConfig:
    """Configuration for inference profiling."""

    dev: str
    n_runs: int
    batch_size: int
    max_seq_len: int
    checkpoint_dir: str
    visualize: bool
    inc_statify_block_size: int
    path: str
    profile: bool


# TODO: integrate into tokenizer.py eventually
def left_pad_token_batch(
    token_lists: List[List[int]],
    max_prompt_len: int,
    tokenizer: RuntimeTokenizer,
) -> RecurrentTensor:
    """
    Left-pad every sequence in *token_lists* to *max_prompt_len* and return a recurrent tensor.
    """
    # Find token id for " "
    if tokenizer.pad_id != -1:
        pad_token = tokenizer.pad_id
    else:
        pad_tokens = tokenizer.encode(" ", bos=False, eos=False)
        assert len(pad_tokens) == 1, f"Pad token {pad_tokens} is not a single token"
        pad_token = pad_tokens[0]

    print(f"Pad token: {pad_token}")
    padded = [[pad_token] * (max_prompt_len - len(seq)) + seq for seq in token_lists]
    arr = np.asarray(padded, dtype=dtypes.to_np(dtypes.default_int))
    ret = RecurrentTensor.lift(arr)
    return ret


class LlamaInferenceRunner:
    """Reusable Tempo‑LLaMA inference & profiling wrapper."""

    def __init__(
        self,
        checkpoint_dir: str,
        prompts: List[str],
        max_seq_len: int = 200,
        batch_size: int = 16,
        *,
        exec_cfg: ExecutionConfig,
    ) -> None:
        dtypes.default_int = dtypes.int64 if exec_cfg.backend == "torch" else dtypes.int32

        # Filled by `run()`
        self.tokenizer: RuntimeTokenizer = Llama.auto_tokenizer_from_checkpoint(checkpoint_dir)

        # dtypes.default_int = dtypes.int32 if self.tokenizer.vocab_size >= 2**16 else dtypes.int16

        # Test tokenizer
        for example_string in prompts:
            tokens = self.tokenizer.encode(example_string)
            decoded_tokens = self.tokenizer.decode(tokens)
            # tokens = self.tokenizer.encode(example_string, bos=True, eos=False)
            # decoded_tokens = self.tokenizer.decode(tokens)
            assert decoded_tokens == example_string, (
                f"{decoded_tokens=} != {example_string=}. Something is wrong with the tokenizer."
            )

        self.checkpoint_dir = checkpoint_dir
        self.prompts = prompts
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        self._sink_data: Dict[str, BackendTensorT] = {}  # type: ignore

        self._build_graph(exec_cfg)

    def _build_graph(self, exec_cfg: ExecutionConfig) -> None:
        self.exec_cfg = exec_cfg

        # TODO: make max_prompt_len a symbol and inject batch using a udf
        tokenized_prompts = [self.tokenizer.encode(p, bos=True, eos=False) for p in self.prompts][
            : self.batch_size
        ]
        self.max_prompt_len = max(
            list(map(len, tokenized_prompts)) + [exec_cfg.inc_statify_block_size]
        )

        # TODO: remove this
        assert self.max_prompt_len == exec_cfg.inc_statify_block_size, (
            f"Max prompt length is {self.max_prompt_len} but  \
            should be {exec_cfg.inc_statify_block_size}"
        )

        ctx = TempoContext(exec_cfg, num_dims=2)
        self.ctx = ctx

        b, s = ctx.variables
        B, S = ctx.upper_bounds

        with ctx:
            with ctx.tag_region("load_left_padded_prompts"):
                prompt_token_batch = left_pad_token_batch(
                    tokenized_prompts, self.max_prompt_len, self.tokenizer
                )

            with ctx.tag_region("model_def"):
                model = Llama.from_checkpoint(
                    checkpoint_dir=self.checkpoint_dir,
                    temporal_dim=s,
                    n_words=self.tokenizer.n_words,
                    dtype=dtypes.default_float,
                    exec_cfg=self.exec_cfg,
                )

            # Ensures model and batch is loaded before we move on.
            # RecurrentTensor.barrier("model_loaded")

            with ctx.tag_region("tokens"):
                token_ids = RecurrentTensor.placeholder(
                    shape=(), dtype=dtypes.default_int, domain=(b, s), requires_grad=False
                )

                # TODO: make max_prompt_len a symbol
                # TODO: logits[s < p] = model.prefill(embedded_tokens[0:B, 0:p])
                # .index(0, b).index(0, t)
                token_ids[s < self.max_prompt_len] = prompt_token_batch.index(dim=0, index=b).index(
                    dim=0, index=s
                )

            with ctx.tag_region("model_fwd"):
                w = 30
                attn_pat = ie.slice_(ie.max(0, s - w + 1), s + 1)
                # attn_pat = None
                logits = model.forward(token_ids, attn_pat)
            with ctx.tag_region("model_sample"):
                nxt = model.sample(logits, greedy=False)

            with ctx.tag_region("tokens"):
                token_ids[True] = nxt[b, s - 1]

            with ctx.tag_region("sink_tokens"):
                RecurrentTensor.sink_udf(token_ids[0:B, 0:S], make_sink("tokens", self._sink_data))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, n_runs: int = 3, profile: bool = False) -> InferenceStats:
        """Compile & execute the graph, collect metrics, populate outputs."""
        # tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        # mon = ResourceMonitorManager(tmp_file.name, fps=10, gpu_ids=[0])
        # with mon:
        start_time = time.time()
        ctx = self.ctx
        B, S = ctx.upper_bounds

        executor = self.ctx.compile({B: self.batch_size, S: self.max_seq_len})
        end_time = time.time()
        compile_time = end_time - start_time

        print(f"Compile time: {compile_time} seconds")

        bend = DLBackend.get_backend(self.exec_cfg.backend)
        non_warmup_run_times = []
        non_warmup_exec_times = []
        for i in range(n_runs):
            start_time = time.time()
            if i == 2 and profile:
                # with TorchProfiler.get(True, "/home/tempo/bind_mnt/"):
                with JaxProfiler.get(True, "/home/tempo/bind_mnt/"):
                    executor.execute()
            else:
                executor.execute()
            exec_done_time = time.time()
            bend.sync()
            end_time = time.time()
            run_time = end_time - start_time
            print(f"Execution time ({i + 1}/{n_runs}): {run_time} seconds")

            self.print_decoded_outputs()

            if i != 0:
                non_warmup_run_times.append(run_time)
                non_warmup_exec_times.append(exec_done_time - start_time)

        if n_runs > 1:
            avg_time = sum(non_warmup_run_times) / len(non_warmup_run_times)
            std_time = float(np.std(non_warmup_run_times))
            print(f"Average execution time (sync): {avg_time}s ± {std_time}s")
            avg_exec_time = sum(non_warmup_exec_times) / len(non_warmup_exec_times)
            std_exec_time = float(np.std(non_warmup_exec_times))
            print(f"Average execution time (no sync): {avg_exec_time}s ±{std_exec_time} s")

        # res = mon.get_results()
        # if res.has_gpu_data:
        #    mean_util = res.mean_gpu_util.get(0, float("nan"))
        #    peak_util = res.peak_gpu_util.get(0, float("nan"))
        # else:
        #    mean_util = float("nan")
        #    peak_util = float("nan")

        ## materialise output arrays so callers can inspect them right away
        # self._materialise_outputs()

        # return InferenceStats(
        #    compile_s=round(compile_time, 3),
        #    exec_s=round(avg_time, 3),
        #    total_s=round(compile_time + avg_time, 3),
        #    #peak_gpu_mb=round(peak_mb, 2),
        #    mean_gpu_util=round(mean_util, 2),
        #    peak_gpu_util=round(peak_util, 2),
        # )

        # Return a default InferenceStats object for now
        return InferenceStats(
            compile_s=round(compile_time, 3),
            exec_s=round(avg_time if n_runs > 1 else 0.0, 3),
            total_s=round(compile_time + (avg_time if n_runs > 1 else 0.0), 3),
            mean_gpu_util=0.0,
            peak_gpu_util=0.0,
        )

    def print_decoded_outputs(self, n_rows: int = 10) -> None:
        """Return decoded strings for each batch element."""
        if self.exec_cfg.backend == "jax":
            import jax

            # Put tokens on CPU in jax
            tok = np.asarray(
                jax.device_put(self._sink_data["tokens"], device=jax.devices("cpu")[0])
            )
        else:
            # Handle the case where BackendTensorT might not have cpu() method
            tokens_tensor = self._sink_data["tokens"]
            if hasattr(tokens_tensor, "cpu"):
                tok = np.asarray(tokens_tensor.cpu())  # type: ignore
            else:
                # Fallback for tensors that don't have cpu() method
                tok = np.asarray(tokens_tensor)

        toks_to_print = tok[: min(n_rows, self.batch_size)]

        tok_rows = [row[row != self.tokenizer.pad_id].tolist() for row in toks_to_print]

        eos_idx = [
            row.index(self.tokenizer.eos_id) if self.tokenizer.eos_id in row else len(row)
            for row in tok_rows
        ]
        print(f"EOS indices: {eos_idx}")

        if any(eos_idx[i] != len(tok_rows[i]) for i in range(len(tok_rows))):
            print("WARNING: Some tokens after EOS were filtered out.")

        tok_rows_filtered = [row[: eos_idx[i]] for i, row in enumerate(tok_rows)]

        try:
            responses = [self.tokenizer.decode(row) for row in tok_rows_filtered]
            for r in responses:
                print(r)
        except Exception as e:
            print(f"Error decoding tokens: {e}")
            print(f"Tokens: {tok_rows_filtered}")


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------


def pretty_print_stats(
    backend: str,
    n_layers: int,
    n_heads: int,
    batch_size: int,
    max_prompt_len: int,
    max_seq_len: int,
    decoded: List[str],
    *,
    compile_s: float,
    exec_s: float,
    total_s: float,
    peak_mb: float,
) -> None:
    """Print the same nicely formatted block used in the original demo."""
    print("=== Inference Params ===")
    print(f"Backend: {backend}")
    print(f"Number of Layers: {n_layers}")
    print(f"Number of Heads: {n_heads}")
    print(f"Batch Size: {batch_size}")
    print(f"Max Prompt Length: {max_prompt_len}")
    print(f"Max Sequence Length: {max_seq_len}")

    print("=== Inference Stats ===")
    print(f"\nCompile time: {compile_s:.3f}s  |  Execute time: {exec_s:.3f}s")
    print(f"Total time:   {total_s:.3f}s  |  Peak GPU memory: {peak_mb:.1f} MiB")

    print("=== Decoded prompts + completions ===")
    for i, text in enumerate(decoded):
        print(f"[{i}] {text[: min(200, len(text))]}")


sample_prompts = [
    "I believe the meaning of life is",
    "In the heart of the ancient forest, I stumbled upon",
    "My favorite book from childhood is",
    "Nothing beats a good cup of coffee when",
    "If I could travel anywhere in the world right now, I'd go to",
    "The secret ingredient in the perfect chocolate cake is",
    "People often forget the importance of",
    "Growing a small herb garden can",
    "The invention that changed the world the most was",
    "My grandfather always used to say",
    "On a misty autumn morning, the old lighthouse keeper discovered",
    "The one song that never fails to lift my spirits is",
    "Whenever I walk past an art gallery, I imagine",
    "If kindness were a currency, the wealthiest person I know would be",
    "The most unforgettable smell from my childhood kitchen is",
    "Late at night, when the city finally quiets down, you can almost hear",
    "The first star I ever wished upon was",
    "When the clock struck midnight, the abandoned carnival suddenly",
    "If walls could talk, the one in my childhood bedroom would whisper",
    "On rainy Sundays, I always find myself",
    "The greatest lesson my favorite teacher taught me was",
    "If dreams were postcards, last night's would have shown",
    "The smell of fresh paint always reminds me of",
    "When I open an old photo album, I'm transported to",
    "Every winter, the frozen lake hides a secret beneath",
    "The one tradition I hope never disappears is",
    "If time travel were possible, I'd first visit",
    "The taste of summer can be found in",
    "My favorite place to watch the sunset is",
    "Whenever I hear distant thunder, I remember",
    "The silent hero in my everyday life is",
    "If I wrote a letter to my future self, I'd start with",
    "The city lights at dawn reveal",
    "In the quiet corner of the library, I discovered",
    "My most cherished family heirloom is more than an object; it's",
    "If laughter were a color, mine would be",
    "The shadow that follows me on late-night walks feels like",
    "When I smell fresh bread baking, I imagine",
    "The moment I realized I'd grown up was",
    "The first star I ever wished upon was",
    "When the clock struck midnight, the abandoned carnival suddenly",
    "If walls could talk, the one in my childhood bedroom would whisper",
    "On rainy Sundays, I always find myself",
    "The greatest lesson my favorite teacher taught me was",
    "If dreams were postcards, last night's would have shown",
    "The most courageous thing I ever witnessed was",
    "If animals could speak, my pet would tell me",
    "Every winter, the frozen lake hides a secret beneath",
    "The one tradition I hope never disappears is",
    "If time travel were possible, I'd first visit",
    "The taste of summer can be found in",
    "My favorite place to watch the sunset is",
    "Whenever I hear distant thunder, I remember",
    "The silent hero in my everyday life is",
    "If I wrote a letter to my future self, I'd start with",
    "The city lights at dawn reveal",
    "In the quiet corner of the library, I discovered",
    "My most cherished family heirloom is more than an object; it's",
    "If laughter were a color, mine would be",
    "The shadow that follows me on late-night walks feels like",
    "When I smell fresh bread baking, I imagine",
    "The moment I realized I'd grown up was",
]

if __name__ == "__main__":
    # model = "Llama-2-7b"
    # model = "Llama3.2-1B"
    model = "Llama3.2-3B"

    gpu_profile = ProfileConfig(
        dev="gpu",
        n_runs=4,
        batch_size=64,
        max_seq_len=150,
        checkpoint_dir=f"/home/tempo/.llama/checkpoints/{model}/",
        visualize=True,
        inc_statify_block_size=50,
        path="/home/tempo/bind_mnt/results_llama",
        profile=False,
    )

    prof_gpu_profile = ProfileConfig(
        dev="gpu",
        n_runs=3,
        batch_size=64,
        max_seq_len=40,
        checkpoint_dir=f"/home/tempo/.llama/checkpoints/{model}/",
        visualize=True,
        inc_statify_block_size=20,
        path="/home/tempo/bind_mnt/results_llama",
        profile=True,
    )

    cpu_profile = ProfileConfig(
        dev="cpu",
        n_runs=1,
        batch_size=2,
        max_seq_len=60,
        checkpoint_dir=f"/home/pedro/.llama/checkpoints/{model}/",
        visualize=True,
        inc_statify_block_size=30,
        path="/home/pedro/results_llama",
        profile=False,
    )

    profile = cpu_profile
    backend = "jax"

    cfg = ExecutionConfig(
        backend=backend,
        dev=profile.dev,
        path=profile.path,
        # options: jit, compile, export, symbolic_trace
        torch_compilation_backend="compile",
        visualize_pipeline_stages=profile.visualize,
        render_schedule=profile.visualize,
        executor_debug_mode=False,
        enable_algebraic_optimizer=True,
        enable_dataflow_grouping=True,
        enable_codegen_dataflows=True,
        enable_hybrid_tensorstore=True,
        enable_statifying_incrementalization=True,
        inc_statify_block_size=profile.inc_statify_block_size,
        enable_fold_pads_into_storage=True,
        enable_pad_mask_removal=True,
        enable_inplace_write=True,
        enable_exec_op_profiling=False,
        exec_op_profiling_sync_after_each=False,
    )

    sampled_prompts = random.choices(sample_prompts, k=profile.batch_size)
    # system_prompt = "Complete the following story."
    # sampled_prompts = [system_prompt + " " + p for p in sampled_prompts]

    runner = LlamaInferenceRunner(
        checkpoint_dir=profile.checkpoint_dir,
        prompts=sampled_prompts,
        max_seq_len=profile.max_seq_len,
        batch_size=profile.batch_size,
        exec_cfg=cfg,
    )
    stats = runner.run(profile.n_runs, profile.profile)
    # pretty_print_stats(**stats)
