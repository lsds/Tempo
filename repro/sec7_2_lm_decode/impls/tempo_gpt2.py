import math

from repro.sec7_2_lm_decode.bench_runner import BenchRunner
from repro.sec7_2_lm_decode.run_measure_tpt import GPT2_SMALL_PARAMS
from repro.sec7_2_lm_decode.shared import run_bench
from tempo.api import nn
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.api.tempo_context_manager import TempoContext
from tempo.core import index_expr as ie
from tempo.core.domain import DomainLike
from tempo.core.dtype import dtypes

"""Simple TEMPO implementation of the GPT-2-Small core.
"""

# NOTE: This is the default statify block size used in the original submission.
# DEFAULT_STATIFY_BLOCK_SIZE = 512

# NOTE: This is the default statify block size obtained in our recent reproduction.
DEFAULT_STATIFY_BLOCK_SIZE = 1024


def get_tempo_gpt2_config(**kwargs):
    from tempo.core.configs import ExecutionConfig

    cfg = ExecutionConfig.default()
    # NOTE: In this experiment we only use the JAX backend
    cfg.backend = "jax"
    cfg.dev = "gpu" if "dev" not in kwargs else kwargs["dev"]
    cfg.path = str(kwargs["results_path"])

    cfg.visualize_pipeline_stages = True
    cfg.render_schedule = True

    cfg.inc_statify_block_size = kwargs.get("statify_block_size", DEFAULT_STATIFY_BLOCK_SIZE)

    # NOTE: We explicitly want to see tempo use its custom tensor storages even if they are worse.
    cfg.enable_point_store_fallback = False

    return cfg


def rms_norm(x: RecurrentTensor, eps: float = 1e-8) -> RecurrentTensor:
    """Applies RMSNorm: x * (weight / rms(x))"""
    rms = RecurrentTensor.sqrt(
        RecurrentTensor.mean(RecurrentTensor.square(x), dims=-1, keepdim=True) + eps
    )
    normed = x / rms
    return normed


class MHA(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_size: int,
        window_size: int,
        attn_type: str,
        t: ie.Symbol,
        domain: DomainLike,
    ):
        super().__init__(domain)
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.embed_size_per_head = embed_size // num_heads
        self.window_size = window_size
        self.attn_type = attn_type
        self.window_size = window_size
        self.t = t

        # self.W_QKV = RecurrentTensor.rand(
        #    (
        #        3,
        #        num_heads,
        #        embed_size,
        #        self.embed_size_per_head,
        #    ),
        #    dtypes.float32,
        #    domain=domain,
        # )
        self.W_Q = RecurrentTensor.rand(
            (num_heads, embed_size, self.embed_size_per_head), domain=domain
        )
        self.W_K = RecurrentTensor.rand(
            (num_heads, embed_size, self.embed_size_per_head), domain=domain
        )
        self.W_V = RecurrentTensor.rand(
            (num_heads, embed_size, self.embed_size_per_head), domain=domain
        )

        self.W_O = RecurrentTensor.rand(
            (num_heads * self.embed_size_per_head, embed_size), domain=domain
        )

    def forward(self, x):
        # x is (embed_size, )
        # projections = x @ self.W_QKV  # (3, num_heads, embed_per_head, embed_size)
        # q, k, v = projections.split(num_splits=3, dim=0)
        # q = q.squeeze(0)  # (num_heads, embed_per_head)
        # k = k.squeeze(0)  # (num_heads, embed_per_head)
        # v = v.squeeze(0)  # (num_heads, embed_per_head)
        q = x @ self.W_Q
        k = x @ self.W_K
        v = x @ self.W_V

        t = self.t

        if self.attn_type == "causal":
            K = k[:, 0 : t + 1, :]  # (t, num_heads, embed_per_head)
            V = v[:, 0 : t + 1, :]  # (t, num_heads, embed_per_head)
            assert K.shape == (
                t + 1,
                self.num_heads,
                self.embed_size_per_head,
            ), f"K.shape: {K.shape}"
        else:
            K = k[
                :, ie.max(0, (t - self.window_size) + 1) : t + 1, :
            ]  # (w, num_heads, embed_per_head)
            V = v[:, ie.max(0, (t - self.window_size) + 1) : t + 1, :]

        Kt = K.permute((1, 2, 0))  # (num_heads, embed_per_head, t)
        qk = (
            q.unsqueeze(1) @ Kt
        )  # (num_heads, 1, embed_per_head) @ (num_heads, embed_per_head, t) -> (num_heads, 1, t)
        # assert qk.shape == (self.num_heads, t+1), f"qk.shape: {qk.shape}"
        qk = qk / math.sqrt(self.embed_size)
        qk = RecurrentTensor.softmax(qk, dim=1)  # (num_heads, 1, t)

        o = qk.unsqueeze(1) @ V.permute(
            (1, 0, 2)
        )  # (num_heads, 1, t) @ (num_heads,t, embed_per_head) -> (num_heads, 1, embed_per_head)
        assert o.shape == (self.num_heads, self.embed_size_per_head), f"o.shape: {o.shape}"

        a = o.flatten() @ self.W_O  # (embed_size)
        assert a.shape == (self.embed_size,)
        return a


class GPT2Block(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_size: int,
        window_size: int,
        attn_type: str,
        t: ie.Symbol,
        domain: DomainLike,
    ):
        super().__init__(domain)
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.embed_size_per_head = embed_size // num_heads
        self.window_size = window_size
        self.attn_type = attn_type
        self.t = t

        self.mha = MHA(num_heads, embed_size, window_size, attn_type, t, domain)
        self.ff1 = RecurrentTensor.rand((embed_size, embed_size * 4), domain=domain)
        self.ff2 = RecurrentTensor.rand((embed_size * 4, embed_size), domain=domain)

    def forward(self, x):
        x_start = x
        x = self.mha(x)
        x_middle = rms_norm(x + x_start)
        x = x_middle @ self.ff1
        x = RecurrentTensor.silu(x)
        x = x @ self.ff2
        x = rms_norm(x + x_middle)
        return x


class GPT2(nn.Module):
    def __init__(
        self,
        ctx: TempoContext,
        num_heads: int,
        embed_size: int,
        seq_len: int,
        window_size: int,
        attn_type: str,
        num_blocks: int,
        n: ie.Symbol,
        t: ie.Symbol,
        b: ie.Symbol,
        domain: DomainLike,
    ):
        super().__init__(domain)
        self.block = GPT2Block(num_heads, embed_size, window_size, attn_type, t, domain)
        self.n, self.t, self.b = n, t, b

    def forward(self, x):
        n, t, b = self.n, self.t, self.b

        x_in = RecurrentTensor.placeholder(x.shape, x.dtype, domain=(n, t, b))
        x_out = self.block(x_in)
        x_in[0, t, b] = x
        x_in[True] = x_out[n - 1, t, b].ident()

        last_block_x_out = x_out[n.as_bound() - 1, t, b].ident()
        return last_block_x_out


class TempoBenchRunner(BenchRunner):
    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        embed_size: int,
        seq_len: int,
        window_size: int,
        attn_type: str,
        batch_size: int,
        **kwargs,
    ):
        cfg = get_tempo_gpt2_config(**kwargs)
        ctx = TempoContext(cfg, num_dims=3)

        n, t, b = ctx.variables
        N, T, B = ctx.upper_bounds

        self.N, self.T, self.B = N, T, B
        self.batch_size, self.num_blocks, self.seq_len = batch_size, num_blocks, seq_len

        with ctx:
            self.model = GPT2(
                ctx,
                num_heads,
                embed_size,
                seq_len,
                window_size,
                attn_type,
                num_blocks,
                n,
                t,
                b,
                (n,),
            )
            s = RecurrentTensor.placeholder(
                (embed_size,), dtypes.float32, domain=(t, b), requires_grad=False
            )
            s[0, b] = 0.0
            s[t, b] = self.model(s)[t - 1, b].ident()

            # NOTE: Prevent Tempo from optimizing away the execution due to no value sinks.
            s[t, 0:B].sink_udf(lambda x: None)

        self.ctx = ctx

    def compile(self):
        self.exec = self.ctx.compile(
            {self.B: self.batch_size, self.N: self.num_blocks, self.T: self.seq_len}
        )

    def warmup(self):
        self.run()
        self.exec.reset()

    def run(self):
        self.exec.execute()
        self.exec.reset()
        self.exec.backend.sync()


def main():
    base_cfg = {
        "runner": run_bench,
        "name": "tempo_gpt2_causal_0_win0_bs4_seq8192",
        "results_path": "./results/tempo_gpt2_causal_0_win0_bs4_seq8192/",
        "use_caching_allocators": True,
        # "attn_type": "window",
        # "window_size": 512,
        "attn_type": "causal",
        "window_size": 0,
        "framework_name": "tempo",
        "batch_size": 64,
        "seq_len": 8192,
        **GPT2_SMALL_PARAMS,
        "max_bench_time_secs": 1 * 60,  # 1 minute
        "dev": "fake-gpu",
        "statify_block_size": DEFAULT_STATIFY_BLOCK_SIZE,
    }

    runner = TempoBenchRunner(**base_cfg)
    runner.compile()
    # runner.warmup()
    # runner.run()


if __name__ == "__main__":
    main()
