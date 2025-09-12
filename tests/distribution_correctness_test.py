import itertools
import math
from dataclasses import replace

import pytest
import torch
from torch import distributions as tor_dist

from tempo.api import distributions as tpo_dist
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.api.tempo_context_manager import TempoContext
from tempo.core import dtype
from tempo.core.configs import ExecutionConfig

# NOTE: We allow quite a high error rate here,
# because the sampling is random and we don't want to fail the test
# We could raise the number of trials to reduce the error rate
# but that would make the test slower
ERROR_ALLOWED = 0.15  # 10% error allowed
NUM_TRIALS = 2500


def histograms_allclose(
    histogram1: list[int], histogram2: list[int], num_trials: int, epsilon: float
) -> bool:
    """
    Compare two histograms with the same buckets to determine if they are close.

    :param histogram1: List of counts for the first histogram.
    :param histogram2: List of counts for the second histogram.
    :param num_trials: Total number of trials used to generate the histograms.
    :param epsilon: Tolerance level for the difference between corresponding buckets.
    :return: True if histograms are close within the specified epsilon, otherwise False.
    """
    # Check if histograms have the same length
    if len(histogram1) != len(histogram2):
        raise ValueError("Histograms must have the same number of buckets")

    # Compute the difference for each bucket and compare with epsilon
    total_error = 0.0

    for count1, count2 in zip(histogram1, histogram2):
        total_error += abs(count1 - count2)

    error = total_error / num_trials
    # print(f"histogram1: {histogram1}")
    # print(f"histogram2: {histogram2}")
    # print(f"error: {error}")
    return error < epsilon


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_categorical_log_prob_and_entropy(
    exec_cfg: ExecutionConfig, backend: str
) -> None:
    for i in range(0, 9):
        num_events = 10

        w = torch.arange(num_events, dtype=torch.float32).softmax(0)
        tor_cat = tor_dist.Categorical(probs=w)

        sample_torch = torch.ones((), dtype=torch.long) * i
        log_probs_torch = tor_cat.log_prob(sample_torch)
        entropy_torch = tor_cat.entropy()

        exec_cfg = replace(exec_cfg, backend=backend)
        ctx = TempoContext(exec_cfg, num_dims=0)
        with ctx:
            w = RecurrentTensor.arange(num_events, dtype=dtype.dtypes.float32).softmax(
                0
            )
            tpo_cat = tpo_dist.Categorical(probs=w)
            sample = RecurrentTensor.ones((), dtype=dtype.dtypes.default_int) * i
            log_probs = tpo_cat.log_prob(sample)
            entropy = tpo_cat.entropy()

            exec = ctx.compile({})
            exec.execute()

            log_probs_computed = exec.get_spatial_tensor_torch(log_probs.tensor_id, ())
            entropy_computed = exec.get_spatial_tensor_torch(entropy.tensor_id, ())

            assert torch.allclose(log_probs_torch, log_probs_computed)
            assert torch.allclose(entropy_torch, entropy_computed)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_categorical_sampling_correctness(
    exec_cfg: ExecutionConfig, backend: str
) -> None:
    num_events = 10

    torch_hist = [0 for _ in range(num_events)]
    tpo_hist = [0 for _ in range(num_events)]

    w = torch.arange(num_events, dtype=torch.float32).softmax(0)
    tor_cat = tor_dist.Categorical(probs=w)

    for _ in range(NUM_TRIALS):
        sample = tor_cat.sample()
        torch_hist[sample] += 1

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg, num_dims=1)
    (t,) = ctx.variables
    (T,) = ctx.upper_bounds
    with ctx:
        w_arr = RecurrentTensor.arange(num_events, dtype=dtype.dtypes.float32)
        w_arr.sink_with_ts_udf(lambda x, ts: print(f"w_arr: {x}, ts: {ts}", flush=True))
        w = w_arr.softmax()
        w.sink_with_ts_udf(lambda x, ts: print(f"w: {x}, ts: {ts}", flush=True))
        tpo_cat = tpo_dist.Categorical(probs=w, domain=(t,))
        sample = tpo_cat.sample()
        exec = ctx.compile({T: NUM_TRIALS})
    exec.execute()

    samples = exec.get_spatial_tensor_torch(
        sample.tensor_id, (slice(0, NUM_TRIALS),)
    )
    for s in samples.unbind(0):
        tpo_hist[int(s)] += 1

    assert histograms_allclose(torch_hist, tpo_hist, NUM_TRIALS, ERROR_ALLOWED)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_uniform_log_prob_and_entropy(exec_cfg: ExecutionConfig, backend: str) -> None:
    for i in range(0, 9):
        low, high = 0, 10

        tor_uniform = tor_dist.Uniform(low=low, high=high)
        # NOTE: we use pi to get interesting non-integer values
        sample = torch.ones((), dtype=torch.float32) * i * math.pi / 3
        log_prob_torch = tor_uniform.log_prob(sample)
        entropy_torch = tor_uniform.entropy()

        exec_cfg = replace(exec_cfg, backend=backend)
        ctx = TempoContext(exec_cfg)
        with ctx:
            tpo_uniform = tpo_dist.Uniform(low=low, high=high)
            sample = (
                RecurrentTensor.ones((), dtype=dtype.dtypes.float32) * i * math.pi / 3
            )
            log_prob = tpo_uniform.log_prob(sample)
            entropy = tpo_uniform.entropy()
            exec = ctx.compile({})
            exec.execute()

            log_probs_computed = exec.get_spatial_tensor_torch(log_prob.tensor_id, ())
            entropy_computed = exec.get_spatial_tensor_torch(entropy.tensor_id, ())

            assert torch.allclose(log_prob_torch, log_probs_computed)
            assert torch.allclose(entropy_torch, entropy_computed)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_uniform_sampling_correctness(exec_cfg: ExecutionConfig, backend: str) -> None:
    low, high = 0, 10
    buckets = 10

    torch_hist = [0 for _ in range(buckets)]
    tpo_hist = [0 for _ in range(buckets)]

    tor_uniform = tor_dist.Uniform(low=low, high=high)

    for _ in range(NUM_TRIALS):
        sample = tor_uniform.sample()
        bucket_idx = int(sample)
        torch_hist[bucket_idx] += 1

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg, num_dims=1)
    (t,) = ctx.variables
    (T,) = ctx.upper_bounds
    with ctx:
        tpo_uniform = tpo_dist.Uniform(low=low, high=high, domain=(t,))
        sample = tpo_uniform.sample()
        exec = ctx.compile({T: NUM_TRIALS})
        exec.execute()

        samples = exec.get_spatial_tensor_torch(
            sample.tensor_id, (slice(0, NUM_TRIALS),)
        )
        for s in samples.unbind(0):
            bucket_idx = int(s)
            tpo_hist[bucket_idx] += 1

        assert histograms_allclose(torch_hist, tpo_hist, NUM_TRIALS, ERROR_ALLOWED)


def define_normal_distribution_buckets(
    mu: float, sigma: float, num_buckets: int
) -> list[tuple[float, float]]:
    num_central_buckets = num_buckets - 2
    bucket_bounds = []

    # Add the extreme buckets
    bucket_bounds.append(
        (-float("inf"), mu - 3 * sigma)
    )  # Covering -infinity to mu - 3*sigma

    # Calculate the width of each central bucket
    central_bucket_width = (
        3 * 2 * sigma / num_central_buckets
    )  # Covering mu - sigma to mu + sigma

    # Add central buckets
    for i in range(num_central_buckets):
        lower_bound = mu - 3 * sigma + i * central_bucket_width
        upper_bound = lower_bound + central_bucket_width
        bucket_bounds.append((lower_bound, upper_bound))

    # Add the last bucket for values greater than mu + 3*sigma to infinity
    bucket_bounds.append((mu + 3 * sigma, float("inf")))

    return bucket_bounds


def get_bucket_idx(sample: float, bucket_bounds: list[tuple[float, float]]) -> int:
    for i, (lower, upper) in enumerate(bucket_bounds):
        if lower <= sample < upper:
            return i
    raise ValueError(f"Sample {sample} does not fall into any bucket")


@pytest.mark.parametrize(
    "backend,loc,scale",
    itertools.product(["torch", "jax"], [0.0, -2.0, 5.0], [0.5, 1.0]),
)
def test_normal_sampling_correctness(
    exec_cfg: ExecutionConfig, backend: str, loc: float, scale: float
) -> None:
    num_buckets = 10
    bucket_bounds = define_normal_distribution_buckets(loc, scale, num_buckets)

    torch_hist = [0 for _ in range(num_buckets)]
    tpo_hist = [0 for _ in range(num_buckets)]

    tor_uniform = tor_dist.Normal(loc=loc, scale=scale)

    for _ in range(NUM_TRIALS):
        sample = tor_uniform.sample()
        bucket_idx = get_bucket_idx(float(sample), bucket_bounds)
        torch_hist[bucket_idx] += 1

    exec_cfg = replace(exec_cfg, backend=backend, enable_vectorization=True)
    ctx = TempoContext(exec_cfg, num_dims=1)
    (t,) = ctx.variables
    (T,) = ctx.upper_bounds
    tpo_samples = []
    with ctx:
        tpo_uniform = tpo_dist.Normal(mu=loc, sigma=scale, domain=(t,))
        sample = tpo_uniform.sample()
        sample.sink_udf(lambda x: tpo_samples.append(x.reshape(())))
        exec = ctx.compile({T: NUM_TRIALS})
        exec.execute()

        for s in tpo_samples:
            bucket_idx = get_bucket_idx(float(s), bucket_bounds)
            tpo_hist[bucket_idx] += 1

        assert histograms_allclose(torch_hist, tpo_hist, NUM_TRIALS, ERROR_ALLOWED)


@pytest.mark.parametrize(
    "backend,loc,scale",
    itertools.product(["torch", "jax"], [0.0, -2.0, 5.0], [0.5, 1.0]),
)
def test_normal_log_prob_and_entropy(
    exec_cfg: ExecutionConfig, backend: str, loc: float, scale: float
) -> None:
    for i in range(0, 9):

        tor_uniform = tor_dist.Normal(loc=loc, scale=scale)
        sample = torch.ones((), dtype=torch.float32) * (i * scale / 3) + (-1) ** i * loc
        log_prob_torch = tor_uniform.log_prob(sample)
        entropy_torch = tor_uniform.entropy()

        exec_cfg = replace(exec_cfg, backend=backend)
        ctx = TempoContext(exec_cfg)
        with ctx:
            tpo_uniform = tpo_dist.Normal(mu=loc, sigma=scale)
            sample = (
                RecurrentTensor.ones((), dtype=dtype.dtypes.float32) * (i * scale / 3)
                + (-1) ** i * loc
            )
            log_prob = tpo_uniform.log_prob(sample)
            entropy = tpo_uniform.entropy()
            exec = ctx.compile({})
            exec.execute()

            log_probs_computed = exec.get_spatial_tensor_torch(log_prob.tensor_id, ())
            entropy_computed = exec.get_spatial_tensor_torch(entropy.tensor_id, ())

            assert torch.allclose(log_prob_torch, log_probs_computed)
            assert torch.allclose(entropy_torch, entropy_computed)


#if __name__ == "__main__":
#    cfg = ExecutionConfig.test_cfg()
#    cfg = replace(cfg, debug_mode=True, visualize_pipeline_stages=True)
#    test_categorical_log_prob_and_entropy(cfg, "torch")
