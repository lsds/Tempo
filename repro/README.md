# Tempo - SOSP'25 Reproducibility

* **Paper Title:** Tempo: Compiled Dynamic Deep Learning with Symbolic Dependence Graphs
* **Paper ID:** #378


Welcome, we have made a good-faith effort to reproduce all of our original results in a (hopefully) simple to use package.

## Hardware/Software Used

All experiments use the same hardware and software configuration.
We use a server with an AMD EPYC 7402P 24-core
CPU, 384 GB of DDR4 RAM (3200 MT/s) at least one NVIDIA
RTX A6000 GPU (48 GB of GDDR6 RAM, PCIe Gen4 ×16
at 32 GB/s). We run Ubuntu v22.04 with Linux kernel v5.15.

Our particular host has 4 A6000 GPUs, and our scripts are prepared to parallelize
experiments across availabel GPUs.

We use Python 3.10.
Originally, we used CUDA v12.1, PyTorch v2.5.1, and JAX v0.4.35, but have since reproduced
the results using some updated dependencies:
* CUDA v12.1 -> v12.8
* PyTorch v2.5.1 -> v2.7.1
* JAX v0.4.35 -> v0.6.2

The full locked list of dependencies file can be seen in [here](../requirements/requirements-repro.txt).

We have packaged the experiments into an easy to use docker container.
Thus, to reproduce these results, you will need access to a similar host with
docker installed (with nvidia container runtime).

Upon request, we can provide SSH access to the host on which results were gathered.

## Artifact Description

### Top-Level Project Structure

Relevant modules in Tempo relating to the claims in the paper:
```text
tempo                                               # source directory
├── api                                             # user-interface (Section 3)
├── core                                            # index expressions, tensor ops (Section 4)
├── runtime                                         # execution runtime (Section 6)
└── transformations                                 # sdg transformations (Section 4&5)
```

### Reproducibility Module Structure

We've packaged the reproducibility process into two shell scripts which should automate the
entire process. Despite this, we describe the structure of the module (shared utilities omitted).

```text
docker
└── gpu.dockerfile                           # Docker file to use in reproducibility efforts

repro                                        # Package containing all reproducibility code
├── build_run_container.sh                   # Script to build above container
├── run_all_exprs_and_plot.sh                # Runs all experiments and plots all results
├─  pinned_buffer_microbenchmark.py          # A microbenchmark discussed in Notes
│  
├── sec7_2_lm_decode/                        # Scripts for running and plotting Section 7.2's experiments
│   │  
│   ├── impls/                               # Implementations of GPT2's architecture in JAX/Torch/Tempo
│   ├── plot/                                # Plotting scripts for Section 7.2
│   │   ├── plot_gpt2_time_per_token.py      # Script to plot Figure 9 and 10
│   │   ├── plot_block_size.py               # Script to plot Figure 11
│   │   └── plot_mem_usage.py                # Script to plot Figure 12
│   │
│   ├── run_measure_tpt.py                   # Runner for time-per-token experiments (Figures 9-10)
│   ├── run_block_size_microbenchmark.py     # Runner for experiment of Figure 11
│   ├── run_mem_usage.py                     # Runner for memory usage experiments (Figure 12)
│   │
├── sec7_3_rl_train/                         # Scripts for running and plotting Section 7.3's experiments
│   │
│   ├── impls/                               # Implementations of PPO in Tempo & baselines
│   ├── plot/                                # Plotting scripts for Section 7.3
│   │   │  
│   │   ├── plot_small_to_med_scale.py       # Script to plot Figure 13
│   │   ├── plot_large_obs.py                # Script to plot Figure 14
│   │   └── speedup_analysis.py              # Additional analysis scripts for aggregate values
│   │
│   ├── run_large_obs.py                     # Runner for large observation experiments (Figure 14)
│   └── run_small_to_med_scale.py            # Runner for small-to-medium scale experiments (Figure 13)
│  
└── sec7_4_algo_specific_sched/              # Scripts for running and plotting Section 7.4's experiments
    ├── impls/                               # Algorithm-specific scheduling implementations
    ├── plot/
    │   └── plot_algo_specific_sched.py      # Script to plot Figure 15
    │
    └── run_algo_specific_sched.py           # Runner for algorithm-specific scheduling experiments (Figure 15)
```

The experiment runners store the input and configuration used in experiments in top-level constants.
By default, runners produce results in ./results, which the plotting scripts read from by default.
All scripts offer usage help with -h.


## Resources and Time Taken

End-to-end, the experiments take roughly 12 hours to complete, with few minutes of active time.
Most experiments will fully utilize the available GPUs, without CPU pressure.
However, run_large_obs.py and run_algo_specific_sched.py will also consume up to 200GB
of CPU memory due to the swapping involved.

Breakdown:
* run_block_size_microbenchmark.py: ~15 minutes
* run_measure_tpt.py: ~1 hour
* run_mem_usage.py: ~3 hours
* run_small_to_med_scale.py: ~1.5 hours
* run_large_obs.py: ~3.5 hours
* run_algo_specific_sched.py: ~1.5 hour

## Minimal Working Example

For this, you can refer to [repro/sec7_3_rl_train/impls/tempo_ppo.py](../repro/sec7_3_rl_train/impls/tempo_ppo.py).
Which can be executed as follows:

```bash

git clone https://github.com/lsds/Tempo/ tempo
cd tempo
chmod +x repro/build_run_container.sh

./repro/build_run_container.sh

# Now in container
python repro/sec7_3_rl_train/impls/tempo_ppo.py

```

## Running All Experiments

We have aimed to make this process as simple as possible:

```bash

git clone https://github.com/lsds/Tempo/ tempo
cd tempo
chmod +x repro/build_run_container.sh

./repro/build_run_container.sh

# Now in container
chmod +x repro/run_all_exprs_and_plot.sh
./repro/run_all_exprs_and_plot.sh

# Before exiting the container, in another shell, copy results out of container
docker cp tempo-repro:/home/tempo/tempo/results ./results
docker cp tempo-repro:/home/tempo/tempo/plots ./plots

# If running in our server infrastructure, you can then scp the results to your local machine
ssh -4 <HOST> "tar -c -C /home/<USER> /path/to/results | xz -c" | xz -d | tar -x
ssh -4 <HOST> "tar -c -C /home/<USER> /path/to/plots | xz -c" | xz -d | tar -x

```

## Working with LaunchLib

We developed a tiny library for parallelizing experiments across gpus.

Individual "run*.py" experiment files support passing '--gpus "0,1,2,3"' to assign gpus and
'--phbgpu 1' to indicate the GPU with best latency to CPU memory. However, note that
"run_all_exprs_and_plot.sh" should auto-detect this.

If you find that some experimental result was compromised (e.g. by another user starting a workload),
or that an experiment fails and you wish to retry it, it is sufficient to delete the results
subdirectory relating to the failed results. For example, if the GPT-2 block-size microbenchmark experiment is
compromised (for batch size=16 and statify block size=1024) and , you can simply remove ./results/gpt2_decode/block_size_microbenchmark/bs16_block1024/ and rerun the experiment. Our scripts
will skip any experiments for which directories already exist.

## Notes

### Pinned (Page-locked) memory

In the original submission, Tempo used page-locked memory buffers to accelerate CPU-to-GPU transfers.
However, we have found that under the current software configuration of our research cluster,
the latency of page-locked memory allocation has increased dramatically. This can be verified
using pinned_buffer_microbenchmark.py.

For this reason, **we have disabled page-locked memory in Tempo
by default**. This has some effect in the plots of Figures 14 and 15, as swapping becomes much more expensive.
If reproducing on a machine where pinned_buffer_microbenchmark.py shows no meaningful difference
in allocation latency, or if we come to understand the source of this latency on our machines, please enable pinned memory in Tempo by default by setting "torch_pinned_memory_enabled"
in tempo/core/configs.py to True.

Finally, we highlight that despite this, the core contribution of automatic scheduling of swap operations through
SDG augmentations, stays intact, and we are happy with the results reproduced with our main backend, JAX.


### RLlib at large scale

The RLlib baseline, in the large observation experiment, can consume all host memory,
causing the machine to fail. For this reason, we have chosen to skip RLlib for this experiment by default.
However, it can be enabled by passing --skip_sys "" to run_large_obs.py.

### Manual alignment

The plot in Figure 15a requires manual alignment to reproduce similar presentation to that in the paper.
It is unlikely that the provided default values will work for reproducers.
However, the alignment parameters are at the top-level of plot_algo_specific_sched.py and can be
tweaked if desired.

### Torch failure at 3x256x256

Under the current system configuration, Tempo's Torch backend will often fail at the largest
setting tested. We are still investigating this issue.

## Warning Messages to expect

The following can be ignored:

```bash
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2025-08-01,21:19:13,685 WARNING  [163403,jumanji_envs.py:47]: Jumanji module not found: No module named 'jumanji'. Likely not installed.
2025-08-01,21:19:13,687 WARNING  [163403,cule_envs.py:79]: CULE module not found: No module named 'torchcule'. Likely not installed.
```

The first is caused by our experiments being forced to use an outdated gym version so every baseline
is functional.
The two warnings are Tempo recognizing that certain RL environments are not available.
