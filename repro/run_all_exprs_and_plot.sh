#! /bin/bash

# Function to print messages with decorative borders
print_section() {
    local message="$1"
    echo "###############################################################################"
    echo "# $message"
    echo "###############################################################################"
    sleep 1
}

# Cleanup function to kill Python and Ray processes
cleanup() {
    echo "Cleaning up processes..."
    ps -axu | grep python | awk '{print $2}' | xargs kill -s9 2>/dev/null || true
    ps -axu | grep ray | awk '{print $2}' | xargs kill -s9 2>/dev/null || true
    ps -axu | grep rllib | awk '{print $2}' | xargs kill -s9 2>/dev/null || true
    echo "Cleanup completed."
}

# Exit handler to ensure cleanup and popd are called
exit_handler() {
    echo "Script exiting, running cleanup..."
    cleanup
    popd
}

# Set up trap to call exit_handler on script exit
trap exit_handler EXIT

git_repo_root=$(git rev-parse --show-toplevel)
pushd $git_repo_root

# Get gpus string from nvidia-smi
gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')


# Identify the PHB GPU (best for D2H and H2D transfers)
phb_gpu=$(nvidia-smi topo -m | \
awk '$1 ~ /^GPU/ {
  for(i=1; i<=NF; i++) if($i=="PHB") {
    sub(/^GPU/,"",$1); print $1; exit
  }
}' 2>/dev/null) || echo "0"

# If phb_gpu is empty or not a number, default to 0
if ! [[ "$phb_gpu" =~ ^[0-9]+$ ]]; then
  phb_gpu=0
fi


print_section "Will run all experiments with GPUs: $gpus and PHB GPU: $phb_gpu..."

# Add the repo root to the python path if not there yet
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

start_time_ms=$(date +%s%3N)

print_section "Will now run the LM decode experiments with GPUs: $gpus and PHB GPU: $phb_gpu..."

# Run all experiments, ensuring cleanup is called after each one
python repro/sec7_2_lm_decode/run_measure_tpt.py --gpus $gpus --phbgpu $phb_gpu
cleanup
python repro/sec7_2_lm_decode/run_mem_usage.py --gpus $gpus --phbgpu $phb_gpu
cleanup
python repro/sec7_2_lm_decode/run_block_size_microbenchmark.py --gpus $gpus --phbgpu $phb_gpu
cleanup

print_section "Will now run the RL experiments with GPUs: $gpus..."
python repro/sec7_3_rl_train/run_small_to_med_scale.py --gpus $gpus --phbgpu $phb_gpu
cleanup

python repro/sec7_3_rl_train/run_large_obs.py --gpus $gpus --phbgpu $phb_gpu --skip_sys rllib
cleanup

print_section "Will now run the algo specific sched experiments with GPUs: $gpus..."
python repro/sec7_4_algo_specific_sched/run_algo_specific_sched.py --gpus $gpus --phbgpu $phb_gpu
cleanup

print_section "Experiments finished! Will now plot the results..."

# Plot the results
print_section "Will now plot the LM decode results..."

python repro/sec7_2_lm_decode/plot/plot_mem_usage.py --mode "paper"
python repro/sec7_2_lm_decode/plot/plot_block_size.py
python repro/sec7_2_lm_decode/plot/plot_gpt2_time_per_token.py

print_section "Will now plot the RL results..."

python repro/sec7_3_rl_train/plot/plot_small_to_med_scale.py
python repro/sec7_3_rl_train/plot/plot_large_obs.py

print_section "Will now plot the algo specific sched results..."
python repro/sec7_4_algo_specific_sched/plot/plot_algo_specific_sched.py

print_section "Will now run the speedup analysis from Section 7.3. Expect Tempo-JAX to be ~2x faster than CleanRL and CleanRL (C) ~70% slower than CleanRL..."
python repro/sec7_3_rl_train/plot/speedup_analysis.py

python repro/sec7_3_rl_train/plot/speedup_analysis.py --base_framework cleanrl

echo "=========================================="

end_time_ms=$(date +%s%3N)
elapsed_ms=$((end_time_ms - start_time_ms))
elapsed_h=$((elapsed_ms / 1000 / 60 / 60))
elapsed_m=$(( (elapsed_ms / 1000 / 60) % 60))
elapsed_s=$(( (elapsed_ms / 1000) % 60))


echo "All done! Took $elapsed_h hours, $elapsed_m minutes, $elapsed_s seconds"

echo ""
echo "You may copy the results from the container to the host machine by running:"
echo ""
echo "    docker cp tempo-repro:/home/tempo/tempo/results ./results"
echo ""

echo ""
echo "You may also copy the plots from the container to the host machine by running:"
echo ""
echo "    docker cp tempo-repro:/home/tempo/tempo/plots ./plots"
echo ""

echo ""
echo "You may now exit the container by running:"
echo "exit"
