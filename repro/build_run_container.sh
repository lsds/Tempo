#! /bin/bash

# Default to not mounting llama volume
MOUNT_LLAMA=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pull)
            if ! git diff-index --quiet HEAD --; then
                echo "Stopping due to uncommitted changes which would prevent pulling. Please commit or stash your changes before running this script."
                exit 1
            fi
            git pull
            shift
            ;;
        --llama32)
            MOUNT_LLAMA=true
            shift
            ;;
        *)
            echo "Unknown option $1"
            echo "Usage: $0 [--pull] [--llama32]"
            echo "  --pull: Pull latest changes from git before building"
            echo "  --llama32: Mount ~/.llama volume for llama32 experiments"
            exit 1
            ;;
    esac
done

# Make sure we are in the repo root
git_repo_root=$(git rev-parse --show-toplevel)
pushd $git_repo_root

# Trap to ensure popd is called on exit
trap 'popd' EXIT

# Build the container
DOCKER_BUILDKIT=1 docker build  -f docker/gpu.dockerfile -t tempo-gpu .

# Run the container
if [ "$MOUNT_LLAMA" = true ]; then
    echo "Mounting ~/.llama volume for llama32 experiments..."
    docker run --name tempo-repro -v ~/.llama:/home/tempo/.llama --gpus 'all' --ipc=host --ulimit memlock=-1:-1 -it --rm tempo-gpu bash
else
    echo "Running container without llama volume mount..."
    docker run --name tempo-repro --gpus 'all' --ipc=host --ulimit memlock=-1:-1 -it --rm tempo-gpu bash
fi
