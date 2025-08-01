#! /bin/bash

# Make sure we are in the repo root
git_repo_root=$(git rev-parse --show-toplevel)
pushd $git_repo_root

# Trap to ensure popd is called on exit
trap 'popd' EXIT

# Check for uncommitted changes and pull if --pull is passed
if [[ "$1" == "--pull" ]]; then
    if ! git diff-index --quiet HEAD --; then
        echo "Stopping due to uncommitted changes which would prevent pulling. Please commit or stash your changes before running this script."
        exit 1
    fi
    git pull
fi

# Build the container
DOCKER_BUILDKIT=1 docker build  -f docker/gpu.dockerfile -t tempo-gpu .

# Run the container
docker run --name tempo-repro --gpus 'all' --ipc=host --ulimit memlock=-1:-1 -it --rm tempo-gpu bash
