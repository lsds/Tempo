
Minimum python version is 3.10.

# Environment set-up

To ensure consistent and readable styling we use black for formatting,
flake8 for linting, mypy for static type checking and follow conventional commit
messages. All of this can be triggered automatically locally with pre-commits and
checked remotely using github actions.

To set-up a dev environment with pre-commit use this:

```bash
git clone https://github.com/LSDS/Tempo
cd Tempo

# Create a virtual python env
python3.10 -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip  

#NOTE: Due to pytorch design, it must be installed seperately
#For CPU support use this:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

#For GPU support use this or cu121
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# JAX is also needed for brax envs. with CPU support
pip install -U "jax[cpu]"

#JAX for GPU. Unfortunately, it looks like JAX has dropped support for CUDA 11
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install -e ".[dev, examples, envs]"

# Run the tests to check everything works
pytest

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
```

# Development Flow

* Develop in small feature branches whenever possible.
* Use (conventional commit)[https://www.conventionalcommits.org/en/v1.0.0/#summary] messages
  * i.e. use feat:, fix:, chore:, refactor: to distinguish commit types.
* Make PRs to request merge onto main (there are branch protections on main).

Then, during development, you can run:

```
# Run the tests locally
pytest

# Run pre-commit to automatically format and run linters
pre-commit run --all-files
```
