# syntax=docker/dockerfile:1.7-labs
# Enable BuildKit

FROM nvidia/cuda:12.8.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}

##############################################################################
# Installation/Basic Utilities
##############################################################################
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        curl wget vim tmux less unzip \
        htop iftop iotop ca-certificates \
        rsync iputils-ping net-tools sudo \
        llvm-dev \
        graphviz \
        pkg-config libcairo2-dev
##############################################################################
# Installation Latest Git
##############################################################################
RUN add-apt-repository ppa:git-core/ppa -y && \
        apt-get update && \
        apt-get install -y git && \
        git --version

##############################################################################
# Python
##############################################################################
ENV DEBIAN_FRONTEND=noninteractive

ENV PY_VERSION=3.10

# Install dependencies and Python
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    #python${PY_VERSION} python${PY_VERSION}-dev python${PY_VERSION}-distutils \
    python${PY_VERSION} python${PY_VERSION}-dev python${PY_VERSION}-venv \
    && rm -f /usr/bin/python && ln -s /usr/bin/python${PY_VERSION} /usr/bin/python \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python get-pip.py && rm get-pip.py \
    && pip install --upgrade pip \
    && python -V && pip -V

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

##############################################################################
# PyTorch
##############################################################################

# NOTE: More recent pytorch version may provide better performance for tempo torch backend and
# for the baselines.
ENV PYTORCH_VERSION=2.7.1
ENV TORCHVISION_VERSION=0.22.1
RUN pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION}  --index-url https://download.pytorch.org/whl/cu128

## NOTE: Original submission used this version of pytorch.
#ENV PYTORCH_VERSION=2.5.1
#ENV TORCHVISION_VERSION=0.20.1
#RUN pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION}  --index-url https://download.pytorch.org/whl/cu121

#https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp310-cp310-linux_x86_64.whl#sha256=92af92c569de5da937dd1afb45ecfdd598ec1254cf2e49e3d698cb24d71aae14
#https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp310-cp310-linux_x86_64.whl#sha256=92af92c569de5da937dd1afb45ecfdd598ec1254cf2e49e3d698cb24d71aae14

##############################################################################
# Jax
##############################################################################
RUN pip install -U "jax[cuda12]"

##############################################################################
## Add tempo user
###############################################################################
RUN useradd --create-home --uid 1000 --shell /bin/bash tempo
RUN usermod -aG sudo tempo
RUN echo "tempo ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Switch to non-root user
USER tempo
# Add Python user directory to PATH for tempo user
ENV PATH="/home/tempo/.local/bin:$PATH"

##############################################################################
# Tempo
##############################################################################

ENV STAGE_DIR=/home/tempo

#Copy the repository to the container
RUN mkdir -p ${STAGE_DIR}/tempo

#NOTE: copy over the requirements dir only, so we cache the libraries
COPY --chown=tempo:tempo ./requirements/ ${STAGE_DIR}/tempo/requirements/

# To install Tempo's dependencies with no constraints, uncomment the following lines:
#RUN pip install --no-cache-dir \
#    -r ${STAGE_DIR}/tempo/requirements/requirements.txt \
#    -r ${STAGE_DIR}/tempo/requirements/requirements-dev.txt \
#    -r ${STAGE_DIR}/tempo/requirements/requirements-examples.txt \
#    -r ${STAGE_DIR}/tempo/requirements/requirements-envs.txt \
#    -r ${STAGE_DIR}/tempo/requirements/requirements-llm.txt

# To install the baselines with no constraints, uncomment the following line:
#RUN pip install --no-deps --no-cache-dir -r ${STAGE_DIR}/tempo/requirements/requirements-baselines.txt

# For now, we install the frozen reproducibility requirements.
RUN pip install --no-deps --no-cache-dir -r ${STAGE_DIR}/tempo/requirements/requirements-repro.txt






#NOTE: finally, copy over our source code and install tempo itself
COPY --chown=tempo:tempo . ${STAGE_DIR}/tempo
RUN cd ${STAGE_DIR}/tempo && pip install  -e "."

#NOTE: add repro/ to the python path
ENV PYTHONPATH="${PYTHONPATH}:/home/tempo/tempo"

WORKDIR ${STAGE_DIR}/tempo
