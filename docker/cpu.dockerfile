# syntax=docker/dockerfile:1.7-labs
# Enable BuildKit

FROM ubuntu:20.04

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
        graphviz

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

ENV PY_VERSION=3.12

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

##############################################################################
# PyTorch
##############################################################################

ENV PYTORCH_VERSION=2.7.1
ENV TORCHVISION_VERSION=0.22.1
RUN pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION}  --index-url https://download.pytorch.org/whl/cpu

##############################################################################
# Jax
##############################################################################
ENV JAX_VERSION=0.6.2
RUN pip install -U "jax[cpu]==${JAX_VERSION}"

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
RUN pip install \
    -r ${STAGE_DIR}/tempo/requirements/requirements.txt \
    -r ${STAGE_DIR}/tempo/requirements/requirements-dev.txt \
    -r ${STAGE_DIR}/tempo/requirements/requirements-examples.txt \
    -r ${STAGE_DIR}/tempo/requirements/requirements-envs.txt \
    -r ${STAGE_DIR}/tempo/requirements/requirements-llm.txt \
    -r ${STAGE_DIR}/tempo/requirements/requirements-baselines.txt

# FIX six version issue
RUN pip install --upgrade --user six



#NOTE: finally, copy over our source code and install tempo itself
COPY --chown=tempo:tempo . ${STAGE_DIR}/tempo
RUN cd ${STAGE_DIR}/tempo && pip install -e "."

RUN mkdir -p ${STAGE_DIR}/bind_mnt/results_llama

WORKDIR ${STAGE_DIR}/tempo
