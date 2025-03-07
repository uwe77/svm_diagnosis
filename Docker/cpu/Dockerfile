FROM ubuntu:22.04 AS main-setup

ARG DEBIAN_FRONTEND=noninteractive

ENV SHELL=/bin/bash \
    NB_USER=uwe \
    NB_UID=1000

RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime

###################################### User Setup #####################################
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER} 

RUN echo "root:root" | chpasswd \
    && echo "${NB_USER}:111111" | chpasswd

ENV HOME=/home/${NB_USER}

###################################### Basic Packages #####################################
RUN apt-get update && \
    apt-get install -q -y --no-install-recommends \
    ccache \
    python3-numpy \
    python3-dev \
    python3-pyparsing \
    python3-serial \
    python3-tk \
    python-is-python3 \
    libpython3-stdlib \
    libtool-bin \
    python3-pip \
    sudo \
    cmake \
    lsb-release \
    libopencv-dev \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

###################################### Python Packages #####################################
RUN apt-get update && apt-get upgrade -y && \
    pip install --upgrade pip && \
    pip install \
    numpy \
    matplotlib \
    opencv-python \
    opencv-python-headless \
    joblib  \
    cvxopt

RUN echo "export DISPLAY=:0" >> /home/${NB_USER}/.bashrc && \
    echo "export MPLBACKEND=Agg" >> /home/${NB_USER}/.bashrc && \
    echo "X11 Forwarding enabled."

# Allow the user to access X11
RUN usermod -a -G video ${NB_USER}

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV HF_HUB_ENABLE_HF_TRANSFER=1
