FROM ubuntu:22.04 AS main-setup


ARG DEBIAN_FRONTEND=noninteractive

ENV SHELL=/bin/bash \
    NB_USER=uwe \
    NB_UID=1000

RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime


###################################### user #####################################
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER} 

RUN echo "root:root" | chpasswd \
    && echo "${NB_USER}:111111" | chpasswd

ENV HOME=/home/${NB_USER}

###################################### basic #####################################
RUN apt-get update && \
    apt-get install -q -y --no-install-recommends \
    ccache \
    g++ \
    gcc \
    gdb \
    gawk \
    make \
    python3-numpy \
    python3-dev \
    python3-pyparsing \
    python3-serial \
    python-is-python3 \
    libpython3-stdlib \
    libtool-bin \
    zip \
    python3-pip \
    vim \
    curl \
    wget \
    sudo \
    cmake \
    net-tools \
    lsb-release \
    libopencv-dev \
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

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV HF_HUB_ENABLE_HF_TRANSFER=1