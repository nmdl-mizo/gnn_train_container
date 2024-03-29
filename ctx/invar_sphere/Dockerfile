FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

COPY . /app

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

ARG GIT_USER
ARG GIT_TOKEN
ARG VERSION

RUN pip3 install \
    ase==3.* \
    numpy==1.* \
    sympy==1.* \
    scipy==1.* \
    pytorch-lightning \
    wandb \
    lmdb \
    && pip3 install \
    torch --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html \
    git+https://${GIT_USER}:${GIT_TOKEN}@github.com/ken2403/invar_spherenet.git \
    && rm -rf /root/.cache/pip

ENTRYPOINT [ "python3", "train.py" ]
