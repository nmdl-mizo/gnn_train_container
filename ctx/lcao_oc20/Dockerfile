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

RUN pip install -U pip && \
    pip install \
    ase==3.* \
    numpy==1.* \
    sympy==1.* \
    scipy==1.* \
    pytorch-lightning \
    wandb \
    lmdb \
    torch \
    torch_geometric \
    pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html \
    git+https://${GIT_USER}:${GIT_TOKEN}@github.com/nmdl-mizo/lcaonet.git@${VERSION} \
    && rm -rf /root/.cache/pip

ENTRYPOINT [ "python3", "train.py" ]
