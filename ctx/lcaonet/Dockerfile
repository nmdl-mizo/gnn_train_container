FROM  pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

COPY . /app

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

ARG GIT_USER
ARG GIT_TOKEN

RUN pip install -U pip && pip install \
    ase==3.* \
    numpy==1.* \
    sympy==1.* \
    scipy==1.* \
    pytorch-lightning==2.* \
    wandb \
    torch_geometric \
    pyglib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html \
    git+https://${GIT_USER}:${GIT_TOKEN}@github.com/nmdl-mizo/lcaonet.git@v1.4.4 \
    && rm -rf /root/.cache/pip

ENTRYPOINT [ "python3", "train.py" ]
