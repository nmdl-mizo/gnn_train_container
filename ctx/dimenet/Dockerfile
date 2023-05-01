FROM  pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

COPY . /app

WORKDIR /app

RUN pip install -U pip && pip install \
    ase==3.* \
    numpy==1.* \
    sympy==1.* \
    scipy==1.* \
    pytorch-lightning==2.* \
    wandb \
    torch_geometric \
    pyglib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html


ENTRYPOINT [ "python3", "train.py" ]