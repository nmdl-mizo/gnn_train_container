FROM  pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

COPY . /app

WORKDIR /app

RUN pip install -U pip && pip install \
    ase==3.* \
    numpy==1.* \
    sympy==1.* \
    pytorch-lightning==1.9.0 \
    wandb \
    schnetpack==2.0.0

ENTRYPOINT [ "python3", "train.py" ]