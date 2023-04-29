FROM  tensorflow/tensorflow:2.12.0-gpu

COPY . /app

WORKDIR /app

RUN pip install -U pip && pip install \
    pymatgen==2022.4.19 \
    ase==3.* \
    numpy==1.* \
    sympy==1.* \
    wandb \
    m3gnet==0.2.4

ENTRYPOINT [ "python3", "train.py" ]