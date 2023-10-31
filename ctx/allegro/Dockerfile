FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

COPY . /app

WORKDIR /app

ARG WANDB_API_KEY

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/mir-group/allegro.git && \
    cd allegro && \
    pip install -U pip && pip install . && \
    pip install wandb

ENV WANDB_API_KEY=$WANDB_API_KEY

# add hyp.yaml as a subcommand.
# example: nequip-train hyp.yaml
ENTRYPOINT [ "nequip-train" ]
