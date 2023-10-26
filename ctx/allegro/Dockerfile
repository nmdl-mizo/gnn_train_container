FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

COPY . /app

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/mir-group/allegro.git && \
    cd allegro && \
    pip install -U pip && pip install . && \
    pip install wandb

ENV WANDB_API_KEY="16c37057ddfcefe79a976f4a4b9c5d4ef489fabe"

ENTRYPOINT [ "nequip-train", "hyp.yaml" ]
