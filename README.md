# GNN Train Containers

## How to use

1. build the image

    ```bash
    make build MODEL=<model name>
    ```

    Model name is the name of the context directory in `ctx/MODEL`.

2. run the trainer container

    ```bash
    make train MODEL=<model name> ARG=<path to argument file>
    ```
