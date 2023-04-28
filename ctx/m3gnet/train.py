from __future__ import annotations

import os
import argparse
import logging
import pickle

from m3gnet.models import M3GNet
from m3gnet.trainers import Trainer
import numpy as np
import tensorflow as tf
import wandb
from wandb.keras import WandbMetricsLogger, WandbCallback

from .utils import json2args

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def main(args: argparse.Namespace):
    save_dir = args.save_dir
    property_name = args.property_name

    # ---------- log info ----------
    logger.info("Start training M3GNet...")
    logger.info(f"property: {property_name}")
    logger.info(f"args: {args}")

    # ---------- load dataset ----------
    logger.info("Loading dataset...")
    with open(args.dataset, "rb") as f:
        dataset = pickle.load(f)
    max_z = 0
    for d in dataset:
        max_z = max(max_z, max(d.z))
    with open(args.idx_file, "rb") as f:
        idx = pickle.load(f)
    d_tr = dataset[idx["train"]]
    d_val = dataset[idx["val"]]
    if idx.get("test") is not None:
        d_test = dataset[idx["test"]]
    else:
        d_test = None
    logger.info(f"max_z: {max_z}")
    logger.info(f"train: {d_tr}, val: {d_val}, test:{d_test}")

    # ---------- setup model ----------
    logger.info("Setting up model...")
    m3gnet = M3GNet(
        max_n=args.max_n,
        max_l=args.max_l,
        n_blocks=args.n_blocks,
        units=args.units,
        cutoff=args.cutoff,
        threebody_cutoff=args.threebody_cutoff,
        n_atom_type=max_z + 1,
        include_states=args.include_states,
        is_intensive=args.is_intensive,
    )
    # when using scheduler, set optimizer with compile method
    m3gnet.compile(optimizer=tf.keras.optimizers.Adam(args.lr))
    trainer = Trainer(model=m3gnet, optimizer=m3gnet.optimizer)

    # ---------- training ----------
    logger.info("Start training...")
    trainer.train(
        graphs_or_structures=d_tr[0],
        targets=d_tr[1],
        validation_graphs_or_structures=d_val[0],
        validation_targets=d_val[1],
        loss=tf.keras.losses.MeanSquaredError(),
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_metrics=[
            tf.keras.metrics.RootMeanSquaredError(),
            tf.keras.metrics.MeanAbsoluteError(),
        ],
        callbacks=[
            tf.keras.callbacks.CSVLogger(save_dir + "/log.csv"),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_mae",
                factor=args.scheduler_factor,
                patience=args.scheduler_patience,
                verbose=0,
                mode="auto",
                min_lr=args.scheduler_min_lr,
            ),
            WandbCallback(save_graph=False, save_model=False, monitor="val_mae"),
            WandbMetricsLogger(),
        ],
        early_stop_patience=args.early_stop_patience,
        verbose=1,
    )
    m3gnet.summary()
    p = m3gnet.count_params()
    logger.info(f"# of parameters: {p}")

    # ---------- predict ----------
    logger.info("Train dataset predicting...")
    y_tr: dict[str, dict[str, float]] = {}
    for i in range(len(d_tr[0])):
        x = trainer.model.graph_converter(d_tr[0][i])
        y_pred = trainer.model(x.as_tf().as_list()).numpy()[0][0]
        y_true = d_tr[1][i]
        y_tr[d_tr[2][i]] = {"y_pred": y_pred, "y_true": y_true}
    with open(save_dir + "/y_tr.pkl", "wb") as f:
        pickle.dump(y_tr, f)

    logger.info("Val dataset predicting...")
    y_val: dict[dict[str, float]] = {}
    for i in range(len(d_val[0])):
        x = trainer.model.graph_converter(d_val[0][i])
        y_pred = trainer.model(x.as_tf().as_list()).numpy()[0][0]
        y_true = d_val[1][i]
        y_val[d_val[2][i]] = {"y_pred": y_pred, "y_true": y_true}
    with open(save_dir + "/y_val.pkl", "wb") as f:
        pickle.dump(y_val, f)

    if d_test is not None:
        logger.info("Test dataset predicting...")
        y_test: dict[dict[str, float]] = {}
        for i in range(len(d_test[0])):
            x = trainer.model.graph_converter(d_test[0][i])
            y_pred = trainer.model(x.as_tf().as_list()).numpy()[0][0]
            y_true = d_test[1][i]
            y_test[d_test[2][i]] = {"y_pred": y_pred, "y_true": y_true}
        with open(save_dir + "/y_test.pkl", "wb") as f:
            pickle.dump(y_test, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arg-file", type=str, default="./results/perovskite/idx0/args.json"
    )
    args = json2args()
    """
    Args:
        wandb_pjname (str): wandb project name
        wandb_jobname (str): wandb job name
        save_dir (str): path to save directory
        property_name (str): property name
        dataset (str): path to dataset. (pickle file of ndarray[tuple(Structure | Atoms, float, str)])
        idx_file (str): path to index file. (pickle file of dict[str, ndarray] {"train": ndarray, "val": ndarray, "test": ndarray})
        max_n (int): number of radial basis
        max_l (int): number of angular basis
        n_blocks (int): number of blocks
        units (int): number of embedding dimension
        cutoff (float): cutoff radius
        threebody_cutoff (float): cutoff radius for three-body interaction
        include_states (bool): whether to include states embedding
        is_intensive (bool): whether to predict intensive property
        lr (float): learning rate
        batch_size (int): batch size
        epochs (int): number of max epochs
        scheduler_factor (float): factor of scheduler
        scheduler_patience (int): patience of scheduler
        scheduler_min_lr (float): minimum learning rate of scheduler
        early_stop_patience (int): patience of early stopping
    """  # noqa: E501

    wandb.login(key=os.environ["WANDB_APIKEY"])
    wandb.init(
        project=args.wandb_pjname,
        name=f"m3g/{args.wandb_jobname}",
        config={"args": args},
    )
    main(args)
