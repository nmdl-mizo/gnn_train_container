from __future__ import annotations

import os
import argparse
import logging
import pickle

from m3gnet.models import M3GNet
from m3gnet.trainers import Trainer
from ase import Atoms
import tensorflow as tf
import wandb
from wandb.keras import WandbMetricsLogger, WandbCallback

from .common.utils import json2args
from .common.data import graphdata2atoms, GraphKeys, GraphDataset

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
        dataset: GraphDataset = pickle.load(f)
    max_z = 0
    for d in dataset:
        max_z = max(max_z, max(d[GraphKeys.Z]))

    # split dataset
    with open(args.idx_file, "rb") as f:
        idx = pickle.load(f)
    tr_struct: list[Atoms] = []
    tr_target: list[float] = []
    tr_key: list[str] = []
    for i in idx["train"]:
        tr_struct.append(graphdata2atoms(dataset[i]))
        tr_target.append(dataset[i][property_name])
        tr_key.append(dataset[i]["key"])
    val_struct: list[Atoms] = []
    val_target: list[float] = []
    val_key: list[str] = []
    for i in idx["val"]:
        val_struct.append(graphdata2atoms(dataset[i]))
        val_target.append(dataset[i][property_name])
        val_key.append(dataset[i]["key"])
    if idx.get("test") is not None:
        test_struct = []
        test_target: list[float] = []
        test_key: list[str] = []
        for i in idx["test"]:
            test_struct.append(graphdata2atoms(dataset[i]))
            test_target.append(dataset[i][property_name])
            test_key.append(dataset[i]["key"])
    else:
        test_struct = None
    logger.info(f"max_z: {max_z}")
    logger.info(f"train: {tr_struct}, val: {val_struct}, test:{test_struct}")

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
        graphs_or_structures=tr_struct,
        targets=tr_target,
        validation_graphs_or_structures=val_struct,
        validation_targets=val_target,
        loss=tf.keras.losses.MeanSquaredError(),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.CSVLogger(save_dir + "/log.csv"),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=save_dir + "/ckpt.pkl",
                monitor="val_mae",
                save_weights_only=False,
                save_best_only=True,
                mode="min",
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_mae",
                factor=args.scheduler_factor,
                patience=args.scheduler_patience,
                verbose=0,
                mode="min",
                min_lr=args.scheduler_min_lr,
            ),
            WandbCallback(save_graph=False, save_model=False, monitor="val_mae"),
            WandbMetricsLogger(),
        ],
        early_stop_patience=args.early_stop_patience,
        save_checkpoint=False,  # callback is added by myself
        verbose=1,
    )
    m3gnet.summary()
    p = m3gnet.count_params()
    logger.info(f"# of parameters: {p}")

    # ---------- predict ----------
    logger.info("Train dataset predicting...")
    y_tr: dict[str, dict[str, float]] = {}
    for i in range(len(tr_struct)):
        x = trainer.model.graph_converter(tr_struct[i])
        y_pred = trainer.model(x.as_tf().as_list()).numpy()[0][0]
        y_true = tr_target[i]
        y_tr[tr_key[i]] = {"y_pred": y_pred, "y_true": y_true}
    with open(save_dir + "/y_tr.pkl", "wb") as f:
        pickle.dump(y_tr, f)

    logger.info("Val dataset predicting...")
    y_val: dict[str, dict[str, float]] = {}
    for i in range(len(val_struct)):
        x = trainer.model.graph_converter(val_struct[i])
        y_pred = trainer.model(x.as_tf().as_list()).numpy()[0][0]
        y_true = val_target[i]
        y_val[val_key[i]] = {"y_pred": y_pred, "y_true": y_true}
    with open(save_dir + "/y_val.pkl", "wb") as f:
        pickle.dump(y_val, f)

    if test_struct is not None:
        logger.info("Test dataset predicting...")
        y_test: dict[str, dict[str, float]] = {}
        for i in range(len(test_struct)):
            x = trainer.model.graph_converter(test_struct[i])
            y_pred = trainer.model(x.as_tf().as_list()).numpy()[0][0]
            y_true = test_target[i]
            y_test[test_key[i]] = {"y_pred": y_pred, "y_true": y_true}
        with open(save_dir + "/y_test.pkl", "wb") as f:
            pickle.dump(y_test, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arg-file", type=str, default="./results/perovskite/m3gnet/idx0/args.json"
    )
    args = json2args()
    """
    Args:
        wandb_pjname (str): wandb project name
        wandb_jobname (str): wandb job name
        save_dir (str): path to save directory
        property_name (str): property name
        dataset (str): path to dataset. (GraphDataset which children of torch_geometric.data.Dataset object)
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
        name=args.wandb_jobname,
        config={"args": args},
    )
    main(args)
