from __future__ import annotations

import os
import sys
import argparse
import logging
import pickle

from m3gnet.models import M3GNet
from m3gnet.trainers import Trainer
from ase import Atoms
import tensorflow as tf
import wandb
from wandb.keras import WandbMetricsLogger, WandbCallback

sys.path.append(os.path.join(os.path.dirname(__file__), "/common"))
from common.utils import json2args, get_data

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
    with open(f"{args.dataset_dir}/atoms_list.p", "rb") as f:
        atoms_list: list[Atoms] = pickle.load(f)
    with open(f"{args.dataset_dir}/prop_dict.p", "rb") as f:
        prop_dict: dict[str, list[float]] = pickle.load(f)
    with open(f"{args.dataset_dir}/keys_list.p", "rb") as f:
        keys_list: list[str] = pickle.load(f)
    # max atomic number
    max_z = 0
    for at in atoms_list:
        max_z = max(max_z, max(at.numbers))

    # split dataset
    with open(args.idx_file, "rb") as f:
        idx = pickle.load(f)
    tr_struct, tr_target, tr_key = get_data(idx["train"], atoms_list, prop_dict[property_name], keys_list)
    val_struct, val_target, val_key = get_data(idx["val"], atoms_list, prop_dict[property_name], keys_list)
    if idx.get("test") is not None:
        test_struct, test_target, test_key = get_data(idx["test"], atoms_list, prop_dict[property_name], keys_list)
    else:
        test_struct = None
    logger.info(f"max_z: {max_z}")
    logger.info(
        f"train: {len(tr_struct)}, val: {len(val_struct)}, test:{len(test_struct) if test_struct is not None else None}"
    )

    # ---------- setup model ----------
    logger.info("Setting up model...")
    m3gnet = M3GNet(
        max_n=args.max_n,
        max_l=args.max_l,
        n_blocks=args.n_blocks,
        units=args.units,
        cutoff=args.cutoff,
        threebody_cutoff=args.threebody_cutoff,
        n_atom_types=max_z + 1,
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
        save_checkpoint=True,
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
    parser.add_argument("arg_file", type=str, default="./results/perovskite/m3gnet/idx0/args.json")
    cli_args = parser.parse_args()
    args = json2args(cli_args.arg_file)
    """
    Args:
        wandb_pjname (str): wandb project name
        wandb_jobname (str): wandb job name
        save_dir (str): path to save directory
        property_name (str): property name
        dataset_dir (str): path to dataset directory. (contains three pickle file of `atoms_list.p`, `prop_dict.p`, `keys_list.p`)
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
