from __future__ import annotations

import os
import sys
import argparse
import logging
import pickle
from typing import Optional, List, Dict
import datetime

from ase import Atoms
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import schnetpack as spk
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), "/common"))
from common.utils import json2args, get_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AtomsDataModuleCustom(spk.data.AtomsDataModule):
    def __init__(
        self,
        batch_size: int,
        train_dataset: spk.data.BaseAtomsData,
        val_dataset: Optional[spk.data.BaseAtomsData] = None,
        test_dataset: Optional[spk.data.BaseAtomsData] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        # train_transforms: Optional[List[torch.nn.Module]] = None,
        # val_transforms: Optional[List[torch.nn.Module]] = None,
        # test_transforms: Optional[List[torch.nn.Module]] = None,
        num_workers: int = 8,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        data_workdir: Optional[str] = None,
        cleanup_workdir_stage: Optional[str] = "test",
        pin_memory: Optional[bool] = False,
    ):
        super().__init__(
            datapath="./dummy",
            batch_size=batch_size,
            num_train=None,
            num_val=None,
            num_test=None,
            split_file="split.npz",
            format=spk.data.AtomsDataFormat.ASE,
            load_properties=None,
            val_batch_size=batch_size,
            test_batch_size=None,
            transforms=transforms,
            train_transforms=None,
            val_transforms=None,
            test_transforms=None,
            num_workers=num_workers,
            num_val_workers=num_workers,
            num_test_workers=num_workers,
            property_units=property_units,
            distance_unit=distance_unit,
            data_workdir=data_workdir,
            cleanup_workdir_stage=cleanup_workdir_stage,
            splitting=None,
            pin_memory=pin_memory,
        )
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._test_dataset = test_dataset

    def setup(self, stage: Optional[str] = None):
        self._setup_transforms()


def n_param(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args: argparse.Namespace):
    save_dir = args.save_dir
    property_name = args.property_name
    property_unit = args.property_unit
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # ---------- log info ----------
    logger.info("Start training SchNet...")
    logger.info(f"property: {property_name} [{property_unit}]")
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
    # train
    tr_struct, tr_target, tr_key = get_data(idx["train"], atoms_list, prop_dict[property_name], keys_list)
    tr_target_dict: List[Dict] = []
    for v in tr_target:
        tr_target_dict.append({property_name: np.array([v])})

    # mean
    if args.add_mean and args.divide_by_n_atoms:
        n_atoms = torch.tensor([len(at) for at in tr_struct])
        mean = (torch.tensor(tr_target) / n_atoms).mean()
    if args.add_mean and not args.divide_by_n_atoms:
        mean = torch.tensor(tr_target).mean()
    else:
        mean = None

    # preprocessers
    if args.subtract_center_of_mass:
        transforms = [spk.transform.SubtractCenterOfMass()]
    else:
        transforms = []
    transforms += [
        spk.transform.RemoveOffsets(
            property=property_name,
            remove_mean=args.add_mean,
            remove_atomrefs=False,
            propery_mean=mean,
            is_extensive=args.is_extensive,
        ),
        spk.transform.ASENeighborList(args.cutoff),
        spk.transform.CastTo32(),
    ]
    property_units = {property_name: property_unit}
    distance_unit = "Ang"

    tr_dataset = spk.data.ASEAtomsData.create(
        f"./data/perovskite/tr_{now}.db",
        property_unit_dict={property_name: property_unit},
        atomrefs=None,
        transforms=transforms,
        distance_unit=distance_unit,
        property_units=property_units,
    )
    tr_dataset.add_systems(tr_target_dict, tr_struct)

    # val
    val_struct, val_target, val_key = get_data(idx["val"], atoms_list, prop_dict[property_name], keys_list)
    val_target_dict: List[Dict] = []
    for v in val_target:
        val_target_dict.append({property_name: np.array([v])})
    val_dataset = spk.data.ASEAtomsData.create(
        f"./data/perovskite/val_{now}.db",
        property_unit_dict={property_name: property_unit},
        atomrefs=None,
        transforms=transforms,
        distance_unit=distance_unit,
        property_units=property_units,
    )
    val_dataset.add_systems(val_target_dict, val_struct)

    # test
    if idx.get("test") is not None:
        test_struct, test_target, test_key = get_data(idx["test"], atoms_list, prop_dict[property_name], keys_list)
        test_target_dict: List[Dict] = []
        for v in test_target:
            test_target_dict.append({property_name: np.array([v])})
        test_dataset = spk.data.ASEAtomsData.create(
            f"./data/perovskite/test_{now}.db",
            property_unit_dict={property_name: property_unit},
            atomrefs=None,
            transforms=transforms,
            distance_unit=distance_unit,
            property_units=property_units,
        )
        test_dataset.add_systems(test_target_dict, test_struct)
    else:
        test_dataset = None

    # datamodule
    datamodule = AtomsDataModuleCustom(
        batch_size=args.batch_size,
        train_dataset=tr_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        num_workers=args.num_workers,
        transforms=transforms,
        distance_unit=distance_unit,
        property_units=property_units,
        pin_memory=False,
    )

    logger.info(f"max_z: {max_z}")
    logger.info(f"mean: {mean}")
    logger.info(
        f"train: {len(tr_dataset)}, val: {len(val_dataset)}, test:{len(test_dataset) if test_dataset is not None else None}"
    )

    # ---------- setup model ----------
    logger.info("Setting up model and trainer...")
    representation = spk.representation.SchNet(
        n_atom_basis=args.n_atom_basis,
        n_interactions=args.n_interactions,
        radial_basis=spk.nn.GaussianRBF(
            n_rbf=args.n_rbf,
            cutoff=args.cutoff,
            start=0.0,
            trainable=args.trainable_gaussians,
        ),
        cutoff_fn=spk.nn.cutoff.CosineCutoff(args.cutoff),
        n_filters=args.n_filters,
        shared_interactions=False,
        max_z=max_z + 1,
        activation=spk.nn.activations.shifted_softplus,
    )
    model = spk.model.NeuralNetworkPotential(
        representation,
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[
            spk.atomistic.Atomwise(
                n_in=args.n_atom_basis,
                n_out=1,
                n_hidden=None,
                n_layers=2,
                activation=F.silu,
                output_key=property_name,
                aggregation_mode="sum" if args.is_extensive else "avg",
                per_atom_output_key=None,
            )
        ],
        postprocessors=[
            spk.transform.CastTo64(),
            spk.transform.AddOffsets(
                property=property_name,
                add_mean=args.add_mean,
                add_atomrefs=False,
                is_extensive=args.is_extensive,
                propery_mean=mean,
            ),
        ],
        input_dtype_str="float32",
        do_postprocessing=True,
    )
    monitor = f"val_{property_name}_mae"
    task = spk.AtomisticTask(
        model=model,
        outputs=[
            spk.task.ModelOutput(
                name=property_name,
                loss_fn=torch.nn.MSELoss(),
                loss_weight=1.0,
                metrics={
                    "mae": torchmetrics.regression.MeanAbsoluteError(),
                    "rmse": torchmetrics.regression.MeanSquaredError(squared=False),
                },
                constraints=None,
            )
        ],
        optimizer_cls=torch.optim.Adam,
        optimizer_args={"lr": args.lr},
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args={
            "patience": args.scheduler_patience,
            "factor": args.scheduler_factor,
            "min_lr": args.scheduler_min_lr,
        },
        scheduler_monitor=monitor,
    )
    # trainer
    wandb_logger = WandbLogger(
        project=args.wandb_pjname,
        name=args.wandb_jobname,
    )
    wandb_logger.experiment.config["args"] = args
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[
            EarlyStopping(monitor=monitor, patience=args.early_stopping_patience),
            ModelCheckpoint(f"{args.save_dir}/checkpoint", monitor=monitor, filename="best", save_last=True),
            LearningRateMonitor("epoch"),
        ],
        logger=[
            CSVLogger(save_dir, name="log"),
            wandb_logger,
        ],
        accelerator="gpu",
    )
    logger.info(f"# of parameters: {n_param(task.model)}")

    # ---------- training ----------
    logger.info("Start training...")
    trainer.fit(model=task, datamodule=datamodule)

    # ---------- predict ----------
    if test_struct is not None:
        logger.info("Start predicting...")
        best_model = task.load_from_checkpoint(
            f"{save_dir}/checkpoint/best.ckpt",
            map_location="cuda",
        )
        best_model.to("cuda")

        y_test: dict[str, dict[str, float]] = {}
        for i in range(len(test_struct)):
            with torch.no_grad():
                x = test_dataset[i]
                idx_m = torch.zeros_like(x[spk.properties.Z])
                x[spk.properties.idx_m] = idx_m
                x = {k: v.to("cuda") for k, v in x.items()}
                out = best_model(x)
                y_pred = out[property_name].detach().cpu().item()
            y_true = test_target[i]
            y_test[test_key[i]] = {"y_pred": y_pred, "y_true": y_true}
        with open(save_dir + "/y_test.pkl", "wb") as f:
            pickle.dump(y_test, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arg_file", type=str, default="./results/perovskite/schnet/idx0/args.json")
    cli_args = parser.parse_args()
    args = json2args(cli_args.arg_file)
    """
    Args:
        save_dir (str): path to save directory
        property_name (str): property name
        property_unit (str): property unit
        dataset_dir (str): path to dataset directory. (contains three pickle file of `atoms_list.p`, `prop_dict.p`, `keys_list.p`)
        idx_file (str): path to index file. (pickle file of dict[str, ndarray] {"train": ndarray, "val": ndarray, "test": ndarray})
        add_mean (bool): whether to add mean
        divide_by_n_atoms (bool): whether to divide by number of atoms
        subtract_center_of_mass (bool): whether to subtract center of mass
        is_extensive (bool): whether to predict extensive property
        cutoff (float): cutoff radius
        batch_size (int): batch size
        num_workers (int): number of workers
        n_atom_basis (int): number of basis of atom embedding
        n_interactions (int): number of interaction layers
        n_rbf (int): number of radial basis functions
        trainable_gaussians (bool): whether to train gaussian width
        n_filters (int): number of filters
        lr (float): learning rate
        scheduler_factor (float): factor of scheduler
        scheduler_patience (int): patience of scheduler
        scheduler_min_lr (float): minimum learning rate of scheduler
        max_epochs (int): number of max epochs
        early_stopping_patience (int): patience of early stopping
        wandb_pjname (str): wandb project name
        wandb_jobname (str): wandb job name
    """  # noqa: E501
    wandb.login(key=os.environ["WANDB_APIKEY"])
    main(args)
