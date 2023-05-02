import os
import sys
import argparse
import logging
import pickle
from typing import List

from ase import Atoms
import torch
import pytorch_lightning as pl
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), "/common"))
from common.utils import json2args, get_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GraphKeys:
    """Class that holds the name of the data key."""

    Lattice = "lattice"  # (B, 3, 3) shape
    PBC = "pbc"  # (B, 3) shape
    Z = "z"  # (N) shape
    Batch_idx = "batch"  # (N) shape
    Pos = "pos"  # (N, 3) shape
    Key = "key"  # (B) shape meta data


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        batch_size,
        num_workers,
        exclude_keys: List[str] = [GraphKeys.Key],
    ):
        super().__init__()
        self.save_hyperparameters("batch_size", "num_workers")
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.exclude_keys = exclude_keys

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            exclude_keys=self.exclude_keys,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            exclude_keys=self.exclude_keys,
        )


class ModelModule(pl.LightningModule):
    def __init__(
        self,
        model,
        property_name,
        batch=64,
        lr=1e-3,
        patience=10,
        factor=0.8,
        min_lr=0,
    ):
        super().__init__()
        self.model = model
        self.num_params = n_param(self.model)
        self.hparams["n_param"] = self.num_params
        self.save_hyperparameters()

        self.property_name = property_name
        self.batch = batch
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr

    def forward(self, x):
        z = x[GraphKeys.Z]
        pos = x[GraphKeys.Pos]
        batch_ind = x.get(GraphKeys.Batch_idx)
        return self.model(z, pos, batch_ind)

    # MSE
    def mse(self, pred, true):
        l = torch.nn.MSELoss()
        return l(pred, true)

    # MAE
    def mae(self, pred, true):
        l = torch.nn.L1Loss()
        return l(pred, true)

    def training_step(self, batch, batch_idx):
        self.log("train_batch_size", float(self.batch), on_step=True, on_epoch=False, logger=False)

        pred = self(batch)
        mse_loss = self.mse(pred.squeeze(-1), batch[self.property_name].squeeze(-1))
        self.log(f"train_loss", mse_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return mse_loss

    def validation_step(self, batch, batch_idx):
        self.log("val_batch_size", float(self.batch), on_step=True, on_epoch=False, logger=False)

        pred = self(batch)
        mse_loss = self.mse(pred.squeeze(-1), batch[self.property_name].squeeze(-1))
        self.log(f"val_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        mae_loss = self.mae(pred.squeeze(-1), batch[self.property_name].squeeze(-1))
        self.log(f"val_{self.property_name}_mae", mae_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return mae_loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sche = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=self.patience, factor=self.factor, min_lr=self.min_lr
        )

        return {
            "optimizer": opt,
            "lr_scheduler": sche,
            "monitor": f"val_{self.property_name}_mae",
        }


def atoms2graphdata(atoms: Atoms, key: str, prop: float, property_name: str) -> Data:
    """Helper function to convert one `Atoms` object to
    `torch_geometric.data.Data` with edge index information include pbc.

    Args:
        atoms (ase.Atoms): one atoms object.

    Returns:
        data (torch_geometric.data.Data): one Data object with edge information include pbc.
    """

    # order is "source_to_target" i.e. [index_j, index_i]
    data = Data()
    data[GraphKeys.Pos] = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    data[GraphKeys.Z] = torch.tensor(atoms.numbers, dtype=torch.long)
    # add batch dimension
    data[GraphKeys.Lattice] = torch.tensor(atoms.cell.array, dtype=torch.float32).unsqueeze(0)
    data[GraphKeys.Key] = key
    data[property_name] = torch.tensor([prop], dtype=torch.float32)
    return data


def n_param(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args: argparse.Namespace):
    save_dir = args.save_dir
    property_name = args.property_name

    # ---------- log info ----------
    logger.info("Start training DimeNet++...")
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
    # train
    tr_struct, tr_target, tr_key = get_data(idx["train"], atoms_list, prop_dict[property_name], keys_list)
    tr_data_list = [atoms2graphdata(at, k, y, property_name) for at, k, y in zip(tr_struct, tr_key, tr_target)]

    # val
    val_struct, val_target, val_key = get_data(idx["val"], atoms_list, prop_dict[property_name], keys_list)
    val_data_list = [atoms2graphdata(at, k, y, property_name) for at, k, y in zip(val_struct, val_key, val_target)]

    # test
    if idx.get("test") is not None:
        test_struct, test_target, test_key = get_data(idx["test"], atoms_list, prop_dict[property_name], keys_list)
        test_data_list = [
            atoms2graphdata(at, k, y, property_name) for at, k, y in zip(test_struct, test_key, test_target)
        ]
    else:
        test_data_list = None

    # datamodule
    datamodule = DataModule(
        train_dataset=tr_data_list,
        val_dataset=val_data_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    logger.info(f"max_z: {max_z}")
    logger.info(
        f"train: {len(tr_data_list)}, val: {len(val_data_list)}, test:{len(test_data_list) if test_data_list is not None else None}"
    )

    # ---------- setup model ----------
    logger.info("Setting up model and trainer...")
    model = torch_geometric.nn.models.DimeNetPlusPlus(
        hidden_channels=args.hidden_channels,
        out_channels=1,
        num_blocks=args.num_blocks,
        int_emb_size=args.int_emb_size,
        basis_emb_size=args.basis_emb_size,
        out_emb_channels=args.out_emb_channels,
        num_spherical=args.num_spherical,
        num_radial=args.num_radial,
        cutoff=args.cutoff,
        max_num_neighbors=args.max_num_neighbors,
        envelope_exponent=args.envelope_exponent,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        act=args.act,
    )
    model_module = ModelModule(
        model=model,
        property_name=property_name,
        batch=args.batch_size,
        lr=args.lr,
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
        min_lr=args.scheduler_min_lr,
    )
    # trainer
    monitor = f"val_{property_name}_mae"
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[
            EarlyStopping(monitor=monitor, patience=args.early_stopping_patience),
            ModelCheckpoint(f"{args.save_dir}/checkpoint", monitor=monitor, filename="best", save_last=True),
            LearningRateMonitor("epoch"),
        ],
        logger=[
            CSVLogger(save_dir, name="log"),
            WandbLogger(
                project=args.wandb_pjname,
                name=args.wandb_jobname,
            ),
        ],
        accelerator="gpu",
    )
    logger.info(f"# of parameters: {model_module.num_params}")

    # ---------- training ----------
    logger.info("Start training...")
    trainer.fit(model=model_module, datamodule=datamodule)

    # ---------- predict ----------
    if test_struct is not None:
        logger.info("Start predicting...")
        best_model = ModelModule.load_from_checkpoint(
            f"{save_dir}/checkpoint/best.ckpt",
            model=model,
            map_location="cuda",
        )
        best_model.to("cuda")

        y_test: dict[str, dict[str, float]] = {}
        for i in range(len(test_struct)):
            with torch.no_grad():
                x = test_data_list[i]
                x.to("cuda")
                y_pred = (
                    best_model.model(x[GraphKeys.Z], x[GraphKeys.Pos], x.get(GraphKeys.Batch_idx)).detach().cpu().item()
                )
            y_true = test_target[i]
            y_test[test_key[i]] = {"y_pred": y_pred, "y_true": y_true}
        with open(save_dir + "/y_test.pkl", "wb") as f:
            pickle.dump(y_test, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arg_file", type=str, default="./results/tmqm/homo/dimenet/idx0/args.json")
    cli_args = parser.parse_args()
    args = json2args(cli_args.arg_file)
    """
    Args:
        save_dir (str): path to save directory
        property_name (str): property name
        property_unit (str): property unit
        dataset_dir (str): path to dataset directory. (contains three pickle file of `atoms_list.p`, `prop_dict.p`, `keys_list.p`)
        idx_file (str): path to index file. (pickle file of dict[str, ndarray] {"train": ndarray, "val": ndarray, "test": ndarray})
        batch_size (int): batch size
        num_workers (int): number of workers
        hidden_channels (int): hidden channels
        num_blocks (int): number of interaction blocks
        int_emb_size (int): size of embedding in the interaction block
        basis_emb_size (int): size of basis embedding in the interaction block
        out_emb_channels (int): size of embedding in the output block
        num_spherical (int): number of spherical harmonics
        num_radial (int): number of radial basis
        cutoff (float): cutoff radius
        max_num_neighbors (int): maximum number of neighbors
        envelope_exponent (float): exponent of smooth cutoff
        act (str): activation function
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
