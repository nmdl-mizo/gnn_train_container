import os
import sys
import argparse
import logging
import pickle
from typing import List

from ase import Atoms
import torch
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lcaonet.data import List2GraphDataset
from lcaonet.data.keys import GraphKeys
from lcaonet.model import LCAONet
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), "/common"))
from common.utils import json2args

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        batch_size,
        num_workers,
        exclude_keys: List[str] = ["key"],
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
        self.num_params = model.n_param
        self.hparams["n_param"] = self.num_params
        self.save_hyperparameters()

        self.property_name = property_name
        self.batch = batch
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr

    def forward(self, x):
        return self.model(x)

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
        mse_loss = self.mse(pred, batch[self.property_name])
        self.log(f"train_loss", mse_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return mse_loss

    def validation_step(self, batch, batch_idx):
        self.log("val_batch_size", float(self.batch), on_step=True, on_epoch=False, logger=False)

        pred = self(batch)
        mse_loss = self.mse(pred, batch[self.property_name])
        self.log(f"val_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        mae_loss = self.mae(pred, batch[self.property_name])
        self.log(f"val_{self.property_name}_mae", mae_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return mae_loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sche = {
            "name": "lr_schedule",
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=self.patience, factor=self.factor, min_lr=self.min_lr
            ),
            "monitor": f"val_{self.property_name}_mae",
        }

        return [opt], [sche]


def main(args: argparse.Namespace):
    save_dir = args.save_dir
    property_name = args.property_name

    # ---------- log info ----------
    logger.info("Start training LCAONet...")
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

    # ---------- make dataset ----------
    logger.info("Making dataset...")
    prop_only_dict = {property_name: prop_dict[property_name], "key": keys_list}
    dataset = List2GraphDataset(
        atoms_list,
        y_values=prop_only_dict,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        self_interaction=False,
        subtract_center_of_mass=args.subtract_center_of_mass,
        remove_batch_key=["key"],
    )

    # split dataset
    with open(args.idx_file, "rb") as f:
        idx = pickle.load(f)
    tr = dataset[idx["train"]]
    val = dataset[idx["val"]]
    if idx.get("test") is not None:
        test = dataset[idx["test"]]
    else:
        test = None

    # datamodule
    datamodule = DataModule(
        train_dataset=tr,
        val_dataset=val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    logger.info(f"max_z: {max_z}")
    logger.info(f"train: {len(tr)}, val: {len(val)}, test:{len(test) if test is not None else None}")

    # ---------- setup model ----------
    logger.info("Setting up model and trainer...")
    model = LCAONet(
        hidden_dim=args.hidden_dim,
        coeffs_dim=args.hidden_dim,
        conv_dim=args.conv_dim,
        out_dim=1,
        n_interaction=args.n_interaction,
        n_per_orb=args.n_per_orb,
        cutoff=None,
        rbf_type=args.rbf_type,
        cutoff_net=args.cutoff_net,
        max_z=max_z,
        max_orb=args.max_orb,
        elec_to_node=args.elec_to_node,
        add_valence=args.add_valence,
        extend_orb=args.extend_orb,
        is_extensive=args.is_extensive,
        activation=args.activation,
        weight_init=args.weight_init,
        atomref=None,
        mean=None,
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
    if test is not None:
        logger.info("Start predicting...")
        best_model = ModelModule.load_from_checkpoint(
            f"{save_dir}/checkpoint/best.ckpt",
            model=model,
            map_location="cuda",
        )
        best_model.to("cuda")

        y_test: dict[str, dict[str, float]] = {}
        for i, x in enumerate(test):
            with torch.no_grad():
                x.to("cuda")
                y_pred = best_model.model(x).detach().cpu().item()
            y_true = x[property_name].item()
            y_test[x["key"]] = {"y_pred": y_pred, "y_true": y_true}
        with open(save_dir + "/y_test.pkl", "wb") as f:
            pickle.dump(y_test, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arg_file", type=str, default="./results/perovskite/lcaonet/idx0/args.json")
    cli_args = parser.parse_args()
    args = json2args(cli_args.arg_file)
    """
    Args:
        save_dir (str): path to save directory
        property_name (str): property name
        dataset_dir (str): path to dataset directory. (contains three pickle file of `atoms_list.p`, `prop_dict.p`, `keys_list.p`)
        idx_file (str): path to index file. (pickle file of dict[str, ndarray] {"train": ndarray, "val": ndarray, "test": ndarray})
        batch_size (int): batch size
        num_workers (int): number of workers
        cutoff (float): cutoff radius
        max_neighbors (int): maximum number of neighbors
        subtract_center_of_mass (bool): whether to subtract center of mass
        hidden_dim (int): hidden dimension
        conv_dim (int): convolution dimension
        n_interaction (int): number of interaction blocks
        n_per_orb (int): number of orbitals per basis
        rbf_type (str): type of radial basis function
        cutoff_net (str): type of cutoff network
        max_orb (str): maximum orbital name
        elec_to_node (bool): whether to convert electrons to nodes
        add_valence (bool): whether to add valence orbitals
        extend_orb (bool): whether to extend orbitals
        is_extensive (bool): whether to use extensive property
        activation (str): activation function
        weight_init (str): weight initialization method
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
