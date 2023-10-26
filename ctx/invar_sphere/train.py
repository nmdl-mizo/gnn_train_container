import os
import sys
import argparse
import logging

import torch
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

import invarsphere
from invarsphere.model import InvarianceSphereNet
from invarsphere.data.dataset import GraphDataset
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
        batch_size: int = 64,
        num_workers: int = 6,
        exclude_keys: list[str] = ["key"],
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
        batch: int = 64,
        lr: float = 1e-3,
        patience: int = 10,
        factor: float = 0.8,
        min_lr: float = 1e-6,
        rho: float = 0.999,
    ):
        super().__init__()
        self.model = model
        self.n_params = self.model.n_param
        self.hparams["n_params"] = self.n_params
        self.save_hyperparameters(ignore=["model"])

        self.batch = batch
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.rho = rho

        self.mse = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.log("train/batch_size", float(self.batch), on_step=False, on_epoch=True, logger=False)

        pred_e, pred_f = self(batch)
        mae_e = self.mae(pred_e, batch["energy"])
        self.log("train/mae_e", mae_e, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        mse_f = self.mse(pred_f, batch["forces"])
        self.log("train/mse_f", mse_f, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        all_loss = (1 - self.rho) * mae_e + self.rho * mse_f
        self.log("train/loss", all_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return all_loss

    def validation_step(self, batch, batch_idx):
        self.log("val/batch_size", float(self.batch), on_step=False, on_epoch=True, logger=False)

        pred_e, pred_f = self(batch)
        mae_e = self.mae(pred_e, batch["energy"])
        self.log("val/mae_e", mae_e, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        mae_f = self.mae(pred_f, batch["forces"])
        self.log("val/mae_f", mae_f, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return mae_f

    def predict_step(self, batch, batch_idx):
        out = self(batch)
        return out

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sche = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": sche,
            "monitor": "val/mae_f",
        }


def main(args: argparse.Namespace):
    save_dir = args.save_dir
    pl.seed_everything(42)

    # ---------- log info ----------
    logger.info("Start training InvarSphereNet...")
    logger.info(f"Version: {invarsphere.__version__}")
    logger.info(f"args: {args}")

    # ---------- make dataset ----------
    logger.info("Making dataset...")
    dataset = GraphDataset(
        save_dir=args.data_dir,
        inmemory=False,
    )

    # split dataset
    max_z = 90
    tr, val = torch.utils.data.random_split(dataset, [1800000, 200000])
    logger.info(f"max_z: {max_z}")
    logger.info(f"tr: {len(tr)}")
    logger.info(f"val: {len(val)}")

    # datamodule
    data_modu = DataModule(
        train_dataset=tr,
        val_dataset=val,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    logger.info(f"max_z: {max_z}")
    logger.info(f"train: {len(tr)}, val: {len(val)}")

    # ---------- setup model ----------
    logger.info("Setting up model and trainer...")
    model = InvarianceSphereNet(
        emb_size_atom=args.emb_size_atom,
        emb_size_edge=args.emb_size_edge,
        emb_size_rbf=args.emb_size_rbf,
        emb_size_cbf=args.emb_size_cbf,
        emb_size_sbf=args.emb_size_sbf,
        emb_triplet=args.emb_triplet,
        emb_quad=args.emb_quad,
        n_blocks=args.n_blocks,
        n_targets=args.n_targets,
        max_n=args.max_n,
        max_l=args.max_l,
        rbf_smooth=args.rbf_smooth,
        triplets_only=args.triplets_only,
        nb_only=args.nb_only,
        cutoff=args.cutoff,
        cutoff_net="envelope",
        cutoff_kwargs={"p": args.p},
        n_residual_output=args.n_residual_output,
        max_z=max_z,
        extensive=args.extensive,
        regress_forces=args.regress_forces,
        direct_forces=args.direct_forces,
        activation=args.activation,
        weight_init=args.weight_init,
        align_initial_weight=args.align_initial_weight,
        scale_file=f"{args.save_dir}/scale_factors.json",
        # scale_file=None,
    )
    model_module = ModelModule(
        model=model,
        batch=args.batch_size,
        rho=args.rho,
        lr=args.lr,
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
        min_lr=args.scheduler_min_lr,
    )
    logger.info("MODEL:")
    logger.info(model_module.model)

    # trainer
    monitor = "val/mae_f"
    wandb_logger = WandbLogger(
        project=args.wandb_pjname,
        name=args.wandb_jobname,
    )
    wandb_logger.experiment.config.update(args)
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
    logger.info(f"# of parameters: {model_module.num_params}")

    # ---------- training ----------
    logger.info("Start training...")
    trainer.fit(model=model_module, datamodule=data_modu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arg_file", type=str, default="./results/oc20_invarsphere/args.json")
    cli_args = parser.parse_args()
    args = json2args(cli_args.arg_file)
    """
    Args:
        save_dir (str): path to save directory
        dataset_dir (str): path to dataset directory. (contains three pickle file of `atoms_list.p`, `prop_dict.p`, `keys_list.p`)
        batch_size (int): batch size
        num_workers (int): number of workers
        emb_size_atom (int): embedding size of atom
        emb_size_edge (int): embedding size of edge
        emb_size_rbf (int): embedding size of radial basis function
        emb_size_cbf (int): embedding size of cosine basis function
        emb_size_sbf (int): embedding size of spherical basis function
        emb_triplet (int): embedding size of triplet
        emb_quad (int): embedding size of quadruplet
        n_blocks (int): number of blocks
        n_targets (int): number of targets
        max_n (int): number of radial basis
        max_l (int): number of spherical basis
        rbf_smooth (float): smooth of radial basis
        triplets_only (bool): whether use only triplets
        nb_only (bool): whether use only neighbor
        cutoff (float): cutoff radius
        p (float): p of cutoff function
        n_residual_output (int): number of residual output
        extensive (bool): whether to predict extensive property
        regress_forces (bool): whether to regress forces
        direct_forces (bool): whether to predict forces directly
        activation (str): activation function
        weight_init (str): weight initialization method
        align_initial_weight (bool): whether to align initial weight
        rho (float): rho of loss function
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
