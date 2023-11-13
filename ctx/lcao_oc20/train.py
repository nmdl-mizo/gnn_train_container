from __future__ import annotations

import bisect
import os
import sys
import argparse
import logging
import pickle
from pathlib import Path

import lmdb
import numpy as np
import wandb
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)

import lcaonet
from lcaonet.model import LCAONet


sys.path.append(os.path.join(os.path.dirname(__file__), "/common"))
from common.utils import json2args

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
)
stdout_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)


class LmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.

    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.

    Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                    (default: :obj:`None`)
    """

    def __init__(self, config, transform=None):
        super(LmdbDataset, self).__init__()
        self.config = config

        assert not self.config.get(
            "train_on_oc20_total_energies", False
        ), "For training on total energies set dataset=oc22_lmdb"

        self.path = Path(self.config["src"])
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"

            self.metadata_path = self.path / "metadata.npz"

            self._keys, self.envs = [], []
            for db_path in db_paths:
                self.envs.append(self.connect_db(db_path))
                length = pickle.loads(
                    self.envs[-1].begin().get("length".encode("ascii"))
                )
                self._keys.append(list(range(length)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            self.metadata_path = self.path.parent / "metadata.npz"
            self.env = self.connect_db(self.path)
            self._keys = [
                f"{j}".encode("ascii") for j in range(self.env.stat()["entries"])
            ]
            self.num_samples = len(self._keys)

        # If specified, limit dataset to only a portion of the entire dataset
        # total_shards: defines total chunks to partition dataset
        # shard: defines dataset shard to make visible
        self.sharded = False
        if "shard" in self.config and "total_shards" in self.config:
            self.sharded = True
            self.indices = range(self.num_samples)
            # split all available indices into 'total_shards' bins
            self.shards = np.array_split(
                self.indices, self.config.get("total_shards", 1)
            )
            # limit each process to see a subset of data based off defined shard
            self.available_indices = self.shards[self.config.get("shard", 0)]
            self.num_samples = len(self.available_indices)

        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # if sharding, remap idx to appropriate idx of the sharded set
        if self.sharded:
            idx = self.available_indices[idx]
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pickle.loads(datapoint_pickled)
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(self._keys[idx])
            data_object = pickle.loads(datapoint_pickled)

        if self.transform is not None:
            data_object = self.transform(data_object)

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()


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
        step_size: int = 10,
        gamma: float = 0.8,
        rho: float = 0.999,
    ):
        super().__init__()
        self.model = model
        self.n_params = self.model.n_param
        self.hparams["n_params"] = self.n_params
        self.save_hyperparameters(ignore=["model"])

        self.batch = batch
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.rho = rho

        self.mse = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.log(
            "train/batch_size",
            float(self.batch),
            on_step=False,
            on_epoch=True,
            logger=False,
        )

        pred_e, pred_f = self(batch)
        mae_e = self.mae(pred_e, batch["y"])
        self.log(
            "train/mae_e",
            mae_e,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        mae_f = self.mae(pred_f, batch["forces"])
        self.log(
            "train/mae_f",
            mae_f,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        all_loss = (1 - self.rho) * mae_e + self.rho * mae_f
        self.log(
            "train/loss",
            all_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return all_loss

    def validation_step(self, batch, batch_idx):
        self.log(
            "val/batch_size",
            float(self.batch),
            on_step=False,
            on_epoch=True,
            logger=False,
        )

        pred_e, pred_f = self(batch)
        mae_e = self.mae(pred_e, batch["y"])
        self.log(
            "val/mae_e", mae_e, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        mae_f = self.mae(pred_f, batch["forces"])
        self.log(
            "val/mae_f", mae_f, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return mae_f

    def predict_step(self, batch, batch_idx):
        out = self(batch)
        return out

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sche = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=self.step_size,
            gamma=self.gamma,
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
    logger.info("Start training LCAONet...")
    logger.info(f"Version: {lcaonet.__version__}")
    logger.info(f"args: {args}")

    # ---------- make dataset ----------
    logger.info("Making dataset...")
    dataset = LmdbDataset({"src": args.dataset_dir})

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
    model = LCAONet(
        emb_size=args.emb_size,
        emb_size_coeff=args.emb_size_coeff,
        emb_size_conv=args.emb_size_conv,
        out_size=args.out_size,
        n_interaction=args.n_interaction,
        n_per_orb=args.n_per_orb,
        cutoff=args.cutoff,
        rbf_type=args.rbf_type,
        cutoff_net=args.cutoff_net,
        max_z=max_z,
        min_orb=args.min_orb,
        max_orb=args.max_orb,
        elec_to_node=args.elec_to_node,
        add_valence=args.add_valence,
        extend_orb=args.extend_orb,
        is_extensive=args.is_extensive,
        activation=args.activation,
        weight_init=args.weight_init,
        atomref=None,
        mean=None,
        regress_forces=True,
        direct_forces=True,
    )
    model_module = ModelModule(
        model=model,
        batch=args.batch_size,
        rho=args.rho,
        lr=args.lr,
        step_size=args.scheduler_step_size,
        gamma=args.scheduler_gamma,
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
        gradient_clip_val=args.grad_norm,
        max_epochs=args.max_epochs,
        callbacks=[
            EarlyStopping(
                monitor=monitor,
                patience=args.early_stopping_patience,
            ),
            ModelCheckpoint(
                f"{args.save_dir}/checkpoint",
                monitor=monitor,
                filename="best",
                save_last=True,
            ),
            LearningRateMonitor("epoch"),
        ],
        logger=[
            CSVLogger(save_dir, name="log"),
            wandb_logger,
        ],
        accelerator="gpu",
    )
    logger.info(f"# of parameters: {model_module.n_params}")

    # ---------- training ----------
    logger.info("Start training...")
    trainer.fit(model=model_module, datamodule=data_modu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arg_file", type=str, default="./results/oc20_lcao_b64/args.json"
    )
    cli_args = parser.parse_args()
    args = json2args(cli_args.arg_file)
    """
    Args:
        save_dir (str): path to save directory
        dataset_dir (str): path to dataset directory. (contains three pickle file of `atoms_list.p`, `prop_dict.p`, `keys_list.p`)
        batch_size (int): batch size
        num_workers (int): number of workers
        emb_size (int): embedding size
        emb_size_coeff (int): embedding size for coefficients
        emb_size_conv (int): embedding size for convolution
        out_size (int): output size
        n_interaction (int): number of interaction blocks
        n_per_orb (int): number of orbitals per atom
        cutoff (float): cutoff radius
        rbf_type (str): radial basis function type
        cutoff_net (str): cutoff network type
        min_orb (int): minimum number of orbitals
        max_orb (int): maximum number of orbitals
        elec_to_node (bool): whether to convert electrons to nodes
        add_valence (bool): whether to add valence
        extend_orb (bool): whether to extend orbitals
        is_extensive (bool): whether to use extensive model
        regress_forces (bool): whether to regress forces
        direct_forces (bool): whether to predict forces directly
        activation (str): activation function
        weight_init (str): weight initialization method
        align_initial_weight (bool): whether to align initial weight
        rho (float): rho of loss function
        lr (float): learning rate
        scheduler_step_size (int): step size of scheduler
        scheduler_gamma (float): gamma of scheduler
        max_epochs (int): number of max epochs
        early_stopping_patience (int): patience of early stopping
        wandb_pjname (str): wandb project name
        wandb_jobname (str): wandb job name
    """  # noqa: E501
    wandb.login(key=os.environ["WANDB_APIKEY"])
    main(args)
