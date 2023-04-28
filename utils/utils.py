from __future__ import annotations

import argparse
import json

from ase import Atoms
from torch_geometric.data import Data


def json2args(json_path: str) -> argparse.Namespace:
    with open(json_path) as f:
        args = argparse.Namespace(**json.load(f))
    return args

class GraphKeys:
    Batch_idx = "batch"
    Edge_idx = "edge_index"  # Attributes marked with "index" are automatically incremented in batch processing
    Position = "pos"
    Z = "z"
    Lattice = "lattice"
    PBC = "pbc"
    Edge_shift = "edge_shift"
    Edge_attr = "edge_attr"

def graphdata2atoms(self, data: Data) -> Atoms:
    """Helper function to convert one `torch_geometric.data.Data` object to
    `ase.Atoms`.

    Args:
        data (torch_geometric.data.Data): one graph data object.
    Returns:
        atoms (ase.Atoms): one Atoms object.
    """
    pos = data[GraphKeys.Position].numpy()
    atom_num = data[GraphKeys.Z].numpy()
    ce = data[GraphKeys.Lattice].numpy()[0]  # remove batch dimension
    atoms = Atoms(numbers=atom_num, positions=pos, pbc=self.pbc, cell=ce)
    return atoms
