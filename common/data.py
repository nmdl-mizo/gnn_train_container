from __future__ import annotations

from ase import Atoms
from torch_geometric.data import Data, Dataset


class GraphKeys:
    Lattice = "lattice"  # (B, 3, 3) shape
    PBC = "pbc"  # (B, 3) shape

    Batch_idx = "batch"  # (N) shape
    Z = "z"  # (N) shape
    Position = "pos"  # (N, 3) shape

    Edge_idx = "edge_index"  # (2, E) shape
    Edge_shift = "edge_shift"  # (E, 3) shape


class GraphDataset(Dataset):
    def __init__(self, data_list: list[Data]):
        self._data_list = data_list

    def get(self, idx: int) -> Data:
        return self._data_list[idx]

    def len(self):
        return len(self._data_list)


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
