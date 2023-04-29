from __future__ import annotations

import argparse
import json

from ase import Atoms


def json2args(json_path: str) -> argparse.Namespace:
    with open(json_path) as f:
        args = argparse.Namespace(**json.load(f))
    return args

def get_data(index: list[int], atoms_list: list[Atoms], prop_list: list[float], keys_list: list[str])-> tuple[list[Atoms], list[float], list[str]]:
    struct: list[Atoms] = []
    target: list[float] = []
    key: list[str] = []
    for i in index:
        struct.append(atoms_list[i])
        target.append(prop_list[i])
        key.append(keys_list[i])
    return struct, target, key
