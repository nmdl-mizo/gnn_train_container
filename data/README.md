# Datasets

This directory contains the **Cubic-Perovskite dataset** and **tmQM**.

## How to use
 
### Load data and Data type
The `dataset.tar.gz` are stored in three pickle files; `atoms_list.p`, `prop_dict.p` and `keys_list.p`.

The dataset is used as follows:

```python3
import pickle

import ase

# load the structure data
with open("atoms_list.p", "rb") as f:
    atoms_list: list[ase.Atoms] = pickle.load(f)

# load the property data
with open("prop_dict.p", "rb") as f:
    # key: property name (str)
    # value: property value (list[float])
    prop_dict: dict[str, list[float]] = pickle.load(f)

# load the key data
with open("keys_list.p", "rb") as f:
    # keys_list holds arbitrary keys of each structure
    keys_list: list[str] = pickle.load(f)
```

### Index file
The index file is a pickle file that contains the split index of the dataset. The index file is used as follows:

```python3
import pickle

with open("idx0.p", "rb") as f:
    # key: split name ("train" | "val" | "test")
    # value: index (list[int])
    index: dict[str, list[int]] = pickle.load(f)
```