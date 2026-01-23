"""
Functions that preprocess data.
"""

from __future__ import annotations

import os
import h5py
import numpy as np
from tqdm import tqdm
from pymatgen.entries.computed_entries import ComputedStructureEntry

def compress_data(cse: ComputedStructureEntry) -> dict:
    """
    Converts the ComputedStructureEntry into a dictionary that contains only the information needed for machine learning.
    We want to learn invariant quantities later. Therefore, we store the direction averaged dielectric function, i.e., Tr[eps(omega)]/3 
    and the RMS of the Drude frequency, i.e., sqrt(mean(omega_drude^2)) (because the Drude frequency is always squared in the equations).
    All arrays are converted to float32, complex64 or uint16 (depending on the data) to save disk space and reduce loading times.
    Input:
        cse:        pymatgen ComputedStructureEntry
    Output:
        data:       Dictionary with minimal information needed for machine learning
    """
    # crystal structure information
    composition = cse.reduced_formula # needed for the data split based on unique compositions
    structure = cse.structure
    lattice = structure.lattice.matrix.astype(np.float32)
    atomic_number = np.array(structure.atomic_numbers, dtype=np.uint16) # in pymatgen this is a tuple
    position = structure.cart_coords.astype(np.float32)
    
    # metadata and material properties we want to learn
    data = cse.data
    mat_id = data["mat_id"]
    e_above_hull = cse.data["e_above_hull"]
    drude = np.sqrt(np.mean(np.array(data["drude_freq"]) ** 2)).astype(np.float32) # root mean square because the Drude frequency is always squared in the equations
    eps = data["eps_inter_re"] + 1j * data["eps_inter_im"]
    eps = np.mean(eps[:, 1:], axis=1).astype(np.complex64) # the first axis is the frequency/energy grid
    
    # gather information in a dictionary
    data = {
        "mat_id": mat_id,
        "e_above_hull": e_above_hull, # good to know in general
        "composition": composition,
        "lattice": lattice,
        "atomic_number": atomic_number,
        "position": position,
        "drude": drude,
        "eps": eps,
    }
    return data

def split_dataset(
    data_list: list, 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    seed: int = 42
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Splits dataset into a training and test set.
    The split is based on unique compositions, as explained in https://www.nature.com/articles/s41586-023-06735-9.
    Input:
        data_list:      List containing the dataset, each entry is a dictionary that must contain a key called 'composition'
        train_ratio     Proportion of the dataset to use for training
        val_ratio       Proportion of the dataset to use for validation
        seed:           Seed for random shuffling to ensure reproducibility
    Output:        
        train_list:     List of dictionaries with training data
        test_list:      List of dictionaries with test data
    """
    print("Train test data splitting based on unique compositions:", flush=True)
    
    # ensure ratios sum to 1
    test_ratio = 1 - train_ratio - val_ratio
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1")

    # get a list of compositions
    compositions = [data["composition"] for data in data_list]

    # get and shuffle all unique compositions
    unique_compositions = np.unique(compositions)
    np.random.seed(seed)
    np.random.shuffle(unique_compositions)

    # determine the split
    total = len(unique_compositions)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)
    train_compositions = unique_compositions[:train_end]
    val_compositions = unique_compositions[train_end:val_end]
    test_compositions = unique_compositions[val_end:]

    # convert the unique compositions in each split into sets (results in much fast membership test, i.e., 'if x in list')
    train_compositions = set(list(train_compositions))
    val_compositions = set(list(val_compositions))
    test_compositions = set(list(test_compositions))
    
    # data split
    train_list = [data for data in data_list if data["composition"] in train_compositions]
    val_list = [data for data in data_list if data["composition"] in val_compositions]
    test_list = [data for data in data_list if data["composition"] in test_compositions]

    # log message
    print(f"Train: {len(train_list):<6d} ({100 * train_ratio:.0f}%)", flush=True)
    print(f"Val:   {len(val_list):<6d} ({100 * val_ratio:.0f}%)", flush=True)
    print(f"Test:  {len(test_list):<6d} ({100 * test_ratio:.0f}%)", flush=True)
    print(f"Total: {len(data_list):<6d}", flush=True)
    return train_list, val_list, test_list
    
def save_as_h5(data_list: list, output_dir: str, filename: str) -> None:
    """
    Saves a list of dictionaries as a h5 file.
    Each dictionary must have a key called 'mat_id'.
    (This function compresses the database into h5 files, saving a lot of disk space, but resulting in slow loading speeds...)
    Input:
        data_list:      List of dictionaries that must have a key called 'mat_id' 
        output_dir:     Path to the directory where the resulting h5 file will be stored (which is created if it does not already exist)
        filename:       Name of the resulting h5 file
    Output:
        Create a h5 file in the 'output_dir' with name 'filename' 
    """
    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(os.path.join(output_dir, f"{filename:s}.h5"), "w") as f:
        for data in tqdm(data_list, desc=f"Storing data in {os.path.join(output_dir, f'{filename:s}.h5'):s}"):
            group = f.create_group(data["mat_id"])
            for key, value in data.items():
                group.create_dataset(key, data=value)
    