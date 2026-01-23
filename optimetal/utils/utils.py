"""
Useful function when using torch or torch_geometric.
"""

from __future__ import annotations

import warnings
from typing import Callable
from prettytable import PrettyTable

import torch

def get_device(index: int = 0) -> torch.device:
    """
    Find a proper 'torch.device' for training or inference.
    Input:
        index:      Index of the GPU to use, default is 0 (< 0 means use cpu)
    Output:
        device:     Device object from torch
    """
    if index < 0:
        device = torch.device("cpu")
        print("Using device: CPU (forced)", flush=True)
        return device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Using device: CPU (CUDA not available)", flush=True)
        return device
    n_gpus = torch.cuda.device_count()
    if index >= n_gpus:
        warnings.warn(f"Requested GPU index {index:d}, but only {n_gpus:d} device(s) detected. Falling back to the CPU.")
        device = torch.device("cpu")
        print("Using device: CPU (invalid index)", flush=True)
        return device
    device = torch.device(f"cuda:{index:d}")
    device_name = torch.cuda.get_device_name(index)
    print(f"Using device: {device} ({device_name:s})", flush=True)
    return device

def get_model_parameters(model: torch.nn.Module) -> int:
    """
    Get the number of trainable parameters in a torch model.
    """
    total_params = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            total_params += parameter.numel()
    return total_params
    
def print_model_parameters(
    model: torch.nn.Module,
    num_param_only:bool = False, 
) -> None:
    """
    Print a summary of the parameter count of each layer of a torch model.
    Input:
        model:              Machine learning model object written in torch 
        num_param_only:     If True, only print number of parameters, 
                            not the names of the layers
    """
    table = PrettyTable([f"{type(model).__name__:s} Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    if num_param_only:
        print(f"Trainable parameters: {total_params:d}", flush=True)
    else:
        print(table, f"\nTrainable parameters: {total_params:d}", flush=True)

"""
Useful functions when working with paths.
"""

import os

def listdir_full(base_dir: str) -> list[str]:
    """
    os.listdir, but the output is the full path to all items.
    """
    return [os.path.join(base_dir, file) for file in os.listdir(base_dir)]

"""
Plot functions.
"""

import re
import numpy as np
import pandas as pd
from collections import Counter
from importlib.resources import files

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

_mplstyle_loaded = False

def load_plot_style() -> None:
    """
    Load custom plot style.
    Inspired by https://github.com/garrettj403/SciencePlots.
    """
    global _mplstyle_loaded
    if _mplstyle_loaded:
        return
    style_path = files("optimetal.files").joinpath("plotstyle.mplstyle")
    plt.style.use(style_path)
    print(f"Loaded plot style: {str(style_path):s}", flush=True)
    _mplstyle_loaded = True
 
def pretty_formula(formula: str) -> str:
    """
    Latex version of composition formulas.
    This just makes numbers a subscript.
    Input:
        formula:    Material composition
    Output:
        Latex version of the formula, where all number are converted 
        to subscripts, i.e., As2O3 -> As$_2$O$_3$
    """
    return re.sub(r"(\d+)", r"$_{\1}$", formula)

def num_sites_dist(data_list: list[dict], fig_path: str | None = None) -> None:
    """
    Plots a histogram showing the distribution of the number of atoms in the unit cell.
    Input:
        data_list:      List containing the dataset, where the entries are dictionaries
        fig_path:       Path to where the plot will be saved
    Output:
        Figure saved to fig_path
    """
    # data preparation
    num_sites = [len(data["atomic_number"]) for data in data_list]
    low, high = min(num_sites), max(num_sites)
    bins = np.arange(low-0.5, high+1.5, 1)
    
    # plot
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.hist(num_sites, bins=bins, color="tab:blue", edgecolor="k", linewidth=0.75)
    ax.set_xlim(bins[0], bins[-1])  
    ax.set_xticks(np.arange(low, high + 1))
    ax.set_xlabel(r"N$_\mathrm{atoms}$")
    ax.set_ylabel(r"Occurrences")
    fig.tight_layout()
    fig.align_labels()
    if fig_path is not None:
        fig.savefig(fig_path)
        print(f"Saved a plot of the distribution of the number of atoms in the unit cell to {fig_path:s}", flush=True)
    
def periodic_table_plot(
    data_list: list[dict], 
    fig_path: str | None = None,
    percentage: bool = False,
) -> None:
    """
    Plots a periodic table highlight the percentage of each element in a dataset.
    Input:
        data_list:      List containing the dataset, where the entries are dictionaries
        fig_path:       Path to where the plot will be saved
        percentage:     Show the occurrence of individual elements in percent
    Output:
        Figure saved to fig_path
    """
    # list generated by ChatGPT (couldn't be bothered...)
    elements = [
        {"Symbol": "H",  "Name": "Hydrogen",   "AtomicNumber": 1,  "Group": 1,  "Period": 1},
        {"Symbol": "He", "Name": "Helium",     "AtomicNumber": 2,  "Group": 18, "Period": 1},
        {"Symbol": "Li", "Name": "Lithium",    "AtomicNumber": 3,  "Group": 1,  "Period": 2},
        {"Symbol": "Be", "Name": "Beryllium",  "AtomicNumber": 4,  "Group": 2,  "Period": 2},
        {"Symbol": "B",  "Name": "Boron",      "AtomicNumber": 5,  "Group": 13, "Period": 2},
        {"Symbol": "C",  "Name": "Carbon",     "AtomicNumber": 6,  "Group": 14, "Period": 2},
        {"Symbol": "N",  "Name": "Nitrogen",   "AtomicNumber": 7,  "Group": 15, "Period": 2},
        {"Symbol": "O",  "Name": "Oxygen",     "AtomicNumber": 8,  "Group": 16, "Period": 2},
        {"Symbol": "F",  "Name": "Fluorine",   "AtomicNumber": 9,  "Group": 17, "Period": 2},
        {"Symbol": "Ne", "Name": "Neon",       "AtomicNumber": 10, "Group": 18, "Period": 2},
        {"Symbol": "Na", "Name": "Sodium",     "AtomicNumber": 11, "Group": 1,  "Period": 3},
        {"Symbol": "Mg", "Name": "Magnesium",  "AtomicNumber": 12, "Group": 2,  "Period": 3},
        {"Symbol": "Al", "Name": "Aluminum",   "AtomicNumber": 13, "Group": 13, "Period": 3},
        {"Symbol": "Si", "Name": "Silicon",    "AtomicNumber": 14, "Group": 14, "Period": 3},
        {"Symbol": "P",  "Name": "Phosphorus", "AtomicNumber": 15, "Group": 15, "Period": 3},
        {"Symbol": "S",  "Name": "Sulfur",     "AtomicNumber": 16, "Group": 16, "Period": 3},
        {"Symbol": "Cl", "Name": "Chlorine",   "AtomicNumber": 17, "Group": 17, "Period": 3},
        {"Symbol": "Ar", "Name": "Argon",      "AtomicNumber": 18, "Group": 18, "Period": 3},
        {"Symbol": "K",  "Name": "Potassium",  "AtomicNumber": 19, "Group": 1,  "Period": 4},
        {"Symbol": "Ca", "Name": "Calcium",    "AtomicNumber": 20, "Group": 2,  "Period": 4},
        {"Symbol": "Sc", "Name": "Scandium",   "AtomicNumber": 21, "Group": 3,  "Period": 4},
        {"Symbol": "Ti", "Name": "Titanium",   "AtomicNumber": 22, "Group": 4,  "Period": 4},
        {"Symbol": "V",  "Name": "Vanadium",   "AtomicNumber": 23, "Group": 5,  "Period": 4},
        {"Symbol": "Cr", "Name": "Chromium",   "AtomicNumber": 24, "Group": 6,  "Period": 4},
        {"Symbol": "Mn", "Name": "Manganese",  "AtomicNumber": 25, "Group": 7,  "Period": 4},
        {"Symbol": "Fe", "Name": "Iron",       "AtomicNumber": 26, "Group": 8,  "Period": 4},
        {"Symbol": "Co", "Name": "Cobalt",     "AtomicNumber": 27, "Group": 9,  "Period": 4},
        {"Symbol": "Ni", "Name": "Nickel",     "AtomicNumber": 28, "Group": 10, "Period": 4},
        {"Symbol": "Cu", "Name": "Copper",     "AtomicNumber": 29, "Group": 11, "Period": 4},
        {"Symbol": "Zn", "Name": "Zinc",       "AtomicNumber": 30, "Group": 12, "Period": 4},
        {"Symbol": "Ga", "Name": "Gallium",    "AtomicNumber": 31, "Group": 13, "Period": 4},
        {"Symbol": "Ge", "Name": "Germanium",  "AtomicNumber": 32, "Group": 14, "Period": 4},
        {"Symbol": "As", "Name": "Arsenic",    "AtomicNumber": 33, "Group": 15, "Period": 4},
        {"Symbol": "Se", "Name": "Selenium",   "AtomicNumber": 34, "Group": 16, "Period": 4},
        {"Symbol": "Br", "Name": "Bromine",    "AtomicNumber": 35, "Group": 17, "Period": 4},
        {"Symbol": "Kr", "Name": "Krypton",    "AtomicNumber": 36, "Group": 18, "Period": 4},
        {"Symbol": "Rb", "Name": "Rubidium",   "AtomicNumber": 37, "Group": 1,  "Period": 5},
        {"Symbol": "Sr", "Name": "Strontium",  "AtomicNumber": 38, "Group": 2,  "Period": 5},
        {"Symbol": "Y",  "Name": "Yttrium",    "AtomicNumber": 39, "Group": 3,  "Period": 5},
        {"Symbol": "Zr", "Name": "Zirconium",  "AtomicNumber": 40, "Group": 4,  "Period": 5},
        {"Symbol": "Nb", "Name": "Niobium",    "AtomicNumber": 41, "Group": 5,  "Period": 5},
        {"Symbol": "Mo", "Name": "Molybdenum", "AtomicNumber": 42, "Group": 6,  "Period": 5},
        {"Symbol": "Tc", "Name": "Technetium", "AtomicNumber": 43, "Group": 7,  "Period": 5},
        {"Symbol": "Ru", "Name": "Ruthenium",  "AtomicNumber": 44, "Group": 8,  "Period": 5},
        {"Symbol": "Rh", "Name": "Rhodium",    "AtomicNumber": 45, "Group": 9,  "Period": 5},
        {"Symbol": "Pd", "Name": "Palladium",  "AtomicNumber": 46, "Group": 10, "Period": 5},
        {"Symbol": "Ag", "Name": "Silver",     "AtomicNumber": 47, "Group": 11, "Period": 5},
        {"Symbol": "Cd", "Name": "Cadmium",    "AtomicNumber": 48, "Group": 12, "Period": 5},
        {"Symbol": "In", "Name": "Indium",     "AtomicNumber": 49, "Group": 13, "Period": 5},
        {"Symbol": "Sn", "Name": "Tin",        "AtomicNumber": 50, "Group": 14, "Period": 5},
        {"Symbol": "Sb", "Name": "Antimony",   "AtomicNumber": 51, "Group": 15, "Period": 5},
        {"Symbol": "Te", "Name": "Tellurium",  "AtomicNumber": 52, "Group": 16, "Period": 5},
        {"Symbol": "I",  "Name": "Iodine",     "AtomicNumber": 53, "Group": 17, "Period": 5},
        {"Symbol": "Xe", "Name": "Xenon",      "AtomicNumber": 54, "Group": 18, "Period": 5},
        {"Symbol": "Cs", "Name": "Cesium",     "AtomicNumber": 55, "Group": 1,  "Period": 6},
        {"Symbol": "Ba", "Name": "Barium",     "AtomicNumber": 56, "Group": 2,  "Period": 6},
        {"Symbol": "La", "Name": "Lanthanum",  "AtomicNumber": 57, "Group": 3,  "Period": 6},
        {"Symbol": "Hf", "Name": "Hafnium",    "AtomicNumber": 72, "Group": 4,  "Period": 6},
        {"Symbol": "Ta", "Name": "Tantalum",   "AtomicNumber": 73, "Group": 5,  "Period": 6},
        {"Symbol": "W",  "Name": "Tungsten",   "AtomicNumber": 74, "Group": 6,  "Period": 6},
        {"Symbol": "Re", "Name": "Rhenium",    "AtomicNumber": 75, "Group": 7,  "Period": 6},
        {"Symbol": "Os", "Name": "Osmium",     "AtomicNumber": 76, "Group": 8,  "Period": 6},
        {"Symbol": "Ir", "Name": "Iridium",    "AtomicNumber": 77, "Group": 9,  "Period": 6},
        {"Symbol": "Pt", "Name": "Platinum",   "AtomicNumber": 78, "Group": 10, "Period": 6},
        {"Symbol": "Au", "Name": "Gold",       "AtomicNumber": 79, "Group": 11, "Period": 6},
        {"Symbol": "Hg", "Name": "Mercury",    "AtomicNumber": 80, "Group": 12, "Period": 6},
        {"Symbol": "Tl", "Name": "Thallium",   "AtomicNumber": 81, "Group": 13, "Period": 6},
        {"Symbol": "Pb", "Name": "Lead",       "AtomicNumber": 82, "Group": 14, "Period": 6},
        {"Symbol": "Bi", "Name": "Bismuth",    "AtomicNumber": 83, "Group": 15, "Period": 6},
        {"Symbol": "Po", "Name": "Polonium",   "AtomicNumber": 84, "Group": 16, "Period": 6},
        {"Symbol": "At", "Name": "Astatine",   "AtomicNumber": 85, "Group": 17, "Period": 6},
        {"Symbol": "Rn", "Name": "Radon",      "AtomicNumber": 86, "Group": 18, "Period": 6},
    ]
    
    # count the number of elements in the dataset
    df = pd.DataFrame(elements)
    counter = Counter()
    for data in data_list:
        counter.update(data["atomic_number"])
    norm = []
    
    # normalize the counts for the colormap
    count_list = []
    for key in counter:
        count_list.append(counter[key])
    norm = np.sum(count_list)
    counts = {}
    for key in counter:
        counts[key] = [counter[key], counter[key] / norm * 100]
        
    # plot the periodic table
    fig, ax = plt.subplots(figsize=(6, 3))
    if percentage:
        cnorm = Normalize(vmin=0, vmax=np.ceil( 100 * np.max(count_list) / norm))
    else:
        cnorm = Normalize(vmin=0, vmax=np.ceil(np.max(count_list)))
    cmap = plt.get_cmap("Blues")
    box_size = 0.9 
    for _, element in df.iterrows():
        xpos = element["Group"]
        ypos = element["Period"]
        try:
            if percentage:
                color = cmap(cnorm(counts[element["AtomicNumber"]][1]), alpha=0.65)
            else:
                color = cmap(cnorm(counts[element["AtomicNumber"]][0]), alpha=0.65)
        except:
            color = cmap(0, alpha=0.65)
        ax.add_patch(
            plt.Rectangle(
                (xpos - box_size / 2, ypos - box_size / 2),  # Position adjusted for smaller box
                box_size,
                box_size,
                color=color,
                ec="black",
                linewidth=0.5,
            )
        )
        ax.text(xpos, ypos + 0.02, element["Symbol"], ha="center", va="center", fontweight="bold", fontsize=8)
        ax.text(xpos, ypos - 0.25, element["AtomicNumber"], ha="center", va="center", fontsize=6)
        try:
            if percentage:
                ax.text(xpos, ypos + 0.3, f"{counts[element['AtomicNumber']][1]:.2f}%", ha="center", va="center", fontsize=6)
            else:
                ax.text(xpos, ypos + 0.3, f"{counts[element['AtomicNumber']][0]:d}", ha="center", va="center", fontsize=6)
        except:
            ax.text(xpos, ypos + 0.3, "0.00", ha="center", va="center", fontsize=6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, 
        ax=ax, 
        location="top",
        shrink=0.50,
        pad=-0.05,
        aspect=30,
    )
    if percentage:
        cbar.set_label(r"Percentage of atomic appearance (\%)", fontsize=8, labelpad=7)
    else:
        cbar.set_label(r"Occurrence", fontsize=8, labelpad=7)
    cbar.ax.tick_params(direction="out", length=3, width=1, labelsize=8)
    cbar.outline.set_linewidth(1)
    ax.set_xlim(0.5, 18.5)
    ax.set_ylim(6.5, 0.5)
    labels = np.arange(1, 19)
    for i in [1, 18]:
        ax.annotate(
            labels[i-1],
            xy=(i, 0.825),
            xycoords=("data", "figure fraction"),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    for i in [2, 13, 14, 15, 16, 17]:
        ax.annotate(
            labels[i-1],
            xy=(i, 0.69),
            xycoords=("data", "figure fraction"),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    for i in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        ax.annotate(
            labels[i-1],
            xy=(i, 0.425),
            xycoords=("data", "figure fraction"),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    ax.grid(False)
    ax.tick_params(left=False, bottom=False, labelleft=True, labelbottom=False, labeltop=False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.align_labels()
    if fig_path is not None:
        fig.savefig(fig_path)
        print(f"Saved periodic table plot to {fig_path:s}", flush=True)
    
"""
Helper functions and variables to validate or print certain inputs.
"""

import os
import stat
import json
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

# list of supported activation functions
ACTIVATIONS = ["relu", "leakyrelu", "silu", "celu", "gelu", "elu", "mish", "softplus", "sigmoid", "tanh"]

class ValidateConfigurationDict(BaseModel):
    """
    Ensure that only valid configuration dictionaries are parsed for training.
    The architecture, optimizer, and learning rate scheduler are validated
    in their respective factory or model definition.
    """
    
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(..., ge=0)
    trial_dir: str
    num_train_data: int = Field(..., ge=0)
    batch_size: int = Field(..., gt=0)
    architecture: dict
    optimizer: dict
    loss_fn_eps: Literal["mae", "mse", "optimate"]
    loss_fn_drude: Literal["mae", "mse"]
    lr_scheduler: dict
    warmup_epochs: int = Field(default=0, ge=0)
    grad_clip: float = Field(default=100.0, gt=0.0)
    eps_weight: float = Field(default=1.0, ge=0.0)
    drude_weight: float = Field(default=1.0, ge=0.0)
    early_stopping: bool
    patience: int = Field(default=200, gt=0)
    num_epoch: int = Field(default=1000, gt=0)
    precision: Literal["auto", "bf16", "fp32"]

def print_dict(some_dict: dict) -> None:    
    """
    Easily readable dictionary printout.
    """
    print(json.dumps(some_dict, indent=4), flush=True)
    
def shutil_remove_readonly(func: Callable, path: str, _) -> None:
    """
    Clear the readonly bit and reattempt the removal (windows only...).
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)