"""
Useful functions.
"""

from __future__ import annotations

import os
import re
import json
import stat
import warnings
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from typing import Literal, Callable
from importlib.resources import files
from pydantic import BaseModel, Field, ConfigDict

import torch

_mplstyle_loaded = False

def load_plot_style() -> None:
    """
    Load custom plot style.
    Inspired by https://github.com/garrettj403/SciencePlots.
    """
    global _mplstyle_loaded
    if _mplstyle_loaded:
        return
    style_path = files("optimetal_e3.files").joinpath("plotstyle.mplstyle")
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

def listdir_full(base_dir: str) -> list[str]:
    """
    os.listdir, but the output is the full path to all items.
    """
    return [os.path.join(base_dir, file) for file in os.listdir(base_dir)]

def shutil_remove_readonly(func: Callable, path: str, _) -> None:
    """
    Clear the readonly bit and reattempt the removal (windows only...).
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)

def print_dict(some_dict: dict) -> None:    
    """
    Easily readable dictionary printout.
    """
    print(json.dumps(some_dict, indent=4), flush=True)

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
    lmax: int = Field(..., ge=0, le=10)
    mmax: int = Field(..., ge=0, le=10)
    hidden_channels: int = Field(..., gt=0)
    ff_type: Literal["spectral", "grid"]
    lr: float = Field(..., gt=0.0)
    weight_decay: float = Field(..., ge=0.0)
    grad_clip: float = Field(default=100.0, gt=0.0)
    eps_weight: float = Field(default=1.0, ge=0.0)
    drude_weight: float = Field(default=1.0, ge=0.0)
    early_stopping: bool
    patience: int = Field(default=200, gt=0)
    num_epoch: int = Field(default=1000, gt=0)
    precision: Literal["fp32", "bf16"]