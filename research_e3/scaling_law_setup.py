"""
Here, we create the input dictionaries for a scaling law study. Specifically, we setup a grid search
scaling law, in which we only vary the number of training data points and the model width, i.e., the number of
model parameters simultaneously. We train each model using three random seeds. The resulting configuration files 
can then be used to easily train models with the 'research_e3/train_model.py' script.

The learning rate and weight decay were optimized for a model with a width of 128 and 20,000 training data points.
For the equivariant version of OptiMetal, we also investigate neural scaling laws with respect to 'lmax'.

Note: The trial directory for each model is hardcoded.
"""

from __future__ import annotations

import os
import json
import shutil
from copy import deepcopy

"""
START OF GLOBAL SETTINGS
"""

# global settings for all scaling laws
batch_size = 256

# training settings (the number of epochs is set very large so that each model is trained until "convergence")
TRAIN_CONFIG = {
    "grad_clip": 100,
    "eps_weight": 1,
    "drude_weight": 1,
    "early_stopping": False,
    "patience": 500,
    "num_epoch": 500,
    "precision": "fp32",
}

"""
END OF GLOBAL SETTINGS
"""

def setup_architectures(
    output_dir: str,
    seed: int,
    lmax_list: list[int],
    num_datapoints: list[int],
    widths: list[int],
) -> None:
    """
    Helper function to set up the architecture configurations for the scaling law study.
    """
    
    for lmax in lmax_list:
        for num_data in num_datapoints:
            for width in widths:
                name = f"lmax{lmax:d}_data{num_data:d}_width{width:d}_seed{seed:d}" 
                # optimzer hyperparameters from 'lmax' lookup and the rest of the configuration dictionary
                config_dict = {
                    "seed": seed,
                    "trial_dir": f"/scratch/magr4985/scaling_law/{name:s}", # hardcoded path
                    "num_train_data": num_data,
                    "batch_size": batch_size,
                    "lmax": lmax,
                    "mmax": lmax,
                    "hidden_channels": width,
                    "ff_type": "spectral", # fixed
                    "lr": 0.002, # "optimal" for width 256 and 20,000 datapoints independently of 'lmax'
                    "weight_decay": 0.0001, # "optimal" for width 256 and 20,000 datapoints independently of 'lmax'
                }
                config_dict.update(deepcopy(TRAIN_CONFIG))
                with open(os.path.join(output_dir, name + ".json"), "w") as f:
                    json.dump(config_dict, f, indent=4)

def main(output_dir: str) -> None:
    # sweep parameters 
    seeds = [1, 42, 137] # the same random seeds were used in the ablation study
    lmax_list = [0, 1, 2, 3] # 'lmax' values for the equivariant OptiMetal version
    num_datapoints = [2500, 5000, 10000, 20000, 40000, 80000, 160000] # Hestness et al.
    widths = [16, 32, 64, 128, 256] # Kaplan et al. (larger widths require too much memory)
    
    # loop over the seeds to create the scaling law configurations
    for seed in seeds:
        setup_architectures(
            output_dir=output_dir,
            seed=seed,
            lmax_list=lmax_list,
            num_datapoints=num_datapoints,
            widths=widths,
        )

if __name__ == "__main__":
    # working directory setup, i.e., enable relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    output_dir = f"./scaling_law_config" # hardcoded
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # create all inputs for the scaling law
    main(output_dir=output_dir)

