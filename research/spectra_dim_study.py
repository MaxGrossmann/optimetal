"""
Here, we examine how changing the multiplier that couples the 'spectra_dim' and 
the 'hidden_dim' affects the performance of a model. We train each model using 
three random seeds. The resulting configuration files can then be used to easily
train models with the 'research/train_model.py' script.

The OptiMetal2B architecture is loaded from the results of the ablation study. 
The trial directory for each model is hardcoded.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
from copy import deepcopy

"""
START OF GLOBAL SETTINGS
"""

# global settings for all scaling laws
batch_size = 256

# training settings
TRAIN_CONFIG = {
    "loss_fn_eps": "mae",
    "loss_fn_drude": "mae",
    "lr_scheduler": {
        "type": "cosineannealing",
        "T_max": 500,
        "eta_min": 0.0,
    },
    "warmup_epochs": 5,
    "grad_clip": 100,
    "eps_weight": 1,
    "drude_weight": 1,
    "early_stopping": False,
    "patience": 500,
    "num_epoch": 500,
    "precision": "bf16",
}

"""
END OF GLOBAL SETTINGS
"""

def load_best_2b_config() -> None | dict:
    """
    This helper function uses the results of the ablation study to retrieve
    the best OptiMetal2B configuration with a Transformer message passing layer.
    """
    config_path = "./ablation_data/2b_interaction_results.json"
    if not os.path.exists(config_path):
        print(f"The file {config_path:s} does not exist. Please run the ablation study first", flush=True)
        return None
    with open(config_path, "r") as f:
        results = json.load(f) # they are already sorted by their validation loss
    idx = 1 # index of the transformer message passing model
    arch_dict = json.loads(results[idx]["perm_str"])
    arch_dict["optimizer"] = {
        "type": "adamw",
        "lr": results[idx]["lr"],
        "weight_decay": results[idx]["wd"],
    }
    return arch_dict

def setup_architectures(
    output_dir: str,
    seed: int,
    spectra_dim_multiplier_list: list[int],
    arch_config: dict,
) -> None:
    """
    Helper function to set up the architecture configurations for a scaling law study.
    """
    # loop over model widths and learning rate scaling exponents
    for spectra_dim_multiplier in spectra_dim_multiplier_list:
        # adjust the 'hidden_dim' and 'spectra_dim'
        parameter_arch_config = deepcopy(arch_config)
        hidden_dim = parameter_arch_config["architecture"]["hidden_dim"]
        parameter_arch_config["architecture"]["spectra_dim"] = spectra_dim_multiplier * hidden_dim
        # name of the configuration file
        model_type = parameter_arch_config["architecture"]["type"]
        model_type = model_type.split("_")[1]
        name = f"{model_type:s}_parameter_scaling_hidden{hidden_dim:d}_spectra_multiplier{spectra_dim_multiplier:d}_seed{seed:d}"
        # create the configuration dictionary
        config_dict = {
            "seed": seed,
            "trial_dir": f"/scratch/magr4985/Spectra_Dim/{name:s}", # hardcoded path
            "num_train_data": 20000,
            "batch_size": batch_size,
        }
        config_dict.update(deepcopy(parameter_arch_config))
        config_dict.update(deepcopy(TRAIN_CONFIG))
        # save the configuration dictionary to a JSON file
        with open(os.path.join(output_dir, name + ".json"), "w") as f:
            json.dump(config_dict, f, indent=4)

def main(output_dir: str) -> None:
    # load the best configuration from the ablation study
    config_2b = load_best_2b_config()
    if config_2b is None:
        sys.exit("Could not load ablation configurations")
    
    # sweep parameters 
    seeds = [1, 42, 137] # the same random seeds were used in the ablation study
    spectra_dim_multiplier_list = [1, 2, 4, 8] 

    # loop over the seeds and over the weight decay scaling options
    for seed in seeds:
        # OptiMetal2B
        setup_architectures(
            output_dir=output_dir,
            seed=seed,
            spectra_dim_multiplier_list=spectra_dim_multiplier_list,
            arch_config=config_2b,
        )

if __name__ == "__main__":
    # working directory setup, i.e., enable relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    output_dir = "./spectra_dim_config" # hardcoded
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # create all inputs for the scaling law
    main(output_dir=output_dir)
