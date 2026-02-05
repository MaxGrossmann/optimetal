"""
Here, we create the input dictionaries for a scaling law study. Specifically, we investigate
how to scale the learning rate when scaling the parameter count of a model with a TransformerConv 
message passing layer for OptiMetal2B. The weight decay is kept constant. We train each model using 
three random seeds. The resulting configuration files can then be used to train models with the 
'research/train_model.py' script.

The OptiMetal2B architecture is loaded from the results of the ablation study. 
The trial directory for each model is hardcoded.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
from copy import deepcopy

import optimetal.factory as factory
from optimetal.utils import get_model_parameters

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
    gamma_list: list[float],
    arch_config: dict,
) -> None:
    """
    Helper function to set up the architecture configurations for the learning rate scaling experiment.
    """
    # optimizer settings and model parameter count of the base architecture
    base_lr = arch_config["optimizer"]["lr"]
    base_parameter = get_model_parameters(
        factory.create_model(arch_config["architecture"])
    )

    # loop over model widths
    for hidden_dim in [16, 32, 64, 128, 256, 512, 1024]:
        # adjust the 'hidden_dim' and 'spectra_dim'
        width_arch = deepcopy(arch_config)
        width_arch["architecture"]["hidden_dim"] = hidden_dim
        width_arch["architecture"]["spectra_dim"] = 4 * hidden_dim
        # compute the number of model parameters once
        model_parameter = get_model_parameters(
            factory.create_model(width_arch["architecture"])
        )
        # loop over learning rate scaling exponents
        for gamma in gamma_list:
            run_arch = deepcopy(width_arch)
            # scale the learning rate
            scale = (model_parameter / base_parameter) ** gamma
            run_arch["optimizer"]["lr"] = base_lr / scale
            # name of the configuration file
            model_type = run_arch["architecture"]["type"]
            model_type = model_type.split("_")[1]
            name = f"{model_type:s}_transformer_scaling_hidden{hidden_dim:d}_gamma{gamma:.3f}_seed{seed:d}"
            # create the configuration dictionary
            config_dict = {
                "seed": seed,
                "trial_dir": f"/scratch/magr4985/Transformer_Scaling/{name:s}", # hardcoded path
                "num_train_data": 20000,
                "batch_size": batch_size,
            }
            config_dict.update(deepcopy(run_arch))
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
    gamma_list = [0.0, 0.1, 0.2, 0.5, 0.9, 1.0, 1.1, 1.2, 1.5, 1.9, 2.0]

    # loop over the seeds and over the weight decay scaling options
    for seed in seeds:
        # OptiMetal2B
        setup_architectures(
            output_dir=output_dir,
            seed=seed,
            gamma_list=gamma_list,
            arch_config=config_2b,
        )

if __name__ == "__main__":
    # working directory setup, i.e., enable relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    output_dir = "./transformer_scaling_config" # hardcoded
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # create all inputs for the scaling law
    main(output_dir=output_dir)
