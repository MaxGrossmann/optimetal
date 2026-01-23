"""
Here, we create the input dictionaries for a scaling law study. Specifically, we setup a grid search
scaling law, in which we only vary the number of training data points and the model width, i.e., the number of
model parameters simultaneously. We train each model using three random seeds. The resulting configuration files 
can then be used to easily train models with the 'research/train_model.py' script.

We scale the learning rate with the model width.
See 'research/transformer_scaling.py' for details.
The learning rate is not tuned per data size.

The model architectures are loaded from the results of the ablation study. 
The trial directory for each model is hardcoded.
"""

from __future__ import annotations

import os
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

# training settings (the number of epochs is set very large so that each model is trained until "convergence")
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

def load_best_3b_config() -> None | dict:
    """
    This helper function uses the results of the ablation study to retrieve
    the best configuration for the message passing layer of OptiMetal3B.
    All other layers are fixed based on the best results from the OptiMetal2B ablation study.
    """
    # load the best OptiMetal2B architecture
    config_path = "./ablation_data/2b_interaction_results.json"
    if not os.path.exists(config_path):
        print(f"The file {config_path:s} does not exist. Please run the ablation study first", flush=True)
        return None
    with open(config_path, "r") as f:
        results = json.load(f) # they are already sorted by their validation loss
    arch_dict_base = json.loads(results[0]["perm_str"])
    arch_dict_base["architecture"]["type"] = "optimetal_3b"
    arch_dict_base["architecture"].pop("message_passing_dict", None)
    arch_dict_base["architecture"]["triplet_block_dict"] = {
        "num_layers": 2, # always fixed to 2 layers
        "edge_graph_mp_dict": {},
        "node_graph_mp_dict": {},
    }
    # load the two best message passing layers for OptiMetal3B
    config_path = "./ablation_data/3b_mp_results.json"
    if not os.path.exists(config_path):
        print(f"The file {config_path:s} does not exist. Please run the ablation study first", flush=True)
        return None
    with open(config_path, "r") as f:
        results = json.load(f) # they are already sorted by their validation loss
    if len(results) < 39: # see 'research/ablation_study.py'
        return None # the ablation study is not finished
    idx = 0 # the best configuration uses the transformer message passing
    arch_dict = deepcopy(arch_dict_base)
    mp_dict = json.loads(results[idx]["perm_str"])
    arch_dict["architecture"]["triplet_block_dict"]["edge_graph_mp_dict"] = deepcopy(mp_dict)
    arch_dict["architecture"]["triplet_block_dict"]["node_graph_mp_dict"] = deepcopy(mp_dict)
    arch_dict["optimizer"] = {
        "type": "adamw",
        "lr": results[idx]["lr"],
        "weight_decay": results[idx]["wd"],
    }
    return arch_dict

def setup_architectures(
    output_dir: str,
    seed: int,
    num_datapoints: list[int],
    widths: list[int],
    arch_config: dict,
) -> None:
    """
    Helper function to set up the architecture configurations for the scaling law study.
    """
    base_lr = arch_config["optimizer"]["lr"]
    base_parameter = get_model_parameters(
        factory.create_model(arch_config["architecture"])
    )
    for num_data in num_datapoints:
        for width in widths:
            scaling_arch_config = deepcopy(arch_config)
            # name of the configuration file and check if the model type is supported
            model_type = scaling_arch_config["architecture"]["type"]
            model_type = model_type.split("_")[1]
            name = f"{model_type:s}_scaling_grid_data{num_data:d}_width{width:d}_seed{seed:d}" 
            # adjust the 'hidden_dim' and 'spectra_dim'
            scaling_arch_config["architecture"]["hidden_dim"] = width
            scaling_arch_config["architecture"]["spectra_dim"] = 4 * width
            # scale the learning rate
            gamma = 1.0 # see 'research/transformer_scaling_results.ipynb'
            model_parameter = get_model_parameters(factory.create_model(scaling_arch_config["architecture"]))
            scale = (model_parameter / base_parameter) ** gamma
            scaling_arch_config["optimizer"]["lr"] = base_lr / scale
            # create the configuration dictionary
            config_dict = {
                "seed": seed,
                "trial_dir": f"/scratch/magr4985/Scaling_Grid/{name:s}", # hardcoded path
                "num_train_data": num_data,
                "batch_size": batch_size,
            }
            config_dict.update(deepcopy(scaling_arch_config))
            config_dict.update(deepcopy(TRAIN_CONFIG))
            with open(os.path.join(output_dir, name + ".json"), "w") as f:
                json.dump(config_dict, f, indent=4)

def main(output_dir: str) -> None:
    # load the best configuration from the ablation study
    config_2b = load_best_2b_config()
    config_3b = load_best_3b_config()

    # sweep parameters 
    seeds = [1, 42, 137] # the same random seeds were used in the ablation study
    num_datapoints = [2500, 5000, 10000, 20000, 40000, 80000, 160000] # Hestness et al.
    widths = [16, 32, 64, 128, 256, 512, 1024] # Kaplan et al.
    
    # loop over the seeds to create the scaling law configurations
    for seed in seeds:
        # create the scaling law configurations for OptiMetal2B
        if config_2b is not None:
            setup_architectures(
                output_dir=output_dir,
                seed=seed,
                num_datapoints=num_datapoints,
                widths=widths,
                arch_config=config_2b,
            )
        # create the scaling law configuration for OptiMetal3B
        if config_3b is not None:
            setup_architectures(
                output_dir=output_dir,
                seed=seed,
                num_datapoints=num_datapoints,
                widths=widths,
                arch_config=config_3b,
            )

if __name__ == "__main__":
    # working directory setup, i.e., enable relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    output_dir = "./scaling_law_grid_config" # hardcoded
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # create all inputs for the scaling law
    main(output_dir=output_dir)

