"""
Here, we create the input dictionaries for a scaling law study. Specifically, we setup a standard scaling 
law study, in which we vary only the number of training data points. The same applies to the model width, i.e.,
the number of model parameters. We train each model using three random seeds. The resulting configuration files 
can then be used to easily train models with the 'research/train_model.py' script.

For OptiMetal2B, we use the two top configurations from the ablation study that differ only in their 
message-passing layer. For OptiMetal3B, we use the best configuration from the ablation study.

When the message passing layer is TransformerConv, we scale the learning rate with scaling model width.
See 'research/transformer_scaling.py' for details.

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

def load_best_2b_configs() -> None | dict:
    """
    This helper function uses the results of the ablation
    study to retrieve the two best OptiMetal2B configurations.
    """
    config_path = "./ablation_data/2b_interaction_results.json"
    if not os.path.exists(config_path):
        print(f"The file {config_path:s} does not exist. Please run the ablation study first", flush=True)
        return None
    with open(config_path, "r") as f:
        results = json.load(f) # they are already sorted by their validation loss
    best_arch = []
    for i in range(2):
        arch_dict = json.loads(results[i]["perm_str"])
        arch_dict["optimizer"] = {
            "type": "adamw",
            "lr": results[i]["lr"],
            "weight_decay": results[i]["wd"],
        }
        best_arch.append(arch_dict)
    return best_arch

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
    idx: int | None = None,
) -> None:
    """
    Helper function to set up the architecture configurations for the scaling law study.
    """
    # Hestness et al. data parameter scaling law
    # https://arxiv.org/abs/1712.00409
    data_arch_config = deepcopy(arch_config)
    for num_data in num_datapoints:
        # name of the configuration file
        model_type = data_arch_config["architecture"]["type"]
        if model_type != "optimate":
            model_type = model_type.split("_")[1]
        if idx is None:
            name = f"{model_type:s}_hestness_data{num_data:d}_seed{seed:d}"
        else:
            name = f"{model_type:s}_variant{idx:d}_hestness_data{num_data:d}_seed{seed:d}"
        # adjust the training settings for OptiMate
        train_config = deepcopy(TRAIN_CONFIG)
        if model_type == "optimate":
            train_config["loss_fn_eps"] = "optimate"
            train_config["drude_weight"] = 0
        # create the configuration dictionary
        config_dict = {
            "seed": seed,
            "trial_dir": f"/scratch/magr4985/Scaling_Base/{name:s}", # hardcoded path
            "num_train_data": num_data,
            "batch_size": batch_size,
        }
        config_dict.update(deepcopy(data_arch_config))
        config_dict.update(deepcopy(train_config))
        # save the configuration dictionary to a JSON file
        with open(os.path.join(output_dir, name + ".json"), "w") as f:
            json.dump(config_dict, f, indent=4)

    # Kaplan et al. model parameter scaling law
    # https://arxiv.org/abs/2001.08361
    do_width_sweep = arch_config["architecture"]["type"] != "optimate"
    transformer_flag = False
    if do_width_sweep:
        if arch_config["architecture"]["type"] == "optimetal_2b":
            transformer_flag = arch_config["architecture"]["message_passing_dict"]["type"] == "transformer"
        elif arch_config["architecture"]["type"] == "optimetal_3b":
            transformer_flag = arch_config["architecture"]["triplet_block_dict"]["node_graph_mp_dict"]["type"] == "transformer"
        if transformer_flag:
            base_lr = arch_config["optimizer"]["lr"]
            base_parameter = get_model_parameters(
                factory.create_model(arch_config["architecture"])
            )
        for width in widths:
            parameter_arch_config = deepcopy(arch_config)
            # name of the configuration file
            model_type = parameter_arch_config["architecture"]["type"]
            model_type = model_type.split("_")[1]
            if idx is None:
                name = f"{model_type:s}_kaplan_width{width:d}_seed{seed:d}" 
            else:
                name = f"{model_type:s}_variant{idx:d}_kaplan_width{width:d}_seed{seed:d}"
            # adjust the 'hidden_dim' and 'spectra_dim'
            parameter_arch_config["architecture"]["hidden_dim"] = width
            parameter_arch_config["architecture"]["spectra_dim"] = 4 * width
            # scale the learning rate (optional)
            if transformer_flag:
                gamma = 1.0 # see 'research/transformer_scaling_results.ipynb'
                model_parameter = get_model_parameters(factory.create_model(parameter_arch_config["architecture"]))
                scale = (model_parameter / base_parameter) ** gamma
                parameter_arch_config["optimizer"]["lr"] = base_lr / scale
            # create the configuration dictionary
            config_dict = {
                "seed": seed,
                "trial_dir": f"/scratch/magr4985/Scaling_Base/{name:s}", # hardcoded path
                "num_train_data": 20000,
                "batch_size": batch_size,
            }
            config_dict.update(deepcopy(parameter_arch_config))
            config_dict.update(deepcopy(TRAIN_CONFIG))
            # save the configuration dictionary to a JSON file
            with open(os.path.join(output_dir, name + ".json"), "w") as f:
                json.dump(config_dict, f, indent=4)

def main(output_dir: str) -> None:
    # model configurations from an older publication (reference scaling law)
    config_optimate = {
        "architecture": {
            "type": "optimate"
        },
        "optimizer": {
            "type": "adam",
            "lr": 1e-3,
            "weight_decay": 1e-5,
        },
    }

    # load the best configurations from the ablation study and combine them into a list
    configs_2b = load_best_2b_configs()
    config_3b = load_best_3b_config()

    # sweep parameters 
    seeds = [1, 42, 137] # the same random seeds were used in the ablation study
    num_datapoints = [2500, 5000, 10000, 20000, 40000, 80000, 160000] # Hestness et al.
    widths = [16, 32, 64, 128, 256, 512, 1024] # Kaplan et al.

    # loop over the seeds to create the scaling law configurations
    for seed in seeds:
        # create the scaling law configurations for OptiMate
        setup_architectures(
            output_dir=output_dir,
            seed=seed,
            num_datapoints=num_datapoints,
            widths=widths,
            arch_config=config_optimate,
            idx=None,
        )
        # create the scaling law configurations for OptiMetal2B
        if configs_2b is not None:
            for idx, arch_config in enumerate(configs_2b):
                setup_architectures(
                    output_dir=output_dir,
                    seed=seed,
                    num_datapoints=num_datapoints,
                    widths=widths,
                    arch_config=arch_config,
                    idx=idx+1, # start with 1
                )
        # create the scaling law configuration for OptiMetal3B
        if config_3b is not None:
            setup_architectures(
                output_dir=output_dir,
                seed=seed,
                num_datapoints=num_datapoints,
                widths=widths,
                arch_config=config_3b,
                idx=None,
            )

if __name__ == "__main__":
    # working directory setup, i.e., enable relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    output_dir = "./scaling_law_base_config" # hardcoded
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # create all inputs for the scaling law
    main(output_dir=output_dir)

