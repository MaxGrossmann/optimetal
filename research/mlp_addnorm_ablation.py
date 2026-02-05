"""
In this study, we examine how adding an 'MLP+AddNorm' block after each message
passing operation affects model performance. We perform this analysis for
different numbers of message passing steps and train each model using three
random seeds. The resulting configuration files can then be used to 
easily train models with either the 'research/train_model.py' or the 
'research/train_model_with_lrwd_optim.py' script.

The OptiMetal2B architecture is loaded from the results of the ablation study.
The trial directory for each model is hardcoded.
"""

from __future__ import annotations

import os
import json
import shutil
from copy import deepcopy

"""
START OF GLOBAL SETTINGS
"""

BASE_CONFIG_2B = {
    "seed": 42,
    "trial_dir": "/scratch/magr4985/MLP_AddNorm", # hardcoded path
    "num_train_data": 20000,
    "batch_size": 256,
    "architecture": {
        "type": "optimetal_2b",
        "node_embedding_dict": {
            "type": "group_period"
        },
        "edge_embedding_dict": {
            "type": "gaussian",
            "num_basis": 64,
            "basis_width": 2.0,
            "apply_envelope": False,
        },
        "message_passing_dict": {
            "num_layers": 2,
            "type": "gatv2",
            "heads": 4,
            "hidden_multiplier": 4
        },
        "pooling_dict": {
            "type": "vector_attention"
        },
        "hidden_dim": 256,
        "spectra_dim": 1024,
        "depth": 2,
        "activation": "relu",
        "twobody_cutoff": 5.5
    },
    "optimizer": {
        "type": "adamw",
        "lr": 0.0008, # will be optimized
        "weight_decay": 1e-06 # will be optimized
    },
    "loss_fn_eps": "mae",
    "loss_fn_drude": "mae",
    "lr_scheduler": {
        "type": "cosineannealing",
        "T_max": 200,
        "eta_min": 0.0
    },
    "warmup_epochs": 5,
    "grad_clip": 100,
    "eps_weight": 1,
    "drude_weight": 1,
    "early_stopping": False,
    "patience": 200,
    "num_epoch": 200,
    "precision": "bf16"
}

BASE_CONFIG_2B_NO_RESIDUAL = {
    "seed": 42,
    "trial_dir": "/scratch/magr4985/MLP_AddNorm", # hardcoded path
    "num_train_data": 20000,
    "batch_size": 256,
    "architecture": {
        "type": "optimetal_2b",
        "node_embedding_dict": {
            "type": "group_period"
        },
        "edge_embedding_dict": {
            "type": "gaussian",
            "num_basis": 64,
            "basis_width": 2.0,
            "apply_envelope": False,
        },
        "message_passing_dict": {
            "num_layers": 2,
            "type": "gatv2",
            "heads": 4,
        },
        "pooling_dict": {
            "type": "vector_attention"
        },
        "hidden_dim": 256,
        "spectra_dim": 1024,
        "depth": 2,
        "activation": "relu",
        "twobody_cutoff": 5.5
    },
    "optimizer": {
        "type": "adamw",
        "lr": 0.0008, # will be optimized
        "weight_decay": 1e-06 # will be optimized
    },
    "loss_fn_eps": "mae",
    "loss_fn_drude": "mae",
    "lr_scheduler": {
        "type": "cosineannealing",
        "T_max": 200,
        "eta_min": 0.0
    },
    "warmup_epochs": 5,
    "grad_clip": 100,
    "eps_weight": 1,
    "drude_weight": 1,
    "early_stopping": False,
    "patience": 200,
    "num_epoch": 200,
    "precision": "bf16"
}

"""
END OF GLOBAL SETTINGS
"""

def main(output_dir: str) -> None:    
    # sweep parameters 
    seeds = [1, 42, 137] # the same random seeds were used in the ablation study
    num_mp_layers = [1, 2, 3, 4, 5, 6, 7, 8]

    # loop over the seeds and over the weight decay scaling options
    for seed in seeds:
        for num_mp_layer in num_mp_layers:
            # version without residual connections
            name = f"optimetal_2b_no_residual_20000_mp{num_mp_layer:d}_seed{seed:d}"
            config_dict = deepcopy(BASE_CONFIG_2B_NO_RESIDUAL)
            config_dict["seed"] = seed
            config_dict["trial_dir"] = os.path.join(config_dict["trial_dir"], name)
            config_dict["architecture"]["type"] = "optimetal_2b_no_residual"
            config_dict["architecture"]["message_passing_dict"]["num_layers"] = num_mp_layer
            with open(os.path.join(output_dir, name + ".json"), "w") as f:
                json.dump(config_dict, f, indent=4)
            # version with residual connections
            name = f"optimetal_2b_20000_mp{num_mp_layer:d}_seed{seed:d}"
            config_dict = deepcopy(BASE_CONFIG_2B)
            config_dict["seed"] = seed
            config_dict["trial_dir"] = os.path.join(config_dict["trial_dir"], name)
            config_dict["architecture"]["message_passing_dict"]["num_layers"] = num_mp_layer
            with open(os.path.join(output_dir, name + ".json"), "w") as f:
                json.dump(config_dict, f, indent=4)

if __name__ == "__main__":
    # working directory setup, i.e., enable relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    output_dir = "./mlp_addnorm_config" # hardcoded
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # create all inputs for the scaling law
    main(output_dir=output_dir)