"""
Helper function for setting up a model based on a dictionary.
"""

from __future__ import annotations

from copy import deepcopy

import torch

from optimetal.nn import OptiMate, OptiMetal2BNoResidual, OptiMetal2B, OptiMetal3B

def create_model(model_config: dict) -> torch.nn.Module:
    """
    Supported model types:
        "optimate"
        "optimetal_2b_no_residual"
        "optimetal_2b"
        "optimetal_3b"
    Input:
        model_config:   Dictionary, which must have the key 'type',
                        additional keys can be included to configure the model
                        (Check the model definition to see how the dictionary needs to look.)
    """
    model_type = model_config["type"].lower()
    param_dict = deepcopy({k: v for k, v in model_config.items() if k != "type"})
    if model_type == "optimate":
        return OptiMate()
    elif model_type == "optimetal_2b_no_residual":
        return OptiMetal2BNoResidual(**param_dict) # the dictionary is validated inside the model
    elif model_type == "optimetal_2b":
        return OptiMetal2B(**param_dict) # the dictionary is validated inside the model
    elif model_type == "optimetal_3b":
        return OptiMetal3B(**param_dict) # the dictionary is validated inside the model
    else:
        raise ValueError(f"Unsupported model type '{model_type.upper():s}'")