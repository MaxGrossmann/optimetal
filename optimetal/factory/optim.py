"""
Helper function to set up an optimizer based on a dictionary.
"""

from __future__ import annotations

from typing import Literal, Annotated, Union
from pydantic import BaseModel, Field, ConfigDict, TypeAdapter

import torch

class AdamValidator(BaseModel):
    """
    Ensure that only valid dictionaries are parsed into the adam optimizer.
    """
    
    model_config = ConfigDict(extra="forbid")
    type: Literal["adam"]
    lr: float = Field(default=1e-3, ge=0)
    weight_decay: float = Field(default=0.0, ge=0)
    
    def make(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

class AdamWValidator(BaseModel):
    """
    Ensure that only valid dictionaries are parsed into the adam optimizer.
    """
    
    model_config = ConfigDict(extra="forbid")
    type: Literal["adamw"]
    lr: float = Field(default=1e-3, ge=0)
    weight_decay: float = Field(default=0.0, ge=0)
    
    def make(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

OptimizerValidator = Annotated[
    Union[AdamValidator, AdamWValidator],
    Field(discriminator="type"),
]

def create_optimizer(
    model: torch.nn.Module, 
    optim_config: dict,
) -> torch.optim.Optimizer:
    """
    Supported optimizer types:
        "adam"
        "adamw"
    Input:
        model:          The model whose parameters the optimizer will update
        optim_config:   Dictionary, which must have the key 'type',
                        additional keys can be included to configure the optimizer
    """
    optim_config = TypeAdapter(OptimizerValidator).validate_python(optim_config)
    return optim_config.make(model)