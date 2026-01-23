"""
Helper function to set up a learning rate scheduler based on a dictionary.
"""

import warnings
from typing import Literal, Annotated, Union
from pydantic import BaseModel, Field, ConfigDict, TypeAdapter

import torch

class NoneValidator(BaseModel):
    """
    For training without a learning rate scheduler.
    We simulate the missing scheduler so that the warmup (i.e., the SequenceLR) still works.
    """
    
    model_config = ConfigDict(extra="forbid")
    type: Literal["None", "none", None]
    
    def make(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1000000, # very large number
            gamma=1.0 # identity
        )

class PlateauValidator(BaseModel):
    """
    Ensure that only valid dictionaries are parsed into the reduce learning rate on plateau learning rate scheduler.
    """
    
    model_config = ConfigDict(extra="forbid")
    type: Literal["reducelronplateau"]
    factor: float = 0.9
    patience: int = Field(10, ge=1)
    eta_min: float = Field(default=0.0, ge=0.0)
    
    def make(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=self.factor,
            patience=self.patience,
            eps=self.eta_min,
        )

class CosineAnnealingLR(BaseModel):
    """
    Ensure that only valid dictionaries are parsed into the cosine annealing learning rate scheduler.
    """
    
    model_config = ConfigDict(extra="forbid")
    type: Literal["cosineannealing"]
    T_max: int = Field(..., gt=0)
    eta_min: float = Field(default=0.0, ge=0.0)
    
    def make(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.T_max, # number of iterations for half of a cosine period
            eta_min=self.eta_min,
        )

class CosineAnnealingWarmRestartsValidator(BaseModel):
    """
    Ensure that only valid dictionaries are parsed into the warm restart cosine annealing learning rate scheduler.
    """
    
    model_config = ConfigDict(extra="forbid")
    type: Literal["cosineannealingwarmrestarts"]
    T_0: int = Field(..., gt=0)
    T_mult: int = Field(1, ge=1)
    eta_min: float = Field(default=0.0, ge=0.0)
    
    def make(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=self.T_0, # number of iterations until we "warm up" again
            T_mult=self.T_mult, # grow length of each cycle by this multipler
            eta_min=self.eta_min,
        )

SchedulerValidator = Annotated[
    Union[NoneValidator, PlateauValidator, CosineAnnealingLR, CosineAnnealingWarmRestartsValidator],
    Field(discriminator="type"),
]

def create_lr_scheduler(
    optimizer: torch.optim.Optimizer, 
    scheduler_config: dict, 
    warmup_epochs: int = 0,
) -> (torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau) | None:
    """
    Supported learning rate scheduler types:
        "cosineannealing"
        "cosineannealingwarmrestarts"
        "reducelronplateau"
    Input:
        optimizer:              torch optimizer object
        scheduler_config:       Dictionary, which must have the key 'type', 
                                additional keys can be included to configure the scheduler
        warmup_epochs:          The number of epochs of linear learning rate warmup before the scheduler takes over.
                                This is ignored (with a warning) for ReduceLROnPlateau, which is event-driven rather than epoch-driven.
    Read: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    """
    try:
        lr_scheduler_config = TypeAdapter(SchedulerValidator).validate_python(scheduler_config)
        main_scheduler = lr_scheduler_config.make(optimizer)
    except Exception:
        warnings.warn(f"Invalid learning rate scheduler configuration, disabling learning rate scheduler (and warmup epochs)")
        return None
    if warmup_epochs > 0:
        if isinstance(main_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            warnings.warn("The warmup epochs are ignored because ReduceLROnPlateau cannot be chained currently")
            return main_scheduler
        base_lr = optimizer.param_groups[0]['lr']
        start_factor = 1e-7 / base_lr # the first epoch should begin with a small learning rate (but not zero)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor, # we multiply the learning rate by the number in the first epoch
            end_factor=1.0, # end the warmup with the base learning rate
            total_iters=warmup_epochs,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
    return main_scheduler