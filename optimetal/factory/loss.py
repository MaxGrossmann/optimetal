"""
Helper function to set up a loss function based on a string.
All losses will use 'reduction="mean"' so that the 'Trainer'
class correctly calculates the average loss per epoch.
"""

from __future__ import annotations

import torch

class OptiMateMAELoss(torch.nn.Module):
    """
    Loss for learning only the imaginary part of the interband dielectric function, 
    adapted to the OptiMate architecture. It just calls loss functions from torch.
    Input:
        eps_pred:       (N, 2) array, where the first column is the real part
                        and the second column is the imaginary part of the dielectric function
        eps_target:     (N, 2) array, where the first column is the real part
                        and the second column is the imaginary part of the dielectric function
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.l1_loss = torch.nn.L1Loss(reduction="mean")

    def forward(
        self, 
        eps_pred: torch.Tensor, 
        eps_target: torch.Tensor,
    ) -> torch.Tensor:
        return self.l1_loss(eps_pred[:, 1], eps_target[:, 1])

def create_loss_fn(loss_type: str) -> torch.nn.Module:
    """
    Supported loss types:
        "mse":          Mean Squared Error Loss
        "mae":          Mean Absolute Error Loss
        "optimate":     Custom OptimateMAELoss 
                        (see above, only use this for the dielectric function loss)
    Input:
        loss_type:      String representing the desired loss function
    """
    loss_type = loss_type.lower()
    if loss_type == "mse":
        return torch.nn.MSELoss(reduction="mean")
    elif loss_type == "mae":
        return torch.nn.L1Loss(reduction="mean")
    elif loss_type == "optimate":
        return OptiMateMAELoss()
    else:
        raise ValueError(f"Unsupported loss function type '{loss_type.upper():s}'")