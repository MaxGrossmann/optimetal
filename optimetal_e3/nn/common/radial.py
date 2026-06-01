"""
Methods of embedding bond lengths.
"""

from __future__ import annotations

import torch
    
class PolynomialEnvelope(torch.nn.Module):
    """
    Polynomial envelope function that ensures a smooth cutoff.
    Refs.:
        https://arxiv.org/pdf/2003.03123
        https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/models/uma/nn/radial.py
    """

    def __init__(
        self, 
        exponent: int = 4, # hardcoded, as it seems to insensitive to the exponent, see https://arxiv.org/abs/2003.03123
    ) -> None:
        super().__init__()
        self.p = float(exponent)
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        env_val = 1 + (d_scaled**self.p) * (self.a + d_scaled * (self.b + self.c * d_scaled))
        return torch.where(d_scaled < 1, env_val, 0)
    
class GaussianEmbedding(torch.nn.Module):
    """
    Bond distance embeddings are obtained using Gaussian radial basis functions. The standard deviation of each Gaussian is given by
    'basis width' multiplied by the spacing between neighboring centers. Therefore, one can control the amount of overlap between neighboring
    Gaussians through 'basis_width'.
    Refs.: 
        https://arxiv.org/pdf/2003.03123
        https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/models/uma/nn/radial.py
    """
    
    def __init__(
        self,
        rcut: float = 5.5,
        num_radial_basis: int = 64,
        basis_width: float = 2.0,
    ) -> None:
        super().__init__()
        offset = torch.linspace(0.0, rcut, num_radial_basis)
        self.register_buffer("offset", offset, persistent=False)
        self.coeff = -0.5 / (basis_width * (offset[1] - offset[0])).item() ** 2

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""   
    
class RadialMLP(torch.nn.Module):
    """
    Construct a radial function (linear layers + layer normalization + SiLU) given a list of channels.
    """

    def __init__(self, channels_list) -> None:
        super().__init__()
        modules = []
        input_channels = channels_list[0]
        for i in range(len(channels_list)):
            if i == 0:
                continue
            modules.append(torch.nn.Linear(input_channels, channels_list[i], bias=True))
            input_channels = channels_list[i]

            if i == len(channels_list) - 1:
                break
            modules.append(torch.nn.LayerNorm(channels_list[i]))
            modules.append(torch.nn.SiLU())
        self.net = torch.nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)
