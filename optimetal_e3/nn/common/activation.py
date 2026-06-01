"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

class GateActivation(torch.nn.Module):
    # m_prime -> order is l0m0, l1m0, l2m0.. , l1m1 , l2m1, ... , l1m-1, l2m-1,...
    def __init__(
        self, lmax: int, mmax: int, num_channels: int, m_prime: bool = False
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.num_channels = num_channels

        # compute 'expand_index' based on 'lmax' and 'mmax'
        num_components = 0
        for lval in range(1, self.lmax + 1):
            num_m_components = min((2 * lval + 1), (2 * self.mmax + 1))
            num_components = num_components + num_m_components
        expand_index = torch.zeros([num_components]).long()

        self.m_prime = m_prime
        if self.m_prime:
            start_idx = 0
            length = self.lmax
            expand_index[start_idx : (start_idx + length)] = torch.arange(self.lmax)
            start_idx = start_idx + length
            for mval in range(1, self.mmax + 1):
                length = 2 * (self.lmax + 1 - mval)
                expand_index[start_idx : (start_idx + length)] = torch.cat(
                    [
                        torch.arange(mval - 1, self.lmax),
                        torch.arange(mval - 1, self.lmax),
                    ]
                )
                start_idx = start_idx + length
        else:
            start_idx = 0
            for lval in range(1, self.lmax + 1):
                length = min((2 * lval + 1), (2 * self.mmax + 1))
                expand_index[start_idx : (start_idx + length)] = lval - 1
                start_idx = start_idx + length
        self.register_buffer("expand_index", expand_index, persistent=False)

        self.scalar_act = torch.nn.SiLU()
        self.gate_act = torch.nn.Sigmoid()

    def forward(self, gating_scalars, input_tensors):
        """
        'gating_scalars': shape [N, lmax * num_channels]
        'input_tensors': shape  [N, (lmax + 1) ** 2, num_channels]
        """
        gating_scalars = self.gate_act(gating_scalars).view(
            gating_scalars.shape[0], self.lmax, self.num_channels
        )

        gating_scalars = torch.index_select(
            gating_scalars, dim=1, index=self.expand_index
        )
        input_tensors_scalars, input_tensors_vectors = input_tensors.split(
            (1, input_tensors.shape[1] - 1), 1
        )

        input_tensors_scalars = self.scalar_act(input_tensors_scalars)
        input_tensors_vectors = input_tensors_vectors * gating_scalars

        return torch.cat((input_tensors_scalars, input_tensors_vectors), dim=1)

