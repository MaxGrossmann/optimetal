"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import math

import torch

from optimetal_e3.nn.common.so3 import CoefficientMapping
from optimetal_e3.nn.common.radial import RadialMLP

class SO2_m_Conv(torch.nn.Module):
    """
    SO(2) Conv: Perform an SO(2) convolution on features corresponding to +- m.
    Input:
        m:                    Order of the spherical harmonic coefficients
        sphere_channels:      Number of spherical channels
        m_output_channels:    Number of output channels used during the SO(2) conv
        lmax:                 Degrees (l)
        mmax:                 Orders (m)
    """

    def __init__(
        self,
        m: int,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
    ) -> None:
        super().__init__()

        self.m = m
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax

        assert self.mmax >= m
        num_coefficents = self.lmax - m + 1
        num_channels = num_coefficents * self.sphere_channels

        self.out_channels_half = self.m_output_channels * (
            num_channels // self.sphere_channels
        )
        self.fc = torch.nn.Linear(
            num_channels,
            2 * self.out_channels_half,
            bias=False,
        )
        self.fc.weight.data.mul_(1 / math.sqrt(2))

    def forward(self, x_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_m = self.fc(x_m)
        x_r_0, x_i_0, x_r_1, x_i_1 = x_m.reshape(
            x_m.shape[0], -1, self.out_channels_half
        ).split(1, dim=1)
        x_m_r = x_r_0 - x_i_1  # x_r[:, 0] - x_i[:, 1]
        x_m_i = x_r_1 + x_i_0  # x_r[:, 1] + x_i[:, 0]
        return (
            x_m_r.view(x_m.shape[0], -1, self.m_output_channels),
            x_m_i.view(x_m.shape[0], -1, self.m_output_channels),
        )


class SO2_Convolution(torch.nn.Module):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders).
    Input:
        sphere_channels:            Number of spherical channels
        m_output_channels:          Number of output channels used during the SO(2) conv
        lmax:                       Degrees (l)
        mmax:                       Orders (m)
        mappingReduced:             Used to extract a subset of m components
        internal_weights:           If True, not using radial function to multiply inputs features
        edge_channel_list:          List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
        extra_m0_output_channels:   If not None, return 'out_embedding' and 'extra_m0_features' (Tensor).
    """

    def __init__(
        self,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
        internal_weights: bool = True,
        edge_channel_list: list[int] | None = None,
        extra_m0_output_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.internal_weights = internal_weights
        self.extra_m0_output_channels = extra_m0_output_channels

        num_channels_m0 = (self.lmax + 1) * self.sphere_channels

        # SO(2) convolution for m = 0
        m0_output_channels = self.m_output_channels * (
            num_channels_m0 // self.sphere_channels
        )
        if self.extra_m0_output_channels is not None:
            m0_output_channels = m0_output_channels + self.extra_m0_output_channels
        self.fc_m0 = torch.nn.Linear(num_channels_m0, m0_output_channels)
        num_channels_rad = self.fc_m0.in_features

        # SO(2) convolution for non-zero m
        self.so2_m_conv = torch.nn.ModuleList()
        for m in range(1, self.mmax + 1):
            self.so2_m_conv.append(
                SO2_m_Conv(
                    m,
                    self.sphere_channels,
                    self.m_output_channels,
                    self.lmax,
                    self.mmax,
                )
            )
            num_channels_rad = num_channels_rad + self.so2_m_conv[-1].fc.in_features

        # embedding function of distance
        self.rad_func = None
        if not self.internal_weights:
            assert edge_channel_list is not None
            edge_channel_list = copy.deepcopy(edge_channel_list)
            edge_channel_list.append(int(num_channels_rad))
            # this can moved outside of SO2 conv and into Edgewise
            self.rad_func = RadialMLP(edge_channel_list)

        self.m_split_sizes = [self.mappingReduced.m_size[0]] + (
            torch.tensor(self.mappingReduced.m_size[1:]) * 2
        ).tolist()
        self.edge_split_sizes = [self.fc_m0.in_features] + [
            mod.fc.in_features for mod in self.so2_m_conv
        ]

    def forward(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # radial function
        if self.rad_func is not None:
            x_edge_by_m = self.rad_func(x_edge).split(self.edge_split_sizes, dim=1)

        x_by_m = x.split(self.m_split_sizes, dim=1)

        num_edges = len(x_edge)
        # compute m=0 coefficients separately since they only have real values (no imaginary)
        x_0 = x_by_m[0].view(num_edges, -1)
        if self.rad_func is not None:
            x_0 = x_0 * x_edge_by_m[0]
        x_0 = self.fc_m0(x_0)

        # extract extra m0 features
        if self.extra_m0_output_channels is not None:
            x_0_extra, x_0 = x_0.split(
                (
                    self.extra_m0_output_channels,
                    self.fc_m0.out_features - self.extra_m0_output_channels,
                ),
                -1,
            )

        out = [x_0.view(num_edges, -1, self.m_output_channels)]  # m0

        # compute the values for the m > 0 coefficients
        for m in range(1, self.mmax + 1):
            x_m = x_by_m[m].view(num_edges, 2, -1)
            if self.rad_func is not None:
                x_m = x_m * x_edge_by_m[m].unsqueeze(1)
            x_m = self.so2_m_conv[m - 1](x_m)
            out.extend(x_m)

        out = torch.cat(out, dim=1)

        if self.extra_m0_output_channels is not None:
            return out, x_0_extra
        else:
            return out