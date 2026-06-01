"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy

import torch

from optimetal_e3.nn.common.so3 import CoefficientMapping
from optimetal_e3.nn.common.radial import RadialMLP

class EdgeDegreeEmbedding(torch.nn.Module):
    """
    Input:
        hidden_channels:        Number of hidden channels
        lmax:                   Degrees (l)
        mmax:                   Orders (m)
        max_num_elements:       Maximum number of atomic numbers
        rescale_factor:         Rescale the sum aggregation
        edge_channel_list:      List of sizes of invariant edge embedding, e.g. , [input_channels, hidden_channels, hidden_channels]
        mappingReduced:         Class to convert l and m indices once node embedding is rotated
    """

    def __init__(
        self,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        rescale_factor: float,
        mappingReduced: CoefficientMapping,
        edge_channel_list: list[int],
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.rescale_factor = rescale_factor
        self.mappingReduced = mappingReduced

        # number of coefficients for m=0 and all m
        self.m_0_num_coefficients = self.mappingReduced.m_size[0]
        self.m_all_num_coefficents = len(self.mappingReduced.l_harmonic)

        # embedding function of distance
        edge_channel_list = copy.deepcopy(edge_channel_list) # avoid modifying the input list
        edge_channel_list.append(self.m_0_num_coefficients * self.hidden_channels)
        self.rad_func = RadialMLP(edge_channel_list)
        
    def forward(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor,
        edge_index: torch.Tensor,
        wigner_and_M_mapping_inv: torch.Tensor,
        edge_envelope: torch.Tensor,
    ):
        x_edge_m_0 = self.rad_func(x_edge) # (num_edges, edge_channel_list[-1])
        x_edge_m_0 = x_edge_m_0.reshape(-1, self.m_0_num_coefficients, self.hidden_channels) # (num_edges, m_0_num_coefficients, hidden_channels)
        x_edge_embedding = torch.nn.functional.pad(
            x_edge_m_0,
            (0, 0, 0, (self.m_all_num_coefficents - self.m_0_num_coefficients)),
        ) # (num_edges, m_all_num_coefficients, hidden_channels)
        x_edge_embedding = torch.bmm(wigner_and_M_mapping_inv, x_edge_embedding) # (num_edges, m_all_num_coefficients, hidden_channels)
        x_edge_embedding = x_edge_embedding * edge_envelope 
        x_edge_embedding = x_edge_embedding.to(x.dtype)
        return x.index_add(0, edge_index[1], x_edge_embedding / self.rescale_factor)