"""
E(3)-equivariant version of the OptiMetal model.
The code is adapted from the eSEN/UMA model of the FAIR Chemistry team of Meta.
Ref.:
    https://github.com/facebookresearch/fairchem 
"""

from __future__ import annotations

from typing import Literal
from importlib.resources import files

import torch
import torch_geometric
from torch_geometric.data import Data
from torch.profiler import record_function

from optimetal_e3.nn.common.embedding import EdgeDegreeEmbedding
from optimetal_e3.nn.common.so3 import CoefficientMapping, SO3_Grid
from optimetal_e3.nn.common.radial import PolynomialEnvelope, GaussianEmbedding, RadialMLP
from optimetal_e3.nn.common.rotation import init_edge_rot_euler_angles, eulers_to_wigner
from optimetal_e3.nn.common.layer_norm import EquivariantRMSNormArraySphericalHarmonics
from optimetal_e3.nn.common.escn_block import eSCNMD_Block

class OptiMetalE3(torch.nn.Module):
    
    def __init__(
        self, 
        max_num_elements: int = 83, # default: all elements up to Bi (Z=83)
        num_layer: int = 2, # number of message passing layers
        hidden_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        rcut: float = 5.5,
        ff_type: Literal["grid", "spectral"] = "spectral",
        check_equivariance: bool = False,
    ) -> None:
        super().__init__()
        
        """
        Initialize architecture parameters.
        """
        
        self.max_num_elements = max_num_elements
        self.num_layer = num_layer
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.rcut = rcut
        self.ff_type = ff_type
        self.check_equivariance = check_equivariance
        if self.check_equivariance:
            print("Equivariance check mode activated", flush=True)
        
        """
        Alignment and indexing setup for rotations and spherical harmonics.
        """
        
        # load precomputed Wigner D-matrices and setup coefficient mapping
        Jd_list = torch.load(files("optimetal_e3.files").joinpath("Jd.pt"), weights_only=False)
        for l in range(self.lmax + 1):
            self.register_buffer(f"Jd_{l:d}", Jd_list[l])
        self.sph_feature_size = int((self.lmax + 1) ** 2)
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)
        
        # (lmax, lmax) for node, (lmax, mmax) for edge
        self.SO3_grid = torch.nn.ModuleDict()
        self.SO3_grid["lmax_lmax"] = SO3_Grid(
            self.lmax, 
            self.lmax, 
        )
        self.SO3_grid["lmax_mmax"] = SO3_Grid(
            self.lmax, 
            self.mmax, 
        )
        
        # precompute coefficient index for the mmax < lmax case        
        coefficient_index = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
            self.lmax, self.mmax
        )
        self.register_buffer("coefficient_index", coefficient_index, persistent=False)
        
        """
        Embedding setup.
        """
        
        # invariant atom embeddings
        self.inv_atom_emb = torch.nn.Embedding(self.max_num_elements, self.hidden_channels)
        
        # invariant edge embedding
        self.inv_edge_emb = GaussianEmbedding(rcut=self.rcut, num_radial_basis=self.hidden_channels)
        
        # polynomial envelope for the edge embedding
        self.envelope = PolynomialEnvelope()
        
        # setup embedding for source and target atoms for equivariant atom embedding
        self.source_emb = torch.nn.Embedding(self.max_num_elements, self.hidden_channels)
        self.target_emb = torch.nn.Embedding(self.max_num_elements, self.hidden_channels)
        torch.nn.init.uniform_(self.source_emb.weight.data, -0.001, 0.001)
        torch.nn.init.uniform_(self.target_emb.weight.data, -0.001, 0.001)
        
        # equivariant atom embedding that includes edge information
        self.edge_channel_list = [
            3 * self.hidden_channels,
            self.hidden_channels,
            self.hidden_channels,
        ]
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            hidden_channels=self.hidden_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            edge_channel_list=self.edge_channel_list,
            rescale_factor=5.0, # hardcoded rescale factor (see UMA implementation)
            mappingReduced=self.mappingReduced,
        )
        
        """
        Message aggregation blocks.
        """
        
        # Initialize the blocks for each layer
        self.mp = torch.nn.ModuleList()
        for _ in range(self.num_layer):
            block = eSCNMD_Block(
                self.hidden_channels, # sphere_channels = hidden_channels
                self.hidden_channels,
                self.lmax,
                self.mmax,
                self.mappingReduced,
                self.SO3_grid,
                self.edge_channel_list,
                self.rcut,
                self.ff_type,
            )
            self.mp.append(block)

        self.norm = EquivariantRMSNormArraySphericalHarmonics(
            lmax=self.lmax,
            num_channels=self.hidden_channels,
        )
        
        """
        Pooling and output heads.
        """
        
        self.pooling = torch_geometric.nn.MeanAggregation()
        self.spectra_mlp = RadialMLP([self.hidden_channels, self.hidden_channels, self.hidden_channels, 4002])
        self.drude_mlp = RadialMLP([self.hidden_channels, self.hidden_channels, self.hidden_channels, 1])
                
    def _get_rotmat_and_wigner(
        self, 
        edge_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Jd_buffers = [
            getattr(self, f"Jd_{l:d}").type(edge_vector.dtype)
            for l in range(self.lmax + 1)
        ]
        with record_function("obtain rotmat wigner original"):
            euler_angles = init_edge_rot_euler_angles(edge_vector, self.check_equivariance)
            wigner = eulers_to_wigner(euler_angles, 0, self.lmax, Jd_buffers)
            wigner_inv = torch.transpose(wigner, 1, 2).contiguous()
        # select subset of coefficients we are using
        if self.mmax != self.lmax:
            wigner = wigner.index_select(1, self.coefficient_index)
            wigner_inv = wigner_inv.index_select(2, self.coefficient_index)
        wigner_and_M_mapping = torch.einsum(
            "mk,nkj->nmj", self.mappingReduced.to_m.to(wigner.dtype), wigner
        )
        wigner_and_M_mapping_inv = torch.einsum(
            "njk,mk->njm", wigner_inv, self.mappingReduced.to_m.to(wigner_inv.dtype)
        )
        return wigner_and_M_mapping, wigner_and_M_mapping_inv
        
    def forward(self, graph: Data) -> tuple[torch.Tensor]:
        
        # unpack graph data
        atomic_number = graph.atomic_number # (num_nodes)
        position = graph.position # (num_nodes, 3)
        lattice = graph.lattice # (batch_size, 3, 3)
        edge_index = graph.edge_index # (2, num_edges)
        pbc_offset = graph.pbc_offset # (num_edges, 3)
        node_batch = graph.batch # (num_nodes)
        edge_batch = graph.edge_index_batch # (num_edges)
        
        # check for edges
        if edge_index.numel() == 0:
            raise ValueError(f"No edges found in input system")
        
        # calculate edge lengths
        with record_function("edge length"):
            shift = torch.einsum('ij,ijk->ik', pbc_offset, lattice.view(-1, 3, 3)[edge_batch]) # (num_edges, 3)
            edge_vector = position[edge_index[0]] - (shift + position[edge_index[1]]) # (num_edges, 3) 
            edge_length = torch.norm(edge_vector, dim=1, p=2) # (num_edges)
            
        # calculate Wigner and M mapping matrices
        with record_function("obtain wigner"):
            # (num_edges, (lmax+1)^2, (lmax+1)^2)
            wigner_and_M_mapping, wigner_and_M_mapping_inv = self._get_rotmat_and_wigner(edge_vector)
            
        # atom embedding
        with record_function("atom embedding"):
            x_message = torch.zeros(
                atomic_number.shape[0],
                self.sph_feature_size,
                self.hidden_channels,
                device=position.device,
                dtype=position.dtype,
            ) # (num_nodes, (lmax+1)^2, hidden_channels)
            # set l=m=0 component with invariant atom embedding
            x_message[:, 0, :] = self.inv_atom_emb(atomic_number - 1) # (num_nodes, hidden_channels)
        
        # equivariant atom embedding which includes edge information
        with record_function("edge embedding"):
            edge_length_scaled = edge_length / self.rcut
            edge_length_envelope = self.envelope(edge_length_scaled).reshape(-1, 1, 1) # (num_edges, 1, 1)
            inv_edge_emb = self.inv_edge_emb(edge_length) # (num_edges, hidden_channels)
            source_emb = self.source_emb(atomic_number[edge_index[0]] - 1) # (num_edges, hidden_channels)
            target_emb = self.target_emb(atomic_number[edge_index[1]] - 1) # (num_edges, hidden_channels)
            x_edge = torch.cat(
                (inv_edge_emb, source_emb, target_emb), 
                dim=1,
            ) # (num_edges, 3 * hidden_channels)
            x_message = self.edge_degree_embedding(
                x_message,
                x_edge,
                edge_index,
                wigner_and_M_mapping_inv,
                edge_length_envelope,
            ) # (num_nodes, (lmax+1)^2, hidden_channels)
            
        # message aggregation layers
        for i in range(self.num_layer):
            with record_function(f"message passing {i:d}"):
                x_message = self.mp[i](
                    x_message,
                    x_edge,
                    edge_index,
                    wigner_and_M_mapping,
                    wigner_and_M_mapping_inv,
                    edge_length_envelope,
                ) # (num_nodes, (lmax+1)^2, hidden_channels)

        # final layer norm
        x_message = self.norm(x_message) # (num_nodes, (lmax+1)^2, hidden_channels)
        
        # check equivariance shortcut
        if self.check_equivariance:
            return x_message
        
        # l=0 component pooling
        x_message_l_0 = x_message[:, 0, :] # (num_nodes, hidden_channels)
        graph_l_0 = self.pooling(x_message_l_0, node_batch) # (batch_size, hidden_channels)
        
        # output heads
        eps = self.spectra_mlp(graph_l_0) # (batch_size, 4002) 
        eps = eps.view(-1, 2) # (batch_size * 2001, 2) 
        drude = self.drude_mlp(graph_l_0).squeeze(-1)  # (batch_size, 1)
        
        return eps, drude