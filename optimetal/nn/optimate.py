"""
OptiMate architecture from https://doi.org/10.1103/PhysRevMaterials.8.L122201 (300 meV broadening).
This model is used here as a reference to see how well it learns the imaginary part of the interband 
dielectric functions of metals. For the real part of the interband dielectric functions and Drude 
frequency it outputs zero. I made some minor adjustments to make it compatible with the data structure.
"""

from __future__ import annotations

import torch
import torch_geometric

from optimetal.data.dataset import ThreebodyData
from optimetal.nn.utils.embedding import GroupPeriodEmbedding, GaussianEmbedding
from optimetal.nn.utils.pooling import VectorAttentionPooling

class OptiMate(torch.nn.Module):
    """
    Relatively simple model. The architecture parameters are taken from https://doi.org/10.1103/PhysRevMaterials.8.L122201.
    Changing them is not intended. If you want to experiment with the architecture, please use the OptiMetal2B model.
    """
    
    def __init__(
        self,
        twobody_cutoff: float = 5.5, # hardcoded, see 'research/db_init.py'
    ) -> None:
        super().__init__()
        
        # check if the twobody cutoff is fine
        if twobody_cutoff < 0:
            raise ValueError("The 'twobody_cutoff' must be greater than zero")
        
        # node embedding functions
        self.node_embedding = GroupPeriodEmbedding()
        
        # edge embedding function
        self.edge_embedding = GaussianEmbedding(
            r_min=0.0, 
            r_max=twobody_cutoff, 
            num_basis=56,
            basis_width=2.0,
            apply_envelope=False,
        )
        
        # mlp to make the node embeddings more expressive
        self.node_mlp = torch_geometric.nn.MLP(
            [24, 48, 48], 
            act="relu",
            plain_last=True,
        )
        
        # message passing layers
        self.gat_mp_1 = torch_geometric.nn.GATv2Conv(
            in_channels=48,
            out_channels=48,
            heads=4,
            concat=True,
            add_self_loops=False,
            edge_dim=56,
        )
        self.gat_mp_2 = torch_geometric.nn.GATv2Conv(
            in_channels=192,
            out_channels=96,
            heads=4,
            concat=True,
            add_self_loops=False,
            edge_dim=56,
        )
        
        # vector attention pooling layer
        self.pooling = VectorAttentionPooling(
            in_channels=384,
            activation="relu",
        )
        
        # mlp to obtain the imaginary part of the interband dielectric function
        self.mlp_eps_imag = torch_geometric.nn.MLP(
            [384, 1024, 1024, 1024, 2001], 
            act="relu",
            plain_last=True,
        )

    def forward(self, graph: ThreebodyData) -> tuple[torch.Tensor]:
        # extract variables
        atomic_number = graph.atomic_number
        position = graph.position
        lattice = graph.lattice
        edge_index = graph.edge_index
        pbc_offset = graph.pbc_offset
        batch_size = graph.num_graphs
        node_batch = graph.batch
        edge_batch = graph.edge_index_batch
        
        # node embeddings
        node_features = self.node_embedding(atomic_number)
        
        # edge embeddings (note that 'torch.bmm' is not deterministic...)
        edge_vectors = torch.bmm(              # this is the best way I found and understood
            pbc_offset.unsqueeze(1),           # shape: (num_edges, 1, 3)
            lattice.view(-1, 3, 3)[edge_batch] # shape: (num_edges, 3, 3)
        ).squeeze(1)                           # shape: (num_edges, 3)
        edge_vectors = edge_vectors + position[edge_index[1]] - position[edge_index[0]]
        edge_length = torch.norm(edge_vectors, dim=1, p=2)
        edge_attr = self.edge_embedding(edge_length)
        
        # the actual network, see https://doi.org/10.1103/PhysRevMaterials.8.L122201
        x = self.node_mlp(node_features)
        x = self.gat_mp_1(x, edge_index, edge_attr)
        x = self.gat_mp_2(x, edge_index, edge_attr)
        x = self.pooling(x, index=node_batch)
        x = self.mlp_eps_imag(x)
        imag_part = torch.nn.functional.leaky_relu(x) # shape: (batch_size, 2001) 
        
        # customize the model output to fit with the rest of code
        real_part = torch.zeros_like(imag_part) # defaults to the device of input
        eps = torch.stack([real_part, imag_part], dim=-1) # shape: (batch_size, 2001, 2)
        eps = eps.view(-1, 2) # shape: (batch_size * 2001, 2)
        drude = torch.zeros(batch_size, dtype=torch.float32, device=atomic_number.device) # shape: (batch_size)
        return eps, drude