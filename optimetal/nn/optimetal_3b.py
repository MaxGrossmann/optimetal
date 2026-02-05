"""
Three-body version of the main model. It is written in a flexible and robust way that allows for easy architecture
and hyperparameter optimization. This version includes three-body interactions, which means it uses atom and bond
distance as well as angular information.
"""

from __future__ import annotations

from copy import deepcopy
from pydantic import TypeAdapter

import torch

import optimetal.nn.utils.config_schema as cs
from optimetal.utils import ACTIVATIONS
from optimetal.data.dataset import ThreebodyData
from optimetal.nn.utils.triplet_block import TripletBlockValidator

class OptiMetal3B(torch.nn.Module):
    """
    Three-body version of the main model.
    Each input is validated using 'pydantic', see 'optimetal.nn.utils.config_schema'.
    """

    def __init__(
        self, 
        node_embedding_dict: dict | None = None,
        edge_embedding_dict: dict | None = None,
        angle_embedding_dict: dict | None = None,
        triplet_block_dict: dict | None = None,
        pooling_dict: dict | None = None,
        hidden_dim: int = 256,
        spectra_dim: int = 2048,
        depth: int = 2,
        activation: str ="relu",
        twobody_cutoff: float = 5.5 # hardcoded, see 'research/db_init.py'
    ) -> None:
        super().__init__()
        
        # check if the activation function is okay
        if activation not in ACTIVATIONS:
            raise ValueError("Unsupported 'activation' function, see 'optimetal.utils'")
        
        # check if the twobody cutoff is fine
        if twobody_cutoff <= 0:
            raise ValueError("The 'twobody_cutoff' must be greater than zero")
        
        # check if the hidden dimension, spectra dimension, and depth is fine
        if hidden_dim <= 0:
            raise ValueError("The 'hidden_dim' must be greater than zero")
        if spectra_dim <= 0:
            raise ValueError("The 'spectra_dim' must be greater than zero")
        if depth <= 0:
            raise ValueError("The 'depth' must be greater than zero")
        
        # defaults as fallback
        if node_embedding_dict is None:
            node_embedding_dict = {
                "type": "atom",
                "embedding_dim": 64,
            }
        if edge_embedding_dict is None:
            edge_embedding_dict = {
                "type": "gaussian",
                "num_basis": 64,
                "basis_width": 2.0,
                "apply_envelope": False,
            }
        if angle_embedding_dict is None:
            angle_embedding_dict = {"l_max": 3}
        if triplet_block_dict is None:
            triplet_block_dict = {
                "num_layers": 2,
                "edge_graph_mp_dict" : {
                    "type": "transformer",
                    "heads": 4,
                },
                "node_graph_mp_dict" : {
                    "type": "transformer",
                    "heads": 4,
                },
            }
        if pooling_dict is None:
            pooling_dict = {
                "type": "vector_attention",
            }
    
        # node embedding function
        node_config = TypeAdapter(cs.NodeEmbeddingValidator).validate_python(node_embedding_dict)
        self.node_embedding = node_config.make()
        node_embedding_dim = self.node_embedding.embedding_dim
        
        # mlp to make the node embeddings more expressive
        self.node_mlp = cs.make_mlp(
            in_channels=node_embedding_dim,
            out_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_hidden=depth,
            activation=activation,
        )
        
        # edge embedding function
        edge_config = TypeAdapter(cs.EdgeEmbeddingValidator).validate_python(edge_embedding_dict)
        self.edge_embedding = edge_config.make(r_max=twobody_cutoff)
        edge_embedding_dim = self.edge_embedding.embedding_dim
        
        # mlp to make the edge embeddings more expressive
        self.edge_mlp = cs.make_mlp(
            in_channels=edge_embedding_dim,
            out_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_hidden=depth,
            activation=activation,
        )
        
        # angle embedding function
        angle_config = TypeAdapter(cs.AngleEmbeddingValidator).validate_python(angle_embedding_dict)
        self.angle_embedding = angle_config.make()
        angle_embedding_dim = self.angle_embedding.embedding_dim
        
        # mlp to make the angle embeddings more expressive
        self.angle_mlp = cs.make_mlp(
            in_channels=angle_embedding_dim,
            out_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_hidden=depth,
            activation=activation,
        )
        
        # manipulate the message-passing sub-dictionaries to ensure that only 
        # one message-passing step occurs for each part of the triplet propagator
        # and that the activation function remains consistent throughout the network
        triplet_block_dict = deepcopy(triplet_block_dict)
        if "edge_graph_mp_dict" in triplet_block_dict:
            triplet_block_dict["edge_graph_mp_dict"]["num_layers"] = 1
            triplet_block_dict["edge_graph_mp_dict"]["activation"] = activation
        else: 
            print("The 'edge_graph_mp_dict' is missing in the 'triplet_block_dict'", flush=True)
            raise KeyError
        if "node_graph_mp_dict" in triplet_block_dict:
            triplet_block_dict["node_graph_mp_dict"]["num_layers"] = 1
            triplet_block_dict["node_graph_mp_dict"]["activation"] = activation
        else:
            print("The 'node_graph_mp_dict' is missing in the 'triplet_block_dict'", flush=True)
            raise KeyError
        
        # triplet block setup
        triplet_config = TypeAdapter(TripletBlockValidator).validate_python(triplet_block_dict)
        self.triplet_block = triplet_config.make(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            angle_dim=hidden_dim,
            activation=activation,
        )
        
        # pooling layer
        if "type" in pooling_dict:
            # fix the activation function to match the rest of the network
            if pooling_dict["type"] in ["scalar_attention", "vector_attention"]:
                pooling_dict["activation"] = activation
        else:
            print("The 'pooling_dict' is not valid as it is missing the 'type' key", flush=True)
            raise KeyError
        pooling_config = TypeAdapter(cs.PoolingValidator).validate_python(pooling_dict)
        self.pooling = pooling_config.make(in_channels=hidden_dim)
        
        # mlp to obtain the real and imaginary part of the interband dielectric function
        self.spectra_mlp = cs.make_mlp(
            in_channels=hidden_dim,
            out_channels=4002,
            hidden_channels=spectra_dim,
            num_hidden=depth,
            activation=activation,
        )
        
        # mlp to obtain the Drude frequency
        self.drude_mlp = cs.make_mlp(
            in_channels=hidden_dim,
            out_channels=1,
            hidden_channels=hidden_dim,
            num_hidden=depth,
            activation=activation,
        )
        
    def forward(self, graph: ThreebodyData) -> tuple[torch.Tensor]:
        # extract variables
        atomic_number = graph.atomic_number
        position = graph.position
        lattice = graph.lattice
        edge_index = graph.edge_index
        pbc_offset = graph.pbc_offset
        threebody_index = graph.threebody_index
        node_batch = graph.batch
        edge_batch = graph.edge_index_batch
        angle_batch = graph.threebody_index_batch
        
        # node embeddings
        node_features = self.node_embedding(atomic_number)
        
        # edge embeddings (note that 'torch.bmm' is not deterministic...)
        edge_vector = torch.bmm(               # this is the best way I found and understood
            pbc_offset.unsqueeze(1),           # shape: (num_edges, 1, 3)
            lattice.view(-1, 3, 3)[edge_batch] # shape: (num_edges, 3, 3)
        ).squeeze(1)                           # shape: (num_edges, 3)
        edge_vector = edge_vector + position[edge_index[1]] - position[edge_index[0]]
        edge_length = torch.norm(edge_vector, dim=1, p=2)
        edge_attr = self.edge_embedding(edge_length)
        
        # angle embeddings ("borrowed" from mattersim)
        vij = edge_vector[threebody_index[0]]
        vik = edge_vector[threebody_index[1]]
        rij = edge_length[threebody_index[0]]
        rik = edge_length[threebody_index[1]]
        cos_jik = torch.sum(vij * vik, dim=1) / (rij * rik)
        cos_jik = torch.clamp(cos_jik, min=-1.0+1e-7, max=1.0-1e-7) # avoid nan
        angle_attr = self.angle_embedding(cos_jik)
        
        # the actual network
        x = self.node_mlp(node_features)
        edge_attr = self.edge_mlp(edge_attr)
        angle_attr = self.angle_mlp(angle_attr)
        x = self.triplet_block(
            x, 
            node_batch,
            edge_index, 
            edge_attr, 
            edge_batch,
            threebody_index, 
            angle_attr,
            angle_batch,
        )
        x = self.pooling(x, index=node_batch)
        eps = self.spectra_mlp(x) # shape: (batch_size, 4002) 
        drude = torch.nn.functional.relu(self.drude_mlp(x)) # shape: (batch_size, 1)
        
        # output reshaping to make it compatible with the rest of the code
        eps = eps.view(-1, 2) # shape: (batch_size * 2001, 2) 
        drude = drude.view(-1) # shape: (batch_size) 
        return eps, drude
    