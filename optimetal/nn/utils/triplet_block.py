"""
Custom threebody message passing block.
"""

from pydantic import BaseModel, Field, ConfigDict, TypeAdapter

import torch
import torch_geometric

import optimetal.nn.utils.config_schema as cs

class TripletBlock(torch.nn.Module):
    """
    Threebody message passing block inspired by the architecture of DimeNet,
    M3GNET and NLP Transformers (see 'optimetal.nn.utils.message_passing').
    """
    
    def __init__(
        self,
        num_layers: int,
        node_dim: int,
        edge_dim: int,
        angle_dim: int,
        edge_graph_mp_dict: dict, # see 'optimetal.nn.utils.message_passing'
        node_graph_mp_dict: dict, # see 'optimetal.nn.utils.message_passing'
        activation: str = "relu", # currently not used
    ) -> None:
        super().__init__()
        
        # validate the configurations of the message passing layer
        edge_graph_config = TypeAdapter(cs.MessagePassingValidator).validate_python(edge_graph_mp_dict)
        node_graph_config = TypeAdapter(cs.MessagePassingValidator).validate_python(node_graph_mp_dict)
        
        # build the triplet blocks
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                torch.nn.ModuleDict(
                    {
                        # triplet encoder
                        # (projects back to the original embedding dimension, i.e., keeps it constant)
                        "node_edge_mlp": torch_geometric.nn.MLP(
                            [2 * node_dim + edge_dim, edge_dim],
                            act=activation, # not used
                            plain_last=True,
                        ),
                        "edge_norm": torch_geometric.nn.LayerNorm(
                            edge_dim,
                            mode="graph",
                        ),
                        "edge_angle_mlp": torch_geometric.nn.MLP(
                            [2 * edge_dim + angle_dim, angle_dim],
                            act=activation, # not used
                            plain_last=True,
                        ),
                        "angle_norm": torch_geometric.nn.LayerNorm(
                            angle_dim,
                            mode="graph",
                        ),
                        # triplet propagator
                        "edge_graph_mp": edge_graph_config.make(
                            in_channels=edge_dim, 
                            edge_dim=angle_dim,
                        ),
                        "node_graph_mp": node_graph_config.make(
                            in_channels=node_dim, 
                            edge_dim=edge_dim,
                        ),
                    }
                )
            )

    def forward(
        self, 
        x: torch.Tensor, 
        node_batch: torch.Tensor,
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor, 
        edge_batch: torch.Tensor, 
        threebody_index: torch.Tensor,
        angle_attr: torch.Tensor,
        angle_batch: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            # node-edge connection with residual connection
            src_e, dst_e = edge_index
            edge_cat = torch.cat([x[src_e], x[dst_e], edge_attr], dim=-1)
            edge_attr_update = layer["node_edge_mlp"](edge_cat)
            edge_attr = edge_attr + edge_attr_update
            edge_attr = layer["edge_norm"](edge_attr, batch=edge_batch)
            # edge-angle connection with residual connection
            src_t, dst_t = threebody_index
            angle_cat = torch.cat([edge_attr[src_t], edge_attr[dst_t], angle_attr], dim=-1)
            angle_attr_update = layer["edge_angle_mlp"](angle_cat)
            angle_attr = angle_attr + angle_attr_update
            angle_attr = layer["angle_norm"](angle_attr, batch=angle_batch)
            # edge-graph message passing with residual connection
            edge_attr = layer["edge_graph_mp"](edge_attr, edge_batch, threebody_index, angle_attr)
            # node-graph message passing with residual connection
            x = layer["node_graph_mp"](x, node_batch, edge_index, edge_attr)
        return x
    
class TripletBlockValidator(BaseModel):
    """
    Ensure that only valid dictionaries are parsed into the triplet message passing block.
    """
    
    model_config = ConfigDict(extra="forbid")
    num_layers: int = Field(..., gt=0)
    edge_graph_mp_dict: dict
    node_graph_mp_dict: dict

    def make(
        self,
        node_dim: int, 
        edge_dim: int, 
        angle_dim: int, 
        activation: str,
    ) -> torch.nn.Module:
        return TripletBlock(
            num_layers=self.num_layers,
            node_dim=node_dim,
            edge_dim=edge_dim,
            angle_dim=angle_dim,
            edge_graph_mp_dict=self.edge_graph_mp_dict,
            node_graph_mp_dict=self.node_graph_mp_dict,
            activation=activation,
        )