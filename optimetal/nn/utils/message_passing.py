"""
Abstractions that simplifies the change to different message passing layer.
Each message passing module must retain the dimensions of node, edge and angle embeddings.
The input for each message passing layer is validated by 'optimetal.nn.utils.config_schema'.
https://pytorch-geometric.readthedocs.io/en/2.5.1/modules/nn.html#convolutional-layers

All of the message passing layers here are inspired by a transformer encoder block. 
That's why they all contain a residual connection, an MLP, and layer normalization after the actual message passing step.
See Fig. 1 at https://arxiv.org/abs/1706.03762 for a visual representation of the transformer encoder block.
"""

import torch
import torch_geometric

class CGConvBlock(torch.nn.Module):
    """
    Message passing block based on successive layers containing the crystal graph convolutional operator.
    For this operator, 'in_channels' equals 'out_channels'.
    Ref.:
        https://doi.org/10.1103/PhysRevLett.120.145301
    """
    
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        edge_dim: int,
        hidden_multiplier: int = 4,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        hidden_channels = hidden_multiplier * in_channels
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                torch.nn.ModuleDict(
                    {
                        "conv": torch_geometric.nn.CGConv(
                            channels=in_channels, 
                            dim=edge_dim,
                            aggr="add", # seems most appropriate
                        ),
                        "mlp": torch_geometric.nn.MLP(
                            [in_channels, hidden_channels, in_channels],
                            act=activation,
                            plain_last=True,
                        ),
                        "norm": torch_geometric.nn.LayerNorm(
                            in_channels,
                            mode="graph",
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
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer["conv"](x, edge_index, edge_attr)
            xp = layer["mlp"](x)
            x = x + xp
            x = layer["norm"](x, batch=node_batch)
        return x

class GATv2Block(torch.nn.Module):
    """
    Message passing block based on successive layers containing the GATv2 operator.
    The attention heads are concatenated after each message passing step. The only free parameter 
    is the number of attention heads as the output dimension is set so that, after concatenating
    the attention heads, we return to the input dimension. This approach ensures consistent embedding
    dimensions and allows for easy stacking of message-passing layers.
    Ref.:
        https://arxiv.org/abs/2105.14491
    """
    
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        heads: int,
        edge_dim: int,
        hidden_multiplier: int = 4,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if in_channels % heads != 0:
            raise ValueError("The 'in_channels' must be divisible by 'heads'")
        out_channels = in_channels // heads
        hidden_channels = hidden_multiplier * in_channels
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                torch.nn.ModuleDict(
                    {
                        "conv": torch_geometric.nn.GATv2Conv(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            heads=heads,
                            concat=True,
                            add_self_loops=False,
                            edge_dim=edge_dim,
                            residual=True, # fixed residual connection
                        ),
                        "mlp": torch_geometric.nn.MLP(
                            [in_channels, hidden_channels, in_channels],
                            act=activation,
                            plain_last=True,
                        ),
                        "norm": torch_geometric.nn.LayerNorm(
                            in_channels,
                            mode="graph",
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
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer["conv"](x, edge_index, edge_attr)
            xp = layer["mlp"](x)
            x = x + xp
            x = layer["norm"](x, batch=node_batch)
        return x

class TransformerBlock(torch.nn.Module):
    """
    Message passing block based on successive layers containing the graph transformer operator.
    The attention heads are concatenated after each message passing step. The only free parameter 
    is the number of attention heads as the output dimension is set so that, after concatenating
    the attention heads, we return to the input dimension. This approach ensures consistent embedding
    dimensions and allows for easy stacking of message-passing layers.
    Ref.:
        https://arxiv.org/abs/2009.03509
    """
    
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        heads: int,
        edge_dim: int,
        hidden_multiplier: int = 4,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if in_channels % heads != 0:
            raise ValueError("The 'in_channels' must be divisible by 'heads'")
        out_channels = in_channels // heads
        hidden_channels = hidden_multiplier * in_channels
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                torch.nn.ModuleDict(
                    {
                        "conv": torch_geometric.nn.TransformerConv(
                            in_channels,
                            out_channels,
                            heads=heads,
                            concat=True,
                            edge_dim=edge_dim,
                            beta=True, # combine aggregation and skip information
                        ),
                        "mlp": torch_geometric.nn.MLP(
                            [in_channels, hidden_channels, in_channels],
                            act=activation,
                            plain_last=True,
                        ),
                        "norm": torch_geometric.nn.LayerNorm(
                            in_channels,
                            mode="graph",
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
    ):
        for layer in self.layers:
            x = layer["conv"](x, edge_index, edge_attr)
            xp = layer["mlp"](x)
            x = x + xp
            x = layer["norm"](x, batch=node_batch)
        return x
