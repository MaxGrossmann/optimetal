"""
Abstractions that simplifies the change to a different pooling layer.
The input for each pooling layer is validated by 'optimetal.nn.utils.config_schema'.
https://pytorch-geometric.readthedocs.io/en/2.5.1/modules/nn.html#pooling-layers
"""

import torch
import torch_geometric

def mean_pooling() -> torch.nn.Module:
    """
    Base pooling operation.
    """
    return torch_geometric.nn.MeanAggregation()

def scalar_attentional_pooling(
    in_channels: int,
    activation: str = "relu",
) -> torch.nn.Module:
    """
    Simple scalar attention pooling.
    Ref.:
        https://arxiv.org/abs/1904.12787
    """
    gate = torch_geometric.nn.MLP(
        [in_channels, 1],
        act=activation,
        plain_last=True,
    )
    return torch_geometric.nn.AttentionalAggregation(gate_nn=gate)

class VectorAttentionPooling(torch.nn.Module):
    """
    Vector attention pooling.
    Ref.:
        https://doi.org/10.1103/PhysRevMaterials.8.L122201
    """
    
    def __init__(
        self,
        in_channels: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.mlp = torch_geometric.nn.MLP(
            [in_channels, in_channels], 
            act=activation,
            plain_last=True,
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        index: torch.Tensor,
    ) -> torch.Tensor:
        att = self.mlp(x)
        att = torch_geometric.utils.softmax(att, index=index)
        return torch_geometric.nn.pool.global_add_pool(x * att, index)
    
class Set2SetPooling(torch.nn.Module):
    """
    Set2Set pooling with a projection back to the original 'in_channels'.
    Ref.:
        https://arxiv.org/abs/1511.06391
    """
    
    def __init__(
        self,
        in_channels: int,
        processing_steps: int = 2,
    ) -> None:
        super().__init__()
        self.pooling = torch_geometric.nn.Set2Set(
            in_channels=in_channels, 
            processing_steps=processing_steps,
        )
        self.proj = torch.nn.Linear(
            2 * in_channels,
            in_channels,
            bias=True
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pooling(x, index=index)
        x = self.proj(x)
        return x
