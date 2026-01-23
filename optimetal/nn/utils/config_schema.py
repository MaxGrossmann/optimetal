"""
The schemas below are used to validate the dictionaries used to create parts of the OptiMetal architecture.
"""

from typing import Literal, Annotated, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator

import torch
import torch_geometric

import optimetal.nn.utils.embedding as embedding
import optimetal.nn.utils.message_passing as message_passing
import optimetal.nn.utils.pooling as pooling
from optimetal.utils import ACTIVATIONS

def make_mlp(
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    num_hidden: int,
    activation: str,
) -> torch_geometric.nn.MLP:
    """
    MLP constructor.
    """
    if in_channels < 1:
        raise ValueError("The 'in_channels' must be greater than zero")
    if out_channels < 1:
        raise ValueError("The 'out_channels' must be greater than zero")
    if hidden_channels < 1:
        raise ValueError("The 'hidden_channels' must be greater than zero")
    if num_hidden < 0:
        raise ValueError("The 'num_hidden' must be zero or greater")
    if activation not in ACTIVATIONS:
        raise ValueError(f"Unsupported 'activation' function, see 'optimetal.utils'")
    dims = [in_channels] + [hidden_channels] * num_hidden + [out_channels]
    return torch_geometric.nn.MLP(dims, act=activation, plain_last=True)

class AtomEmbeddingValidator(BaseModel):
    """
    Ensure that only valid dictionaries are parsed into the atom embedding.
    """
    
    model_config = ConfigDict(extra="forbid")
    type: Literal["atom"]
    embedding_dim: int = Field(..., gt=0)
    num_elements: int = Field(default=83, gt=0)
    
    def make(self) -> torch.nn.Module:
        return embedding.AtomEmbedding(
            embedding_dim=self.embedding_dim,
        )

class GroupPeriodEmbeddingValidator(BaseModel):
    """
    Ensure that only valid dictionaries are parsed into the group period embedding.
    """
    
    model_config = ConfigDict(extra="forbid")
    type: Literal["group_period"]
    
    def make(self) -> torch.nn.Module:
        return embedding.GroupPeriodEmbedding()

# validator for node embedding that switches based on the type of embedding
NodeEmbeddingValidator = Annotated[
    Union[AtomEmbeddingValidator, GroupPeriodEmbeddingValidator], 
    Field(discriminator="type"),
]

class GaussianEmbeddingValidator(BaseModel):
    """
    Ensure that only valid dictionaries are parsed into the gaussian embedding.
    """
    
    model_config = ConfigDict(extra="forbid")
    type: Literal["gaussian"]
    num_basis: int = Field(..., gt=0)
    basis_width: float = Field(default=2.0, gt=0)
    apply_envelope: bool = False
    envelope_exp: int = Field(default=4, gt=0)
    
    def make(self, r_max: float) -> torch.nn.Module:
        return embedding.GaussianEmbedding(
            r_min=0.0,
            r_max=r_max,
            num_basis=self.num_basis,
            basis_width=self.basis_width,
            apply_envelope=self.apply_envelope,
            envelope_exp=self.envelope_exp,
        )


class BesselEmbeddingValidator(BaseModel):
    """
    Ensure that only valid dictionaries are parsed into the bessel embedding.
    """
    
    model_config = ConfigDict(extra="forbid")
    type: Literal["bessel"]
    num_basis: int = Field(..., gt=0)
    trainable: bool = False
    apply_envelope: bool = False
    envelope_exp: int = Field(default=4, gt=0)
    
    def make(self, r_max: float) -> torch.nn.Module:
        return embedding.BesselEmbedding(
            r_max=r_max,
            num_basis=self.num_basis,
            trainable=self.trainable,
            apply_envelope=self.apply_envelope,
            envelope_exp=self.envelope_exp,
        )

# validator for edge embedding that switches based on the type of embedding
EdgeEmbeddingValidator = Annotated[
    Union[GaussianEmbeddingValidator, BesselEmbeddingValidator], 
    Field(discriminator="type"),
]

class AngleEmbeddingValidator(BaseModel):
    """
    Ensure that only valid dictionaries are parsed into the angle embedding.
    """
    
    model_config = ConfigDict(extra="forbid")
    l_max: int = Field(..., gt=0, lt=5)
    
    def make(self) -> torch.nn.Module:
        return embedding.AngleEmbedding(l_max=self.l_max)

class CGConvValidator(BaseModel):
    """
    Ensure that only valid dictionaries are parsed into the crystal graph convolutional message passing block.
    """
    
    model_config = ConfigDict(extra="forbid")
    num_layers: int = Field(..., gt=0)
    type: Literal["cgconv"]
    hidden_multiplier: int = Field(default=4, gt=0)
    activation: str = "relu"

    @field_validator("activation")
    def check_activation(cls, v: str) -> str:
        if v not in ACTIVATIONS:
            raise ValueError(f"'activation' must be one of {ACTIVATIONS}, not {v!r}")
        return v
    
    def make(self, in_channels: int, edge_dim: int) -> torch.nn.Module:
        return message_passing.CGConvBlock(
            num_layers=self.num_layers,
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_multiplier=self.hidden_multiplier,
            activation=self.activation,
        )
        
class GATv2Validator(BaseModel):
    """
    Ensure that only valid dictionaries are parsed into the GATv2 message passing block.
    """
    
    model_config = ConfigDict(extra="forbid")
    num_layers: int = Field(..., gt=0)
    type: Literal["gatv2"]
    heads: int = Field(..., gt=0)
    hidden_multiplier: int = Field(default=4, gt=0)
    activation: str = "relu"

    @field_validator("activation")
    def check_activation(cls, v: str) -> str:
        if v not in ACTIVATIONS:
            raise ValueError(f"'activation' must be one of {ACTIVATIONS}, not {v!r}")
        return v
    
    def make(self, in_channels: int, edge_dim: int) -> torch.nn.Module:
        return message_passing.GATv2Block(
            num_layers=self.num_layers,
            in_channels=in_channels,
            heads=self.heads,
            edge_dim=edge_dim,
            hidden_multiplier=self.hidden_multiplier,
            activation=self.activation,
        )
        
class TransformerValidator(BaseModel):
    """
    Ensure that only valid dictionaries are parsed into the transformer message passing block.
    """
    
    model_config = ConfigDict(extra="forbid")
    num_layers: int = Field(..., gt=0)
    type: Literal["transformer"]
    heads: int = Field(..., gt=0)
    hidden_multiplier: int = Field(default=4, gt=0)
    activation: str = "relu"

    @field_validator("activation")
    def check_activation(cls, v: str) -> str:
        if v not in ACTIVATIONS:
            raise ValueError(f"'activation' must be one of {ACTIVATIONS}, not {v!r}")
        return v
    
    def make(self, in_channels: int, edge_dim: int) -> torch.nn.Module:
        return message_passing.TransformerBlock(
            num_layers=self.num_layers,
            in_channels=in_channels,
            heads=self.heads,
            edge_dim=edge_dim,
            hidden_multiplier=self.hidden_multiplier,
            activation=self.activation,
        )

# validator for message passing blocks that switches based on the type of message passing layer
MessagePassingValidator = Annotated[
    Union[CGConvValidator, GATv2Validator, TransformerValidator], 
    Field(discriminator="type"),
]

class MeanPoolingValidator(BaseModel):
    """
    Ensure that only valid dictionaries are parsed into the mean pooling function.
    """
    
    model_config = ConfigDict(extra="forbid")
    type: Literal["mean"]
    
    def make(self, in_channels: int) -> torch.nn.Module:
        return pooling.mean_pooling()

class ScalarAttentionPoolingValidator(BaseModel):
    """
    Ensure that only valid dictionaries are parsed into the scalar attention pooling function.
    """
    
    model_config = ConfigDict(extra="forbid")
    type: Literal["scalar_attention"]
    activation: str = "relu"

    @field_validator("activation")
    def check_activation(cls, v: str) -> str:
        if v not in ACTIVATIONS:
            raise ValueError(f"'activation' must be one of {ACTIVATIONS}, not {v!r}")
        return v
    
    def make(self, in_channels: int) -> torch.nn.Module:
        return pooling.scalar_attentional_pooling(
            in_channels=in_channels,
            activation=self.activation,
        )

class VectorAttentionPoolingValidator(BaseModel):
    """
    Ensure that only valid dictionaries are parsed into the vector attention pooling function.
    """
    
    model_config = ConfigDict(extra="forbid")
    type: Literal["vector_attention"]
    activation: str = "relu"

    @field_validator("activation")
    def check_activation(cls, v: str) -> str:
        if v not in ACTIVATIONS:
            raise ValueError(f"'activation' must be one of {ACTIVATIONS}, not {v!r}")
        return v
    
    def make(self, in_channels: int) -> torch.nn.Module:
        return pooling.VectorAttentionPooling(
            in_channels=in_channels,
            activation=self.activation,
        )

class Set2SetPoolingValidator(BaseModel):
    """
    Ensure that only valid dictionaries are parsed into the set2set pooling function.
    """
    
    model_config = ConfigDict(extra="forbid")
    type: Literal["set2set"]
    processing_steps: int = Field(default=3, gt=0)
    
    def make(self, in_channels: int) -> torch.nn.Module:
        return pooling.Set2SetPooling(
            in_channels=in_channels,
            processing_steps=self.processing_steps,
        )

# validator for pooling layers that switches based on the type of pooling
PoolingValidator = Annotated[
    Union[
        MeanPoolingValidator, 
        ScalarAttentionPoolingValidator,
        VectorAttentionPoolingValidator,
        Set2SetPoolingValidator,
    ],
    Field(discriminator="type"),
]