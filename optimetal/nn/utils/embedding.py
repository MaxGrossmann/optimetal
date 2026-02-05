"""
Methods of embedding atoms, bonds and angles.
Each embedding module must define a class variable called 'embedding_dim'.
The input for embedding is validated by 'optimetal.nn.utils.config_schema'.
"""

from __future__ import annotations

import torch

class AtomEmbedding(torch.nn.Module):
    """
    A simple embedding layer for atomic numbers to obtain node embeddings.
    """

    def __init__(
        self, 
        embedding_dim: int, 
        num_elements: int = 83, # hardcoded because the database only contains elements up to and including bismuth
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim # needs to be a class variable
        self.embeddings = torch.nn.Embedding(num_elements, embedding_dim, sparse=False) # maybe try 'sparse=True'?

    def forward(self, atomic_number: torch.Tensor) -> torch.Tensor:
        return self.embeddings(atomic_number - 1) # convert the atomic numbers to zero-based
    
class GroupPeriodEmbedding(torch.nn.Module):
    """
    We concatenate the one-hot encoding of an atom's group and row in 
    the periodic table to obtain a node embedding from atomic numbers.
    The current implementation only works for elements up to and including bismuth.
    Ref.:
        https://doi.org/10.1103/PhysRevMaterials.8.L122201.
    """
    
    def __init__(self) -> None:
        super().__init__()
        # mapping atomic numbers to group numbers using a zero-based index for one-hot encoding
        group_mapping = [
            0, # dummy index
            1, 18, 
            1, 2, 13, 14, 15, 16, 17, 18,
            1, 2, 13, 14, 15, 16, 17, 18,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
            1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        ]
        group_mapping = torch.tensor(group_mapping, dtype=torch.long) - 1
        self.register_buffer("group_mapping", group_mapping, persistent=False)
        # mapping atomic numbers to row numbers using a zero-based index for one-hot encoding
        row_mapping = [
            0, # dummy index
            1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 
            3, 3, 3, 3, 3, 3, 3, 3,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6
        ]
        row_mapping = torch.tensor(row_mapping, dtype=torch.long) - 1
        self.register_buffer("row_mapping", row_mapping, persistent=False)
        # define the embedding dimension
        self.embedding_dim = 24 # needs to be a class variable
    
    def forward(self, atomic_number: torch.Tensor) -> torch.Tensor:
        groups = self.group_mapping[atomic_number]
        rows = self.row_mapping[atomic_number]
        one_hot_groups = torch.nn.functional.one_hot(groups, num_classes=18).to(torch.float32)
        one_hot_rows = torch.nn.functional.one_hot(rows, num_classes=6).to(torch.float32)
        embedding = torch.cat([one_hot_groups, one_hot_rows], dim=1)
        return embedding
    
class PolynomialEnvelope(torch.nn.Module):
    """
    Polynomial envelope function that ensures a smooth cutoff.
    Refs.:
        https://arxiv.org/pdf/2003.03123
        https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/models/uma/nn/radial.py
    """

    def __init__(
        self, 
        exponent: int = 4, # hardcoded, as it seems to insensitive to the exponent, see https://arxiv.org/abs/2003.03123
    ) -> None:
        super().__init__()
        self.p = float(exponent)
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        env_val = 1 + (d_scaled**self.p) * (self.a + d_scaled * (self.b + self.c * d_scaled))
        return torch.where(d_scaled < 1, env_val, 0)
    
    
class GaussianEmbedding(torch.nn.Module):
    """
    Bond distance embeddings are obtained using Gaussian radial basis functions. The standard deviation of each Gaussian is given by 
    'basis width' multiplied by the spacing between neighboring centers. Therefore, one can control the amount of overlap between neighboring
    Gaussians through 'basis_width'. The inclusion of a polynomial envelope is optional. However, it tends to improve model smoothness and performance.
    Refs.: 
        https://arxiv.org/pdf/2003.03123
        https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/models/uma/nn/radial.py
    """
    
    def __init__(
        self,
        r_min: float = 0.0,
        r_max: float = 5.5,
        num_basis: int = 64,
        basis_width: float = 2.0,
        apply_envelope: bool = False,
        envelope_exp: int = 4, # hardcoded, as it seems to insensitive to the exponent, see https://arxiv.org/abs/2003.03123
    ) -> None:
        super().__init__()
        self.embedding_dim = num_basis # needs to be a class variable
        offset = torch.linspace(r_min, r_max, num_basis)
        self.register_buffer("offset", offset, persistent=False)
        self.coeff = -0.5 / (basis_width * (offset[1] - offset[0])).item() ** 2
        self.apply_envelope = apply_envelope
        if self.apply_envelope:
            self.r_max = r_max
            self.envelope = PolynomialEnvelope(envelope_exp)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.unsqueeze(-1)
        diff = dist - self.offset.unsqueeze(0)
        gaussian = torch.exp(self.coeff * torch.pow(diff, 2))
        if self.apply_envelope:
            d_scaled = dist / self.r_max
            return self.envelope(d_scaled) * gaussian
        else:
            return gaussian

class BesselEmbedding(torch.nn.Module):
    """
    Bond distance embeddings are obtained using Bessel radial basis functions. The inclusion of a polynomial envelope is optional. 
    However, it tends to improve model smoothness and performance. The Bessel wave numbers (k_n = n*pi/c) can be made trainable parameters.
    Refs.: 
        https://arxiv.org/pdf/2003.03123
        https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html
    """
    
    def __init__(
        self,
        r_max: float = 5.5, 
        num_basis: int = 8, 
        trainable: bool = False,
        apply_envelope: bool = False,
        envelope_exp: int = 4, # hardcoded, as it seems to insensitive to the exponent, see https://arxiv.org/abs/2003.03123
    ) -> None:
        super().__init__()
        self.embedding_dim = num_basis # needs to be a class variable
        self.r_max = torch.tensor(r_max)
        self.prefactor = torch.sqrt(2.0 / self.r_max)
        bessel_weights = torch.linspace(start=1.0, end=num_basis, steps=num_basis) * torch.pi
        self.trainable = trainable
        if self.trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)
        self.apply_envelope = apply_envelope
        if self.apply_envelope:
            self.envelope = PolynomialEnvelope(envelope_exp)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.unsqueeze(-1)
        bessel = self.prefactor * torch.sin(self.bessel_weights * dist / self.r_max) / dist
        if self.apply_envelope:
            d_scaled = dist / self.r_max
            return self.envelope(d_scaled) * bessel
        else:
            return bessel
    
class AngleEmbedding(torch.nn.Module):
    """
    Embedding cosines of angles through m=0 spherical harmonics.
    Refs.:
        https://doi.org/10.1038/s43588-022-00349-3
        https://arxiv.org/abs/2405.04967
    """

    def __init__(
        self, 
        l_max: int = 4,
    ) -> None:
        super().__init__()
        if l_max > 4:
            raise ValueError("lmax is only supported up to 4...")
        self.l_max = l_max
        self.embedding_dim = l_max + 1 # needs to be a class variable
        pi = torch.tensor(torch.pi)
        self.register_buffer("c0", 0.5 * torch.sqrt(1.0 / pi))
        self.register_buffer("c1", torch.sqrt(3.0 / (4.0 * pi)))
        self.register_buffer("c2", torch.sqrt(5.0 / (16.0 * pi)))
        self.register_buffer("c3", torch.sqrt(7.0 / (16.0 * pi)))
        self.register_buffer("c4", torch.sqrt(9.0 / (256.0 * pi)))

    def forward(self, cos_angle: torch.Tensor) -> torch.Tensor:
        features = [self.c0.expand_as(cos_angle)]
        if self.l_max >= 1:
            features.append(self.c1 * cos_angle)
        if self.l_max >= 2:
            features.append(self.c2 * (3.0 * cos_angle**2 - 1.0))
        if self.l_max >= 3:
            features.append(self.c3 * cos_angle * (5.0 * cos_angle**2 - 3.0))
        if self.l_max >= 4:
            features.append(self.c4 * (35.0 * cos_angle**4 - 30.0 * cos_angle**2 + 3.0))
        return torch.stack(features, dim=-1)