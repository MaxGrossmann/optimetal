"""
Function that transform the data. 
Examples are:
    - Unit conversions.
    - Calculate the intraband dielectric function from the drude frequency.
    - Calculate the color perceived by humans from the dielectric function.
    - Convert the data into PyG 'Data' objects (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html).
"""

from __future__ import annotations

import warnings
import numpy as np
from tqdm import tqdm
from pymatgen.core import Structure
from importlib.resources import files
from scipy.interpolate import CubicSpline

import torch
from torch_geometric.data import Data

def ev2nm(eV: float) -> float:
    """
    Unit converstion from eV to nm. (Also works with numpy and torch arrays...)
    """
    return 1239.84193 / eV

def get_omega() -> np.ndarray:
    """
    To reduce the size of the dataset, we have not included the frequency axis in the database.
    This function only outputs it. The same frequency grid is used throughout the dataset.
    Output:
        omega:      Frequency/energy grid
    """
    omega = np.linspace(0, 20, 2001)
    return omega

def calc_eps_intra(df: np.ndarray, gamma: float = 0.1) -> np.ndarray:
    """
    Calculates the intraband dielectric function from the drude frequency.
    (We tested this function by reconstructing the total dielectric found in the database from the compressed database, see 'db_sanity_check.py'.)
    Input:
        df:             drude Frequency (eV)
        gamma:          Broadening parameter (eV) (~ 1 / scattering time)
    Output:
        eps_intra:      Complex intraband dielectric function
    """
    omega = get_omega()
    # numerically better than Eq. 2 in https://doi.org/10.1038/s41524-019-0266-0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eps_intra_re = -(df ** 2) / (omega ** 2 + gamma ** 2)
        eps_intra_im = (df ** 2) * gamma / (omega * (omega ** 2 + gamma ** 2))
        eps_intra = eps_intra_re + 1j * eps_intra_im
    return eps_intra

def load_cie_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the reference CIE data needed to calculate the perceived color of a material.
    The data was downloaded from: 
        https://cie.co.at/datatable/cie-standard-illuminant-d65
        https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer
    Output:
        lbd_range:          Wave length range in nm (part of the spectrum that is visible to the human eye)  
        d65_illuminant:     Standardized light source that represents average daylight
        cmf:                CIE 1931 2 degree standard observer
    """
    # wavelength range visible to the human eye
    lbd_range = np.array([380, 780])
    
    # load the CIE standard illuminant D65
    cie_std_path = files("optimetal_e3.files").joinpath("CIE_std_illum_D65.csv")
    data = np.loadtxt(cie_std_path, delimiter=",")
    idx_vis = (data[:, 0] >= lbd_range[0]) & (data[:, 0] <= lbd_range[1])
    d65_illuminant = data[idx_vis, 1]
    
    # load the CIE 1931 2 degree standard observer (used more often than the 1964 10 degree standard observer)
    cie_xyz_path = files("optimetal_e3.files").joinpath("CIE_xyz_1931_2deg.csv")
    data = np.loadtxt(cie_xyz_path, delimiter=",")
    idx_vis = (data[:, 0] >= lbd_range[0]) & (data[:, 0] <= lbd_range[1])
    cmf = data[idx_vis, :]
    return lbd_range, d65_illuminant, cmf

def calc_color(eps: np.ndarray) -> np.ndarray:
    """
    Calculate the color perceived by humans from the dielectric function.
    Input:
        eps:        Complex dielectric function (interband + intraband, averaged over all crystal axes)
    Output:
        color:      CIE L*a*b* color (L*, a*, b*)
    """
    # load the CIE data
    lbd_range, d65_illuminant, cmf = load_cie_data()
    
    # remove the 'omega=0' and convert from frequency to wavelength
    # (flip the array so the order goes from small to large wavelength)
    omega = get_omega()[1:] # ignore 'omega=0'
    lbd = np.flip(ev2nm(omega))
    eps = np.flip(eps[1:]) # ignore 'omega=0'
    idx_vis = (lbd >= lbd_range[0]) & (lbd <= lbd_range[1])
    lbd = lbd[idx_vis]
    eps = eps[idx_vis]

    # convert to refractive index and extinction coefficient
    norm_eps = np.sqrt(eps.real ** 2 + eps.imag ** 2)
    n = np.sqrt((norm_eps + eps.real) / 2)
    k = np.sqrt((norm_eps - eps.real) / 2)

    # calculate the reflectivity at the air-metal interface at normal incidence (assuming 'n_air=1' and 'k_air=0')
    reflec = ((1 - n) ** 2 + k ** 2) / ((1 + n) ** 2 + k ** 2)

    # interpolate the reflectivity on the wavelength grid of the CIE data
    reflec_itp = CubicSpline(lbd, reflec)
    lbd_grid_vis = np.linspace(
        lbd_range[0], lbd_range[1], lbd_range[1] - lbd_range[0] + 1, dtype=int
    )
    reflec = reflec_itp(lbd_grid_vis)

    # split the CIE 1931 2 degree standard observer into x, y and z
    cmf_x = cmf[:, 1]
    cmf_y = cmf[:, 2]
    cmf_z = cmf[:, 3]

    # calculate renormalization constant
    k_norm = 100.0 / np.sum(cmf_y * d65_illuminant)

    # calculate x_n, y_n and z_n (y_n=100)
    x_n = k_norm * np.sum(cmf_x * d65_illuminant)
    y_n = k_norm * np.sum(cmf_y * d65_illuminant)
    z_n = k_norm * np.sum(cmf_z * d65_illuminant)

    # tristimulus values
    x = k_norm * np.sum(cmf_x * reflec * d65_illuminant)
    y = k_norm * np.sum(cmf_y * reflec * d65_illuminant)
    z = k_norm * np.sum(cmf_z * reflec * d65_illuminant)

    # calculate the color in the CIE L*a*b* color space
    # (https://en.wikipedia.org/wiki/CIELAB_color_space)
    def f(x: float) -> float:
        delta = 6 / 29
        if x > delta ** 3:
            return x ** (1 / 3)
        elif x <= delta ** 3:
            return (1 / 3) * (delta ** (-2)) * x + (4 / 29)
    l = 116 * f(y/y_n) - 16
    a = 500 * (f(x/x_n) - f(y/y_n))
    b = 200 * (f(y/y_n) - f(z/z_n))
    return np.array([l, a, b])

def split_complex_vector(complex_vector: np.ndarray) -> np.ndarray:
    """
    Splits a vector of complex numbers into an array of size (n, 2), separating the real and imaginary parts.
    Input:
        complex_vector:     1D array of complex numbers of size (n)
    Output:
        stack:              2D array of size (n, 2), where the first column is 
                            the real part and the second column is the imaginary part
    """
    real_part = complex_vector.real
    imag_part = complex_vector.imag
    stack = np.stack((real_part, imag_part), axis=-1)
    return stack

def graph_setup(
    data_list: list, 
    rcut: float = 5.5, 
) -> list[Data]:
    """
    Converts the input data into a custom PyG 'Data' object.
    Input:
        data_list:      List of custom PyG 'Data' objects containing the crystal structure and material properties
        rcut:           Cutoff radius in angstroms for finding bonds
    Output:
        graph_list:     List with PyG 'Data' objects
    """
    # create the graph and store them in the custom 'Data' object
    graph_list = []
    for data in tqdm(data_list, desc="Setting up graphs"):
        # useful variables
        lattice = data["lattice"]
        atomic_number = data["atomic_number"]
        position = data["position"]
        
        # obtain bond information
        struct = Structure(lattice, atomic_number, position, coords_are_cartesian=True)
        src_idx, dst_idx, pbc_offset, bond_lengths = struct.get_neighbor_list(
            rcut,
            numerical_tol=1e-8, # pymatgen default
            exclude_self=True, # pymatgen default
        )
        edge_index = np.array([src_idx, dst_idx]) # COO format, (2, num_edge)
        num_edge = edge_index.shape[1]
        
        # target properties
        drude = data["drude"]
        eps_intra = calc_eps_intra(drude)
        eps_inter = data["eps"]
        eps_tot = eps_intra + eps_inter
        eps_matrix = split_complex_vector(eps_inter)
        color = calc_color(eps_tot)
        
        # convert everything to torch tensors
        atomic_number = torch.tensor(atomic_number, dtype=torch.long) # (num_node)
        position = torch.tensor(position, dtype=torch.float) # (num_node, 3)
        lattice = torch.tensor(lattice, dtype=torch.float) # (3, 3)
        num_edge = torch.tensor(num_edge, dtype=torch.long) # (1)
        edge_index = torch.tensor(edge_index, dtype=torch.long) # (2, num_edge)
        pbc_offset = torch.tensor(pbc_offset, dtype=torch.float) # (num_edge, 3)
        drude = torch.tensor(drude, dtype=torch.float) # (1)
        eps = torch.tensor(eps_matrix, dtype=torch.float) # (2001, 2)
        color = torch.tensor(color, dtype=torch.float) # (3)
        
        # metadata
        mat_id = data["mat_id"]
        e_above_hull = torch.tensor(data["e_above_hull"], dtype=torch.float)
        composition = data["composition"]
        
        # create a data object and append to the list
        graph = Data(
            num_nodes=atomic_number.shape[0], # needed because I omitted 'data.x' in the object
            atomic_number=atomic_number,
            position=position,
            lattice=lattice,
            num_edge=num_edge,
            edge_index=edge_index,
            pbc_offset=pbc_offset, 
            drude=drude,
            eps=eps,
            color=color,
            mat_id=mat_id,
            e_above_hull=e_above_hull,
            composition=composition,
        )
        graph_list.append(graph)
    return graph_list