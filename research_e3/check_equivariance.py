"""
This script tests the equivariance of the OptiMetalE3 model by applying random 
rotations to the input structures. It then verifies that the final node embeddings,
right before pooling, transform accordingly while the invariant outputs remain unchanged.
"""

from __future__ import annotations

import math 
import argparse
import numpy as np
from pymatgen.core import Structure

from e3nn.math import direct_sum
from e3nn.o3 import matrix_to_angles, wigner_D

import torch
from torch_geometric.loader import DataLoader

"""
-----------------------------------------------------------------------------------------------------------------------
PARSE INPUT ARGUMENTS:
"""

parser = argparse.ArgumentParser(description="Model training script")
parser.add_argument(
    "--device_index", 
    type=int, 
    default=0, 
    help="Index of the GPU to use for training. The default is 0, i.e., the first GPU (-1 for CPU)",
)
args = parser.parse_args()

"""
-----------------------------------------------------------------------------------------------------------------------
"""

EPS = 1e-7

def save_structure_as_poscar(data: dict, filename: str) -> None:
    lattice = data["lattice"]
    atomic_number = data["atomic_number"]
    position = data["position"]
    struct = Structure(lattice, atomic_number, position, coords_are_cartesian=True)
    struct.to(fmt="poscar", filename=filename)

def random_rotation_matrix() -> np.ndarray:
    """
    Generate a random 3D rotation matrix using the quaternion method.
    Ref.:
        https://en.wikipedia.org/wiki/3D_rotation_group#Using_quaternions_of_unit_norm
    """
    u1, u2, u3 = np.random.rand(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * math.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * math.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * math.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * math.pi * u3)
    x, y, z, w = q1, q2, q3, q4
    R = np.empty((3, 3))
    R[0, 0] = 1 - 2 * (y*y + z*z)
    R[0, 1] = 2 * (x*y - z*w)
    R[0, 2] = 2 * (x*z + y*w)
    R[1, 0] = 2 * (x*y + z*w)
    R[1, 1] = 1 - 2 * (x*x + z*z)
    R[1, 2] = 2 * (y*z - x*w)
    R[2, 0] = 2 * (x*z - y*w)
    R[2, 1] = 2 * (y*z + x*w)
    R[2, 2] = 1 - 2 * (x*x + y*y)
    return R

@torch.no_grad()
def check_node_embedding_equivariance(
    device: torch.device, 
    model: torch.nn.Module, 
    dataloader: DataLoader,
    dataloader_rot: DataLoader, 
    R: torch.Tensor,
    atol: float = 5e-4,
    rtol: float = 5e-4,
) -> None:
    model.eval()
    model.to(device)
    R = torch.as_tensor(R, dtype=torch.float32)
    a, b, c = matrix_to_angles(R)
    wigner_d = direct_sum( # swap Euler angles to the internal convention
        *[wigner_D(l, *[-c, -b, -a]) for l in range(model.lmax + 1)]
    )
    wigner_d = wigner_d.to(device)
    for graph, graph_rot in zip(dataloader, dataloader_rot):
        graph.to(device)
        graph_rot.to(device)
        x_message = model(graph)
        x_message_rot = model(graph_rot)
        x_message_rot_aligned = torch.einsum("ij,njk->nik", wigner_d, x_message_rot)
        diff = x_message - x_message_rot_aligned
        err_max = diff.abs().max().item()
        rel = (diff.norm() / x_message.norm().clamp_min(EPS)).item()
        ok = torch.allclose(x_message, x_message_rot_aligned, rtol=rtol, atol=atol)
        print(f"    equivariance={str(ok):5s} |  rel_err={rel:.3e}  |  max_abs={err_max:.3e}")
        
@torch.no_grad()
def compare_outputs(
    eps: torch.Tensor, 
    drude: torch.Tensor, 
    eps_rot: torch.Tensor, 
    drude_rot: torch.Tensor, 
    batch_size: int, 
    rtol: float = 5e-4, 
    atol: float = 5e-4,
) -> dict:
    stat_dict = {
        "drude_ok": [],
        "eps_ok": [],
        "drude_max_abs": [],
        "eps_max_abs": [],
        "drude_max_rel": [],
        "eps_max_rel": [],
    }
    eps = eps.view(batch_size, -1, 2)
    eps_rot = eps_rot.view(batch_size, -1, 2)
    for b in range(batch_size):
        stat_dict["drude_ok"].append(torch.allclose(drude[b], drude_rot[b], rtol=rtol, atol=atol))
        stat_dict["eps_ok"].append(torch.allclose(eps[b], eps_rot[b], rtol=rtol, atol=atol))
        stat_dict["drude_max_abs"].append((drude[b] - drude_rot[b]).abs().max().item())
        stat_dict["eps_max_abs"].append((eps[b] - eps_rot[b]).abs().max().item())
        stat_dict["drude_max_rel"].append(((drude[b] - drude_rot[b]).abs() / (drude[b].abs() + EPS)).max().item())
        stat_dict["eps_max_rel"].append(((eps[b] - eps_rot[b]).abs() / (eps[b].abs() + EPS)).max().item())
    return stat_dict

@torch.no_grad()
def check_output_equivariance(
    device: torch.device, 
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    dataloader_rot: DataLoader,
) -> None:
    model.eval()
    model.to(device)
    for graph, graph_rot in zip(dataloader, dataloader_rot):
        graph.to(device)
        graph_rot.to(device)
        batch_size = graph.batch_size
        batch_size_rot = graph_rot.batch_size
        if batch_size != batch_size_rot:
            raise ValueError("Batch size mismatch between original and rotated graphs")
        eps, drude = model(graph)
        eps_rot, drude_rot = model(graph_rot)
        stat_dict = compare_outputs(eps, drude, eps_rot, drude_rot, batch_size)
        for b in range(batch_size):
            print(
                f"    drude_ok={str(stat_dict['drude_ok'][b]):5s} |  drude_abs={stat_dict['drude_max_abs'][b]:.3e}  |  " +
                f"drude_rel={stat_dict['drude_max_rel'][b]:.3e}  |  eps_ok={str(stat_dict['eps_ok'][b]):5s} |  " +
                f"eps_abs={stat_dict['eps_max_abs'][b]:.3e}  |  eps_rel={stat_dict['eps_max_rel'][b]:.3e}"
            )

if __name__ == "__main__":
    
    import os
    import math
    from copy import deepcopy
    from pymatgen.core import Structure
    from importlib.resources import files

    import torch

    from optimetal_e3.utils import get_device
    from optimetal_e3.data.loader import parse_cse_entry, create_dataloader
    from optimetal_e3.data.preprocess import compress_data
    from optimetal_e3.data.transform import graph_setup
    from optimetal_e3.nn import OptiMetalE3
    
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    poscar_dir = "./test_struct_rot"
    os.makedirs(poscar_dir, exist_ok=True)

    json_files = [f for f in os.listdir(files("optimetal_e3.files")) if f.endswith(".json")]
    data_list = []
    data_list_rot = []
    R = random_rotation_matrix()

    for json_file in json_files:
        
        # load structure, compress it, and store original data
        json_path = files("optimetal_e3.files").joinpath(json_file)
        cse = parse_cse_entry(json_path)
        data = compress_data(cse)
        data_list.append(data)
        save_structure_as_poscar(data, os.path.join(poscar_dir, f"{json_file.split('.')[0]}.vasp"))
        
        # apply rotation to positions and lattice and store rotated data
        data_rot = deepcopy(data)
        data_rot["position"] = data["position"] @ R.T
        data_rot["lattice"] = data["lattice"] @ R.T
        data_list_rot.append(data_rot)
        save_structure_as_poscar(data_rot, os.path.join(poscar_dir, f"{json_file.split('.')[0]}_rot.vasp"))
    
    graph = graph_setup(data_list)
    graph_rot = graph_setup(data_list_rot)

    batch_size = 1
    dataloader = create_dataloader(
        graph, 
        num_data=-1, 
        batch_size=batch_size, 
        shuffle=False,
    )
    dataloader_rot = create_dataloader(
        graph_rot,
        num_data=-1,
        batch_size=batch_size,
        shuffle=False,
    )
    
    lmax = 3
    hidden_channels = 128
    ff_types = ["spectral", "grid"]
    device = get_device(args.device_index)
    
    for ff_type in ff_types:
        for l in range(lmax + 1):
            print(
                f"\n==================================== " + 
                f"Testing lmax={l:d} and ff_type={ff_type:s}" + 
                " ===================================="
            )
            print("Check node embedding equivariance:")
            model = OptiMetalE3(
                lmax=l,
                mmax=l,
                hidden_channels=hidden_channels,
                ff_type=ff_type,
                check_equivariance=True, # model returns final node embeddings
            )
            check_node_embedding_equivariance(device, model, dataloader, dataloader_rot, R)
            
            print("Check equivariance of outputs:")
            model = OptiMetalE3(
                lmax=l,
                mmax=l,
                hidden_channels=hidden_channels,
                ff_type=ff_type,
            )
            check_output_equivariance(device, model, dataloader, dataloader_rot)