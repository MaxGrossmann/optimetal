"""
Run 'research/db_compression.py' first.

Loads the compressed database ('train.h5', 'val.h5' and 'test.h5') 
and creates a multigraph representation of the materials usable in torch_geometric. 
It also calculates some target values, e.g., the material color.
(See the file 'optimetal/data/transform.py' for details).
The created '.pt' contains a list torch_geometric 'Data' objects. 
(See https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html)

We convert the crystal structures into multigraph by creating a node for each atom in the unit cell, 
and creating an edge between nodes if the distance between corresponding atoms is less than 'twobody_cutoff' angstroms.
Since some architectures include angles, e.g., for edge updates like M3GNet, we also construct three-body interactions:
We identify bond pairs that share a central atom and where both bonds are shorter than a given 'threebody_cutoff'.

The resulting torch_geometric 'Data' object contains the following information:
- 'num_nodes':          Number of atoms in the unit cell
- 'atomic_number':      Atomic numbers of the atoms (used to generate the node features)
- 'position':           Atomic positions in Cartesian coordinates
- 'lattice':            The lattice matrix of the unit cell, where each row is a lattice vector
- 'num_twobody':        Number of valid three-body interactions (bonds, i.e., edges)
- 'edge_index':         COO-style tensor defining source and destination atoms of each bond, size: (2, num_twobody)
- 'pbc_offset':         Periodic boundary corrections for each bond
- 'num_threebody':      Number of valid three-body interactions (angles)
- 'threebody_index':    COO-style tensor defining source and destination bonds of each angle, size: (2, num_threebody)
- 'drude':              Drude frequency (intraband contributions)
- 'eps':                Complex dielectric function, interband contribution, split into real and imaginary parts, size: (2001, 2)
- 'color':              CIE L*a*b* material color derived from the total dielectric function
- 'mat_id':             Unique identifier for the material (Alexandria database)
- 'composition':        String representation of the chemical composition

(This script takes some time to run. But we did not care enough to optimize/parallelize it.)
"""

import os
import sys

import torch

import optimetal.data.loader as loader
import optimetal.data.transform as transform

"""
-----------------------------------------------------------------------------------------------------------------------
START OF USER INPUT:
"""

# cutoff radius for finding bonds (edges) between atoms (nodes)
twobody_cutoff = 5.5 # angstrom

# cutoff radius for finding bond angles
threebody_cutoff = 4.0 # angstrom

"""
END OF USER INPUT:
-----------------------------------------------------------------------------------------------------------------------
"""

def main(twobody_cutoff: float, threebody_cutoff: float) -> None:
    """
    Processes dataset files by converting raw data into graph representations and saving them.
    """
    # path setup and checks
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = current_dir + "/../dataset" 
    output_dir = current_dir + "/../graph"
    os.makedirs(output_dir, exist_ok=True)
    file_list = [
        "train.h5",
        "val.h5",
        "test.h5",
    ]
    flag = True
    for file in file_list:
        file_path = os.path.join(db_path, file)
        if not os.path.exists(file_path):
            flag = False
            print(f"'{file:s}' is missing, run 'research/db_compression.py' first", flush=True)
    if not flag:
        sys.exit("Some compressed database files are missing (see above)...")
    else:
        print("The files 'train.h5', 'val.h5' and 'test.h5' exist in the 'dataset' directory", flush=True)
        
    # setup and save the graphs for the train, val and test split
    for file in file_list:
        name = file.split(".")[0]
        file_path = os.path.join(db_path, file)
        data_list = loader.parse_h5_file(file_path)
        graph_list = transform.graph_setup(
            data_list, 
            twobody_cutoff=twobody_cutoff, 
            threebody_cutoff=threebody_cutoff,
        )
        torch.save(graph_list, os.path.join(output_dir, f"{name:s}.pt"))

if __name__ == "__main__":
    main(twobody_cutoff, threebody_cutoff)