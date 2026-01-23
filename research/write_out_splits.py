"""
Run 'research/db_init.py' first.
"""

import os
import sys

from optimetal.data.loader import load_torch_data, create_dataloader

def get_mat_ids(db_path: str, file: str, num_data: int, output_dir: str) -> None:
    data = load_torch_data(os.path.join(db_path, file))
    loader = create_dataloader(
        data, 
        num_data=num_data,
        batch_size=256, # not important here, can be any number
        shuffle=False,
    )
    mat_id_list = []
    for batch in loader:
        batch_size = batch.num_graphs
        mat_ids = batch.mat_id
        for i in range(batch_size):
            mat_id_list.append(mat_ids[i].item().decode())
    name = file.split(".")[0]
    print(f"Writing out material IDs for the {name:s} dataset")
    with open(f"{output_dir}/{name:s}.txt", "w") as f:
        for mat_id in mat_id_list:
            f.write(f"{mat_id:s}\n")
            
def main() -> None:
    """
    Loads the training, validation and test graphs, and writes out the material
    IDs for all datasets and the training subsets used in the scaling laws.
    """
    # path setup and checks
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "data_splits")
    os.makedirs(output_dir, exist_ok=True)
    db_path = current_dir + "/../graph" 
    file_list = [
        "train.pt",
        "val.pt",
        "test.pt",
    ]
    flag = True
    for file in file_list:
        file_path = os.path.join(db_path, file)
        if not os.path.exists(file_path):
            flag = False
            print(f"'{file:s}' is missing, run 'research/db_init.py' first", flush=True)
    if not flag:
        sys.exit("Some compressed database files are missing (see above)...")
    else:
        print("The files 'train.pt', 'val.pt' and 'test.pt' exist in the 'dataset' directory", flush=True)
        
    # write out the materials in each dataset split
    num_datapoints = [2500, 5000, 10000, 20000, 40000, 80000, 160000, -1] # only for training set
    for file in file_list:
        if file == "train.pt":
            for num_data in num_datapoints:
                get_mat_ids(db_path=db_path, file=file, num_data=num_data)
        else:
            get_mat_ids(db_path=db_path, file=file, num_data=-1)

if __name__ == "__main__":
    main()