"""
Run 'research/db_compression.py' first.
"""

import os
import sys

import optimetal.utils as utils
import optimetal.data.loader as loader

utils.load_plot_style() 

def main() -> None:
    """
    Loads the compressed database and generates periodic table plots showing the percentage
    of each element in a train, validation, and test dataset, as well as histograms showing
    the distribution of the number of atoms in the unit cell.
    """
    # path setup and checks
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "dataset_plots")
    os.makedirs(output_dir, exist_ok=True)
    db_path = current_dir + "/../dataset" 
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
        
    # create plots for the dataset splits
    for file in file_list:
        name = file.split(".")[0]
        file_path = os.path.join(db_path, file)
        data_list = loader.parse_h5_file(file_path)
        # periodic table plot showing how many times each element occurs in the dataset
        print("Creating periodic table showing how many times each element occurs in the dataset", flush=True)
        fig_path = os.path.join(output_dir, f"periodic_table_{name:s}.pdf")
        utils.periodic_table_plot(data_list, fig_path)
        # histogram showing the distribution of the number of atoms in the unit cell
        print("Creating histogram showing the distribution of the number of atoms in the unit cell", flush=True)
        fig_path = os.path.join(output_dir, f"num_site_dist_{name:s}.pdf")
        utils.num_sites_dist(data_list, fig_path)
        
    # create plots for the whole dataset
    data_list = []
    for file in file_list:
        name = file.split(".")[0]
        file_path = os.path.join(db_path, file)
        data_list.extend(loader.parse_h5_file(file_path))
    # periodic table plot showing how many times each element occurs in the dataset
    print("Creating periodic table showing how many times each element occurs in the dataset", flush=True)
    fig_path = os.path.join(output_dir, f"periodic_table.pdf")
    utils.periodic_table_plot(data_list, fig_path)
    # histogram showing the distribution of the number of atoms in the unit cell
    print("Creating histogram showing the distribution of the number of atoms in the unit cell", flush=True)
    fig_path = os.path.join(output_dir, f"num_site_dist.pdf")
    utils.num_sites_dist(data_list, fig_path)
       
if __name__ == "__main__":
    main()
