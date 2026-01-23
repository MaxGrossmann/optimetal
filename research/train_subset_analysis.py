"""
Run 'research/db_init.py' and 'research/write_out_splits.py' first.
"""

import os
import sys

import optimetal.utils as utils
from optimetal.data.loader import load_torch_data

utils.load_plot_style() 

def main() -> None:
    """
    Loads the training graphs and generates periodic table plots showing the percentage
    of each element, as well as histograms displaying the distribution of atoms within 
    the unit cell for the subsets used to derive the scaling laws. Unlike 'db_analysis.py', 
    which reads compressed H5 splits of the full dataset, this script operates on the graph
    dataset stored in 'graph/train.pt'. To ensure that the plotted subsets match those used 
    during training, the material identifiers for each subset were written out on our cluster 
    using 'research/write_out_splits.py', see 'research/data_splits'. This is because I was 
    initially unsure whether the Python random number generator would produce the same splits
    on both Windows and Linux. However, later tests confirmed that the outputs are consistent
    across both platforms.
    """
    # path setup and checks
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "dataset_plots")
    os.makedirs(output_dir, exist_ok=True)
    train_path = current_dir + "/../graph/train.pt"
    if not os.path.exists(train_path):
        sys.exit(f"'{train_path:s}' is missing, run 'research/db_init.py' first")
    train_data = load_torch_data(train_path)
        
    # create plots for the training data subsets used for the scaling laws 
    num_datapoints = [2500, 5000, 10000, 20000, 40000, 80000, 160000]
    for num_data in num_datapoints:
        # load the list of materials contained in this subset (see 'write_out_splits.py', which was created on our cluster)
        mat_list_path = current_dir + f"/data_splits/train_{num_data:d}.txt"
        if not os.path.exists(mat_list_path):
            sys.exit(f"'{mat_list_path:s}' is missing, run 'research/write_out_splits.py' first")
        with open(mat_list_path, "r") as f:
            mat_list = f.readlines()
        mat_list = [mat_id.split("\n")[0] for mat_id in mat_list]
        mat_list = set(mat_list) # faster membership check
        # create the a list of dictionaries containing the atomic number from each graph
        data_list = []
        for data in train_data:
            mat_id = data.mat_id.item().decode()
            if mat_id in mat_list:
                data_list.append({"atomic_number": list(data.atomic_number.cpu().numpy())})
        # periodic table plot showing how many times each element occurs in the dataset
        print("Creating periodic table showing how many times each element occurs in the dataset", flush=True)
        fig_path = os.path.join(output_dir, f"periodic_table_train_{num_data:d}.pdf")
        utils.periodic_table_plot(data_list, fig_path)
        # histogram showing the distribution of the number of atoms in the unit cell
        print("Creating histogram showing the distribution of the number of atoms in the unit cell", flush=True)
        fig_path = os.path.join(output_dir, f"num_site_dist_train_{num_data:d}.pdf")
        utils.num_sites_dist(data_list, fig_path)
       
if __name__ == "__main__":
    main()
