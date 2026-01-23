"""
Database compression and training, validation and test split:

Running this script is optional. To skip it, do the following:
Download the h5 version of the database and place the 'train.h5', 'val.h5' and 'test.h5' files
into a directory called `dataset' in the top-level directory of this repository.

Due to the large size of the database, running this script will take some time.
(IMPORTANT: More than 16 GB of memory is recommended when running this script..., 32 GB are fine though.)

Before you run this script, do the following:
- Download and extract the complete database of dielectric functions for intermetallics, see README.
- Set the `full_db_path` variable to the directory where the downloaded database is located.

This script will then compress the database and split it into three h5 files ('train.h5', 'val.h5' and 'test.h5').
To see what data we keep when compressing, see the function 'compress_data()' in 'optimetal.data.preprocess'.

The training, validation and test split is based on unique compositions, cf., https://www.nature.com/articles/s41586-023-06735-9.

(This script is parallelized and still takes some time.)
"""

import os
from tqdm import tqdm

import optimetal.utils as utils
import optimetal.data.loader as loader
import optimetal.data.preprocess as preprocess

"""
-----------------------------------------------------------------------------------------------------------------------
START OF USER INPUT:
"""

# directory where the downloaded database is located
full_db_path = r"C:\Database" # CHANGE THIS TO THE PATH WHERE THE DOWNLOADED DATABASE IS LOCATED

# dataset split ratios
train_ratio = 0.8
val_ratio = 0.1
# test_ratio = 1 - train_ratio - val_ratio

"""
END OF USER INPUT:
-----------------------------------------------------------------------------------------------------------------------
"""

def check_full_db(full_db_path: str) -> bool:
    """
    Checks the directory structure and file counts of the unpacked full database.
    Input:
        base_dir (str):         The root directory of the unpacked database
        db_dir_dict (dict):     A dictionary where keys are directory names and values are the expected number of JSON files
    Output:
        bool:                   True if the database structure matches exactly, False if the database structure does not match
    """
    print("Validating the full database", flush=True)

    # list all directory names in the full database and how many JSON files are in each.
    db_dir_dict = {
        "database_nsite_1_part_1": 68,
        "database_nsite_2_part_1": 924,
        "database_nsite_3_part_1": 11854,
        "database_nsite_4_part_1": 14785,
        "database_nsite_4_part_2": 14830,
        "database_nsite_4_part_3": 14876, 
        "database_nsite_4_part_4": 14905,
        "database_nsite_4_part_5": 14007,
        "database_nsite_5_part_1": 14722,
        "database_nsite_5_part_2": 14216,
        "database_nsite_6_part_1": 14831,
        "database_nsite_6_part_2": 4237,
        "database_nsite_7_part_1": 7905,
        "database_nsite_8_part_1": 14402,
        "database_nsite_8_part_2": 3744,
        "database_nsite_9_part_1": 9287,
        "database_nsite_10_part_1": 14562,
        "database_nsite_10_part_2": 14819,
        "database_nsite_10_part_3": 2387,
    }

    # get the set of expected directories
    expected_dirs = set(db_dir_dict.keys())

    # get the set of actual directories in the base directory
    actual_dirs = set([d for d in os.listdir(full_db_path) if ("database" in d) and os.path.isdir(os.path.join(full_db_path, d))]) 

    # check for missing directories
    missing_dirs = expected_dirs - actual_dirs
    if missing_dirs:
        print("Validation failed.\nMissing directories found:", flush=True)
        for dir_name in missing_dirs:
            print(f" - {dir_name:s}", flush=True)
        print("The full database appears to be corrupted", flush=True)
        return False

    # check for extra directories
    extra_dirs = actual_dirs - expected_dirs
    if extra_dirs:
        print("Validation failed.\nUnexpected directories found:", flush=True)
        for dir_name in extra_dirs:
            print(f" - {dir_name:s}", flush=True)
        print("The full database appears to be corrupted", flush=True)
        return False

    # validate the file counts for each directory
    for dir_name, expected_count in db_dir_dict.items():
        dir_path = os.path.join(full_db_path, dir_name)
        # count all JSON files in the directory
        json_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
        actual_count = len(json_files)
        # compare file counts
        if actual_count != expected_count:
            print(f"Validation failed: Directory '{dir_name:s}' has {actual_count:d} JSON files, expected {expected_count:d}", flush=True)
        else:
            print(f"Directory '{dir_name:s}' validated successfully with {actual_count:d} JSON files", flush=True)

    print("The full database was successfully validated", flush=True)
    return True

def main(full_db_path: str) -> None:
    """
    Initialize the database, i.e., convert it to a format usable for machine learning.
    """
    # hardcoded path for the compressed database (train test split)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = current_dir + "/../dataset"
    
    if check_full_db(full_db_path):
        # load all materials into memory (keep only the important data)
        print("Creating a compressed version of the database", flush=True)
        db_dir_list = [db for db in utils.listdir_full(full_db_path) if "database" in db]
        data_list = []
        for db_dir in db_dir_list:
            print(f"Parsing and compressing database entries from {str(db_dir):s}", flush=True)
            db_file_list = [db for db in utils.listdir_full(db_dir)]
            cse_list = loader.parallel_parse_cse_entries(db_file_list)
            for cse in tqdm(cse_list, desc="CSE data compression"):
                data_list.append(
                    preprocess.compress_data(cse)
                )
        # create a train test split
        train_list, val_list, test_list = preprocess.split_dataset(data_list, train_ratio=train_ratio, val_ratio=val_ratio)
        # save the splits into separte h5 files
        os.makedirs(output_dir, exist_ok=True)
        preprocess.save_as_h5(train_list, output_dir, "train")    
        preprocess.save_as_h5(val_list, output_dir, "val")   
        preprocess.save_as_h5(test_list, output_dir, "test")   
        print("Database compression and training, validation and test split completed", flush=True)

if __name__ == "__main__":
    main(full_db_path)