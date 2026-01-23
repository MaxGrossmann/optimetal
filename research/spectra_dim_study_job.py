"""
Automatically generate and submit batch jobs for a scaling law study.

This script has been written to run on the TU Ilmenau cluster with my specific setup.
"""

import os
import sys
from itertools import islice
from typing import Iterable, Iterator, TypeVar
T = TypeVar("T")

# paths (depends on your setup)
config_dir  = "/scratch/magr4985/Metals/research/spectra_dim_config"
base_script = "/scratch/magr4985/Metals/research/train_model.py"
base_train  = "/scratch/magr4985/Metals/graph/train.pt"
base_val    = "/scratch/magr4985/Metals/graph/val.pt"
job_dir     = "/scratch/magr4985/Spectra_Dim"

# bashrc and conda environment
bashrc_path = "/home/magr4985/.bashrc"
conda_env   = "ml"

# number of models to be trained as one job
job_batch_size = 8

# check that the directory containing the model configurations exists
if not os.path.exists(config_dir):
    sys.exit(f"The directory {config_dir:s} containing the model configurations does not exist!")

# ensure job directory exists
os.makedirs(job_dir, exist_ok=True)

# read list of already batched configs
submitted_file = os.path.join(job_dir, "submitted.txt")
def read_submitted() -> set:
    if not os.path.exists(submitted_file):
        return set()
    with open(submitted_file, "r") as f:
        return {line.strip() for line in f if line.strip()}

# append newly batched configs to the record file
def write_submitted(configs: list[str]) -> None:
    with open(submitted_file, "a") as f:
        for config in configs:
            f.write(f"{config:s}\n")

# helper function to chunk lists
def chunked(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    it = iter(iterable)
    for first in it:
        yield [first] + list(islice(it, size - 1))

# helper to find the first available job index
def next_free_idx(job_dir: str) -> int:
    taken = {
        int(fname.split("_")[1].split(".")[0])
        for fname in os.listdir(job_dir)
        if fname.startswith("batch_") and fname.endswith(".job")
    }
    idx = 1
    while idx in taken:
        idx += 1
    return idx

# generate one batch job script for a list of configs
def create_batch_script(batch_idx: int, batch_configs: list[str]) -> bool:
    job_name = f"batch_{batch_idx:d}.job"
    job_path = os.path.join(job_dir, job_name)
    if os.path.exists(job_path):
        print(f"Job script {job_path:s} already exists, skipping")
        return False
    with open(job_path, "w") as f:
        f.write("#!/bin/sh\n")
        f.write("# batch transformer parameter scaling law job\n\n")
        f.write(f"source {bashrc_path:s}\n")
        f.write(f"conda activate {conda_env:s}\n")
        for config in batch_configs:
            config_path = os.path.join(config_dir, config)
            f.write(f"python {base_script:s} \\\n")
            f.write(f"--config_path={config_path:s} \\\n")
            f.write(f"--train_path={base_train:s} \\\n")
            f.write(f"--val_path={base_val:s} \\\n\n")
    os.system(f"chmod 755 {job_path:s}")
    os.system(f"batch.1gpu {job_path:s}")
    print(f"Created {job_path:s} containing {len(batch_configs):d} model training runs")
    return True

def main(job_batch_size: int) -> None:
    os.makedirs(job_dir, exist_ok=True)
    os.chdir(job_dir) # sumbit all jobs from the job directory (the log and error files will be stored there)
    all_configs = sorted([f for f in os.listdir(config_dir) if f.endswith(".json")])
    submitted = read_submitted()
    to_process = [config for config in all_configs if config not in submitted]
    if not to_process:
        print("All configurations have already been batched and submitted")
        return
    batch_idx = next_free_idx(job_dir)
    for batch in chunked(to_process, job_batch_size):
        success = create_batch_script(batch_idx, batch)
        if success:
            write_submitted(batch)
            batch_idx = next_free_idx(job_dir)

if __name__ == "__main__":
    main(job_batch_size)