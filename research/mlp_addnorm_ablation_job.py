"""
Generate and submit batch jobs for training models automatically.

This script has been written to run on the TU Ilmenau cluster with my specific setup.
"""

import os
import re
import sys
import textwrap

# random seeds (the script checks the file names to select the seeds)
seed = 137 # do this for multiple seeds, e.g., 1, 42, 137

# paths (depends on your setup)
config_dir  = "/scratch/magr4985/Metals/research/mlp_addnorm_config"
base_script = "/scratch/magr4985/Metals/research/train_model_with_lrwd_optim.py"
base_train  = "/scratch/magr4985/Metals/graph/train.pt"
base_val    = "/scratch/magr4985/Metals/graph/val.pt"
job_dir     = "/scratch/magr4985/MLP_AddNorm"

# bashrc and conda environment
bashrc_path = "/home/magr4985/.bashrc"
conda_env   = "ml"

# base seed run from which we reuse the learning rate and weight decay
base_seed = 42 # DO NOT CHANGE THIS!

# check that the directory containing the model configurations exists
if not os.path.exists(config_dir):
    sys.exit(f"The directory {config_dir:s} containing the model configurations does not exist!")

# ensure job directory exists
os.makedirs(job_dir, exist_ok=True)

# setup job files and submit them (if they dont always exist)
for config_name in os.listdir(config_dir):
    
    # check to see if the seed matches the one we selected
    if f"seed{seed:d}.json" not in config_name:
        continue

    # setup paths and file names
    job_filename = re.sub(".json", ".job", config_name)
    job_path = os.path.join(job_dir, job_filename)
    
    # write out the job script (if it doesn't already exist)
    if os.path.exists(job_path):
        print(f"Job file {job_path:s} already exists", flush=True)
        continue
    with open(job_path, "w") as f:
        f.write(
            textwrap.dedent(
            f"""\
            #!/bin/sh
            # train model job

            source {bashrc_path:s}
            conda activate {conda_env:s}

            python {base_script:s} \\
            --config_path={os.path.join(config_dir, config_name):s} \\
            --train_path={base_train:s} \\
            --val_path={base_val:s} \\
            """
            )
        )
        if seed != base_seed:
                f.write(f"--reuse_lrwd_from_seed={base_seed:d} \\")

    # submit the job from the study directory
    os.chdir(job_dir)
    os.system(f"chmod 755 {job_path:s}")
    os.system(f"batch.1gpu {job_path:s}")
    print(f"Submitted {job_path:s}", flush=True)