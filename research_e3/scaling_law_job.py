"""
Automatically generate and submit batch jobs for a scaling law study.

This script has been written to run on the TU Ilmenau cluster with my specific setup.
"""

from __future__ import annotations

import re
import os
import sys
import textwrap

# paths (depends on your setup)
config_dir  = "/scratch/magr4985/optimetal_e3/research_e3/scaling_law_config"
base_script = "/scratch/magr4985/optimetal_e3/research_e3/train_model.py"
base_train  = "/scratch/magr4985/optimetal_e3/graph_e3/train.pt"
base_val    = "/scratch/magr4985/optimetal_e3/graph_e3/val.pt"
job_dir     = "/scratch/magr4985/scaling_law"

# bashrc and conda environment
bashrc_path = "/home/magr4985/.bashrc"
conda_env   = "ml"

# check that the directory containing the model configurations exists
if not os.path.exists(config_dir):
    sys.exit(f"The directory '{config_dir:s}' containing the model configurations does not exist!")

# ensure the job directory exists
os.makedirs(job_dir, exist_ok=True)

# setup job files and submit them (if they dont always exist)
for config_name in os.listdir(config_dir):

    # setup paths and file names
    job_filename = re.sub(".json", ".lsf", config_name)
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
            --val_path={base_val:s}
            """
            )
        )

    # submit the job from the study directory
    os.chdir(job_dir)
    os.system(f"chmod 755 {job_path:s}")
    os.system(f"batch.1gpu {job_path:s}")
    print(f"Submitted {job_path:s}", flush=True)