"""
Automatically generate and submit batch jobs for the OptiMetal ablation study, where
each layer variation is farmed out to multiple parallel workers, which self-balance
via the 'ablation_study.py' directory-level locking.

This script has been written to run on the TU Ilmenau cluster with my specific setup.
"""

import os
import textwrap

# parameters
model_type = "3b" # "2b" or "3b" are available ("3b" is too expensive...)
layer_vars = ["node", "edge", "mp", "pool"] # "node", "edge", "mp", and "pool" are available
seed = 137 # do this for multiple seeds, e.g., 1, 42, 137

# paths (depends on your setup)
base_script = "/scratch/magr4985/Metals/research/ablation_study.py"
base_train  = "/scratch/magr4985/Metals/graph/train.pt"
base_val    = "/scratch/magr4985/Metals/graph/val.pt"
job_dir     = "/scratch/magr4985/Ablation"

# bashrc and conda environment
bashrc_path = "/home/magr4985/.bashrc"
conda_env   = "ml"

# fixed parameters
num_train_data = 20000 # DO NOT CHANGE THIS!
batch_size = 256 # DO NOT CHANGE THIS!

# base seed run from which we reuse the learning rate and weight decay
base_seed = 42 # DO NOT CHANGE THIS!

# ensure job directory exists
os.makedirs(job_dir, exist_ok=True)

# setup job files and submit them (if they dont always exist)
for layer in layer_vars:

    # depending on the layer, we may need to adjust the number of workers
    if model_type == "2b":
        if layer == "node" or layer == "pool":
            num_worker = 1
        else:
            num_worker = 4
    elif model_type == "3b":
        num_worker = 8
        # we only run the ablation study for the message passing layers
        if layer != "mp":
            continue
    else:
        raise ValueError(f"Unknown model type: {model_type:s}")

    # loop over the number of workers
    for worker_id in range(num_worker):

        # setup paths and file names
        study_path = f"/scratch/magr4985/Ablation/optimetal{model_type:s}_seed{seed:d}"
        job_filename = f"optimetal{model_type:s}_{layer:s}_seed{seed:d}_w{worker_id}.job"
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
                # ablation study job
                # model={model_type:s}, layer={layer:s}, seed={seed:d}, worker={worker_id:d}

                source {bashrc_path:s}
                conda activate {conda_env:s}

                python {base_script:s} \\
                --model_type={model_type:s} \\
                --layer_variation={layer:s} \\
                --study_path={study_path:s} \\
                --seed={seed:d} \\
                --num_train_data={num_train_data:d} \\
                --batch_size={batch_size:d} \\
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