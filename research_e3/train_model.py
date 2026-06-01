"""
Run 'research_e3/db_init.py' first.

Train a model based on a configuration file (JSON file containing a dictionary).

The configuration file contains training settings:
{
    "seed": int                 # Random seed
    "trial_dir": str            # Path to the directory where the results will be saved (will be created if it does not exist)
    "num_train_data": int       # Number of training data points (selected as a random subset)
    "batch_size": int           # Batch size
    "lmax": int                 # Maximum degree of spherical harmonics (see 'optimetal_e3.nn.OptiMetalE3')
    "mmax": int                 # Maximum order of spherical harmonics (see 'optimetal_e3.nn.OptiMetalE3')
    "hidden_channels": int      # Number of hidden channels (see 'optimetal_e3.nn.OptiMetalE3')
    "ff_type": str              # Type of equivariant feedforward network ("spectral" or "grid")
    "lr": str                   # Learning rate for AdamW optimizer
    "weight_decay": float       # Weight decay for AdamW optimizer
    "grad_clip": float          # Gradient norm clipping threshold
    "eps_weight": float         # Interband dielectric function loss weight
    "drude_weight": float       # drude frequency loss weight
    "early_stopping": bool      # Flag to activate early stopping
    "patience": int             # Early stopping patience (requires 'early_stopping=true')
    "num_epoch": int            # Number of training epochs
    "precision": str            # Automatic mixed precision training ("fp32" or "bf16")
}

Example configuration:
    ./lr_wd_tuning_config/lmax2_width256_seed42.json
    
The configuration dictionary is valid using pydantic. The architecture, learning rate scheduler, 
and optimizer dictionaries, however, are validated in their respective factories and model definitions.

Track the training progress though something like this (adjust the path accordingly):
    tensorboard --logdir ./scratch/magr4985/scaling_law/lmax2_data10000_width256_seed42
https://docs.pytorch.org/docs/stable//tensorboard.html
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pydantic import TypeAdapter

from optimetal_e3.utils import ValidateConfigurationDict, get_device, print_model_parameters, print_dict
from optimetal_e3.data.loader import load_torch_data, create_dataloader
from optimetal_e3.nn import OptiMetalE3
from optimetal_e3.training import Trainer

"""
-----------------------------------------------------------------------------------------------------------------------
PARSE INPUT ARGUMENTS:
"""

parser = argparse.ArgumentParser(description="Model training script")
parser.add_argument(
    "--config_path",
    type=str, 
    default="./lr_wd_tuning_config/lmax2_width256_seed42.json", 
    help="Path to a configuration file that contains all training settings (relative paths are possible)",
)
parser.add_argument(
    "--train_path", 
    type=str, 
    default="../graph_e3/train.pt", 
    help="Path to the training graphs (relative paths are possible)",
)
parser.add_argument(
    "--val_path", 
    type=str, 
    default="../graph_e3/val.pt", 
    help="Path to the validation graphs (relative paths are possible)",
)
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

def main(args: argparse.Namespace) -> None:
    """
    Train a model based on inputs from a configuration file.
    """
    
    # path setup and checks
    if not os.path.exists(args.train_path):
        sys.exit("The path 'train_path' does not exist (training data not found)")
    if not os.path.exists(args.val_path):
        sys.exit("The path 'val_path' does not exist (validation data not found)")
    if not os.path.exists(args.config_path):
        sys.exit("The path 'config_path' does not exist")
        
    # load and validate the configuration dictionary
    with open(args.config_path, "r") as f:
        config_dict = json.load(f)
    print(f"Parameters from configuration file: {args.config_path:s}")
    print_dict(config_dict) # debugging
    config_dict = TypeAdapter(ValidateConfigurationDict).validate_python(config_dict)
         
    # load the datasets
    train_data = load_torch_data(args.train_path)
    val_data = load_torch_data(args.val_path)
    
    # create dataloader objects
    train_loader = create_dataloader(
        train_data, 
        num_data=config_dict.num_train_data, 
        batch_size=config_dict.batch_size, 
        shuffle=True, # shuffle the training set
        seed=config_dict.seed,
    )
    print(f"Using {len(train_loader.dataset):d} graph from the training set", flush=True)
    val_loader = create_dataloader(
        val_data, 
        num_data=-1, # use the whole validation dataset 
        batch_size=config_dict.batch_size, 
        shuffle=False, # do not shuffle the validation set
        seed=config_dict.seed, # not needed here, but set it anyway
    )
    print(f"Using the entire validation set", flush=True)
    
    # training device
    device = get_device(index=args.device_index)

    # build the model
    model = OptiMetalE3(
        lmax=config_dict.lmax,
        mmax=config_dict.mmax,
        hidden_channels=config_dict.hidden_channels,
        ff_type=config_dict.ff_type,
    ).to(device)
    
    # build the trainer dictionary
    trainer_dict = {
        "trial_dir": config_dict.trial_dir,
        "device": device,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "model": model,
        "model_kwargs": {
            "lmax": config_dict.lmax,
            "mmax": config_dict.mmax,
            "hidden_channels": config_dict.hidden_channels,
            "ff_type": config_dict.ff_type,
        }, 
        "lr": config_dict.lr,
        "weight_decay": config_dict.weight_decay,
        "T_max": config_dict.num_epoch,
        "grad_clip": config_dict.grad_clip,
        "eps_weight": config_dict.eps_weight,
        "drude_weight": config_dict.drude_weight,
        "early_stopping": config_dict.early_stopping,
        "patience": config_dict.patience,
        "seed": config_dict.seed,
        "precision": config_dict.precision,
    }

    # print a summary of the trainable model parameters
    print_model_parameters(model)
    
    # build the trainer and train the model
    trainer = Trainer(
        trainer_dict=trainer_dict,
        checkpoint_every=10,
        best_model_start_epoch=10,
    )
    trainer.train(config_dict.num_epoch)
    
    # log the best validation loss
    with open(os.path.join(config_dict.trial_dir, "val_loss.txt"), "w") as f:
        f.write(f"{trainer.best_val_loss:.4f}")

if __name__ == "__main__":
    # working directory setup, i.e., enable relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    # train a model
    main(args)