"""
Run 'research/db_init.py' first.

Train a model based on a configuration file (JSON file containing a dictionary).

The configuration file contains training settings:
{
    "seed": int                 # Random seed
    "trial_dir": str            # Path to the directory where the results will be saved (will be created if it does not exist)
    "num_train_data": int       # Number of training data points (selected as a random subset)
    "batch_size": int           # Batch size
    "architecture": dict        # See 'optimetal.factory.model'
    "optimizer": dict           # See 'optimetal.factory.optim'
    "loss_fn_eps": str          # Loss function for the dielectric function, see 'optimetal.factory.loss'
    "loss_fn_drude": str        # Loss function for the Drude frequency, see 'optimetal.factory.loss'
    "lr_scheduler": str         # See 'optimetal.factory.lr_scheduler'
    "warmup_epochs": int        # Number of learning rate warmup epochs
    "grad_clip": float          # Gradient norm clipping threshold
    "eps_weight": float         # Interband dielectric function loss weight
    "drude_weight": float       # Drude frequency loss weight
    "early_stopping": bool      # Flag to activate early stopping
    "patience": int             # Early stopping patience (requires 'early_stopping=true')
    "num_epoch": int            # Number of training epochs
    "precision": str            # Automatic mixed precision training ("auto", "bf16" or "fp32")
}

Example configuration:
    ./scaling_law_base_config/2b_variant2_hestness_data20000_seed42.json
    
The configuration dictionary is valid using pydantic. The architecture, learning rate scheduler, 
and optimizer dictionaries, however, are validated in their respective factories and model definitions.

Track the training progress through something like this (adjust the path accordingly):
    tensorboard --logdir ./scratch/magr4985/Scaling_Base/2b_variant2_hestness_data20000_seed42
https://docs.pytorch.org/docs/stable//tensorboard.html
"""

import os
import sys
import json
import argparse
from pydantic import TypeAdapter

import optimetal.factory as factory
from optimetal.training import Trainer
from optimetal.data.loader import load_torch_data, create_dataloader
from optimetal.utils import ValidateConfigurationDict, get_device, print_model_parameters, print_dict

"""
-----------------------------------------------------------------------------------------------------------------------
PARSE INPUT ARGUMENTS:
"""

parser = argparse.ArgumentParser(description="Model training script")
parser.add_argument(
    "--config_path",
    type=str, 
    default="./scaling_law_base_config/2b_variant2_hestness_data20000_seed42.json", 
    help="Path to a configuration file that contains all training settings (relative paths are possible)",
)
parser.add_argument(
    "--train_path", 
    type=str, 
    default="../graph/train.pt", 
    help="Path to the training graphs (relative paths are possible)",
)
parser.add_argument(
    "--val_path", 
    type=str, 
    default="../graph/val.pt", 
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
    model_config = config_dict.architecture
    model = factory.create_model(model_config)

    # setup the optimizer
    optim_config = config_dict.optimizer
    optimizer = factory.create_optimizer(model, optim_config)

    # setup the loss functions for the dielectric function and Drude frequency
    loss_type_eps = config_dict.loss_fn_eps
    loss_fn_eps = factory.create_loss_fn(loss_type_eps)
    loss_type_drude = config_dict.loss_fn_drude
    loss_fn_drude = factory.create_loss_fn(loss_type_drude)

    # setup the learning rate scheduler
    lr_scheduler_config = config_dict.lr_scheduler
    lr_scheduler = factory.create_lr_scheduler(
        optimizer, 
        lr_scheduler_config,
        warmup_epochs=config_dict.warmup_epochs,
    )

    # build the trainer dictionary
    trainer_dict = {
        "config_dict": config_dict, # metadata
        "trial_dir": config_dict.trial_dir,
        "device": device,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "model": model,
        "optimizer": optimizer,
        "loss_fn_eps": loss_fn_eps,
        "loss_fn_drude": loss_fn_drude,  
        "lr_scheduler": lr_scheduler,
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