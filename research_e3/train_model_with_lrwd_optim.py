"""
Run 'research_e3/db_init.py' first.

Find a good learning rate and weight decay for a model in a configuration file (JSON file containing a dictionary).
The process is the same as in the ablation study. To optimize the learning rate and weight decay, we train for half
of the epochs specified in the configuration file. Then, the model is fully trained.

The learning rate and weight decay obtained are updated in the configuration file and stored in the trial directory.
If the learning rate and weight decay were already optimized, this would run just like the normal training script.

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
import re
import sys
import json
import optuna
import argparse
from copy import deepcopy
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
    "--reuse_lrwd_from_seed",
    type=int,
    default=None,
    help=(
        "If set, try to load the learning rate and weight decay from the " + 
        "study directory of this seed. Falls back to optimization " +
        "when the cached values are not found",
    ),
)
parser.add_argument(
    "--n_trials", 
    type=int, 
    default=25,
    help="The number of Optuna trials used to determine the optimal learning rate and weight decay",   
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

# learning rate and weight decay grid
LR_GRID = [
    1e-5,  2e-5,  4e-5, 6e-5, 8e-5,
    1e-4,  2e-4,  4e-4, 6e-4, 8e-4,
    1e-3,  2e-3,  4e-3, 6e-3, 8e-3,
]
WD_GRID = [
    0.0,
    1e-6,  2e-6,  4e-6, 6e-6, 8e-6,
    1e-5,  2e-5,  4e-5, 6e-5, 8e-5,
    1e-4,  2e-4,  4e-4, 6e-4, 8e-4,
    1e-3,
]

def trainer_from_cfg(
    config_dict: dict, 
    train_loader,
    val_loader,
    trial: optuna.Trial | None = None,
) -> Trainer:
    # validate the configuration dictionary
    config_dict = TypeAdapter(ValidateConfigurationDict).validate_python(config_dict)
    
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
        "trial": trial,
    }
    return Trainer(
        trainer_dict=trainer_dict,
        checkpoint_every=config_dict.num_epoch, # do not store checkpoints
        best_model_start_epoch=10, # just store the best model
    )

def lrwd_tuner(
    config_dict: dict,
    train_loader, 
    val_loader,
    n_trials: int,
    ) -> dict:
    """
    Helper function to quickly optimize the learning rate
    and weight decay for a given architecture permutation.
    """
    # copy the learning rate and weight decay grids
    lr_grid = deepcopy(LR_GRID)
    wd_grid = deepcopy(WD_GRID)
    
    # setup a Optuna study
    trial_dir = config_dict["trial_dir"]
    study = optuna.create_study(
        storage=f"sqlite:///{os.path.join(trial_dir, 'study.db')}",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=10, # number of random trials to run before the sampler takes over
            seed=config_dict["seed"],
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10, # let 10 trials run without pruning
            n_warmup_steps=25, # do not prune before epoch 25
            interval_steps=25 # check every 25 epochs
        ),
        study_name="lr_wd",
        load_if_exists=False, # if one already exists, crash to script
        direction="minimize",
    )
    
    # setup a local objective function
    def objective(trial: optuna.Trial) -> float:
        local_config_dict = deepcopy(config_dict)
        local_config_dict["lr"] = trial.suggest_categorical("lr", lr_grid)
        local_config_dict["weight_decay"] = trial.suggest_categorical("wd", wd_grid)
        local_trial_dir = os.path.join(local_config_dict["trial_dir"], f"trial_{trial.number:06d}")
        os.makedirs(local_trial_dir, exist_ok=True)
        local_config_dict["trial_dir"] = local_trial_dir
        trainer = trainer_from_cfg(
            config_dict=local_config_dict, 
            train_loader=train_loader, 
            val_loader=val_loader,
            trial=trial
        )
        trainer.train(local_config_dict["num_epoch"])
        return trainer.best_val_loss
    
    # find a good learning rate and weight decay
    study.optimize(objective, n_trials=n_trials, timeout=None)
    return study.best_trial.params

def main(args: argparse.Namespace) -> None:
    """
    Train a model based on inputs from a configuration file.
    """
    # path checks
    if not os.path.exists(args.train_path):
        sys.exit("The path 'train_path' does not exist (training data not found)")
    if not os.path.exists(args.val_path):
        sys.exit("The path 'val_path' does not exist (validation data not found)")
    if not os.path.exists(args.config_path):
        sys.exit("The path 'config_path' does not exist")
        
    # load the datasets
    train_data = load_torch_data(args.train_path)
    val_data = load_torch_data(args.val_path)
    
    # load and validate the configuration dictionary
    with open(args.config_path, "r") as f:
        config_dict = json.load(f)
    print(f"Parameters from configuration file: {args.config_path:s}")
    print_dict(config_dict) # debugging
         
    # create dataloader objects
    train_loader = create_dataloader(
        train_data, 
        num_data=config_dict["num_train_data"], 
        batch_size=config_dict["batch_size"], 
        shuffle=True, # shuffle the training set
        seed=config_dict["seed"],
    )
    print(f"Using {len(train_loader.dataset):d} graph from the training set", flush=True)
    val_loader = create_dataloader(
        val_data, 
        num_data=-1, # use the whole validation dataset 
        batch_size=config_dict["batch_size"], 
        shuffle=False, # do not shuffle the validation set
        seed=config_dict["seed"], # not needed here, but set it anyway
    )
    print(f"Using the entire validation set", flush=True)
    
    """
    Optimize the learning rate and weight decay.
    """
    
    # path setup
    lrwd_dir = os.path.join(config_dict["trial_dir"], "lrwd")
    lrwd_json_path = os.path.join(config_dict["trial_dir"], "lrwd.json")
    
    # check if we already optimized the learning rate and weight decay 
    # for this specific configuration using a different seed
    # (this assumes that the configurations are equal, but does not verify it)
    cache_best_params = None
    if args.reuse_lrwd_from_seed is not None and args.reuse_lrwd_from_seed != config_dict["seed"]:
        cache_config_path = re.sub(r"_seed\d+", f"_seed{args.reuse_lrwd_from_seed:d}", args.config_path)
        if os.path.exists(cache_config_path):
            print(f"\nConfiguration file with reference seed {args.reuse_lrwd_from_seed:d} exists")
            with open(cache_config_path, "r") as f:
                cache_config_dict = json.load(f)
            if os.path.exists(cache_config_dict["trial_dir"]):
                if os.path.exists(os.path.join(cache_config_dict["trial_dir"], "lrwd.json")):
                    print(f"The learning rate and weight decay were already optimized for the reference seed {args.reuse_lrwd_from_seed:d}")
                    with open(os.path.join(cache_config_dict["trial_dir"], "lrwd.json"), "r") as f:
                        cache_best_params = json.load(f)

    # find a good learning rate and weight decay for a model
    if not os.path.exists(lrwd_json_path) and cache_best_params is None:
        # basis setup
        print(
            f"\nRun a series of {args.n_trials:d} short training runs with different learning rates\n" +
            "and weight decay values to find the good values for them before fully training the model\n",
            flush=True,
        )
        os.makedirs(lrwd_dir, exist_ok=True)
        lrwd_config_dict = deepcopy(config_dict)
        lrwd_config_dict["trial_dir"] = lrwd_dir
        
        # 50% of the total epoch should be enough to see if the learning rate and weight decay are good
        lrwd_epoch = int(0.5 * config_dict["num_epoch"])
        if lrwd_epoch <= 10:
            raise ValueError("'lrwd_epoch' must be greater than 10 to be effective")
        lrwd_config_dict["num_epoch"] = lrwd_epoch
        
        # run the learning rate and weight decay optimization
        best_params = lrwd_tuner(
            config_dict=lrwd_config_dict,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=args.n_trials,
        )
        
        # adjust the learning rate and weight decay in the original configuration dictionary and store the results 
        with open(lrwd_json_path, "w") as f:
            json.dump(best_params, f, indent=4)
        print(
            f"\nOptimal learning rate and weight decay: lr={best_params['lr']:.2e}, wd={best_params['wd']:.2e}",
            flush=True,
        )
    else:
        # load the best learning rate and weight decay from a previous optimization and update the configuration dictionary
        if cache_best_params is not None and args.reuse_lrwd_from_seed != config_dict["seed"]:
            best_params = cache_best_params
        else:
            with open(lrwd_json_path, "r") as f:
                best_params = json.load(f)
        print(
            f"\nUsing learning rate and weight decay from a previous optimization: lr={best_params['lr']:.2e}, wd={best_params['wd']:.2e}",
            flush=True,
        )
    config_dict["lr"] = best_params["lr"]
    config_dict["weight_decay"] = best_params["wd"]
    
    # either way, update the configuration dictionary with the optimized learning rate and weight decay
    with open(args.config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    
    """
    Train the final model with the optimized learning rate and weight decay.
    If the learning rate and weight decay were already optimized, this would run just like the normal training script.
    """
    
    # log message
    print(f"\nRunning full model training with lr={best_params['lr']:.2e}, wd={best_params['wd']:.2e}\n")
    
    # valididate the original configuration dictionary
    config_dict = TypeAdapter(ValidateConfigurationDict).validate_python(config_dict)
    
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
    # train a model and optimize the learning rate and weight decay
    main(args)