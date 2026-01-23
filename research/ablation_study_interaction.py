"""
Run 'research/db_init.py' first.

This script runs an ablation study investigating layer interactions for OptiMetal2B.
We start with a fixed, manually designed architecture that performs well during training.
Then, for the selected architectural parameter (e.g., the node embedding layer type), 
we try all sensible permutations, while keeping everything else fixed. The learning
rate and weight decay are optimized for each though a small Optuna study.

This is a two-stage training process:
    1. In the first stage, the model is trained 25 times for a small number of epochs (e.g. 100) in order to determine an appropriate learning rate and weight decay.
    2. In the second stage, we train the model for the full number of epochs (e.g. 200), using the learning rate and weight decay found in the first stage.

The tested architectural permutations are generated in the following notebook: 'research/ablation_results.ipynb'.

First, we ran 'research/ablation_study.py' to try all possible architectural permutations for OptiMetal2B.
Then, we used this script to run all combinations from the two best-performing layer variations to investigate the impact of layer interactions.
We then used the final OptiMetal2B architecture as a starting point for the OptiMetal3B architecture. Next, we ran an ablation study on the message
passing layer used in the triplet block of OptiMetal3B, fixing all other layers to the configurations that performed best in the OptiMetal2B study. 
We used 'research/ablation_study.py' again for this step.

Disclaimer:
We never ran the final part of the ablation study for the triplet block of OptiMetal3B, i.e., the message passing layer ablation study.
This would have required around 10 A100s for multiple weeks, which is why we did not run it.
"""

from __future__ import annotations

import os
import time
import json
import socket
import optuna
import shutil
import argparse
from copy import deepcopy
from pydantic import TypeAdapter

import optimetal.factory as factory
from optimetal.training import Trainer
from optimetal.data.loader import load_torch_data, create_dataloader
from optimetal.utils import get_device, ValidateConfigurationDict, shutil_remove_readonly, print_dict

"""
-----------------------------------------------------------------------------------------------------------------------
PARSE INPUT ARGUMENTS:
"""

parser = argparse.ArgumentParser(description="OptiMetal ablation study")
parser.add_argument(
    "--perm_path", 
    type=str, 
    default="./ablation_data/2b_interaction_configs.json",
    help="Path to the JSON file containing the architectural permutations (relative paths are possible)",
)
parser.add_argument(
    "--study_path", 
    type=str, 
    default="./ablation_interaction_result",
    help="Path where the Optuna study and associated files will be stored (relative paths are possible)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Run this script multiple times with different seeds to get statistics"
)
parser.add_argument(
    "--num_train_data", 
    type=int, 
    default=20000, 
    help="Number of training data point the model is trained on",       
)
parser.add_argument(
    "--batch_size", 
    type=int, 
    default=256, 
    help="Training and validation batch size (depends on the available GPU memory)",        
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
    "--reuse_lrwd_from_seed",
    type=int,
    default=None,
    help=(
        "If set, try to load the learning rate and weight decay from the " + 
        "study directory of this seed. Falls back to optimisation " +
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

class AblationStudyInteraction:

    # training settings
    TRAIN_CONFIG = {
        "optimizer": {
            "type": "adamw",
        },
        "loss_fn_eps": "mae",
        "loss_fn_drude": "mae",
        "lr_scheduler": {
            "type": "cosineannealing",
            "T_max": 200,
            "eta_min": 0.0,
        },
        "warmup_epochs": 5,
        "grad_clip": 100,
        "eps_weight": 1,
        "drude_weight": 1,
        "early_stopping": False,
        "patience": 200,
        "num_epoch": 200,
        "precision": "bf16",
    }
    
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
    
    def __init__(
        self,
        perm_path: str,
        study_path: str,
        seed: int,
        num_train_data: int = 20000,
        batch_size: int = 256,
        train_path: str = "../graph/train.pt",
        val_path: str = "../graph/val.pt",
        reuse_lrwd_from_seed: int | None = None,
        n_trials: int = 25,
        device_index: int = 0,
        cleanup_timeout_hours: int = 24,
    ) -> None:
        # initialize class variables
        self.perm_path = perm_path
        self.study_path = study_path
        self.seed = seed
        self.num_train_data = num_train_data
        self.batch_size = batch_size
        self.train_path = train_path
        self.val_path = val_path
        self.n_trials = n_trials
        
        # global training settings
        self.train_config = deepcopy(self.TRAIN_CONFIG)
        
        # selected architecture permutations and key paths for the architecture dictionary
        if not os.path.exists(self.perm_path):
            raise FileNotFoundError(f"Permutations file '{self.perm_path:s}' does not exist")
        with open(self.perm_path, "r") as f:
            self.perms = json.load(f)
        print(
            f"\nRunning all {len(self.perms):d} architecture permutations for OptiMetal2B",
            flush=True,
        )
        print(
            f"\nFor each architecture permutation, we run a series of {self.n_trials:d} short\n" +
            "training runs with different learning rates and weight decay\nvalues to find the " +
            "good values for them before fully training the permutation\n",
            flush=True,
        )
        
        # cleanup stale runs before starting
        if os.path.isdir(self.study_path):
            self._cleanup_stale_runs(dir_path=self.study_path, timeout_hours=cleanup_timeout_hours)
        
        # setup things that do not change across trials
        self._build_dataloaders()
        self.device = get_device(index=device_index)
        self.loss_fn_eps = factory.create_loss_fn(self.train_config["loss_fn_eps"])
        self.loss_fn_drude = factory.create_loss_fn(self.train_config["loss_fn_drude"])

        # reusing the learning rate and weight decay only works if this script is run in a specific way, see "ablation_job_interaction.py"
        # I coded it poorly, but I doubt anyone will recreate it since the ablation study requires so many GPU hours...
        self.reuse_lrwd_from_seed = reuse_lrwd_from_seed
        if self.reuse_lrwd_from_seed is not None:
            self.cache_study_path = f"{os.path.dirname(self.study_path)}/optimetal2b_interaction_seed{self.reuse_lrwd_from_seed:d}"

    def _build_dataloaders(self) -> None:
        """
        Create the training and validation dataloaders that will be used throughout the study.
        """
        train_data = load_torch_data(self.train_path)
        val_data = load_torch_data(self.val_path)
        self.train_loader = create_dataloader(
            train_data, 
            num_data=self.num_train_data, 
            batch_size=self.batch_size, 
            shuffle=True, # shuffle the training set
            seed=self.seed,
        )
        print(f"Using {len(self.train_loader.dataset):d} graph from the training set", flush=True)
        self.val_loader = create_dataloader(
            val_data, 
            num_data=-1, # use the whole validation dataset 
            batch_size=self.batch_size, 
            shuffle=False, # do not shuffle the validation set
            seed=self.seed, # not needed here, but set it anyway
        )
        print(f"Using the entire validation set", flush=True)

    @staticmethod
    def _try_claim(dir_path: str) -> bool:
        """
        Attempt to secure a directory using an architecture permutation so that no other jobs attempt to run it.
        """
        run_flag = os.path.join(dir_path, "_running")
        try:
            os.mkdir(run_flag)
            with open(os.path.join(run_flag, "info.txt"), "w") as f: # add metadata
                f.write(f"{socket.gethostname()} pid={os.getpid()} ts={time.time()}\n")
            return True
        except FileExistsError:
            return False

    @staticmethod
    def _cleanup_stale_runs(dir_path: str, timeout_hours: float) -> None:
        """
        Remove stale '_running' flags older than 'timeout_hours'.
        input:
            timeout_hours:    Remove runs with timestamp older than this
        """
        cutoff = time.time() - timeout_hours * 3600
        for root, dirs, _ in os.walk(dir_path):
            if "_running" in dirs:
                running_dir = os.path.join(root, "_running")
                info_file = os.path.join(running_dir, "info.txt")
                try:
                    ts = float(open(info_file).read().split("ts=")[-1])
                except Exception:
                    ts = 0
                if ts < cutoff:
                    print(f"The {root:s} directory has probably crashed, so it will be deleted\n", flush=True)
                    shutil.rmtree(root, onexc=shutil_remove_readonly)

    @staticmethod
    def _mark_done(
        dir_path: str,
        seed: int,
        perm: dict,
        val_loss: float,
        lr_wd: dict,
        num_train_data: int,
        batch_size: int,
        n_trials: int,
        study_path: str,
    ) -> None:
        """
        Mark the directory of the architectural permutation as finished.
        Also save the results alongside the associated metadata for later analysis.
        """
        summary = {
            "seed": seed,
            "perm": perm,
            "val_loss": val_loss,
            "lr_wd": lr_wd,
            "num_train_data": num_train_data,
            "batch_size": batch_size,
            "n_trials": n_trials,
            "study_path": study_path,
        }
        run_flag = os.path.join(dir_path, "_running")
        done_flag = os.path.join(dir_path, "_done")
        with open(done_flag, "w") as f:
            json.dump(summary, f, indent=4)
        shutil.rmtree(run_flag, onexc=shutil_remove_readonly)

    def _load_cached_lrwd(self, perm_name: str) -> dict | None:
        """
        Try to load a cached learning rate / weight decay dictionary for
        the given permutation name from the reference seed. Returns 'None' if no cache is found.
        """
        if self.reuse_lrwd_from_seed is None:
            return None
        cache_file = os.path.join(
            self.cache_study_path,
            perm_name,
            "_done",
        )
        try:
            with open(cache_file, "r") as f:
                return json.load(f)["lr_wd"]
        except FileNotFoundError:
            return None

    def _trainer_from_cfg(self, config_dict: dict, trial: optuna.Trial | None = None) -> Trainer:
        # validate the configuration dictionary
        config_dict = TypeAdapter(ValidateConfigurationDict).validate_python(config_dict)
        
        # build the model
        model_config = config_dict.architecture
        model = factory.create_model(model_config)

        # setup the optimizer
        optim_config = config_dict.optimizer
        optimizer = factory.create_optimizer(model, optim_config)

        # setup the learning rate scheduler
        lr_scheduler_config = config_dict.lr_scheduler
        lr_scheduler = factory.create_lr_scheduler(
            optimizer, 
            lr_scheduler_config,
            warmup_epochs=config_dict.warmup_epochs,
        )
        
        # setup the training dictionary
        trainer_dict = {
            "config_dict": config_dict,
            "trial_dir": config_dict.trial_dir,
            "device": self.device,
            "train_loader": self.train_loader,
            "val_loader": self.val_loader,
            "model": model,
            "optimizer": optimizer,
            "loss_fn_eps": self.loss_fn_eps,
            "loss_fn_drude": self.loss_fn_drude,
            "lr_scheduler": lr_scheduler,
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

    def _lrwd_tuner(self, config_dict: dict) -> dict:
        """
        Helper function to quickly optimize the learning rate
        and weight decay for a given architecture permutation.
        """
        # copy the learning rate and weight decay grids
        lr_grid = deepcopy(self.LR_GRID)
        wd_grid = deepcopy(self.WD_GRID)
        
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
        def _objective(trial: optuna.Trial) -> float:
            local_config_dict = deepcopy(config_dict)
            local_config_dict["optimizer"]["lr"] = trial.suggest_categorical(
                "lr", 
                lr_grid,
            )
            local_config_dict["optimizer"]["weight_decay"] = trial.suggest_categorical(
                "wd", 
                wd_grid,
            )
            local_trial_dir = os.path.join(local_config_dict["trial_dir"], f"trial_{trial.number:06d}")
            os.makedirs(local_trial_dir, exist_ok=True)
            local_config_dict["trial_dir"] = local_trial_dir
            trainer = self._trainer_from_cfg(local_config_dict, trial=trial)
            trainer.train(local_config_dict["num_epoch"])
            return trainer.best_val_loss
        
        # find a good learning rate and weight decay
        study.optimize(_objective, n_trials=self.n_trials, timeout=None)
        return study.best_trial.params

    def _two_stage_training(self, perm: dict, trial_dir: str, perm_name: str) -> float:
        # prepare the configuration dictionary
        config_dict = {
            "seed": self.seed,
            "trial_dir": trial_dir,
            "num_train_data": self.num_train_data,
            "batch_size": self.batch_size,
        }
        config_dict.update(deepcopy(perm))
        config_dict.update(deepcopy(self.train_config))
            
        # 50% of the total epoch should be enough to see if the learning rate and weight decay are good
        train_epoch = config_dict["num_epoch"]
        lrwd_epoch = int(0.5 * train_epoch)
        if lrwd_epoch <= 10:
            raise ValueError("'lrwd_epoch' must be greater than 10 to be effective")

        # check if we can reuse the cached learning rate and weight decay
        cached_lrwd = self._load_cached_lrwd(perm_name)
        if cached_lrwd is None:
            # stage 1 - learning rate and weight decay optimization
            print(
                f"\nSearching for a good learning rate and weight decay using {lrwd_epoch:d} training epochs per trial\n",
                flush=True,
            )
            config_dict["num_epoch"] = lrwd_epoch
            config_dict["lr_scheduler"]["T_max"] = lrwd_epoch
            stage1_dir = os.path.join(trial_dir, "stage1")
            os.makedirs(stage1_dir, exist_ok=True)
            config_dict["trial_dir"] = stage1_dir
            best_params = self._lrwd_tuner(config_dict)
        else:
            print(f"\nReusing cached learning rate and weight decay", flush=True)
            best_params = cached_lrwd

        # stage 2 - full training run
        print(
            f"\nFull model training using {train_epoch:d} training epochs: lr={best_params['lr']:.2e} | wd={best_params['wd']:.2e}\n",
            flush=True,
        )
        config_dict_full = deepcopy(config_dict)
        config_dict_full["num_epoch"] = train_epoch
        config_dict_full["lr_scheduler"]["T_max"] = train_epoch
        config_dict_full["optimizer"]["lr"] = best_params["lr"]
        config_dict_full["optimizer"]["weight_decay"] = best_params["wd"]
        stage2_dir = os.path.join(trial_dir, "stage2")
        os.makedirs(stage2_dir, exist_ok=True)
        config_dict_full["trial_dir"] = stage2_dir
        trainer = self._trainer_from_cfg(config_dict_full, trial=None)
        trainer.train(config_dict_full["num_epoch"])
        return best_params, trainer.best_val_loss

    def run(self) -> None:
        """
        Main entry point: This is where we iterate over the selected permutations.
        """
        for i, perm in enumerate(self.perms):
            # try to setup the trial directory
            perm_name = f"architecture_{i:d}"
            trial_dir = os.path.join(
                self.study_path,
                perm_name,
            )
            os.makedirs(trial_dir, exist_ok=True)
            
            # check if this permutation has been completed
            if os.path.exists(os.path.join(trial_dir, "_done")):
                print(f"\nPermutation {i:d} already finished", flush=True)
                continue
            
            # check whether this permutation is currently being run as part of another job
            if not self._try_claim(trial_dir):
                print(f"\nAnother job probably performs the permutation {i:d}", flush=True)
                continue
            
            # run the permutation
            print(f"\nRunning permutation {i:d}", flush=True)
            print("")
            print_dict(perm)
            lr_wd, val_loss = self._two_stage_training(
                perm=perm, 
                trial_dir=trial_dir,
                perm_name=perm_name,
            )
            self._mark_done(
                dir_path=trial_dir,
                seed=self.seed,
                perm=perm,
                val_loss=val_loss,
                lr_wd=lr_wd,
                num_train_data=self.num_train_data,
                batch_size=self.batch_size,
                n_trials=self.n_trials,
                study_path=self.study_path,
            )
            print(f"[done] {perm_name:s} | seed={self.seed:d} | val_loss={val_loss:.4f}", flush=True)
                    
if __name__ == "__main__":
    # working directory setup, i.e., enable relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    # set and run the ablation study
    study = AblationStudyInteraction(
        perm_path=args.perm_path,
        study_path=args.study_path,
        seed=args.seed,
        num_train_data=args.num_train_data,
        batch_size=args.batch_size,
        train_path=args.train_path,
        val_path=args.val_path,
        reuse_lrwd_from_seed=args.reuse_lrwd_from_seed,
        n_trials=args.n_trials,
        device_index=args.device_index,
    )
    study.run()