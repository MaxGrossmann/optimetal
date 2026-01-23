"""
This is a rudimentary class for training a graph neural network, 
specifically on the dielectric function and Drude frequency of metals.

The trial directory is reset when training is started again, even if 
valid checkpoints and the best model checkpoint are found. No restart 
logic has been implemented.
"""

from __future__ import annotations

import os
import optuna
import random
import warnings
import numpy as np
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter # required when using the CUDA 12.8 nightly build for the RTX 5000 series

from optimetal.utils import listdir_full

class Trainer:
    """
    Custom training class, optimized for learning the dielectric function and Drude frequency of metals.
    """
    
    def __init__(
        self, 
        trainer_dict: dict,
        checkpoint_every: int = 10,
        best_model_start_epoch: int = 10,
    ) -> None:
        """
        The 'trainer_dict' dictionary contains:
            "config_dict":      Metadata (Which model was trained with which data and training settings)
            "trial_dir":        Directory where the checkpoint, best model and tensorboard logs are stored
            "device":           torch device
            "train_loader":     Dataloader for the training data 
            "val_loader":       Dataloader for the validation data
            "model":            Graph neural network model (must output the dielectric function and Drude frequency, see below)
            "optimizer":        Optimizer instance
            "loss_fn_eps":      Loss function for the dielectric function
            "loss_fn_drude":    Loss function for the Drude frequency
            "lr_scheduler":     None or learning rate scheduler instance
            "grad_clip:"        None or maximum norm of the gradients
            "eps_weight":       Interband dielectric function loss weight
            "drude_weight":     Drude frequency loss weight
            "early_stopping:    Flag to activate early stopping
            "patience":         Early stopping patience (only used if early_stopping=True)
            "seed":             Random seed
            "precision":        Automatic mixed precision training ("auto", "bf16" or "fp32")
            "trial":            (Optional) Optuna trial object, set it to 'None' or omit it
        The variables 'checkpoint_every' and 'best_model_start_epoch' are self-explanatory.
        The 'trainer_dict' is validated in the training script/notebook, check the 'research' directory.
        """
        # metadata (useful for reloading the model or training settings)
        self.config_dict = trainer_dict["config_dict"]
        
        # logging and checkpoint directory setup
        self.trial_dir = trainer_dict["trial_dir"]
        self.checkpoint_path = os.path.join(self.trial_dir, "checkpoint.pt")
        self.best_model_path = os.path.join(self.trial_dir, "best_model.pt")
        
        # core objects
        self.device = trainer_dict["device"]
        self.train_loader = trainer_dict["train_loader"]
        self.val_loader = trainer_dict["val_loader"]
        self.model = trainer_dict["model"]
        self.optimizer = trainer_dict["optimizer"]
        self.loss_fn_eps = trainer_dict["loss_fn_eps"]
        self.loss_fn_drude = trainer_dict["loss_fn_drude"]

        # training hyperparameters
        self.lr_scheduler = trainer_dict["lr_scheduler"]
        self.grad_clip = trainer_dict.get("grad_clip", None)
        self.max_grad_norm = self.grad_clip if self.grad_clip is not None else float("inf")
        self.eps_weight = trainer_dict["eps_weight"]
        self.drude_weight = trainer_dict["drude_weight"]
        self.early_stopping = trainer_dict["early_stopping"]
        self.patience = trainer_dict["patience"]
        
        # automatic mixed precision training
        self.amp_request = trainer_dict["precision"]
        if self.amp_request == "bf16":
            self.use_amp = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        elif self.amp_request == "fp32":
            self.use_amp = False
        else:
            raise ValueError(f"Invalid precision request: {self.amp_request:s}. Only 'bf16', 'fp32' or 'auto' are supported")
        self.amp_dtype = torch.bfloat16 if self.use_amp else None
        
        # random seed (reproducibility)
        self.seed = trainer_dict["seed"]
        self._set_random_seed(self.seed)
        
        # (optional) Optuna trial
        self.trial = trainer_dict.get("trial", None)
        if self.trial is not None:
            assert isinstance(self.trial, optuna.Trial)
        
        # epoch counter and early stopping tracking variables
        self.epoch = 0
        self.bad_epochs = 0
        self.best_val_loss = float("inf")
        
        # if there are any checkpoint files from a previous training run, delete them
        if os.path.exists(self.trial_dir):
            paths = listdir_full(self.trial_dir)
            for path in paths:
                # keep the learning rate and weight decay results
                if not os.path.isdir(path) and os.path.basename(path) != "lrwd.json":
                    os.remove(path)
            
        # setup the trial directory
        os.makedirs(self.trial_dir, exist_ok=True)
            
        # checkpoint frequency and when to start saving the best models
        self.checkpoint_every = checkpoint_every
        self.best_model_start_epoch = best_model_start_epoch
            
    def _set_random_seed(self, seed: int) -> None:
        """
        Sets a fixed random seed for reproducibility.
        Input:
            seed:       Integer
        """
        print(f"Setting RNG seed to {seed:d}", flush=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
            
    def _train_step(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function to perform one training epoch.
        Returns the average train loss and the gradient norm.
        """
        train_loss = 0.0
        total_grad_norm = 0.0
        num_samples = 0
        num_batches = 0
        self.model.to(self.device) # needed when calling this function by itself
        self.model.train()
        for batch in self.train_loader:
            batch.to(self.device)
            # optimizer setup
            self.optimizer.zero_grad()
            # automated mixed precision
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                # call the model (must return the dielectric function and the Drude frequency)
                # 'eps'   - shape: (batch_size * 2001, 2), 1st column is the real part, 2nd column is the imaginary part
                # 'drude' - shape: (batch_size) 
                eps, drude = self.model(batch)
                # loss functions
                # 'batch.eps'   - shape: (batch_size * 2001, 2), 1st column is the real part, 2nd column is the imaginary part
                # 'batch.drude' - shape: (batch_size) 
                eps_loss = self.loss_fn_eps(eps, batch.eps)
                drude_loss = self.loss_fn_drude(drude, batch.drude)
                loss = self.eps_weight * eps_loss + self.drude_weight * drude_loss
            # backpropagation
            loss.backward()
            # optional gradient clipping and compute gradient norm (useful to log)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            # perform a optimization step
            self.optimizer.step()
            # aggregate the loss
            train_loss += loss.detach().item() * batch.num_graphs
            total_grad_norm += grad_norm.detach().item()
            num_samples += batch.num_graphs
            num_batches += 1
        avg_train_loss = train_loss / num_samples
        avg_grad_norm = total_grad_norm / num_batches
        return avg_train_loss, avg_grad_norm

    def _val_step(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate model performance on the validation set.
        """
        eps_val_loss = 0.0
        drude_val_loss = 0.0
        val_loss = 0.0
        num_samples = 0
        self.model.to(self.device) # needed when calling this function by itself
        self.model.eval()
        for batch in self.val_loader:
            # no gradient tracking and automated mixed precision
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                batch.to(self.device)
                # call the model (must return the dielectric function and the Drude frequency)
                # 'eps'   - shape: (batch_size * 2001, 2), 1st column is the real part, 2nd column is the imaginary part
                # 'drude' - shape: (batch_size) 
                eps, drude = self.model(batch)
                # loss functions
                # 'batch.eps'   - shape: (batch_size * 2001, 2), 1st column is the real part, 2nd column is the imaginary part
                # 'batch.drude' - shape: (batch_size) 
                eps_loss = self.loss_fn_eps(eps, batch.eps)
                drude_loss = self.loss_fn_drude(drude, batch.drude)
                loss = self.eps_weight * eps_loss + self.drude_weight * drude_loss
                # aggregate the losses
                val_loss += loss.detach().item() * batch.num_graphs
                eps_val_loss += eps_loss.detach().item() * batch.num_graphs
                drude_val_loss += drude_loss.detach().item() * batch.num_graphs
                num_samples += batch.num_graphs
        avg_val_loss = val_loss / num_samples
        avg_eps_val_loss = eps_val_loss / num_samples
        avg_drude_val_loss = drude_val_loss / num_samples
        return avg_val_loss, avg_eps_val_loss, avg_drude_val_loss
    
    def train(self, num_epoch: int) -> None:
        """
        Main training loop.
        Input:
            num_epochs:     Number of training epochs
        """
        # clear the GPU memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        # initialize tensorboard and print some logs
        self.writer = SummaryWriter(log_dir=self.trial_dir)
        precision_str = "Bfloat16" if self.use_amp else "Float32"
        print(
            f"Training for {num_epoch:d} epochs on device={self.device.type.upper():s} " + 
            f"(batch size={self.train_loader.batch_size:d}, precision={precision_str:s})",
            flush=True
        )
        print(f"The best model is stored/updated after the initial {self.best_model_start_epoch:d} epochs", flush=True)
        print(f"Checkpoints are saved every {self.checkpoint_every:d} epochs", flush=True)
        self.model.to(self.device)
        for _ in range(num_epoch):
            # training step 
            train_loss, grad_norm = self._train_step()
            # validation step
            self.val_loss, self.eps_val_loss, self.drude_val_loss = self._val_step()
            # command line log
            print(
                f"    {datetime.now():%Y%m%d-%H%M%S}: epoch {self.epoch:04d} | train_loss={train_loss:.4f} | " +
                f"val_loss={self.val_loss:.4f} | eps_val_loss={self.eps_val_loss:.4f} | drude_val_loss={self.drude_val_loss:.4f} ",
                flush=True,
            )
            # log metrics
            self.writer.add_scalar("train/loss", train_loss, self.epoch)
            self.writer.add_scalar("train/grad_norm", grad_norm, self.epoch)
            self.writer.add_scalar("val/loss", self.val_loss, self.epoch)
            self.writer.add_scalar("val/eps", self.eps_val_loss, self.epoch)
            self.writer.add_scalar("val/drude", self.drude_val_loss, self.epoch)
            if self.lr_scheduler is not None:
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/learning_rate", current_lr, self.epoch)
            self.writer.flush()
            # save a checkpoint
            if (self.epoch > 0) and (self.epoch % self.checkpoint_every) == 0:
                self._save_checkpoint()
            # check if we have a new best model
            if self.val_loss < (self.best_val_loss - 1e-3): # loss is always minimized
                self.bad_epochs = 0
                self.best_val_loss = self.val_loss
                if self.epoch >= self.best_model_start_epoch:
                    self._save_model()
            else:
                self.bad_epochs += 1
            # check patience for early stopping
            if self.early_stopping and self.bad_epochs >= self.patience:
                print("Early stopping", flush=True)
                break
            # update the learning rate scheduler
            if self.lr_scheduler is not None:
                # the learning rate scheduler 'ReduceLROnPlateau' needs a metric (validation loss)
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(self.val_loss)
                else:
                    # https://github.com/pytorch/pytorch/issues/76113
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.SequentialLR):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            self.lr_scheduler.step()
                    else:
                        self.lr_scheduler.step()
            # report to Optuna if a trial is provided
            if self.trial is not None:
                self.trial.report(self.val_loss, self.epoch)
                if self.trial.should_prune():
                    self.writer.close()
                    raise optuna.exceptions.TrialPruned(f"Trial pruned at epoch {self.epoch:d}")
            # increment the epoch counter
            self.epoch += 1
        print(f"Training complete (best_val_loss={self.best_val_loss:.4f})\n", flush=True)
        self.writer.close()
        # clear the GPU memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
                
    def _save_model(self) -> None:
        """
        Save a model. The 'config_dict' stores all the relevant settings used to train this model.
        """    
        checkpoint = {
            "epoch": self.epoch, # good to know
            "model_state_dict": self.model.state_dict(),
            "config_dict": self.config_dict, # needed to run 'optimetal.Evaluator'
            "precision": self.amp_dtype, # good to know (None = float32)
        }
        torch.save(checkpoint, self.best_model_path) 
        
    def _save_checkpoint(self) -> None:
        """
        Saves a checkpoint containing all relevant information about the current training state.
        """
        checkpoint = {
            "epoch": self.epoch, # so we know where we left off with the training
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": None,
            "config_dict": self.config_dict, # contains all training parameters
            "precision": self.amp_dtype, # good to know (None = float32)
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        checkpoint["numpy_rng_state"] = np.random.get_state()
        checkpoint["torch_rng_state"] = torch.get_rng_state()
        if self.device.type == "cuda":
            checkpoint["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
        torch.save(checkpoint, self.checkpoint_path)