"""
Class to evaluate the performance of a graph neural network
trained to predict the dielectric function of metals using an
ensemble of models, that is, a set of models that have been
trained using different random seeds.
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Any
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec

import torch
import torch_geometric

import optimetal.utils as utils
import optimetal.factory as factory
import optimetal.data.transform as transform
from optimetal.data.dataset import ThreebodyData
from optimetal.data.loader import load_torch_data, create_dataloader

utils.load_plot_style()

def deep_diff(a: Any, b: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    """
    Recursive deep difference between python containers (dict/list/tuple).
    """
    diffs: list[tuple[str, Any, Any]] = []
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a.keys()) | set(b.keys())
        for k in sorted(keys):
            p = f"{path}.{k}" if path else str(k)
            if k not in a:
                diffs.append((p, "<missing>", b[k]))
            elif k not in b:
                diffs.append((p, a[k], "<missing>"))
            else:
                diffs.extend(deep_diff(a[k], b[k], p))
        return diffs
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append((f"{path}.length", len(a), len(b)))
        for i, (ai, bi) in enumerate(zip(a, b)):
            p = f"{path}[{i}]"
            diffs.extend(deep_diff(ai, bi, p))
        if len(a) > len(b):
            diffs.append((f"{path}[{len(b)}:]", a[len(b):], "<missing>"))
        elif len(b) > len(a):
            diffs.append((f"{path}[{len(a)}:]", "<missing>", b[len(a):]))
        return diffs
    if a != b:
        diffs.append((path, a, b))
    return diffs

def normalized_config(
    config: utils.ValidateConfigurationDict,
    allowed_to_differ: list[str] = ["seed", "trial_dir"],
) -> dict:
    """
    Helper function to convert the model configuration to a 
    dictionary and remove keys, we do not want to check for equality.
    """
    d = config.model_dump()
    # remove the allowed-to-differ key
    for key in allowed_to_differ:
        d.pop(key, None)
    return {k: v for k, v in d.items()}

def assert_configs_equal(
    configs: list[utils.ValidateConfigurationDict],
    paths: list[str],
) -> None:
    """
    Helper function to validate the the configuration dictionaries of from
    training runs with different seeds and training directories are equal.
    """
    norm_config = [normalized_config(c) for c in configs]
    ref = norm_config[0]
    for i, other in enumerate(norm_config[1:], start=1):
        diffs = deep_diff(ref, other, path="")
        if diffs:
            lines = [
                f"configuration mismatch (only 'seed' may differ):",
                f"- reference: {paths[0]}",
                f"- compared : {paths[i]}",
            ]
            for p, va, vb in diffs[:200]:
                lines.append(f"    {p}: {va!r} != {vb!r}")
            if len(diffs) > 200:
                lines.append(f"    ... and {len(diffs) - 200} more differences")
            raise ValueError("\n".join(lines))

class EnsembleEvaluator:
    """
    Custom class to evaluate the performance of a graph neural network predicting the dielectric
    function and Drude frequency of metals. Metrics and plots are inspired by the following publications: 
    https://doi.org/10.1103/PhysRevMaterials.8.L122201, https://doi.org/10.1002/adma.202409175 
    """
    
    def __init__(
        self, 
        best_model_paths: list[str],
        dataloader: torch_geometric.loader.DataLoader,
        device_index: int = 0, # default to the first GPU
        turn_off_progress_bar: bool = False,
        dataset_name: str = "val",
    ) -> None:
        # sanity checks
        for best_model_path in best_model_paths:
            if not os.path.exists(best_model_path):
                sys.exit(f"The 'best_model_path' {best_model_path!s} does not exist")
        
        # initialize class variables
        self.dataloader = dataloader
        self.best_model_paths = best_model_paths
        self.num_models = len(self.best_model_paths)
        self.best_model_dicts = []
        self.config_dicts = []
        self.amp_dtypes = []
        for best_model_path in self.best_model_paths:
            best_model_dict = load_torch_data(best_model_path)
            config_dict = best_model_dict["config_dict"]
            assert isinstance(config_dict, utils.ValidateConfigurationDict)
            amp_dtype = best_model_dict["precision"]
            self.best_model_dicts.append(best_model_dict)
            self.config_dicts.append(config_dict)
            self.amp_dtypes.append(amp_dtype)
            
        # check that all training configurations dictionaries match
        assert_configs_equal(self.config_dicts, self.best_model_paths)
        self.config_dict = self.config_dicts[0]
            
        # evaluation device
        self.device = utils.get_device(index=device_index)

        # ensure that the same level of precision is used here as in the model training
        # (and that all models where training with the same precision)
        if not all(self.amp_dtypes[0] == amp_dtype for amp_dtype in self.amp_dtypes):
            sys.exit("All models must be trained using the same precision")
        self.amp_dtype = self.amp_dtypes[0]
        self.use_amp = False # default to fp32
        if self.amp_dtype == torch.bfloat16:
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                self.use_amp = True
            else:
                raise SystemError(
                    "The selected models were trained using Bfloat16, " +
                    "however the current machine does not support it"
                )
        print(f"The model was trained using '{self.amp_dtype}'", flush=True)
        
        # progress bar flag
        self.turn_off_progress_bar = turn_off_progress_bar

        # load the best models and set them up for evaluation
        self._load_models()

        # recreate the total loss used in model training (this is the metric used to define quantiles)
        self.eps_weight = self.config_dict.eps_weight
        self.drude_weight = self.config_dict.drude_weight
        loss_type_eps = self.config_dict.loss_fn_eps
        self.loss_fn_eps = factory.create_loss_fn(loss_type_eps)
        loss_type_drude = self.config_dict.loss_fn_drude
        self.loss_fn_drude = factory.create_loss_fn(loss_type_drude)
        
        # helper variables for nicer plot labels
        self.dataset_name = dataset_name
        self.label_dict = {
            "loss": rf"$L_\mathrm{{{self.dataset_name:s}}}$",
            "eps_real_mae": r"MAE$[\mathrm{Re}(\overline{\varepsilon}_\mathrm{inter})]$",
            "eps_real_r2": r"R$^2[\mathrm{Re}(\overline{\varepsilon}_\mathrm{inter})]$",
            "eps_real_sc": r"SC$[\mathrm{Re}(\overline{\varepsilon}_\mathrm{inter})]$",
            "eps_imag_mae": r"MAE$[\mathrm{Im}(\overline{\varepsilon}_\mathrm{inter})]$",
            "eps_imag_r2": r"R$^2[\mathrm{Im}(\overline{\varepsilon}_\mathrm{inter})]$",
            "eps_imag_sc": r"SC$[\mathrm{Im}(\overline{\varepsilon}_\mathrm{inter})]$",
            "drude_diff": r"$\Delta\overline{\omega}_\mathrm{D}$ (eV)",
            "delta_e": r"$\Delta E$",
        }
        
        # load the plot style
        utils.load_plot_style()
        
    def _load_models(self) -> None:
        """
        It builds the best models, puts them on the device and then puts them in evaluation mode.
        We also store the number of model parameters in the object.
        """
        self.models = []
        model_config = self.config_dicts[0].architecture
        for i, best_model_dict in enumerate(self.best_model_dicts):
            model = factory.create_model(model_config)
            model.load_state_dict(best_model_dict["model_state_dict"])
            model.to(device=self.device)
            model.eval()
            self.models.append(model)
            print(f"Loaded model from {self.best_model_paths[i]!s}", flush=True)
        self.num_parameter = utils.get_model_parameters(self.models[0])
        
    def _ensemble_prediction(self, batch: ThreebodyData) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Obtain the average prediction from the ensemble of models 
        trained using different random seeds for a batch of graphs.
        """
        eps_list = []
        drude_list = []
        for model in self.models:
            eps, drude = model(batch)
            eps_list.append(eps)
            drude_list.append(drude)
        eps_stack = torch.stack(eps_list, dim=0) # (n_models, ...)
        drude_stack = torch.stack(drude_list, dim=0) # (n_models, ...)
        eps_mean = torch.mean(eps_stack, dim=0)
        drude_mean = torch.mean(drude_stack, dim=0)
        return eps_mean, drude_mean

    def _ensemble_prediction_min_max(
        self, 
        batch: ThreebodyData,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Obtain the average prediction from the ensemble of models 
        trained using different random seeds for a batch of graphs.
        Additionally, the output includes minimum and maximum predictions.
        """
        eps_list = []
        drude_list = []
        for model in self.models:
            eps, drude = model(batch)
            eps_list.append(eps)
            drude_list.append(drude)
        eps_stack = torch.stack(eps_list, dim=0) # (n_models, ...)
        drude_stack = torch.stack(drude_list, dim=0) # (n_models, ...)
        eps_min = torch.amin(eps_stack, dim=0)
        drude_min = torch.amin(drude_stack, dim=0)
        eps_mean = torch.mean(eps_stack, dim=0)
        drude_mean = torch.mean(drude_stack, dim=0)
        eps_max = torch.amax(eps_stack, dim=0)
        drude_max = torch.amax(drude_stack, dim=0)
        return eps_mean, drude_mean, eps_min, eps_max, drude_min, drude_max
        
    def _mae(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred - target))

    def _sc(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Similarity coefficient (SC) tailored to learning dielectric function, 
        see https://doi.org/10.1103/PhysRevMaterials.8.L122201.
        """
        return 1.0 - (torch.trapezoid(torch.abs(pred - target)) / torch.trapezoid(torch.abs(target)))
    
    def _r2(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - (torch.sum((pred - target)**2) / torch.sum((target - torch.mean(target))**2) + 1e-7)
    
    def evaluate(self) -> None:
        """
        Evaluates model performance using many different metrics. 
        Individual and average metrics are stored. The individual 
        metrics can be used to select materials from different quantiles.
        """
        # helper variables   
        drude_diff_list = []
        drude_rel_diff_list = []
        drude_pred_list = []
        drude_target_list = []
                
        # here we store the metrics for each material
        self.metrics = []
        
        # main evaluation
        for batch in tqdm(
            self.dataloader, 
            desc="Performing batch evaluation to obtain individual metrics", 
            disable=self.turn_off_progress_bar,
        ):
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                # prepare the batch
                batch.to(self.device)
                bsz = batch.num_graphs
                # call the model
                eps_pred, drude_pred = self._ensemble_prediction(batch)
                # needed because the model may output bf16
                eps_pred = eps_pred.to(torch.float32)
                drude_pred = drude_pred.to(torch.float32)
                # helper variables and output reshape
                mat_ids = batch.mat_id
                comps = batch.composition
                eps_pred = eps_pred.reshape(bsz, 2001, 2)
                eps_target = batch.eps.reshape(bsz, 2001, 2)
                drude_target = batch.drude
                color_target = batch.color.reshape(bsz, 3).detach().cpu().numpy()
                # evaluate each material individually
                for i in range(bsz):
                    # reference loss functions
                    eps_loss = self.loss_fn_eps(eps_pred[i, :, :], eps_target[i, :, :]).detach().item()
                    drude_loss = self.loss_fn_drude(drude_pred[i], drude_target[i]).detach().item()
                    loss = self.eps_weight * eps_loss + self.drude_weight * drude_loss
                    # interband dielectric function metrics for real and imaginary part
                    eps_real_mae = self._mae(eps_pred[i, :, 0], eps_target[i, :, 0]).detach().item()
                    eps_real_r2 = self._r2(eps_pred[i, :, 0], eps_target[i, :, 0]).detach().item()
                    eps_real_sc = self._sc(eps_pred[i, :, 0], eps_target[i, :, 0]).detach().item()
                    eps_imag_mae = self._mae(eps_pred[i, :, 1], eps_target[i, :, 1]).detach().item()
                    eps_imag_r2 = self._r2(eps_pred[i, :, 1], eps_target[i, :, 1]).detach().item()
                    eps_imag_sc = self._sc(eps_pred[i, :, 1], eps_target[i, :, 1]).detach().item()
                    # Drude frequency differences (metrics will be calculated later on)
                    drude_pred_val = drude_pred[i].detach().item()
                    drude_pred_list.append(drude_pred_val)
                    drude_target_val = drude_target[i].detach().item()
                    drude_target_list.append(drude_target_val)
                    drude_diff = (drude_pred[i] - drude_target[i]).detach().item()
                    drude_diff_list.append(drude_diff)
                    drude_rel_diff = ((drude_pred[i] - drude_target[i]) / (drude_target[i] + 1e-7)).detach().item()
                    drude_rel_diff_list.append(drude_rel_diff)
                    # color metric
                    eps_pred_np = eps_pred[i].detach().cpu().numpy()
                    eps_total_pred = (
                        eps_pred_np[:, 0] + 1j * eps_pred_np[:, 1]
                        + transform.calc_eps_intra(drude_pred_val)
                    )
                    color_pred = transform.calc_color(eps_total_pred)
                    delta_e  = np.linalg.norm(color_pred - color_target[i])
                    # store individual metrics
                    metric_dict = {
                        "mat_id": mat_ids[i].item().decode(),
                        "composition": comps[i].item().decode(),
                        "loss": loss,
                        "eps_real_mae": eps_real_mae,
                        "eps_real_sc": eps_real_sc,
                        "eps_real_r2": eps_real_r2,
                        "eps_imag_mae": eps_imag_mae,
                        "eps_imag_sc": eps_imag_sc,
                        "eps_imag_r2": eps_imag_r2,
                        "drude_diff": drude_diff,
                        "drude_rel_diff": drude_rel_diff,
                        "delta_e":  delta_e,
                    }
                    self.metrics.append(metric_dict)
        drude_pred_list = np.array(drude_pred_list)
        drude_target_list = np.array(drude_target_list)
        drude_diff_list = np.array(drude_diff_list)
        drude_rel_diff_list = np.array(drude_rel_diff_list)
        
        # calculate the global Drude frequency R2 metric
        drude_ss_res = np.sum((drude_target_list - drude_pred_list)**2)
        drude_ss_tot = np.sum((drude_target_list - np.mean(drude_target_list))**2) + 1e-7
        self.drude_r2 = 1.0 - (drude_ss_res / drude_ss_tot)
                        
        # calculate mean metrics for the dielectric function over all materials
        mean_loss = np.mean([m["loss"] for m in self.metrics]).item()
        mean_eps_real_mae = np.mean([m["eps_real_mae"] for m in self.metrics]).item()
        mean_eps_real_r2 = np.mean([m["eps_real_r2"] for m in self.metrics]).item()
        mean_eps_real_sc  = np.mean([m["eps_real_sc"] for m in self.metrics]).item()
        mean_eps_imag_mae = np.mean([m["eps_imag_mae"] for m in self.metrics]).item()
        mean_eps_imag_r2 = np.mean([m["eps_imag_r2"] for m in self.metrics]).item()
        mean_eps_imag_sc  = np.mean([m["eps_imag_sc"] for m in self.metrics]).item()
        mean_drude_mae = np.mean(np.abs(drude_diff_list))
        mean_drude_mape = 100.0 * np.mean(np.abs(drude_rel_diff_list))
        mean_delta_e  = np.mean([m["delta_e"] for m in self.metrics]).item()
        self.mean_metrics = {
            "loss": mean_loss,
            "eps_real_mae": mean_eps_real_mae,
            "eps_real_r2": mean_eps_real_r2,
            "eps_real_sc": mean_eps_real_sc,
            "eps_imag_mae": mean_eps_imag_mae,
            "eps_imag_r2": mean_eps_imag_r2,
            "eps_imag_sc": mean_eps_imag_sc,
            "drude_mae": mean_drude_mae,
            "drude_mape": mean_drude_mape,
            "delta_e": mean_delta_e,
        }
        
        # calculate the median metrics for the dielectric function over all materials
        median_loss = np.median([m["loss"] for m in self.metrics]).item()
        median_eps_real_mae = np.median([m["eps_real_mae"] for m in self.metrics]).item()
        median_eps_real_r2  = np.median([m["eps_real_r2"] for m in self.metrics]).item()
        median_eps_real_sc  = np.median([m["eps_real_sc"] for m in self.metrics]).item()
        median_eps_imag_mae = np.median([m["eps_imag_mae"] for m in self.metrics]).item()
        median_eps_imag_r2  = np.median([m["eps_imag_r2"] for m in self.metrics]).item()
        median_eps_imag_sc  = np.median([m["eps_imag_sc"] for m in self.metrics]).item()
        median_drude_mae = np.median(np.abs(drude_diff_list))
        median_drude_mape = 100.0 * np.median(np.abs(drude_rel_diff_list))
        median_delta_e  = np.median([m["delta_e"] for m in self.metrics]).item()
        self.median_metrics = {
            "loss": median_loss,
            "eps_real_mae": median_eps_real_mae,
            "eps_real_r2": median_eps_real_r2,
            "eps_real_sc": median_eps_real_sc,
            "eps_imag_mae": median_eps_imag_mae,
            "eps_imag_r2": median_eps_imag_r2,
            "eps_imag_sc": median_eps_imag_sc,
            "drude_mae": median_drude_mae,
            "drude_mape": median_drude_mape,
            "delta_e": median_delta_e,
        }
        
        # calculate the standard deviations of the metrics for the dielectric function over all materials
        std_loss = np.std([m["loss"] for m in self.metrics]).item()
        std_eps_real_mae = np.std([m["eps_real_mae"] for m in self.metrics]).item()
        std_eps_real_r2  = np.std([m["eps_real_r2"] for m in self.metrics]).item()
        std_eps_real_sc  = np.std([m["eps_real_sc"] for m in self.metrics]).item()
        std_eps_imag_mae = np.std([m["eps_imag_mae"] for m in self.metrics]).item()
        std_eps_imag_r2  = np.std([m["eps_imag_r2"] for m in self.metrics]).item()
        std_eps_imag_sc  = np.std([m["eps_imag_sc"] for m in self.metrics]).item()
        std_drude_mae = np.std(np.abs(drude_diff_list))
        std_drude_mape = 100.0 * np.std(np.abs(drude_rel_diff_list))
        std_delta_e  = np.std([m["delta_e"] for m in self.metrics]).item()
        self.std_metrics = {
            "loss": std_loss,
            "eps_real_mae": std_eps_real_mae,
            "eps_real_r2": std_eps_real_r2,
            "eps_real_sc": std_eps_real_sc,
            "eps_imag_mae": std_eps_imag_mae,
            "eps_imag_r2": std_eps_imag_r2,
            "eps_imag_sc": std_eps_imag_sc,
            "drude_mae": std_drude_mae,
            "drude_mape": std_drude_mape,
            "delta_e": std_delta_e,
        }
        
        # clear the GPU memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # log message if the process bar was turned off
        print("Evaluation finished", flush=True)
        
    def print_metrics(self) -> None:
        """
        Prints the metrics as a table.
        """
        if not self.metrics:
            raise ValueError("The dictionary 'self.metrics' is empty, call 'self.evaluate()' first")
        table = pd.DataFrame(
            {
                "Mean": self.mean_metrics,
                "Median": self.median_metrics,
                "Std": self.std_metrics,
            }
        ).round(3)
        print(table, flush=True)
        print(40*"-", flush=True)
        print(f"Drude Frequency R2: {self.drude_r2:.3f}", flush=True)
        
    def print_metric_latex_table(self) -> None:
        """
        Prints the metrics as a table in a latex format, ready to copy.
        Output:
            latex_table_str:    String of the LaTeX table (can be saved to  file...)
        """
        if not self.metrics:
            raise ValueError("The dictionary 'self.metrics' is empty, call 'self.evaluate()' first")
        def smart_fmt(x: float) -> str:
            return f"{int(round(x))}" if np.isclose(x, round(x)) else f"{x:.3f}"
        latex_label_dict = {
            "loss": rf"$L_\mathrm{{{self.dataset_name:s}}}$",
            "eps_real_mae": r"MAE$[\mathrm{Re}(\overline{\varepsilon}_\mathrm{inter})]$",
            "eps_real_r2": r"R$^2[\mathrm{Re}(\overline{\varepsilon}_\mathrm{inter})]$",
            "eps_real_sc": r"SC$[\mathrm{Re}(\overline{\varepsilon}_\mathrm{inter})]$",
            "eps_imag_mae": r"MAE$[\mathrm{Im}(\overline{\varepsilon}_\mathrm{inter})]$",
            "eps_imag_r2": r"R$^2[\mathrm{Im}(\overline{\varepsilon}_\mathrm{inter})]$",
            "eps_imag_sc": r"SC$[\mathrm{Im}(\overline{\varepsilon}_\mathrm{inter})]$",
            "drude_mae" : r"$\vert\Delta\overline{\omega}_\mathrm{D}\vert$ (eV)",
            "drude_mape" : r"$\delta\overline{\omega}_\mathrm{D}$ (\%)",
            "delta_e": r"$\Delta E$",
        }
        table = pd.DataFrame(
            {
                r"Mean": self.mean_metrics,
                r"Median": self.median_metrics,
                r"$\sigma$": self.std_metrics,
            }
        ).round(3)
        table.index = table.index.map(lambda k: latex_label_dict.get(k, k))
        table_str = table.to_latex(index=True, float_format=smart_fmt, column_format="l"+3*"c")
        break_after = [
            rf"$L_\mathrm{{{self.dataset_name:s}}}$",
            r"SC$[\mathrm{Re}(\overline{\varepsilon}_\mathrm{inter})]$",
            r"SC$[\mathrm{Im}(\overline{\varepsilon}_\mathrm{inter})]$",
            r"$\delta\overline{\omega}_\mathrm{D}$ (\%)",
        ]
        lines = table_str.splitlines()
        out = []
        for ln in lines:
            out.append(ln)
            for key in break_after:
                if ln.lstrip().startswith(key):
                    out.append(r"\midrule")
                    break
        latex_table_str = "\n".join(out)
        print(latex_table_str, flush=True)
        return latex_table_str
        
    def metric_histogram(self, metric_name: str , fig_path: str | None = None) -> tuple[Figure, Axes]:
        """
        Pick a metric and plot a histogram of its distribution.
        Input:
            metric_name:    Key that must exist in each material dictionary, e.g., 'eps_imag_mae'
            fig_path:       Optional path for storing the figure (must include a file extension, e.g., '.pdf' or '.png')
        Output:
            fig:            Matplotlib figure object
            ax:             Matplotlib axis object   
        """
        if metric_name not in self.label_dict: 
            raise KeyError(f"Unknown metric {metric_name:s}")
        if not self.metrics:
            raise ValueError("The dictionary 'self.metrics' is empty, call 'self.evaluate()' first")
        metric_list = [m[metric_name] for m in self.metrics] 
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.hist(metric_list, bins=100, color="tab:blue", edgecolor="k", linewidth=0.1)
        ax.set_xlabel(self.label_dict[metric_name])
        ax.set_ylabel(r"Occurrences")
        if ("_r2" in metric_name) or ("_sc" in metric_name):
            ax.set_xlim(left=0.0, right=1.0)
        fig.tight_layout()
        fig.align_labels()
        if fig_path is not None:
            fig.savefig(fig_path)
        return fig, ax
            
    def metric_violinplot(self, metric_name: str, fig_path: str | None = None) -> tuple[Figure, Axes]:
        """
        Pick a metric and create a violin plot of its distribution.
        Input:
            metric_name:    Key that must exist in each sample dictionary, e.g., 'eps_imag_mae'
            fig_path:       Optional path for storing the figure (must include a file extension, e.g., '.pdf' or '.png')
        Output:
            fig:            Matplotlib figure object
            ax:             Matplotlib axis object
        """
        if metric_name not in self.label_dict: 
            raise KeyError(f"Unknown metric {metric_name:s}")
        if not self.metrics:
            raise ValueError("The dictionary 'self.metrics' is empty, call 'self.evaluate()' first")
        metric_list = [m[metric_name] for m in self.metrics] 
        fig, ax = plt.subplots(figsize=(3, 3))
        vp = ax.violinplot(metric_list)
        for pc in vp["bodies"]:
            pc.set_facecolor("tab:blue")
            pc.set_edgecolor("k")
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_ylabel(self.label_dict[metric_name])
        if ("_r2" in metric_name) or ("_sc" in metric_name):
            ax.set_xlim(left=0.0, right=1.0)
        fig.tight_layout()
        fig.align_labels()
        if fig_path is not None:
            fig.savefig(fig_path)
        return fig, ax 
            
    def plot_rand_mats(
        self, 
        num_mats: int = 30, 
        gamma: float = 0.1, 
        interband_only: bool = False,
        fig_path: str | None = None, 
        batch_size: int | None = None, 
        rng_seed: int | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Plots comparing the predicted and target dielectric functions for a number of random materials.
        Depending on the model, we plot only the imaginary part of the dielectric function or the total
        dielectric function.
        Input:
            num_mats:           The number of random materials that we want to plot 
                                (currently, only 9, 16, or 30 are supported)
            gamma:              Broadening parameter for the intraband dielectric function (eV)
                                (~ 1 / scattering time)
            interband_only:     Only plots the interband part of the total dielectric function
            fig_path:           Optional path for storing the figure (must include a file extension, e.g., '.pdf' or '.png')
            batch_size:         Batch size for the dataloader, if 'None' it will be set to 'num_mats'
            rng_seed:           Optional seed for a reproducible sampling of materials
        Output:
            fig:                Matplotlib figure object
            axes:               Matplotlib axis array
        """
        if num_mats not in (9, 16, 30):
            raise ValueError(f"Currently, we only support plots of 9, 16 or 30 materials")
        # pick 'num_mats' random materials and put them into a dataloader
        # this could be problematic if the model is either very large or you have limited GPU memory...
        data_list = list(self.dataloader.dataset)
        rng = np.random.default_rng(rng_seed)
        rand_idx = rng.choice(len(data_list), size=num_mats, replace=False)
        plt_data = [data_list[idx] for idx in rand_idx]
        if batch_size is None:
            batch_size = num_mats
        dataloader = create_dataloader(
            plt_data, 
            num_data=-1,
            batch_size=batch_size, 
            shuffle=False,
            seed=self.config_dict.seed, # not needed here, but set it anyway
        )
        
        # get the predictions and targets
        samples = []
        for batch in dataloader:
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                # prepare the batch
                batch.to(self.device)
                bsz = batch.num_graphs
                # call the model
                output = self._ensemble_prediction_min_max(batch)
                eps_mean, drude_mean, eps_min, eps_max, drude_min, drude_max = output
                # needed because the model may output bf16
                eps_mean = eps_mean.to(torch.float32)
                drude_mean = drude_mean.to(torch.float32)
                eps_min = eps_min.to(torch.float32)
                drude_min = drude_min.to(torch.float32)
                eps_max = eps_max.to(torch.float32)
                drude_max = drude_max.to(torch.float32)
                # helper variables and output reshape
                comps = batch.composition
                eps_mean = eps_mean.reshape(bsz, 2001, 2)
                eps_min = eps_min.reshape(bsz, 2001, 2)
                eps_max = eps_max.reshape(bsz, 2001, 2)
                eps_target = batch.eps.reshape(bsz, 2001, 2)
                drude_target = batch.drude
                # get predictions for each material individually
                for i in range(bsz):
                    sample_info = {
                        "composition": comps[i].item().decode(),
                        "eps_mean": eps_mean[i, :, :].cpu().detach().numpy(),
                        "eps_min": eps_min[i, :, :].cpu().detach().numpy(),
                        "eps_max": eps_max[i, :, :].cpu().detach().numpy(),
                        "eps_target": eps_target[i, :, :].cpu().detach().numpy(),
                        "drude_mean": drude_mean[i].cpu().detach().numpy().item(),
                        "drude_min": drude_min[i].cpu().detach().numpy().item(),
                        "drude_max": drude_max[i].cpu().detach().numpy().item(),
                        "drude_target": drude_target[i].cpu().detach().numpy().item(),
                    }
                    samples.append(sample_info)
                    
        # create the plot
        if num_mats == 9:
            fig, axes = plt.subplots(3, 3, figsize=(6, 6))
            xlabel_xy = [0.54, 0.015]
            ylabel_xy = [0.017, 0.54]
        elif num_mats == 16:
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            xlabel_xy = [0.525, 0.01]
            ylabel_xy = [0.012, 0.52]
        elif num_mats == 30:
            fig, axes = plt.subplots(5, 5, figsize=(10, 10))
            xlabel_xy = [0.525, 0.01]
            ylabel_xy = [0.012, 0.52]
        else:
            raise ValueError(f"Currently, we only support plots of 9, 16 or 30 materials")
        for i, ax in enumerate(axes.ravel()):
            data = samples[i]
            omega = transform.get_omega() # shape: (2001)
            eps_inter_target = data["eps_target"] # shape: (2001, 2)
            eps_inter_pred = data["eps_mean"] # shape: (2001, 2)
            if self.config_dict.architecture["type"].lower() == "optimate":
                # the original OptiMate predicts only the imaginary part of the interband dielectric function
                ax.plot(omega, eps_inter_target[:, 1], "-", color="tab:blue")
                ax.plot(omega, eps_inter_pred[:, 1], "-", color="tab:orange",
                        label=utils.pretty_formula(data["composition"]))
                ymax = np.max([eps_inter_pred[:, 1], eps_inter_target[:, 1]])
                ax.set_ylim([0, 1.05 * ymax])
            else:
                # imaginary part: solid line & real part: dashed line
                if interband_only:
                    ax.plot(omega, eps_inter_target[:, 1], "-", color="tab:blue")
                    ax.plot(omega, eps_inter_target[:, 0], "--", color="tab:blue")
                    ax.plot(omega, eps_inter_pred[:, 1], "-", color="tab:orange", 
                            label=utils.pretty_formula(data["composition"]))
                    ax.plot(omega, eps_inter_pred[:, 0], "--", color="tab:orange")
                    ymin = np.min(eps_inter_target[:, 0])
                    ymax = np.max([eps_inter_target[:, 0], eps_inter_target[:, 1]])
                else:
                    eps_intra_target = transform.calc_eps_intra(data["drude_target"], gamma=gamma)
                    ax.plot(omega, eps_inter_target[:, 1] + eps_intra_target.imag, "-", color="tab:blue")
                    ax.plot(omega, eps_inter_target[:, 0] + eps_intra_target.real, "--", color="tab:blue")
                    eps_intra_pred = transform.calc_eps_intra(data["drude_mean"], gamma=gamma)
                    ax.plot(omega, eps_inter_pred[:, 1] + eps_intra_pred.imag, "-", color="tab:orange",
                            label=utils.pretty_formula(data["composition"]))
                    ax.plot(omega, eps_inter_pred[:, 0] + eps_intra_pred.real, "--", color="tab:orange")
                    omega_idx = omega > 0.75 # this proved to be fine when tested
                    ymin = np.min(eps_inter_target[omega_idx, 0])
                    ymax = np.max([eps_inter_target[omega_idx, 0], eps_inter_target[omega_idx, 1]])
                ax.set_ylim([ymin - 5, ymax + 5])
            ax.set_xticks([0, 5, 10, 15, 20])
            if i == 0:
                ax.plot([-2, -1], [0, 0], "-", color="tab:blue", label="DFT")
                ax.plot([-2, -1], [0, 0], "-", color="tab:orange", label="ML")
            ax.set_xlim(omega[0], omega[-1])
            leg = ax.legend(handlelength=1.25)
            for counter, item in enumerate(leg.legend_handles):
                if counter > 0:
                    break
                item.set_visible(False)
        fig.supxlabel(r"Energy (eV)", x=xlabel_xy[0], y=xlabel_xy[1])
        if self.config_dict.architecture["type"].lower() == "optimate" or interband_only:
            fig.supylabel(r"$\overline{\varepsilon}_\mathrm{inter}(\omega)$", x=ylabel_xy[0], y=ylabel_xy[1])
        else:
            fig.supylabel(r"$\overline{\varepsilon}(\omega)$", x=ylabel_xy[0], y=ylabel_xy[1])
        fig.tight_layout()
        fig.align_labels()
        if fig_path is not None:
            fig.savefig(fig_path)
        return fig, axes
        
    def quantile_plot(
        self,
        gamma: float = 0.1, 
        interband_only: bool = False,
        fig_path: str | None = None,
        batch_size: int | None = None, 
        rng_seed: int | None = None,
        colors: list[str] = ["tab:orange", "tab:blue", "tab:green", "tab:purple"],
    ) -> tuple[Figure, GridSpec]:
        """
        Create a figure similar to Fig. 2g in 10.1002/adma.202409175.
        Input:
            gamma:              Broadening for intraband dielectric function (eV)
            interband_only:     If true, plot only interband parts
            fig_path:           Optional path for storing the figure (must include a file extension, e.g., '.pdf' or '.png')
            batch_size:         Batch size for the dataloader, if 'None' it will be set to 'num_mats'
            rng_seed:           Optional seed for sampling materials from the quantiles
            colors:             Quantile colors, only adjust this if you increase 'n_quantiles'
        Output:
            fig:                Matplotlib figure object
            gs:                 Matplotlib gridSpec object
        """
        # hardcoded
        n_quantiles = 4
        n_per_quartile = 3
        # sanity check (not needed, because I hardcoded 'n_quantiles' and 'n_per_quartile')
        if n_per_quartile % 2 == 0:
            raise ValueError("'n_per_quartile' needs to be an odd number")
        if n_quantiles != len(colors):
            raise ValueError("Not enough colors were provided")
        if self.config_dict.architecture["type"].lower() == "optimate":
            raise NotImplementedError("This function is not supported for the OptiMate architecture")
        metric_vals = np.array([m["loss"] for m in self.metrics]) # only supported for total loss at the moment
        q_edges = np.quantile(metric_vals, np.linspace(0, 1, n_quantiles + 1))
        if rng_seed is None:
            rng = np.random.default_rng(self.config_dict.seed)
        else:
            rng = np.random.default_rng(rng_seed)
        quartile_idxs = []
        for i in range(n_quantiles):
            lo, hi = q_edges[i], q_edges[i+1]
            if i < n_quantiles - 1:
                mask = (metric_vals >= lo) & (metric_vals < hi)
            else:
                mask = (metric_vals >= lo) & (metric_vals <= hi)
            idxs = np.flatnonzero(mask)
            k = min(n_per_quartile, idxs.size)  # avoid choice error
            chosen = rng.choice(idxs, size=k, replace=False)
            # sort by metric for nicer display
            chosen = chosen[np.argsort(metric_vals[chosen], kind="mergesort")]
            quartile_idxs.append(chosen)
            
        # setup the dataloader
        mat_idxs = np.concatenate(quartile_idxs)
        plt_data = [self.dataloader.dataset[i] for i in mat_idxs]
        if batch_size is None:
            batch_size = n_quantiles * n_per_quartile
        dataloader = create_dataloader(
            plt_data, 
            num_data=-1,
            batch_size=batch_size, 
            shuffle=False,
        )
        
        # get the predictions and targets
        samples = []
        counter = 0
        for batch in dataloader:
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                # prepare the batch
                batch.to(self.device)
                bsz = batch.num_graphs
                # call the model
                output = self._ensemble_prediction_min_max(batch)
                eps_mean, drude_mean, eps_min, eps_max, drude_min, drude_max = output
                # needed because the model may output bf16
                eps_mean = eps_mean.to(torch.float32)
                drude_mean = drude_mean.to(torch.float32)
                eps_min = eps_min.to(torch.float32)
                drude_min = drude_min.to(torch.float32)
                eps_max = eps_max.to(torch.float32)
                drude_max = drude_max.to(torch.float32)
                # helper variables and output reshape
                comps = batch.composition
                eps_mean = eps_mean.reshape(bsz, 2001, 2)
                eps_min = eps_min.reshape(bsz, 2001, 2)
                eps_max = eps_max.reshape(bsz, 2001, 2)
                eps_target = batch.eps.reshape(bsz, 2001, 2)
                drude_target = batch.drude
                # get predictions for each material individually
                for i in range(bsz):
                    sample = {
                        "index": mat_idxs[counter],
                        "loss": metric_vals[mat_idxs[counter]],
                        "composition": comps[i].item().decode(),
                        "eps_mean": eps_mean[i, :, :].cpu().detach().numpy(),
                        "eps_min": eps_min[i, :, :].cpu().detach().numpy(),
                        "eps_max": eps_max[i, :, :].cpu().detach().numpy(),
                        "eps_target": eps_target[i, :, :].cpu().detach().numpy(),
                        "drude_mean": drude_mean[i].cpu().detach().numpy().item(),
                        "drude_min": drude_min[i].cpu().detach().numpy().item(),
                        "drude_max": drude_max[i].cpu().detach().numpy().item(),
                        "drude_target": drude_target[i].cpu().detach().numpy().item(),
                    }
                    samples.append(sample)
                    counter += 1
        
        # create the plot
        fig = plt.figure(figsize=(6.5, 4.5), constrained_layout=True)
        gs = fig.add_gridspec(n_quantiles, n_per_quartile + 2, width_ratios=[1] + [0.12] + [1]*n_per_quartile)
        ax = fig.add_subplot(gs[:, 0])
        n_quantiles = 4
        data = metric_vals
        bin_edges = np.histogram_bin_edges(data, bins=np.arange(0, 6, 0.025))
        bin_widths = np.diff(bin_edges)
        counts_all, _ = np.histogram(data, bins=bin_edges)
        start_bin = 0
        q_edges = np.quantile(data, np.linspace(0, 1, n_quantiles + 1))
        start_bin = 0
        for i in range(n_quantiles):
            hi = q_edges[i + 1]
            boundary_bin = np.searchsorted(bin_edges, hi, side="right") - 1
            boundary_bin = np.clip(boundary_bin, 0, len(bin_widths) - 1)
            idx_bins = np.arange(start_bin, boundary_bin + 1)
            ax.barh(
                bin_edges[idx_bins],
                counts_all[idx_bins],
                height=bin_widths[idx_bins],
                align="edge",
                color=colors[i],
                alpha=1,
                edgecolor="k",
                linewidth=0.1,
                label=rf"$Q_{{{(i + 1):d}}}$",
            )
            start_bin = boundary_bin + 1
        ax.set_ylim([0, 2]) # you may need to adjust this
        ax.invert_yaxis()
        ax.legend(handlelength=1.25, loc="lower right")
        ax.set_xlabel(r"Occurrences")
        ax.set_ylabel(rf"$L_\mathrm{{{self.dataset_name:s}}}$")
        for j in range(n_quantiles):
            for i in range(n_per_quartile):
                ax = fig.add_subplot(gs[j, i + 2])
                data = samples[i + j * (n_per_quartile)]
                omega = transform.get_omega() # shape: (2001)
                eps_inter_target = data["eps_target"] # shape: (2001, 2)
                eps_inter_pred = data["eps_mean"] # shape: (2001, 2)
                # imaginary part: solid line & real part: dashed line
                if interband_only:
                    ax.plot(omega, eps_inter_target[:, 1], "-", color="k")
                    ax.plot(omega, eps_inter_target[:, 0], "--", color="k")
                    ax.plot(omega, eps_inter_pred[:, 1], "-", color=colors[j], 
                            label=utils.pretty_formula(data["composition"]))
                    ax.plot(omega, eps_inter_pred[:, 0], "--", color=colors[j],
                            label=rf"$L_\mathrm{{{self.dataset_name:s}}} = {{{data['loss']:.2f}}}$")
                    ymin = np.min(eps_inter_target[:, 0])
                    ymax = np.max([eps_inter_target[:, 0], eps_inter_target[:, 1]])
                else:
                    eps_intra_target = transform.calc_eps_intra(data["drude_target"], gamma=gamma)
                    ax.plot(omega, eps_inter_target[:, 1] + eps_intra_target.imag, "-", color="k")
                    ax.plot(omega, eps_inter_target[:, 0] + eps_intra_target.real, "--", color="k")
                    eps_intra_pred = transform.calc_eps_intra(data["drude_mean"], gamma=gamma)
                    ax.plot(omega, eps_inter_pred[:, 1] + eps_intra_pred.imag, "-", color=colors[j],
                            label=utils.pretty_formula(data["composition"]))
                    ax.plot(omega, eps_inter_pred[:, 0] + eps_intra_pred.real, "--", color=colors[j],
                            label=rf"$L_\mathrm{{{self.dataset_name:s}}} = {{{data['loss']:.2f}}}$")
                    omega_idx = omega > 0.75 # this proved to be fine when tested
                    ymin = np.min(eps_inter_target[omega_idx, 0])
                    ymax = np.max([eps_inter_target[omega_idx, 0], eps_inter_target[omega_idx, 1]])
                ax.set_xlim(omega[0], omega[-1])
                ax.set_ylim([ymin - 10, ymax + 5])
                ax.set_xticks([0, 5, 10, 15, 20])
                ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=3, nbins=4))
                leg = ax.legend(handlelength=1.25, loc="upper right")
                for counter, item in enumerate(leg.legend_handles):
                    item.set_visible(False)
                if (j == n_quantiles - 1) and (i == n_per_quartile // 2):
                    ax.set_xlabel(r"Energy (eV)")
        if interband_only:
            fig.supylabel(r"$\overline{\varepsilon}_\mathrm{inter}(\omega)$", x=0.27, y=0.535)
        else:
            fig.supylabel(r"$\overline{\varepsilon}(\omega)$", x=0.27, y=0.535)  
        if fig_path is not None:
            fig.savefig(fig_path)
        return fig, gs
    
    def store_metrics(self, metric_path: str) -> None:
        """
        Store all obtained metrics in a JSON file.
        metric_path:       Path where all metrics should be stored (must end in '.json')
        """
        if not metric_path.endswith(".json"):
            raise ValueError("The variable 'metric_path' must end with '.json'")
        if not self.metrics:
            raise ValueError("The dictionary 'self.metrics' is empty, call 'self.evaluate()' first")
        json_dict = {
            "metrics": self.metrics,
            "mean_metrics": self.mean_metrics,
            "median_metrics": self.median_metrics,
            "std_metrics": self.std_metrics,
            "drude_r2": self.drude_r2,
        }
        with open(metric_path, "w") as f:
            json.dump(json_dict, f, indent=4)