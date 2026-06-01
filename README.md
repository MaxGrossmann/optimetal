# OptiMetal

This repository contains the implementation of the OptiMetal model family: a set of invariant and equivariant graph neural networks for predicting the optical properties of metals from structural information alone.

In addition, the repository provides all code and derived results used in the publication [*Broken neural scaling laws in learning the optical properties of solids*](https://arxiv.org/abs/2602.05702), including dataset analytics, architecture optimization results, and neural scaling law analyses, as well as instructions for obtaining the associated datasets.

---

### Project structure

| Path                   | Purpose                                                                                                                     
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`./optimetal/`** | Source code for the invariant OptiMetal models, including data handling, graph construction, training, and evaluation utilities. |
| **`./optimetal_e3/`** | Source code for the equivariant OptiMetalE3 model based on the UMA architecture [^1]. |
| **`./research/`** | Scripts and notebooks for dataset preparation, architecture optimization, model training, evaluation, and neural scaling law analyses for the invariant models. |
| **`./research_e3/`** | Scripts and notebooks for graph preparation, equivariance checks, training, evaluation, and neural scaling law analyses for the equivariant model. |

Most scripts and notebooks contain a brief description at the beginning, in the form of a docstring or Markdown, explaining their purpose and expected inputs. 
The notebook `./nsl_analysis.ipynb` provides the main neural scaling law analysis and was used to produce the central results presented in the publication.
Additional entry points are the training and evaluation notebooks in the `./research` directory and the corresponding equivariant model scripts and notebooks in the `./research_e3` directory.

---

### Installation

#### CPU-only installation

```
# 1. clone the repository
git clone https://github.com/MaxGrossmann/optimetal.git
cd optimetal

# 2. (optional) create an isolated python environment, e.g., using conda
conda create -n optimetal python=3.12
conda activate optimetal

# 3. install the code and its dependencies
pip install -e .
```

#### Installation with NVIDIA GPU support

```
# 1. clone the repository
git clone https://github.com/MaxGrossmann/optimetal.git
cd optimetal

# 2. (optional) create an isolated python environment, e.g., using conda
conda create -n optimetal python=3.12
conda activate optimetal

# 3. install PyTorch
# Choose the command matching your system from:
# https://pytorch.org/get-started/locally/
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# 4. install PyTorch Geometric matching your PyTorch version
# See:
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html

# 5. install e3nn, used by the equivariant model
pip install --upgrade e3nn

# 6. install the code and remaining dependencies
pip install -e .
```

### Data download

There are two ways to set up the dataset containing the dielectric functions and Drude frequencies of over 200,000 intermetallic compounds used in the study:

1. Download the complete ab initio dataset, which is distributed across two repositories due to its size: [*Part 1*](https://doi.org/10.6084/m9.figshare.31111798) and [*Part 2*](https://doi.org/10.6084/m9.figshare.31112491). First run `./research/db_compression.py` (see documentation given in the script). Then run `./research/db_init.py` (see documentation given in the script). The directories `./dataset` and `./graph` should then be created and contain the files `train.h5`, `val.h5`, `test.h5`, and `train.pt`, `val.pt`, `test.pt`, respectively.

2. Download the version of the dataset already preprocessed for machine learning from [*here*](https://doi.org/10.6084/m9.figshare.31112554). Extract the directories `./dataset` and `./graph` into the main directory of this repository.

To ensure full reproducibility, the dataset splits produced by the script `./research/db_compression.py` can be found in the directory `./research/data_splits`.

---

### Notes on pretrained models and inference precision

The pretrained invariant OptiMetal models in the `./optimetal/files` directory were trained using mixed-precision arithmetic with `bfloat16` to reduce memory usage and accelerate computation. 
However, we observed small but systematic numerical differences between inference using `bfloat16` compared to `fp32`. 
Therefore, to reproduce the reported results as closely as possible, inference with these models should be performed using `bfloat16`.

`bfloat16` inference requires NVIDIA GPUs of the Ampere generation or newer. 
Inference using `fp32` is generally possible on older hardware, but may lead to small numerical deviations from the reported values.

The equivariant OptiMetalE3 models based on the UMA architecture were trained and evaluated using `fp32`. 
Due to their file size, the pretrained OptiMetalE3 model checkpoints are not included directly in this repository. 
They can be downloaded from [*here*](https://doi.org/10.6084/m9.figshare.31112554) and should be placed in `./optimetal_e3/files` before running the corresponding evaluation scripts.

Example training workflows can be found in `./research/train_model.ipynb` and `./research_e3/train_model.py`.
Example single-model evaluation workflows are provided in `./research/evaluate_model.ipynb` and `./research_e3/evaluate_model.ipynb`. 
For ensemble evaluation, see `./research/evaluate_model_ensemble.ipynb` and `./research_e3/evaluate_model_ensemble.ipynb`.

---

### Naming conventions

Some scripts, notebooks, and configuration files follow naming conventions used throughout the development process. 
In particular, filenames containing `ablation` refer to experiments performed to optimize the architecture. 
For scaling-law studies, configuration names containing `hestness`[^2] refer to data-scaling experiments, while `kaplan`[^3] refers to parameter-scaling experiments. 
These names are used consistently throughout the repository.

---

### References 

[^1]: B. Wood, M. Dzamba, X. Fu, M. Gao, M. Shuaibi, L. Barroso-Luque, K. Abdelmaqsoud, V. Gharakhanyan, J. Kitchin, D. Levine, K. Michel, A. Sriram, T. Cohen, A. Das, S. Sahoo, A. Rizvi, Z. Ulissi, and L. Zitnick, UMA: A family of universal models for atoms, in Advances in Neural Information Processing Systems, Vol. 38, edited by D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (Curran Associates, Inc., 2025) p. 129391–129427, https://doi.org/10.48550/arXiv.2506.23971
[^2]: J. Hestness, S. Narang, N. Ardalani, G. Diamos, H. Jun, H. Kianinejad, M. M. A. Patwary, Y. Yang, and Y. Zhou, Deep learning scaling is predictable, empirically, arXiv:1712.00409 (2017), https://doi.org/10.48550/arXiv.1712.00409
[^3]: J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, Scaling laws for neural language models, arXiv:2001.08361 (2020), https://doi.org/10.48550/arXiv.2001.08361
