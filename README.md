# OptiMetal

This repository contains the implementation of the OptiMetal model family: a set of graph neural networks for predicting the optical properties of metals from structural information alone.

In addition, the repository provides all code and derived results used in the publication [*Broken neural scaling laws in materials science*](https://arxiv.org/), including dataset analytics, architecture optimization results, and neural scaling law analyses, as well as instructions for obtaining the associated datasets.

---

### Project structure

| Path                   | Purpose                                                                                                                     
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`./optimetal/`** | This directory contains the source code for the OptiMetal model family. It also includes code for dataset analytics, data preprocessing, graph setup, model training, and evaluation. Currently, the inference class is still a work in progress. The models are written in a flexible manner that allows for systematic adjustments to their architecture. |
| **`./research/`** | This directory contains the scripts used for analyzing the ab initio dataset, converting crystal structures into graphs, optimizing the model architecture, investigating neural scaling laws, and analyzing other machine learning results presented in the publication and Supplementary Information. A summary of the results is provided as JSON files and stored in subdirectories. |

---

### Installation

#### Code

```
# 1. clone the repository
git clone https://github.com/MaxGrossmann/optimetal.git
cd optimetal

# 2. (optional) create an isolated python environment, e.g., using conda
conda create -n optimetal python=3.12
conda activate optimetal

# 3. install torch (choose the correct command for your system)
# see https://pytorch.org/get-started/locally/

# 4. install torch_geometric matching your torch version
# see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# 5. install the code and its dependencies
pip install -e .

```

**Note:** The default installation via `pip install -e .` installs the CPU-only versions of PyTorch and PyTorch Geometric. 
Users with NVIDIA GPUs who wish to use CUDA or `bfloat16` inference should first install a CUDA-enabled version of PyTorch and the corresponding PyTorch Geometric packages.
In other words, users with NVIDIA GPUs should follow and complete steps 3 and 4 before running `pip install -e .`.

#### Data

There are two paths to obtain the dataset containing the dielectric functions and Drude frequencies of over 200,000 intermetallic compounds used in the study, as well as preparing it for machine learning (see `./research/db_compression.py` and `./research/db_init.py`):

1. Download the complete ab initio dataset which is distributed across two repositories due to its size: [*Part 1*](https://doi.org/10.6084/m9.figshare.31111798) and [*Part 2*](https://doi.org/10.6084/m9.figshare.31112491). First run `./research/db_compression.py` (see documentation given in the script). Then run `./research/db_init.py` (see documentation given in the script). The directories `./dataset` and `./graph` should then be created and contain the files `train.h5`, `val.h5`, `test.h5`, and `train.pt`, `val.pt`, `test.pt`, respectively.

2. Download the version of the dataset already preprocessed for machine learning from [*here*](https://doi.org/10.6084/m9.figshare.31112554). Extract the directories `./dataset` and `./graph` into the main directory of this repository.

---

### Model disclaimer

All models provided in this repository (see `./optimetal/files`) were trained using mixed-precision arithmetic with `bfloat16` to reduce memory usage and accelerate computation. 
We observed small yet systematic differences between inference performed in `bfloat16` and `fp32`. 
Therefore, to reproduce the results reported in the publication as closely as possible, inference should be performed using `bfloat16`.

Running inference in `bfloat16` requires NVIDIA GPUs of the Ampere generation or newer. 
Inference in full precision (`fp32`) is generally possible on older hardware, though it may result in slight numerical discrepancies from the reported results.

Example training and model evaluation workflows can be tested using the notebooks `./research/train_model.ipynb` and `./research/evaluate_model.ipynb`, respectively.

---

### Detailed project structure (with selected files)

Here, we only mention some exemplary scripts and notebooks that are important for the main results.
We attempted to write all scripts and notebooks in a self-documenting manner. 
Therefore, please refer to the docstring at the top of each one for details.

```
./optimetal/
├── data/                       # Data loading, preprocessing, and transformations
│   ├── dataset.py              # Custom torch_geometric 'Data' classes to handle threebody indices
│   ├── loader.py               # Data loading utilities
│   ├── preprocess.py           # Preprocessing functions
│   └── transform.py            # Data transformations (crystal -> graph, color calculation, etc.)
├── nn/                         # Graph neural networks
│   ├── optimate.py             # OptiMate reference model
│   ├── optimetal_2b.py         # Flexible implementation of OptiMetal2B
│   ├── optimetal_3b.py         # Flexible implementation of OptiMetal3B
│   └── utils/                  # Embeddings, message passing, pooling layers, etc.
├── factory/                    # Factory functions for models, optimizers, loss functions
├── evaluation/                 # Class for model evaluation
├── training/                   # Training loop and callbacks
├── utils/                      # Utility functions and helpers (mostly for plotting)
├── inference/                  # Inference utilities (work in progress)
└── files/                      # Pretrained models and data files
./research/
├── train_model.ipynb           # Training example
├── evaluate_model.ipynb        # Model evaluation (an ensemble version also exists)
├── ablation_results.ipynb      # Architecture optimization results
├── scaling_law_results.ipynb   # Neural scaling law analysis
├── ...
└── other_results.ipynb         # Additional results, see Supplementary Information
```

The directory `./research/` contains many subdirectories.
For instance, the directory `./research/scaling_law_base_config` contains the model configurations for the 1D neural scaling laws.
The corresponding directory `./research/scaling_data` contains the results, which are stored in JSON files. 
Similarly, the results of the model evaluations on the test set can be found in the directory `./research/best_model_eval`.

---

### Naming conventions

Some scripts, notebooks, and configuration files follow naming conventions used throughout the development process. 
In particular, filenames containing `ablation` refer to experiments performed to optimize the architecture. 
For scaling-law studies, configuration names containing `hestness`[^1] refer to data-scaling experiments, while `kaplan`[^2] refers to parameter-scaling experiments. 
These names are used consistently throughout the repository.

[^1]: J. Hestness, S. Narang, N. Ardalani, G. Diamos, H. Jun, H. Kianinejad, M. M. A. Patwary, Y. Yang, and Y. Zhou, Deep learning scaling is predictable, empirically, arXiv:1712.00409 (2017), https://doi.org/10.48550/arXiv.1712.00409
[^2]: J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, Scaling laws for neural language models, arXiv:2001.08361 (2020), https://doi.org/10.48550/arXiv.2001.08361