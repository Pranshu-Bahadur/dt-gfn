[Making this notebook a Repo/Package](https://www.kaggle.com/code/pranshubahadur/drw-dt-gfn-generate-trees-using-gflownets)

this repo is a WIP we thank creators of DT-GFN and their original repo. This is just an attempt to make dt-gfn useable in production pipelines / competitions.

# Fast-DT-GFN-Boost

This repository implements a boosting variant of Decision Tree Generative Flow Networks (DT-GFN), inspired by the original DT-GFN framework for amortized structure inference in decision trees. It incorporates optimizations for faster training and inference, including parallel batched rollouts and vectorized Redundancy-Aware Selection (RAS), adapted from techniques in the Fast Monte Carlo Tree Diffusion (Fast-MCTD) paper. The implementation is designed for tabular data tasks, with a focus on scalability for larger datasets like those in Numerai competitions.

## Description

DT-GFN frames decision tree learning as a sequential decision-making process using Generative Flow Networks (GFNs) to sample trees proportionally to a Bayesian posterior. This repo extends it with a boosting ensemble approach, where each iteration fits trees to residuals, similar to gradient boosting. 

Key enhancements:
- **Parallel Batched Rollouts**: Enables GPU-parallel sampling of multiple trajectories (tree constructions) in batches, reducing rollout time.
- **Vectorized RAS**: Diversifies explorations by penalizing redundant actions in a vectorized manner, improving efficiency without significant overhead.

These features draw from parallel and redundancy-aware methods in Fast-MCTD, while building on GFN-based structure learning from DT-GFN and graph problems in "Let the flows tell".

@TODO: Add Sparse Planning

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Pranshu-Bahadur/dt-gfn.git
cd dt-gfn
pip install -r requirements.txt
```

Requirements include PyTorch, NumPy, Pandas, and tqdm. Tested on Python 3.10+ with CUDA support.

## Usage

### Quick Start

For local use (Numerai Data):

```python
from src.wrapper import DTGFNRegressor
import pandas as pd
from pathlib import Path

DATA_DIR = Path('/content/dt-gfn/v5.0')
train = pd.read_parquet(DATA_DIR / "train.parquet")

FEATURES = features['feature_sets']['all']

model = DTGFNRegressor(
    feature_cols=FEATURES,
    n_bins=5,
    updates=100,
    rollouts=10,
    batch_size=len(train),
    max_depth=9,
    boosting_lr=0.1,
    device="cuda",

)

model.fit(train[FEATURES], train["target"])

# Predict
preds = model.predict(df_test[FEATURES], 'ensemble') #'policy' for n trees generator
```

Adjust `num_parallel` in Config for VRAM constraints (e.g., 10 on A100 with 40GB).

### Training Parameters

- `updates`: Number of boosting iterations (default: 50).
- `rollouts`: Rollouts per update (default: 60).
- `max_depth`: Maximum tree depth (default: 7).
- `num_parallel`: Batch size for parallel rollouts (default: 10; tune for VRAM).

For deeper trees, increase `max_depth` and monitor VRAM (current rollout stage uses ~19GB on full Numerai dataset).

## Citations

This implementation builds on the following works:

- DT-GFN Paper: Mahfoud, M., et al. "Learning Decision Trees as Amortized Structure Inference." Frontiers in Probabilistic Inference Workshop, ICLR 2025.
- Fast-MCTD Paper: Yoon, J., et al. "Fast Monte Carlo Tree Diffusion: 100x Speedup via Parallel Sparse Planning." arXiv preprint arXiv:2506.09498, 2025.
- Let the Flows Tell Paper: Zhang, D., et al. "Let the Flows Tell: Solving Graph Combinatorial Problems with GFlowNets." NeurIPS 2023.
- [Original DT-GFN Repo](https://github.com/GFNOrg/dt-gfn)
- DT-GFN Kaggle Notebook: [DRW DT-GFN: generate trees using GFlowNets](https://www.kaggle.com/code/pranshubahadur/drw-dt-gfn-generate-trees-using-gflownets)
