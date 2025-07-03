[Making this notebook a Repo/Package](https://www.kaggle.com/code/pranshubahadur/drw-dt-gfn-generate-trees-using-gflownets)

this repo is a WIP we thank creators of DT-GFN and their original repo. This is just an attempt to make dt-gfn useable in production pipelines / competitions.
# DT-GFN: GFlowNet-Boosted Decision Tree s

A clean, **modular**, and **high-performance** implementation of the Decision-Tree GFlowNet (DT-GFN) algorithm.  
The project builds on the research of **Mahfoud et al. (2025)** and was inspired by the Kaggle exploration *“drw-dt-gfn-generate-trees-using-gflownets”* by **Pranshu Bahadur**.

`DT-GFN` frames decision-tree construction as a *sequential decision process*.  
A **Generative Flow Network** learns to *draw* trees from an approximate Bayesian posterior, yielding a **diverse ensemble** of compact, interpretable models that excel on complex tabular data.

| Update | Complete Rollouts | TB Loss | FL Loss | Train ρ |
|:-----:|:-----------------:|:------:|:------:|:-------:|
| 01 → 10 | 10 / 10 | 58.6 → 72.5 | 0.003 → 0.0032 | **+0.146 → +0.262** |
| 20 | 10 / 10 | 58.9 | 0.0044 | **+0.300** |

---

## ✨ Key Features
| | |
|---|---|
| ⚡ **High-Performance Policy Nets** | PyTorch + `torch.jit.script` for graph-mode speed. |
| 🧩 **Flexible Pre-processing** | Quantile or uniform binning; supports **pre-binned** integer features. |
| 📦 **Zero-Dependency Tokenizer** | Pure-Python, HF-friendly, no external libs. |
| ⚙️ **Optimized Environment** | `TabularEnv` stores int8 tensors, CUDA-ready. |
| 🚀 **Advanced GFN Training** | Replay-buffered boosting; Trajectory Balance **+** Flow Matching losses (Zhang et al., 2023). |
| 🤖 **Scikit-Learn Wrapper** | `DTGFNRegressor` exposes familiar `.fit()` / `.predict()`. |
| ✅ **Fully Tested & Typed** | 100 % `pytest` coverage, mypy-clean. |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/pranshu-bahadur/dt-gfn.git
cd dt-gfn

# Install (add [dev] to get black, mypy, pytest, etc.)
pip install -e ".[dev]"

# Run tests
pytest -q
````

---

## 🚀 Quick-Start — Kaggle *DRW Crypto Market* Example

```python
import pandas as pd
from pathlib import Path
from src.wrappers.sklearn import DTGFNRegressor

# 1. Load data
DATA = Path("/kaggle/input/drw-crypto-market-prediction")
train = pd.read_parquet(DATA / "train.parquet")
test  = pd.read_parquet(DATA / "test.parquet")

# 2. Feature list
FEATURES = [c for c in train.columns if c.startswith("X") or
            c in ("bid_qty","ask_qty","buy_qty","sell_qty","volume")]

# 3. Configure & train
model = DTGFNRegressor(
    feature_cols = FEATURES,
    n_bins       = 256,
    bin_strategy = "feature_quantile",
    updates      = 20,
    rollouts     = 40,
    batch_size   = 8192,
    max_depth    = 4,
    boosting_lr  = 0.15,
    device       = "cuda",   # use "cpu" if no GPU
)

print("Fitting...")
model.fit(train[FEATURES], train["label"])

# 4. Predict & submit
print("Predicting...")
preds = model.predict(test[FEATURES])

subm = pd.read_csv(DATA / "sample_submission.csv")
subm["prediction"] = preds
subm.to_csv("dtgfn_submission.csv", index=False)
```

A **20-update** run on a single T4 GPU reaches **≈ 0.09–0.10** public LB.
Scale to **50–100** updates for the original 0.097 @ one-T4 result.

---

## Command-Line Interface

```bash
python -m src.cli \
  --train data/train.parquet \
  --test  data/test.parquet \
  --feature-cols X863 X856 X344 ... volume \
  --n-bins 256 --updates 50 --rollouts 60 \
  --batch-size 8192 --device cuda
```

Run `python -m src.cli --help` for all flags.

---

## Project Structure

```
dtgfn/
├─ src/
│  ├─ binning/         # Binner & BinConfig
│  ├─ env.py           # TabularEnv (int8, CUDA)
│  ├─ tokenizer.py     # Pure-Python Tokenizer
│  ├─ trees/
│  │   └─ policy.py    # PolicyPaperMLP
│  ├─ utils.py         # predictor, losses, replay, sampling
│  ├─ trainer.py       # Boost loop (.fit / .predict)
│  └─ wrappers/
│      └─ sklearn.py   # DTGFNRegressor
├─ tests/              # pytest suite
└─ docs/               # diagrams, assets
```

---

## Development & CI

* **Formatting** — `black .`
* **Static typing** — `mypy src/`
* **Testing** — `pytest` (≈ 6 s CPU)
* **GitHub Actions** — lint + tests on every push / PR

---

## Citation & Acknowledgements

If this repo helps your research or competition entry, please cite:

```bibtex
@misc{dtgfn_repo_2025,
  title  = {DT-GFN: A Modular Implementation of GFlowNet-Boosted Decision Trees},
  author = {Pranshu Bahadur and Timothy DeLise},
  year   = {2025},
  url    = {https://github.com/pranshu-bahadur/dt-gfn}
}

@article{mahfoud2025learning,
  title   = {Learning Decision Trees as Amortized Structure Inference},
  author  = {Mahfoud, Mohammed and Boukachab, Ghait and Koziarski, Michal and Hernandez-Garcia, Alex and Bauer, Stefan and Bengio, Yoshua and Malkin, Nikolay},
  journal = {arXiv preprint arXiv:2503.06985},
  year    = {2025}
}

@inproceedings{zhang2023let,
  title     = {Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets},
  author    = {Zhang, Dinghuai and Dai, Hanjun and Malkin, Nikolay and Courville, Aaron and Bengio, Yoshua and Pan, Ling},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023}
}
```

---

## License

**MIT** — see `LICENSE` for full text.


