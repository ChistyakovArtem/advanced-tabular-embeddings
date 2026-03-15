# PAF Experiments

Pilot study extending **"On Embeddings for Numerical Features in Tabular Deep Learning"**
(Gorishniy et al., NeurIPS 2022).

## Research questions

1. **Grid σ vs single σ** — instead of tuning a single `σ` for Periodic embeddings,
   initialise `k` coefficients from a log-spaced grid of σ values
   `σᵢ = exp(linspace(log(1e-3), log(100), k))`, covering the full frequency range
   simultaneously without HPO.

2. **Skip connections in Periodic** — augment `concat[sin, cos]` with either
   raw `x` (sc) or `relu(x)` (sc_af) to prevent information loss from non-monotonic activations.

3. **PAF-Net** — instead of using Periodic only as an input embedding,
   replace every `Linear→ReLU` hidden block with `Linear→PAF`.

## Model variants (13 total)

| Group          | Name                    | Description                              |
|----------------|-------------------------|------------------------------------------|
| Baseline       | MLP                     | Standard MLP                             |
| Embedding-only | EmbMLP-orig-plain       | OriginalPeriodic(plain) + MLP            |
|                | EmbMLP-orig-sc          | OriginalPeriodic + skip(x)               |
|                | EmbMLP-orig-sc_af       | OriginalPeriodic + skip(relu(x))         |
|                | EmbMLP-grid-plain       | GridPeriodic(plain) + MLP                |
|                | EmbMLP-grid-sc          | GridPeriodic + skip(x)                   |
|                | EmbMLP-grid-sc_af       | GridPeriodic + skip(relu(x))             |
| PAF-Net        | PAFNet-plain-const      | PAF hidden layers, single σ              |
|                | PAFNet-sc-const         | PAF + skip(x), single σ                  |
|                | PAFNet-sc_af-const      | PAF + skip(relu(x)), single σ            |
|                | PAFNet-plain-grid       | PAF hidden layers, grid σ                |
|                | PAFNet-sc-grid          | PAF + skip(x), grid σ                    |
|                | PAFNet-sc_af-grid       | PAF + skip(relu(x)), grid σ              |

## Project structure

```
paf_experiments/
├── run.py                      # main entry point (CLI)
├── requirements.txt
├── data/
│   ├── loader.py               # dataset loading + quantile preprocessing
│   └── <dataset_name>/         # place TabM datasets here
│       ├── N_train.npy
│       ├── N_val.npy
│       ├── N_test.npy
│       ├── y_train.npy
│       ├── y_val.npy
│       ├── y_test.npy
│       └── info.json           # {"task_type": "regression"|"binclass"|"multiclass"}
├── models/
│   ├── embeddings.py           # OriginalPeriodic, GridPeriodic
│   └── backbones.py            # MLP, PAFNet, EmbeddingMLP
├── experiments/
│   ├── runner.py               # model factory + experiment grid
│   └── trainer.py              # training loop, early stopping
└── results/
    ├── analysis.py             # summary table printer
    └── all_results.json        # written after each run
```

## Setup

```bash
pip install -r requirements.txt
```

Datasets: download from https://www.kaggle.com/datasets/artemchistyakov/datasets-in-tabm
and place each dataset folder inside `data/`.

## Usage

```bash
# California Housing, all 13 variants, 3 seeds
python run.py --datasets california

# Multiple datasets
python run.py --datasets california house adult

# Specific variants only
python run.py --variants MLP EmbMLP-grid-sc_af PAFNet-sc_af-grid

# Quick smoke test (3 variants, 1 seed, 5 epochs)
python run.py --smoke

# Print results table from saved JSON
python run.py --analyse

# All options
python run.py --help
```

## Key design decisions

- **No HPO** — all hyperparameters are fixed (`DefaultHParams` in `runner.py`).
  This is intentional: the goal is to compare embedding/activation strategies
  under identical conditions, not to optimise each model individually.

- **Quantile transform** applied to all numerical features (matches original paper).

- **3 seeds** per variant by default — gives mean ± std for each metric.

- **Early stopping** on validation metric (patience=16).

- Metrics: RMSE for regression (stored as −RMSE internally, higher = better),
  accuracy for classification.
