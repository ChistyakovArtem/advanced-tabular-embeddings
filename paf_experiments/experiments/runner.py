"""
Experiment runner.

Two modes:
  1. Fixed HPs (default): run all variants with DefaultHParams, no tuning.
     Good for quick comparisons and smoke tests.

  2. Tuned HPs: run Optuna TPE search per variant, then evaluate best config
     with n_seeds. Activated by passing do_tune=True to run_experiments().

Model variants (13 total)
--------------------------
MLP  |  EmbMLP-{orig,grid}-{plain,sc,sc_af}  |  PAFNet-{plain,sc,sc_af}-{ln,noln}

DefaultHParams targets ~3-5K parameters for datasets with ~10-50K samples.
  MLP(n_features=8, hidden_dim=64, n_layers=2):  8*64 + 64*64 + 64 = 4,672 params
"""

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.embeddings import build_embedding
from models.backbones  import MLP, PAFNet, EmbeddingMLP
from experiments.trainer import train


# ---------------------------------------------------------------------------
# Default hyperparameters  (no-HPO baseline)
# ---------------------------------------------------------------------------

@dataclass
class DefaultHParams:
    # Architecture — ~3-5K params for 8-feature datasets
    hidden_dim   : int   = 64
    n_layers     : int   = 2
    k            : int   = 16     # frequencies for Periodic / PAF layers
    dropout      : float = 0.1
    sigma_init   : float = 1.0    # OriginalPeriodic init std only

    # Training
    lr           : float = 1e-3
    weight_decay : float = 1e-5
    n_epochs     : int   = 100
    patience     : int   = 16
    batch_size   : int   = 256

    # Infra
    seed         : int   = 42
    n_seeds      : int   = 3


# ---------------------------------------------------------------------------
# Model factory  (fixed-HP path)
# ---------------------------------------------------------------------------

def _build_model(
    name: str,
    n_features: int,
    n_classes: int,
    task_type: str,
    hp: DefaultHParams,
) -> nn.Module:
    out_dim = n_classes if task_type == "multiclass" else 1

    if name == "MLP":
        return MLP(in_dim=n_features, hidden_dim=hp.hidden_dim,
                   n_layers=hp.n_layers, out_dim=out_dim, dropout=hp.dropout)

    if name.startswith("EmbMLP-"):
        parts    = name.split("-")
        emb_type = parts[1]
        variant  = parts[2]
        emb_name = "original" if emb_type == "orig" else "grid"
        extra    = {} if emb_name == "grid" else {"sigma": hp.sigma_init}
        embedding = build_embedding(name=emb_name, n_features=n_features,
                                    k=hp.k, variant=variant, **extra)
        return EmbeddingMLP(embedding=embedding, n_features=n_features,
                            hidden_dim=hp.hidden_dim, n_layers=hp.n_layers,
                            out_dim=out_dim, dropout=hp.dropout)

    if name.startswith("PAFNet-"):
        parts         = name.split("-")
        variant       = parts[1]
        use_layernorm = parts[2] == "ln"
        return PAFNet(in_dim=n_features, k=hp.k, n_layers=hp.n_layers,
                      out_dim=out_dim, paf_variant=variant,
                      use_layernorm=use_layernorm, dropout=hp.dropout)

    raise ValueError(f"Unknown model name: {name!r}")


ALL_VARIANTS: list[str] = [
    "MLP",
    "EmbMLP-orig-plain",
    "EmbMLP-orig-sc",
    "EmbMLP-orig-sc_af",
    "EmbMLP-grid-plain",
    "EmbMLP-grid-sc",
    "EmbMLP-grid-sc_af",
    "PAFNet-plain-ln",
    "PAFNet-sc-ln",
    "PAFNet-sc_af-ln",
    "PAFNet-plain-noln",
    "PAFNet-sc-noln",
    "PAFNet-sc_af-noln",
]


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(
    model_name: str,
    dataset: dict,
    hp: DefaultHParams,
    results_dir: Path,
    device: torch.device,
    seed: int,
    verbose: bool = True,
    # optional: override with tuned flat-hp dict
    tuned_hp: dict | None = None,
    tuned_lr: float | None = None,
    tuned_wd: float | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)

    if tuned_hp is not None:
        # Tuned path: build from flat hp dict
        from experiments.tuner import build_model_from_hp
        out_dim = dataset["n_classes"] if dataset["task_type"] == "multiclass" else 1
        model   = build_model_from_hp(model_name, dict(tuned_hp), dataset["n_features"], out_dim)
        lr = tuned_lr or hp.lr
        wd = tuned_wd or hp.weight_decay
    else:
        model = _build_model(model_name, dataset["n_features"],
                             dataset["n_classes"], dataset["task_type"], hp)
        lr = hp.lr
        wd = hp.weight_decay

    n_params = sum(p.numel() for p in model.parameters())
    ckpt = results_dir / dataset["dataset_name"] / model_name / f"seed{seed}.pt"

    result = train(model=model, dataset=dataset, lr=lr, weight_decay=wd,
                   n_epochs=hp.n_epochs, patience=hp.patience,
                   device=device, checkpoint_path=ckpt, verbose=verbose)

    result["model_name"]   = model_name
    result["dataset_name"] = dataset["dataset_name"]
    result["seed"]         = seed
    result["n_params"]     = n_params
    return result


# ---------------------------------------------------------------------------
# Full experiment grid
# ---------------------------------------------------------------------------

def run_experiments(
    dataset_names: list[str],
    data_root:     str | Path = "data",
    results_dir:   str | Path = "results",
    variants:      list[str] | None = None,
    hp:            DefaultHParams | None = None,
    device:        torch.device | None = None,
    verbose:       bool = True,
    # Tuning options
    do_tune:       bool = False,
    n_trials:      int  = 50,
) -> list[dict[str, Any]]:
    """
    Run all variants on all datasets.

    do_tune=True  : run Optuna TPE search before evaluating each variant.
                    Best HP is used for final n_seeds evaluation.
    do_tune=False : use fixed DefaultHParams (fast, no HPO).
    """
    from data.loader import load_dataset

    if hp       is None: hp       = DefaultHParams()
    if variants is None: variants = ALL_VARIANTS
    if device   is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for ds_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")

        dataset = load_dataset(ds_name, data_root=data_root, batch_size=hp.batch_size)
        print(f"  n_features={dataset['n_features']}  "
              f"task={dataset['task_type']}  n_classes={dataset['n_classes']}")

        # --- optional tuning pass ---
        tuning_results: dict[str, dict] = {}
        if do_tune:
            from experiments.tuner import tune
            for model_name in variants:
                print(f"\n  [HPO] Tuning {model_name} ({n_trials} trials)...")
                t = tune(model_name=model_name, dataset=dataset,
                         n_trials=n_trials, device=device, seed=hp.seed,
                         n_epochs=hp.n_epochs, patience=hp.patience)
                tuning_results[model_name] = t
                print(f"  [HPO] best_val_R2={t['best_val_r2']:.4f}  "
                      f"hp={t['best_hp']}  lr={t['best_lr']:.2e}")

        # --- evaluation pass ---
        for model_name in variants:
            seed_results = []
            tuned = tuning_results.get(model_name)

            for seed_idx in range(hp.n_seeds):
                actual_seed = hp.seed + seed_idx
                print(f"\n  [{model_name}]  seed={actual_seed}"
                      f"{'  (tuned)' if tuned else ''}")
                r = run_one(
                    model_name=model_name,
                    dataset=dataset,
                    hp=hp,
                    results_dir=results_dir,
                    device=device,
                    seed=actual_seed,
                    verbose=verbose,
                    tuned_hp=tuned["best_hp"]  if tuned else None,
                    tuned_lr=tuned["best_lr"]  if tuned else None,
                    tuned_wd=tuned["best_wd"]  if tuned else None,
                )
                seed_results.append(r)
                all_results.append(r)

            vals  = [r["best_val_metric"] for r in seed_results]
            tests = [r["test_metric"]     for r in seed_results]
            std_v = statistics.stdev(vals)  if len(vals)  > 1 else 0.0
            std_t = statistics.stdev(tests) if len(tests) > 1 else 0.0
            print(f"  → {model_name:30s} "
                  f"val_R2={statistics.mean(vals):.4f}±{std_v:.4f}  "
                  f"test_R2={statistics.mean(tests):.4f}±{std_t:.4f}  "
                  f"n_params={seed_results[0]['n_params']:,}")

    out_path = results_dir / "all_results.json"
    slim = [{k: v for k, v in r.items() if k != "history"} for r in all_results]
    out_path.write_text(json.dumps(slim, indent=2))
    print(f"\nResults saved to {out_path}")

    return all_results