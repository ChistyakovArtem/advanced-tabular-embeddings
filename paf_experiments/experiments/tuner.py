"""
Hyperparameter tuning via Optuna TPE sampler.

Search spaces are taken from:
  - MLP/MLP-PLR: Gorishniy et al. (NeurIPS 2022) Appendix E
    + TabM paper (ICLR 2025) tuning table (the two are consistent)
  - PAFNet: adapted from MLP space with PAF-specific additions

Tuning protocol (matches the papers):
  - TPE sampler
  - 100 trials on smaller datasets, 50 on larger (>100K samples)
  - 1 seed per trial (fast), then re-evaluate best config with n_seeds
  - Early stopping within each trial (patience=16)

Usage
-----
    from experiments.tuner import tune
    best_hp = tune("MLP", dataset, n_trials=50)
    # best_hp is a dict that can be passed to _build_model_from_hp()
"""

import math
from typing import Any

import optuna
import torch
import torch.nn as nn

from experiments.trainer import train


optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Search spaces (per model family)
# ---------------------------------------------------------------------------

def _suggest_mlp(trial: optuna.Trial) -> dict:
    """
    MLP search space from Gorishniy et al. (NeurIPS 2022) Appendix E
    and TabM paper tuning table.

    n_layers  : 1..4  (n_blocks in paper notation)
    hidden_dim: 64, 128, 256, 512, 1024  (d_block / d_layers)
    dropout   : 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
    lr        : log-uniform [1e-5, 1e-2]
    weight_decay: log-uniform [1e-6, 1e-3]
    """
    n_layers   = trial.suggest_int("n_layers", 1, 4)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512, 1024])
    dropout    = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    lr         = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    wd         = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    return dict(n_layers=n_layers, hidden_dim=hidden_dim, dropout=dropout,
                lr=lr, weight_decay=wd)


def _suggest_emb_mlp(trial: optuna.Trial) -> dict:
    """
    MLP-PLR / EmbMLP search space from Gorishniy et al. (NeurIPS 2022).

    Same MLP backbone HPs, plus:
    k     : 8, 16, 32, 48, 64, 96, 128  (n_frequencies)
    sigma : log-uniform [0.001, 100]     (frequency_init_scale, only for orig)
    """
    hp = _suggest_mlp(trial)
    hp["k"]     = trial.suggest_categorical("k", [8, 16, 32, 48, 64, 96, 128])
    hp["sigma"] = trial.suggest_float("sigma", 1e-3, 100.0, log=True)
    return hp


def _suggest_pafnet(trial: optuna.Trial) -> dict:
    """
    PAFNet search space.

    Same backbone HPs as MLP, but hidden_dim is replaced by k
    (frequencies per PAF layer, effective width ≈ 2*k or 2*k+in_dim).
    use_layernorm is also tuned.
    """
    n_layers      = trial.suggest_int("n_layers", 1, 4)
    k             = trial.suggest_categorical("k", [16, 32, 48, 64, 96, 128])
    dropout       = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3])
    lr            = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    wd            = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    use_layernorm = trial.suggest_categorical("use_layernorm", [True, False])
    return dict(n_layers=n_layers, k=k, dropout=dropout,
                lr=lr, weight_decay=wd, use_layernorm=use_layernorm)


SUGGEST_FN = {
    "MLP":          _suggest_mlp,
    "EmbMLP-orig":  _suggest_emb_mlp,
    "EmbMLP-grid":  _suggest_emb_mlp,   # sigma is suggested but ignored for grid
    "PAFNet":       _suggest_pafnet,
}


def _model_family(model_name: str) -> str:
    if model_name == "MLP":
        return "MLP"
    if model_name.startswith("EmbMLP-orig"):
        return "EmbMLP-orig"
    if model_name.startswith("EmbMLP-grid"):
        return "EmbMLP-grid"
    if model_name.startswith("PAFNet"):
        return "PAFNet"
    raise ValueError(f"Unknown model family for: {model_name}")


# ---------------------------------------------------------------------------
# Build model from flat HP dict
# ---------------------------------------------------------------------------

def build_model_from_hp(
    model_name: str,
    hp: dict,
    n_features: int,
    out_dim: int,
) -> nn.Module:
    """
    Construct a model from a flat HP dict as returned by suggest functions.
    Variant info (plain/sc/sc_af, ln/noln) is encoded in model_name.
    """
    from models.embeddings import build_embedding
    from models.backbones   import MLP, PAFNet, EmbeddingMLP

    if model_name == "MLP":
        return MLP(
            in_dim=n_features,
            hidden_dim=hp["hidden_dim"],
            n_layers=hp["n_layers"],
            out_dim=out_dim,
            dropout=hp["dropout"],
        )

    if model_name.startswith("EmbMLP-"):
        parts    = model_name.split("-")   # ["EmbMLP", "orig"|"grid", variant]
        emb_type = parts[1]
        variant  = parts[2]
        emb_name = "original" if emb_type == "orig" else "grid"
        extra    = {} if emb_name == "grid" else {"sigma": hp.get("sigma", 1.0)}
        embedding = build_embedding(
            name=emb_name,
            n_features=n_features,
            k=hp["k"],
            variant=variant,
            **extra,
        )
        return EmbeddingMLP(
            embedding=embedding,
            n_features=n_features,
            hidden_dim=hp["hidden_dim"],
            n_layers=hp["n_layers"],
            out_dim=out_dim,
            dropout=hp["dropout"],
        )

    if model_name.startswith("PAFNet-"):
        parts         = model_name.split("-")   # ["PAFNet", variant, "ln"|"noln"]
        variant       = parts[1]
        # use_layernorm can be overridden by HP dict; fallback to name
        use_layernorm = hp.get("use_layernorm", parts[2] == "ln")
        return PAFNet(
            in_dim=n_features,
            k=hp["k"],
            n_layers=hp["n_layers"],
            out_dim=out_dim,
            paf_variant=variant,
            use_layernorm=use_layernorm,
            dropout=hp["dropout"],
        )

    raise ValueError(f"Unknown model_name: {model_name!r}")


# ---------------------------------------------------------------------------
# Single Optuna trial
# ---------------------------------------------------------------------------

def _objective(
    trial: optuna.Trial,
    model_name: str,
    dataset: dict,
    device: torch.device,
    seed: int,
    n_epochs: int,
    patience: int,
) -> float:
    family = _model_family(model_name)
    hp     = SUGGEST_FN[family](trial)

    # Also suggest lr / weight_decay separately if not already in hp
    # (they are always included in all suggest functions above)
    lr = hp.pop("lr")
    wd = hp.pop("weight_decay")

    torch.manual_seed(seed)

    out_dim = dataset["n_classes"] if dataset["task_type"] == "multiclass" else 1
    model   = build_model_from_hp(model_name, hp, dataset["n_features"], out_dim)

    result = train(
        model=model,
        dataset=dataset,
        lr=lr,
        weight_decay=wd,
        n_epochs=n_epochs,
        patience=patience,
        device=device,
        checkpoint_path=None,
        verbose=False,
    )
    return result["best_val_metric"]   # R² (higher = better)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tune(
    model_name: str,
    dataset: dict,
    n_trials: int = 50,
    device: torch.device | None = None,
    seed: int = 42,
    n_epochs: int = 100,
    patience: int = 16,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Run Optuna TPE hyperparameter search for one model on one dataset.

    Returns
    -------
    dict with keys:
        best_hp      : flat HP dict (no lr/wd — those are separate)
        best_lr      : float
        best_wd      : float
        best_val_r2  : float
        study        : optuna.Study  (for further inspection)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        return _objective(trial, model_name, dataset, device, seed, n_epochs, patience)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress,
    )

    best_params = study.best_params.copy()
    best_lr = best_params.pop("lr")
    best_wd = best_params.pop("weight_decay")

    return {
        "best_hp":     best_params,
        "best_lr":     best_lr,
        "best_wd":     best_wd,
        "best_val_r2": study.best_value,
        "study":       study,
    }


def tune_all(
    model_names: list[str],
    dataset: dict,
    n_trials: int = 50,
    device: torch.device | None = None,
    seed: int = 42,
    n_epochs: int = 100,
    patience: int = 16,
) -> dict[str, dict]:
    """
    Tune all model_names on one dataset.
    Returns dict: {model_name: tune() result}
    """
    results = {}
    for name in model_names:
        print(f"\n  Tuning {name} ({n_trials} trials)...")
        results[name] = tune(
            model_name=name,
            dataset=dataset,
            n_trials=n_trials,
            device=device,
            seed=seed,
            n_epochs=n_epochs,
            patience=patience,
            show_progress=True,
        )
        print(f"  → best_val_R2={results[name]['best_val_r2']:.4f}  "
              f"hp={results[name]['best_hp']}  "
              f"lr={results[name]['best_lr']:.2e}  "
              f"wd={results[name]['best_wd']:.2e}")
    return results