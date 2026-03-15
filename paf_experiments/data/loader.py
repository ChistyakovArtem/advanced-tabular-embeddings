"""
Data loading utilities.

Dataset layout (TabM format):
    <data_root>/<dataset_name>/
        info.json
        X_num_train.npy, X_num_val.npy, X_num_test.npy   — numerical (may be absent)
        X_cat_train.npy, X_cat_val.npy, X_cat_test.npy   — categorical (may be absent)
        X_bin_train.npy, X_bin_val.npy, X_bin_test.npy   — binary (may be absent)
        Y_train.npy,     Y_val.npy,     Y_test.npy        — targets

Preprocessing pipeline (applied in this order):
    1. Categorical features:
       - If a cat column has <= OHE_MAX_CARDINALITY unique values (fit on train)
         → round to nearest int → one-hot encode
       - Otherwise → treat as numerical (label-encoded integer, passed to quantile)
    2. Binary features: passed as-is (already 0/1)
    3. Numerical features: quantile transform (fit on train)
    4. All processed features are concatenated:
       [qt_transformed(num + high_card_cat) | binary | ohe_cat]
    5. Regression target: standardised on train mean/std

Note: task_type is always "regression" for now (as per experiment design).
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder


OHE_MAX_CARDINALITY = 50   # cat columns with more unique values -> numeric path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_npy(path: Path):
    return np.load(path) if path.exists() else None


def _load_split_arrays(dataset_dir: Path, split: str) -> dict:
    return {
        "num": _load_npy(dataset_dir / f"X_num_{split}.npy"),
        "cat": _load_npy(dataset_dir / f"X_cat_{split}.npy"),
        "bin": _load_npy(dataset_dir / f"X_bin_{split}.npy"),
        "y":   _load_npy(dataset_dir / f"Y_{split}.npy"),
    }


# ---------------------------------------------------------------------------
# Categorical preprocessing
# ---------------------------------------------------------------------------

def _fit_cat_preprocessor(X_cat_train: np.ndarray,
                           ohe_max_cardinality: int = OHE_MAX_CARDINALITY) -> dict:
    """
    Decide per-column: OHE (low cardinality) or numeric (high cardinality).
    Returns a preprocessor dict to be applied consistently across splits.
    """
    n_cols = X_cat_train.shape[1]
    X_int  = np.round(X_cat_train).astype(int)

    ohe_cols:    list[int]       = []
    num_cols:    list[int]       = []
    unique_vals: list[np.ndarray] = []

    for col in range(n_cols):
        uniq = np.unique(X_int[:, col])
        unique_vals.append(uniq)
        if len(uniq) <= ohe_max_cardinality:
            ohe_cols.append(col)
        else:
            num_cols.append(col)

    ohe = None
    if ohe_cols:
        ohe = OneHotEncoder(
            categories=[unique_vals[c] for c in ohe_cols],
            sparse_output=False,
            handle_unknown="ignore",
        )
        ohe.fit(X_int[:, ohe_cols])

    return {"ohe_cols": ohe_cols, "num_cols": num_cols, "ohe": ohe}


def _apply_cat_preprocessor(X_cat: np.ndarray, prep: dict):
    """Returns (ohe_block, numeric_block) — either may be None."""
    X_int     = np.round(X_cat).astype(int)
    ohe_block = None
    num_block = None

    if prep["ohe_cols"] and prep["ohe"] is not None:
        ohe_block = prep["ohe"].transform(
            X_int[:, prep["ohe_cols"]]
        ).astype(np.float32)

    if prep["num_cols"]:
        num_block = X_int[:, prep["num_cols"]].astype(np.float32)

    return ohe_block, num_block


# ---------------------------------------------------------------------------
# Quantile transform
# ---------------------------------------------------------------------------

def _fit_quantile(X_train: np.ndarray,
                  n_quantiles: int = 1000,
                  output_distribution: str = "normal") -> QuantileTransformer:
    qt = QuantileTransformer(
        n_quantiles=min(n_quantiles, X_train.shape[0]),
        output_distribution=output_distribution,
        subsample=int(1e9),
        random_state=0,
    )
    qt.fit(X_train)
    return qt


# ---------------------------------------------------------------------------
# Target normalisation
# ---------------------------------------------------------------------------

def _normalise_target(y_train, y_val, y_test):
    mean = float(y_train.mean())
    std  = float(y_train.std()) + 1e-8
    norm = lambda y: ((y - mean) / std).astype(np.float32)
    return norm(y_train), norm(y_val), norm(y_test), mean, std


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def load_dataset(
    dataset_name: str,
    data_root="data",
    apply_quantile: bool = True,
    ohe_max_cardinality: int = OHE_MAX_CARDINALITY,
    batch_size: int = 256,
    num_workers: int = 0,
) -> dict:
    """
    Load a single dataset and return a dict with:
        loaders      : {"train": DataLoader, "val": DataLoader, "test": DataLoader}
        n_features   : int   (total after preprocessing)
        task_type    : str   always "regression" for now
        n_classes    : int   always 1
        y_stats      : {"mean": float, "std": float}
        dataset_name : str
    """
    data_root   = Path(data_root)
    dataset_dir = data_root / dataset_name

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            f"Expected: {data_root}/<dataset_name>/X_num_train.npy ..."
        )

    # ---- Load raw arrays ----
    splits = {s: _load_split_arrays(dataset_dir, s) for s in ("train", "val", "test")}

    for s in ("train", "val", "test"):
        if splits[s]["y"] is None:
            raise FileNotFoundError(f"Y_{s}.npy not found in {dataset_dir}")

    # ---- Categorical preprocessing ----
    cat_prep = None
    cat_ohe  = {s: None for s in ("train", "val", "test")}
    cat_num  = {s: None for s in ("train", "val", "test")}

    if splits["train"]["cat"] is not None:
        cat_prep = _fit_cat_preprocessor(splits["train"]["cat"], ohe_max_cardinality)
        for s in ("train", "val", "test"):
            if splits[s]["cat"] is not None:
                cat_ohe[s], cat_num[s] = _apply_cat_preprocessor(
                    splits[s]["cat"], cat_prep
                )

    # ---- Collect columns for quantile transform ----
    # num features + high-cardinality cat columns (numeric path)
    def _concat_for_qt(split: str):
        parts = []
        if splits[split]["num"] is not None:
            parts.append(splits[split]["num"].astype(np.float32))
        if cat_num[split] is not None:
            parts.append(cat_num[split])
        return np.concatenate(parts, axis=1) if parts else None

    qt_blocks = {s: _concat_for_qt(s) for s in ("train", "val", "test")}

    if apply_quantile and qt_blocks["train"] is not None:
        qt = _fit_quantile(qt_blocks["train"])
        for s in ("train", "val", "test"):
            if qt_blocks[s] is not None:
                qt_blocks[s] = qt.transform(qt_blocks[s]).astype(np.float32)

    # ---- Assemble final feature matrix ----
    # Order: [qt_transformed | binary | cat_ohe]
    def _build_X(split: str) -> np.ndarray:
        parts = []
        if qt_blocks[split] is not None:
            parts.append(qt_blocks[split])
        if splits[split]["bin"] is not None:
            parts.append(splits[split]["bin"].astype(np.float32))
        if cat_ohe[split] is not None:
            parts.append(cat_ohe[split])
        if not parts:
            raise ValueError(
                f"Dataset {dataset_name}: no features found in split '{split}'"
            )
        return np.concatenate(parts, axis=1)

    X = {s: _build_X(s) for s in ("train", "val", "test")}

    # ---- Target normalisation ----
    y_train, y_val, y_test, y_mean, y_std = _normalise_target(
        splits["train"]["y"].astype(np.float32),
        splits["val"]["y"].astype(np.float32),
        splits["test"]["y"].astype(np.float32),
    )

    # ---- DataLoaders ----
    def _make_loader(X_arr: np.ndarray, y_arr: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(X_arr, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.float32),
        )
        return DataLoader(
            ds, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers,
            pin_memory=True,
        )

    n_features = X["train"].shape[1]

    # ---- Debug print ----
    num_dim = splits["train"]["num"].shape[1] if splits["train"]["num"] is not None else 0
    bin_dim = splits["train"]["bin"].shape[1] if splits["train"]["bin"] is not None else 0
    ohe_dim = cat_ohe["train"].shape[1]       if cat_ohe["train"] is not None       else 0
    cnum_dim = cat_num["train"].shape[1]      if cat_num["train"] is not None       else 0
    print(
        f"  [{dataset_name}] n_features={n_features} "
        f"(num={num_dim}, bin={bin_dim}, cat_ohe={ohe_dim}, cat_num={cnum_dim})"
    )

    return {
        "loaders": {
            "train": _make_loader(X["train"], y_train, shuffle=True),
            "val":   _make_loader(X["val"],   y_val,   shuffle=False),
            "test":  _make_loader(X["test"],  y_test,  shuffle=False),
        },
        "n_features":   n_features,
        "task_type":    "regression",
        "n_classes":    1,
        "y_stats":      {"mean": y_mean, "std": y_std},
        "dataset_name": dataset_name,
    }


def load_datasets(dataset_names: list, data_root="data", **kwargs) -> dict:
    return {
        name: load_dataset(name, data_root=data_root, **kwargs)
        for name in dataset_names
    }