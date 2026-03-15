"""
Training loop and evaluation.

Metric: R² score for regression (higher = better).
Stored and compared as-is (no sign flip needed).
"""

import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Loss / metric
# ---------------------------------------------------------------------------

def _get_loss_fn(task_type: str) -> nn.Module:
    if task_type == "regression":
        return nn.MSELoss()
    if task_type == "binclass":
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def _r2_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """R² = 1 - SS_res / SS_tot"""
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _compute_metric(
    logits: torch.Tensor,
    y: torch.Tensor,
    task_type: str,
) -> float:
    with torch.no_grad():
        if task_type == "regression":
            return _r2_score(logits.squeeze(), y)
        if task_type == "binclass":
            preds = (logits.squeeze() > 0).long()
            return (preds == y.long()).float().mean().item()
        preds = logits.argmax(dim=-1)
        return (preds == y).float().mean().item()


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    task_type: str,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (avg_loss, metric)."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_y:      list[torch.Tensor] = []

    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)

            if task_type in ("regression", "binclass"):
                loss = loss_fn(logits.squeeze(), y_batch)
            else:
                loss = loss_fn(logits, y_batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            all_logits.append(logits.detach().cpu())
            all_y.append(y_batch.detach().cpu())

    logits_all = torch.cat(all_logits)
    y_all      = torch.cat(all_y)
    metric     = _compute_metric(logits_all, y_all, task_type)
    avg_loss   = total_loss / len(y_all)

    return avg_loss, metric


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    dataset: dict,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    n_epochs: int = 100,
    patience: int = 16,
    device: torch.device | None = None,
    checkpoint_path: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    task_type = dataset["task_type"]
    loss_fn   = _get_loss_fn(task_type)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loaders = dataset["loaders"]
    best_val_metric = -math.inf
    best_epoch      = 0
    no_improve      = 0
    best_state      = None
    history: list[dict] = []

    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_metric = _run_epoch(
            model, loaders["train"], loss_fn, task_type, optimizer, device
        )
        val_loss, val_metric = _run_epoch(
            model, loaders["val"], loss_fn, task_type, None, device
        )

        history.append({
            "epoch":        epoch,
            "train_loss":   tr_loss,
            "val_loss":     val_loss,
            "train_metric": tr_metric,
            "val_metric":   val_metric,
        })

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_epoch      = epoch
            no_improve      = 0
            best_state      = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"  [{epoch:>4}] "
                f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_R2={val_metric:.4f}"
                f"{'  *' if no_improve == 0 else ''}"
            )

        if no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_metric = _run_epoch(
        model, loaders["test"], loss_fn, task_type, None, device
    )
    train_time = time.time() - t0

    if verbose:
        print(
            f"  → best_epoch={best_epoch}  "
            f"best_val_R2={best_val_metric:.4f}  "
            f"test_R2={test_metric:.4f}  "
            f"({train_time:.1f}s)"
        )

    return {
        "best_val_metric": best_val_metric,
        "test_metric":     test_metric,
        "best_epoch":      best_epoch,
        "history":         history,
        "train_time_s":    train_time,
    }