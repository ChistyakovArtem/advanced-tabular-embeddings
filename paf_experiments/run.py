#!/usr/bin/env python3
"""
PAF Experiments — main entry point.

Usage
-----
# Run on California Housing only (default)
python run.py

# Run on multiple datasets
python run.py --datasets california house

# Run specific variants only
python run.py --variants MLP PAFNet-sc-grid EmbMLP-grid-sc_af

# Quick smoke-test (1 seed, 5 epochs)
python run.py --smoke

# Analyse existing results
python run.py --analyse
"""

import argparse
import sys
from pathlib import Path

# Make sure project root is on the path when running from any working dir
sys.path.insert(0, str(Path(__file__).parent))

import torch

from experiments.runner import run_experiments, DefaultHParams, ALL_VARIANTS
from results.analysis   import main as analyse_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PAF embedding experiments")

    p.add_argument(
        "--datasets", nargs="+", default=["california"],
        help="Dataset names (folders inside --data_root). Default: california",
    )
    p.add_argument(
        "--data_root", default="data",
        help="Root directory for datasets. Default: data/",
    )
    p.add_argument(
        "--results_dir", default="results",
        help="Where to save checkpoints and result JSON. Default: results/",
    )
    p.add_argument(
        "--variants", nargs="+", default=None,
        help=(
            "Which model variants to run. "
            f"Default: all {len(ALL_VARIANTS)} variants.\n"
            f"Available: {', '.join(ALL_VARIANTS)}"
        ),
    )
    p.add_argument(
        "--n_seeds", type=int, default=3,
        help="Number of random seeds per experiment. Default: 3",
    )
    p.add_argument(
        "--n_epochs", type=int, default=100,
        help="Max training epochs. Default: 100",
    )
    p.add_argument(
        "--patience", type=int, default=16,
        help="Early stopping patience. Default: 16",
    )
    p.add_argument(
        "--hidden_dim", type=int, default=256,
        help="Hidden dimension for MLP / EmbeddingMLP. Default: 256",
    )
    p.add_argument(
        "--k", type=int, default=48,
        help="Number of frequencies k for Periodic / PAF. Default: 48",
    )
    p.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size. Default: 256",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate. Default: 1e-3",
    )
    p.add_argument(
        "--device", default=None,
        help="PyTorch device string (e.g. 'cuda', 'cpu'). Default: auto",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: 1 seed, 5 epochs, MLP + 2 variants",
    )
    p.add_argument(
        "--analyse", action="store_true",
        help="Only analyse existing results (no training)",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-epoch output",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Analyse-only mode ----
    if args.analyse:
        analyse_results(str(Path(args.results_dir) / "all_results.json"))
        return

    # ---- Device ----
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Hyperparameters ----
    hp = DefaultHParams(
        hidden_dim=args.hidden_dim,
        k=args.k,
        lr=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        patience=args.patience,
        n_seeds=args.n_seeds,
    )

    # ---- Smoke-test overrides ----
    variants = args.variants
    if args.smoke:
        hp.n_epochs = 5
        hp.patience = 5
        hp.n_seeds  = 1
        variants    = ["MLP", "EmbMLP-grid-sc_af", "PAFNet-sc-grid"]
        print("Smoke test mode: 1 seed, 5 epochs, 3 variants")

    # ---- Run ----
    results = run_experiments(
        dataset_names=args.datasets,
        data_root=args.data_root,
        results_dir=args.results_dir,
        variants=variants,
        hp=hp,
        device=device,
        verbose=not args.quiet,
    )

    # ---- Print summary table ----
    print("\n" + "="*72)
    print("  SUMMARY")
    print("="*72)
    from results.analysis import aggregate, print_table
    agg = aggregate(results)
    print_table(agg)


if __name__ == "__main__":
    main()
