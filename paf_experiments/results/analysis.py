"""
Results analysis. Metric is R² (higher = better) for all tasks.
"""

import json
import statistics
from pathlib import Path
from collections import defaultdict

from experiments.runner import ALL_VARIANTS


def aggregate(results: list[dict]) -> dict[str, dict[str, dict]]:
    groups: dict[tuple, list] = defaultdict(list)
    for r in results:
        groups[(r["dataset_name"], r["model_name"])].append(r)

    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for (ds, model), runs in groups.items():
        vals  = [r["best_val_metric"] for r in runs]
        tests = [r["test_metric"]     for r in runs]
        out[ds][model] = {
            "val_mean":  statistics.mean(vals),
            "val_std":   statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "test_mean": statistics.mean(tests),
            "test_std":  statistics.stdev(tests) if len(tests) > 1 else 0.0,
            "n_params":  runs[0].get("n_params", 0),
            "n_seeds":   len(runs),
        }
    return dict(out)


def print_table(agg: dict[str, dict[str, dict]]) -> None:
    for ds_name, models in agg.items():
        print(f"\n{'='*72}")
        print(f"  Dataset: {ds_name}  (R² ↑)")
        print(f"{'='*72}")
        print(f"  {'Model':<30}  {'Val R²':>16}  {'Test R²':>16}  {'ΔTest':>8}  {'#params':>10}")
        print(f"  {'-'*30}  {'-'*16}  {'-'*16}  {'-'*8}  {'-'*10}")

        ordered  = [v for v in ALL_VARIANTS if v in models]
        ordered += [v for v in sorted(models) if v not in ordered]

        baseline_test = None
        for model_name in ordered:
            if model_name not in models:
                continue
            m = models[model_name]

            val_s  = f"{m['val_mean']:.4f} ±{m['val_std']:.4f}"
            test_s = f"{m['test_mean']:.4f} ±{m['test_std']:.4f}"

            delta_str = ""
            if model_name == "MLP":
                baseline_test = m["test_mean"]
            elif baseline_test is not None:
                delta = m["test_mean"] - baseline_test
                arrow = "▲" if delta > 0 else "▼"
                delta_str = f"{arrow}{abs(delta):.4f}"

            print(
                f"  {model_name:<30}  {val_s:>16}  {test_s:>16}  "
                f"{delta_str:>8}  {m['n_params']:>10,}"
            )
    print()


def main(results_path: str = "results/all_results.json") -> None:
    p = Path(results_path)
    if not p.exists():
        print(f"Results file not found: {p}")
        return
    results = json.loads(p.read_text())
    print_table(aggregate(results))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "results/all_results.json")