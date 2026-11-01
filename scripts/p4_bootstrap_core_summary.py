#!/usr/bin/env python3
"""Create deterministic bootstrap intervals for the core-method summary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
METRICS = PAPER / "outputs/core_formal_grid/results/core_metrics_all.csv"
OUTPUT = PAPER / "figures/source_data/Fig3_method_bootstrap.csv"

METHOD_MAP = {
    "sample_prevalence": "Sample prevalence",
    "project_balanced_prevalence": "Project-balanced",
    "beta_binomial_eb": "Beta-binomial EB",
    "hierarchical_occupancy": "Hierarchical occupancy",
}
METHOD_ORDER = [
    "Sample prevalence",
    "Project-balanced",
    "Beta-binomial EB",
    "Hierarchical occupancy",
]


def main() -> None:
    metrics = pd.read_csv(METRICS)
    metrics = metrics[metrics["scenario"].ne("core_null")].copy()
    metrics["method_label"] = metrics["method_id"].map(METHOD_MAP)
    rng = np.random.default_rng(20261029)

    rows = []
    for method in METHOD_ORDER:
        values = metrics.loc[
            metrics["method_label"].eq(method),
            "f1",
        ].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        indices = rng.integers(0, values.size, size=(5000, values.size))
        bootstrap = values[indices].mean(axis=1)
        rows.append(
            {
                "method_label": method,
                "f1_mean": float(values.mean()),
                "f1_ci_low": float(np.quantile(bootstrap, 0.025)),
                "f1_ci_high": float(np.quantile(bootstrap, 0.975)),
                "n_metrics": int(values.size),
                "bootstrap_draws": int(bootstrap.size),
                "bootstrap_seed": 20261029,
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(OUTPUT)


if __name__ == "__main__":
    main()
