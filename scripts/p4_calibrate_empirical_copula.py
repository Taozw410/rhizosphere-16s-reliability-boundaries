#!/usr/bin/env python3
"""Calibrate the empirical-margin Gaussian-copula DA generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p4_calibrate_mvp import TOLERANCES, distance_profile, normalized_score
from p4_core_sim_common import load_core_context
from p4_sim_common import compute_summary_components
from p4_sim_empirical_copula import simulate_empirical_copula_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-matrix",
        default="out_analysis/feature_matrix.csv",
    )
    parser.add_argument(
        "--design",
        default="outputs/effect_interface/sample_design.csv",
    )
    parser.add_argument(
        "--formal-config",
        default=(
            "outputs/formal_grid/"
            "formal_grid_config.json"
        ),
    )
    parser.add_argument(
        "--panel",
        default=(
            "outputs/formal_grid/data/"
            "sparse20_fc2/feature_panel.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/da_generator_copula",
    )
    parser.add_argument("--seeds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    context = load_core_context(
        Path(args.feature_matrix),
        Path(args.design),
        Path(args.formal_config),
        Path(args.panel),
    )
    real_components = compute_summary_components(
        context.real_counts,
        context.meta,
        context.feature_names,
    )
    effects = pd.DataFrame(
        columns=["scenario", "feature", "requested_log2_effect"]
    )
    rows = []
    settings = [
        (0.10, 2.5),
        (0.20, 2.5),
        (0.20, 3.0),
        (0.20, 4.0),
        (0.35, 3.0),
        (0.35, 4.0),
        (0.50, 3.0),
        (0.50, 4.0),
    ]
    for shrinkage, correlation_scale in settings:
        distances = []
        for seed in range(7000, 7000 + args.seeds):
            simulation = simulate_empirical_copula_scenario(
                context,
                effects,
                "no_effect",
                shrinkage,
                correlation_scale,
                seed,
            )
            components = compute_summary_components(
                simulation.counts,
                context.meta,
                context.feature_names,
            )
            distances.append(distance_profile(real_components, components))
        means = {
            key: float(np.mean([row[key] for row in distances]))
            for key in distances[0]
        }
        rows.append(
            {
                "correlation_shrinkage": shrinkage,
                "correlation_scale": correlation_scale,
                "normalized_score": normalized_score(means),
                **{
                    f"distance_{key}": value
                    for key, value in means.items()
                },
                **{
                    f"pass_{key}": means[key] <= TOLERANCES[key]
                    for key in TOLERANCES
                },
                "n_pass": sum(
                    means[key] <= TOLERANCES[key]
                    for key in TOLERANCES
                ),
            }
        )
    search = pd.DataFrame(rows).sort_values(
        ["n_pass", "normalized_score"],
        ascending=[False, True],
    )
    search.to_csv(out_dir / "calibration_search.csv", index=False)
    print(search.to_string(index=False))
    (out_dir / "calibration_status.json").write_text(
        json.dumps(
            {
                "generator": "empirical-margin-gaussian-copula-v1",
                "best": search.iloc[0].to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
