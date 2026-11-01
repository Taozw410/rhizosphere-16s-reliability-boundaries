#!/usr/bin/env python3
"""Build the frozen Paper 4 core/occupancy formal seed grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from p4_run_core_smoke import SCENARIOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="outputs/core_formal_grid",
    )
    parser.add_argument("--primary-seeds", type=int, default=10)
    parser.add_argument("--extension-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=6200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(
        range(
            args.seed_start,
            args.seed_start + args.primary_seeds + args.extension_seeds,
        )
    )
    rows = []
    for scenario in SCENARIOS:
        for index, seed in enumerate(seeds):
            rows.append(
                {
                    "stage": (
                        "primary"
                        if index < args.primary_seeds
                        else "extension"
                    ),
                    "scenario": scenario,
                    "seed": seed,
                }
            )
    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "core_seed_grid.csv", index=False)
    config = {
        "version": "core_formal_grid_v1",
        "scenarios": SCENARIOS,
        "seed_start": args.seed_start,
        "primary_seeds": seeds[: args.primary_seeds],
        "extension_seeds": seeds[args.primary_seeds :],
        "n_datasets": len(grid),
        "methods": [
            "sample_prevalence",
            "project_balanced_prevalence",
            "beta_binomial_eb",
            "hierarchical_occupancy",
        ],
        "method_thresholds_file": (
            "outputs/core_occupancy/"
            "core_method_thresholds.json"
        ),
        "scenario_targets_file": (
            "outputs/core_occupancy/"
            "core_scenario_targets.json"
        ),
    }
    (out_dir / "core_formal_grid_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(grid)} core datasets to {out_dir}")


if __name__ == "__main__":
    main()
