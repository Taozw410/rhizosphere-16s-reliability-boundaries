#!/usr/bin/env python3
"""Build a cross-generator DA validation grid using correlated perturbation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p4_core_sim_common import load_core_context
from p4_sim_perturbed import simulate_perturbed_scenario


SCENARIOS = [
    "no_effect",
    "sparse20_fc1_19",
    "sparse20_fc1_41",
    "sparse20_fc2",
    "dense40_fc1_41",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-data",
        default="outputs/formal_grid/data",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/formal_grid_generator2",
    )
    parser.add_argument("--seed-start", type=int, default=7200)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--perturbation-sd", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formal_data = Path(args.formal_data)
    out_dir = Path(args.out_dir)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        effects = pd.read_csv(formal_data / scenario / "effects.csv")
        context = load_core_context(
            Path("out_analysis/feature_matrix.csv"),
            Path(
                "outputs/effect_interface/"
                "sample_design.csv"
            ),
            Path(
                "outputs/formal_grid/"
                "formal_grid_config.json"
            ),
            formal_data / scenario / "feature_panel.csv",
        )
        scenario_dir = data_dir / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            simulation = simulate_perturbed_scenario(
                context,
                effects,
                scenario,
                args.perturbation_sd,
                seed,
            )
            run_dir = scenario_dir / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                run_dir / "counts.npz",
                counts=simulation.counts.astype(np.int32),
                feature_names=np.array(context.feature_names, dtype=str),
                sample=np.array(context.meta["sample"], dtype=str),
                project=np.array(context.meta["project"], dtype=str),
            )
            context.meta.to_csv(run_dir / "metadata.csv", index=False)
            simulation.truth.to_csv(run_dir / "truth.csv", index=False)
            rows.append(
                {
                    "stage": "primary",
                    "scenario": scenario,
                    "seed": seed,
                    "n_targets": int(simulation.truth["is_target"].sum()),
                    "absolute_log2_effect": (
                        0.0
                        if scenario == "no_effect"
                        else float(
                            simulation.truth.loc[
                                simulation.truth["is_target"],
                                "requested_log2_effect",
                            ].abs().median()
                        )
                    ),
                    "n_projects": context.meta["project"].nunique(),
                    "n_samples": len(context.meta),
                    "n_features": len(context.feature_names),
                    "counts_file": str((run_dir / "counts.npz").resolve()),
                    "truth_file": str((run_dir / "truth.csv").resolve()),
                    "metadata_file": str(
                        (run_dir / "metadata.csv").resolve()
                    ),
                }
            )
    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "seed_grid.csv", index=False)
    config = {
        "version": "generator2_perturbed_v1",
        "generator": "correlated-template-perturbation-v1",
        "generator_status": (
            "semi-parametric second family; not a fully independent "
            "ecological generator"
        ),
        "perturbation_sd": args.perturbation_sd,
        "seed_start": args.seed_start,
        "seeds": list(
            range(args.seed_start, args.seed_start + args.seeds)
        ),
        "scenarios": SCENARIOS,
    }
    (out_dir / "generator_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(grid)} generator-2 datasets to {out_dir}")


if __name__ == "__main__":
    main()
