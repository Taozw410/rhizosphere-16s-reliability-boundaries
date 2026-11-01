#!/usr/bin/env python3
"""Audit structural calibration of the completed core formal grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p4_calibrate_mvp import TOLERANCES, distance_profile
from p4_core_sim_common import load_core_context
from p4_sim_common import compute_summary_components


NULL_STATISTICS = [
    "prevalence_distribution",
    "rank_abundance",
    "sample_depth",
    "within_between_variance_ratio",
    "bray_curtis_distribution",
    "aitchison_distribution",
    "zero_fraction",
    "network_topology",
]
INVARIANTS = [
    "sample_depth",
    "within_between_variance_ratio",
    "bray_curtis_distribution",
]
TARGETS = [
    "prevalence_distribution",
    "rank_abundance",
    "zero_fraction",
    "aitchison_distribution",
    "network_topology",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="outputs/core_formal_grid/results",
    )
    parser.add_argument(
        "--targets",
        default=(
            "outputs/core_occupancy/"
            "core_scenario_targets.json"
        ),
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
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
    with Path(args.targets).open("r", encoding="utf-8") as handle:
        targets = json.load(handle)["scenarios"]
    rows = []
    for path in sorted(root.glob("*/*/seed_*/counts.npz")):
        stage = path.parts[-4]
        scenario = path.parts[-3]
        seed = int(path.parts[-2].replace("seed_", ""))
        archive = np.load(path, allow_pickle=True)
        counts = archive["counts"].astype(np.int64)
        components = compute_summary_components(
            counts,
            context.meta,
            context.feature_names,
        )
        distances = distance_profile(real_components, components)
        rows.append(
            {
                "stage": stage,
                "scenario": scenario,
                "seed": seed,
                **{name: distances[name] for name in NULL_STATISTICS},
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "core_formal_calibration_distances.csv", index=False)
    summary_rows = []
    for (stage, scenario), group in frame.groupby(["stage", "scenario"]):
        if scenario == "core_null":
            failed = [
                name
                for name in NULL_STATISTICS
                if group[name].mean() > TOLERANCES[name]
            ]
        else:
            failed = [
                f"invariant:{name}"
                for name in INVARIANTS
                if group[name].mean() > TOLERANCES[name]
            ]
            for name in TARGETS:
                lower = targets[scenario][name]["envelope_low"]
                upper = targets[scenario][name]["envelope_high"]
                if not lower <= group[name].mean() <= upper:
                    failed.append(f"target:{name}")
        summary_rows.append(
            {
                "stage": stage,
                "scenario": scenario,
                "seeds": len(group),
                "pass": not failed,
                "failed_checks": ";".join(failed),
                **{
                    f"mean_{name}": group[name].mean()
                    for name in NULL_STATISTICS
                },
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "core_formal_calibration_summary.csv", index=False)
    print(summary[["stage", "scenario", "seeds", "pass", "failed_checks"]].to_string(index=False))


if __name__ == "__main__":
    main()
