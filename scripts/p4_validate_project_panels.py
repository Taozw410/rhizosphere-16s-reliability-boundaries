#!/usr/bin/env python3
"""Validate data-generating structure across three disjoint 15-project panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from p4_calibrate_mvp import TOLERANCES, distance_profile
from p4_core_sim_common import load_core_context
from p4_sim_common import (
    compute_summary_components,
    simulate_template_multinomial,
)
from p4_sim_perturbed import simulate_perturbed_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-meta",
        default="out_analysis/sample_meta.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/project_panel_validation",
    )
    parser.add_argument("--seeds", type=int, default=3)
    return parser.parse_args()


def make_panels(sample_meta: pd.DataFrame) -> dict[str, list[str]]:
    project_crop = (
        sample_meta[["project", "crop"]]
        .drop_duplicates()
        .sort_values(["crop", "project"])
    )
    panels = {"panel_1": [], "panel_2": [], "panel_3": []}
    for _, group in project_crop.groupby("crop"):
        projects = group["project"].tolist()
        if len(projects) != 15:
            raise ValueError("Expected 15 projects per crop")
        for index, key in enumerate(panels):
            panels[key].extend(projects[index * 5 : (index + 1) * 5])
    return {key: sorted(value) for key, value in panels.items()}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_meta = pd.read_csv(args.sample_meta)
    panels = make_panels(sample_meta)
    panel_rows = [
        {"panel": panel, "project": project}
        for panel, projects in panels.items()
        for project in projects
    ]
    pd.DataFrame(panel_rows).to_csv(out_dir / "project_panels.csv", index=False)
    rows = []
    for panel, projects in panels.items():
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
            Path(
                "outputs/formal_grid/data/"
                "sparse20_fc2/feature_panel.csv"
            ),
            project_panel_override=projects,
        )
        real = compute_summary_components(
            context.real_counts,
            context.meta,
            context.feature_names,
        )
        global_profile = context.real_counts.sum(axis=0).astype(float)
        global_profile /= global_profile.sum()
        effects = pd.DataFrame(
            columns=["scenario", "feature", "requested_log2_effect"]
        )
        for generator in ["template_multinomial", "correlated_perturbation"]:
            for seed in range(7300, 7300 + args.seeds):
                if generator == "template_multinomial":
                    counts = simulate_template_multinomial(
                        context.real_counts,
                        global_profile,
                        0.5,
                        seed,
                    )
                else:
                    counts = simulate_perturbed_scenario(
                        context,
                        effects,
                        "no_effect",
                        0.03,
                        seed,
                    ).counts
                simulated = compute_summary_components(
                    counts,
                    context.meta,
                    context.feature_names,
                )
                distances = distance_profile(real, simulated)
                rows.append(
                    {
                        "panel": panel,
                        "generator": generator,
                        "seed": seed,
                        **distances,
                        **{
                            f"pass_{name}": (
                                distances[name] <= TOLERANCES[name]
                            )
                            for name in TOLERANCES
                        },
                        "n_pass": sum(
                            distances[name] <= TOLERANCES[name]
                            for name in TOLERANCES
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "panel_calibration_by_seed.csv", index=False)
    summary = (
        frame.groupby(["panel", "generator"])
        .agg(
            seeds=("seed", "nunique"),
            mean_n_pass=("n_pass", "mean"),
            min_n_pass=("n_pass", "min"),
            **{
                f"mean_{name}": (name, "mean")
                for name in TOLERANCES
            },
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "panel_calibration_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
