#!/usr/bin/env python3
"""Leave-panel-out validation for the independent hurdle generator."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from p4_calibrate_mvp import TOLERANCES, distance_profile
from p4_real_project_holdout import make_panels
from p4_sim_common import compute_summary_components, load_analysis_data
from p4_sim_independent import fit_independent_model, simulate_independent_projects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-matrix", default="out_analysis/feature_matrix.csv")
    parser.add_argument(
        "--design",
        default="outputs/effect_interface/sample_design.csv",
    )
    parser.add_argument("--sample-meta", default="out_analysis/sample_meta.csv")
    parser.add_argument(
        "--panel",
        default=(
            "outputs/formal_grid/data/"
            "sparse20_fc2/feature_panel.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/independent_generator_holdout",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--between-shrinkage", type=float, default=0.35)
    parser.add_argument("--within-shrinkage", type=float, default=0.20)
    parser.add_argument("--presence-shrinkage", type=float, default=0.50)
    parser.add_argument("--depth-slope", type=float, default=0.80)
    parser.add_argument("--dirichlet-concentration", type=float, default=100.0)
    parser.add_argument("--project-effect-sd", type=float, default=0.50)
    parser.add_argument("--presence-offset-logit", type=float, default=0.0)
    parser.add_argument("--presence-between-multiplier", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_analysis_data(Path(args.feature_matrix), min_depth=1000)
    design = pd.read_csv(args.design, dtype={"sample": str}).set_index("sample")
    meta = design.loc[data.meta["sample"]].reset_index()
    sample_meta = pd.read_csv(args.sample_meta)
    crop_map = (
        sample_meta[["project", "crop"]]
        .drop_duplicates("project")
        .set_index("project")["crop"]
    )
    meta["crop"] = meta["project"].map(crop_map)
    panel_features = pd.read_csv(args.panel)["feature"].tolist()
    feature_index = {
        name: index for index, name in enumerate(data.feature_names)
    }
    idx = np.array([feature_index[name] for name in panel_features], dtype=int)
    counts = data.counts[:, idx]
    valid = counts.sum(axis=1) > 0
    counts = counts[valid]
    meta = meta.loc[valid].reset_index(drop=True)
    meta["feature_depth"] = counts.sum(axis=1)
    panels = make_panels(sample_meta)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for held_out_panel, held_out_projects in panels.items():
        held_out = meta["project"].isin(held_out_projects).to_numpy()
        training = ~held_out
        model = fit_independent_model(
            counts[training],
            meta[training].reset_index(drop=True),
            panel_features,
            meta.loc[training, "crop"].to_numpy(),
            between_shrinkage=args.between_shrinkage,
            within_shrinkage=args.within_shrinkage,
            presence_shrinkage=args.presence_shrinkage,
        )
        real_components = compute_summary_components(
            counts[held_out],
            meta[held_out].reset_index(drop=True),
            panel_features,
        )
        for seed in range(7500, 7500 + args.seeds):
            simulated = simulate_independent_projects(
                model,
                meta[held_out].reset_index(drop=True),
                meta.loc[held_out, "crop"].to_numpy(),
                seed,
                depth_slope=args.depth_slope,
                dirichlet_concentration=args.dirichlet_concentration,
                project_effect_sd=args.project_effect_sd,
                presence_offset_logit=args.presence_offset_logit,
                presence_between_multiplier=args.presence_between_multiplier,
            )
            simulated_components = compute_summary_components(
                simulated,
                meta[held_out].reset_index(drop=True),
                panel_features,
            )
            distances = distance_profile(real_components, simulated_components)
            rows.append(
                {
                    "held_out_panel": held_out_panel,
                    "seed": seed,
                    **distances,
                    **{
                        f"pass_{name}": distances[name] <= TOLERANCES[name]
                        for name in TOLERANCES
                    },
                    "n_pass": sum(
                        distances[name] <= TOLERANCES[name]
                        for name in TOLERANCES
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "independent_holdout_distances.csv", index=False)
    summary = (
        frame.groupby("held_out_panel")
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
    summary.to_csv(out_dir / "independent_holdout_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
