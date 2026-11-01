#!/usr/bin/env python3
"""Leave-project-panel-out validation against real 45-project data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from p4_calibrate_mvp import TOLERANCES, distance_profile
from p4_sim_common import compute_summary_components, load_analysis_data


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
        "--sample-meta",
        default="out_analysis/sample_meta.csv",
    )
    parser.add_argument(
        "--study-covariates",
        default="out_analysis/study_covariates.csv",
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
        default="outputs/real_project_holdout",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--panel-seed", type=int, default=None)
    parser.add_argument("--covariate-match", action="store_true")
    return parser.parse_args()


def make_panels(
    sample_meta: pd.DataFrame,
    seed: int | None = None,
) -> dict[str, list[str]]:
    project_crop = (
        sample_meta[["project", "crop"]]
        .drop_duplicates()
        .sort_values(["crop", "project"])
    )
    panels = {"panel_1": [], "panel_2": [], "panel_3": []}
    rng = np.random.default_rng(seed) if seed is not None else None
    for _, group in project_crop.groupby("crop"):
        projects = group["project"].tolist()
        if rng is not None:
            rng.shuffle(projects)
        for index, key in enumerate(panels):
            panels[key].extend(projects[index * 5 : (index + 1) * 5])
    return {key: sorted(value) for key, value in panels.items()}


def main() -> None:
    args = parse_args()
    data = load_analysis_data(Path(args.feature_matrix), min_depth=1000)
    design = pd.read_csv(args.design, dtype={"sample": str}).set_index("sample")
    design = design.loc[data.meta["sample"]].reset_index()
    sample_meta = pd.read_csv(args.sample_meta)
    study_covariates = pd.read_csv(args.study_covariates).set_index("project")
    crop_map = (
        sample_meta[["project", "crop"]]
        .drop_duplicates("project")
        .set_index("project")["crop"]
    )
    panel_frame = pd.read_csv(args.panel)
    feature_index = {
        feature: index for index, feature in enumerate(data.feature_names)
    }
    feature_indices = np.array(
        [feature_index[feature] for feature in panel_frame["feature"]],
        dtype=int,
    )
    counts = data.counts[:, feature_indices]
    feature_names = panel_frame["feature"].tolist()
    meta = design.copy()
    meta["crop"] = meta["project"].map(crop_map)
    meta["feature_depth"] = counts.sum(axis=1)
    panels = make_panels(sample_meta, seed=args.panel_seed)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for held_out_panel, held_out_projects in panels.items():
        held_out = meta["project"].isin(held_out_projects).to_numpy()
        training = ~held_out
        training_projects = meta.loc[training, "project"].unique().tolist()
        training_projects_by_crop = {
            crop: [
                project
                for project in training_projects
                if crop_map.loc[project] == crop
            ]
            for crop in meta["crop"].dropna().unique()
        }
        global_profile = counts[training].sum(axis=0).astype(float)
        global_profile /= global_profile.sum()
        covariate_columns = [
            "depth_median",
            "asv_len_mode",
            "n_asv_median",
            "unassigned_pct",
            "genus_pct",
            "family_pct",
        ]
        covariates = study_covariates[covariate_columns].astype(float)
        covariates["depth_median"] = np.log1p(covariates["depth_median"])
        train_covariates = covariates.loc[training_projects]
        covariate_mean = train_covariates.mean(axis=0)
        covariate_sd = train_covariates.std(axis=0).replace(0, 1)
        real_components = compute_summary_components(
            counts[held_out],
            meta[held_out].reset_index(drop=True),
            feature_names,
        )
        for seed in range(7400, 7400 + args.seeds):
            rng = np.random.default_rng(seed)
            simulated = np.zeros_like(counts[held_out])
            held_rows = np.flatnonzero(held_out)
            for held_project in held_out_projects:
                project_rows = np.flatnonzero(
                    meta["project"].eq(held_project).to_numpy()
                    & held_out
                )
                crop = meta.iloc[project_rows[0]]["crop"]
                if args.covariate_match:
                    candidates = training_projects_by_crop[crop]
                    target = (
                        covariates.loc[held_project] - covariate_mean
                    ) / covariate_sd
                    candidate_values = (
                        covariates.loc[candidates] - covariate_mean
                    ) / covariate_sd
                    distances = np.sqrt(
                        ((candidate_values - target) ** 2).sum(axis=1)
                    )
                    donor_project = candidates[int(np.argmin(distances))]
                else:
                    donor_project = rng.choice(training_projects_by_crop[crop])
                donor_rows = np.flatnonzero(
                    meta["project"].eq(donor_project).to_numpy()
                    & training
                )
                for global_row in project_rows:
                    donor = rng.choice(donor_rows)
                    depth = int(meta.iloc[global_row]["feature_depth"])
                    probability = (
                        counts[donor] + 0.5 * global_profile
                    ) / (counts[donor].sum() + 0.5)
                    simulated[np.flatnonzero(held_rows == global_row)[0]] = (
                        rng.multinomial(depth, probability)
                    )
            simulated_components = compute_summary_components(
                simulated,
                meta[held_out].reset_index(drop=True),
                feature_names,
            )
            distances = distance_profile(
                real_components,
                simulated_components,
            )
            rows.append(
                {
                    "held_out_panel": held_out_panel,
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
    frame.to_csv(output_dir / "holdout_distances_by_seed.csv", index=False)
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
    summary.to_csv(output_dir / "holdout_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
