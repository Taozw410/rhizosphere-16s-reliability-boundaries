#!/usr/bin/env python3
"""Project-panel posterior predictive check using covariate-matched donors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from p4_real_project_holdout import make_panels
from p4_sim_common import compute_summary_components, load_analysis_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-matrix", default="out_analysis/feature_matrix.csv")
    parser.add_argument(
        "--design",
        default="outputs/effect_interface/sample_design.csv",
    )
    parser.add_argument("--sample-meta", default="out_analysis/sample_meta.csv")
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
        default="outputs/real_project_holdout_predictive",
    )
    parser.add_argument("--predictive-draws", type=int, default=100)
    parser.add_argument("--panel-seed", type=int, default=20261028)
    parser.add_argument(
        "--donor-mode",
        choices=["nearest3", "random_training"],
        default="nearest3",
    )
    return parser.parse_args()


def scalar_summary(counts, meta, features) -> dict[str, float]:
    components = compute_summary_components(counts, meta, features)
    return {
        "prevalence_mean": float(np.mean(components["prevalence"])),
        "rank_abundance_log_median": float(
            np.median(np.log10(np.maximum(components["rank_abundance"], 1e-10)))
        ),
        "depth_log_median": float(np.median(np.log1p(components["depth"]))),
        "variance_ratio": float(components["variance_ratio"]),
        "bray_median": float(np.median(components["bray"])),
        "aitchison_median": float(np.median(components["aitchison"])),
        "zero_fraction": float(components["global_zero_fraction"]),
        "network_density_median": float(
            components["network"]["edge_density"].median()
        ),
        "network_centrality_median": float(
            components["network"]["centrality_cv"].median()
        ),
    }


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
    indices = np.array(
        [feature_index[name] for name in panel_features],
        dtype=int,
    )
    counts = data.counts[:, indices]
    valid = counts.sum(axis=1) > 0
    counts = counts[valid]
    meta = meta.loc[valid].reset_index(drop=True)
    meta["feature_depth"] = counts.sum(axis=1)
    study = pd.read_csv(args.study_covariates).set_index("project")
    covariates = study[
        [
            "depth_median",
            "asv_len_mode",
            "n_asv_median",
            "unassigned_pct",
            "genus_pct",
            "family_pct",
        ]
    ].astype(float)
    covariates["depth_median"] = np.log1p(covariates["depth_median"])
    panels = make_panels(sample_meta, seed=args.panel_seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_rows = []
    prediction_rows = []
    for held_out_panel, held_out_projects in panels.items():
        held_out = meta["project"].isin(held_out_projects).to_numpy()
        training = ~held_out
        training_projects = meta.loc[training, "project"].unique().tolist()
        training_by_crop = {
            crop: [
                project
                for project in training_projects
                if crop_map.loc[project] == crop
            ]
            for crop in meta["crop"].dropna().unique()
        }
        train_covariates = covariates.loc[training_projects]
        covariate_mean = train_covariates.mean(axis=0)
        covariate_sd = train_covariates.std(axis=0).replace(0, 1)
        global_profile = counts[training].sum(axis=0).astype(float)
        global_profile /= global_profile.sum()
        real_summary = scalar_summary(
            counts[held_out],
            meta[held_out].reset_index(drop=True),
            panel_features,
        )
        rng = np.random.default_rng(args.panel_seed)
        for draw in range(args.predictive_draws):
            simulated = np.zeros_like(counts[held_out])
            held_rows = np.flatnonzero(held_out)
            for held_project in held_out_projects:
                project_rows = np.flatnonzero(
                    meta["project"].eq(held_project).to_numpy() & held_out
                )
                crop = meta.iloc[project_rows[0]]["crop"]
                candidates = training_by_crop[crop]
                target = (
                    covariates.loc[held_project] - covariate_mean
                ) / covariate_sd
                candidate_values = (
                    covariates.loc[candidates] - covariate_mean
                ) / covariate_sd
                distances = np.sqrt(
                    ((candidate_values - target) ** 2).sum(axis=1)
                )
                if args.donor_mode == "nearest3":
                    nearest = np.argsort(distances)[: min(3, len(candidates))]
                    donor_project = candidates[int(rng.choice(nearest))]
                else:
                    donor_project = str(rng.choice(candidates))
                donor_rows = np.flatnonzero(
                    meta["project"].eq(donor_project).to_numpy() & training
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
            simulated_summary = scalar_summary(
                simulated,
                meta[held_out].reset_index(drop=True),
                panel_features,
            )
            prediction_rows.append(
                {
                    "held_out_panel": held_out_panel,
                    "draw": draw,
                    **simulated_summary,
                }
            )
        predictions = pd.DataFrame(
            [
                row
                for row in prediction_rows
                if row["held_out_panel"] == held_out_panel
            ]
        )
        for metric, real_value in real_summary.items():
            lower, upper = np.percentile(predictions[metric], [2.5, 97.5])
            validation_rows.append(
                {
                    "held_out_panel": held_out_panel,
                    "metric": metric,
                    "real_value": real_value,
                    "predictive_2.5": lower,
                    "predictive_50": np.median(predictions[metric]),
                    "predictive_97.5": upper,
                    "pass": lower <= real_value <= upper,
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(out_dir / "predictive_draws.csv", index=False)
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(out_dir / "predictive_check.csv", index=False)
    summary = (
        validation.groupby("held_out_panel")
        .agg(metrics=("metric", "size"), metrics_pass=("pass", "sum"))
        .reset_index()
    )
    summary.to_csv(out_dir / "predictive_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
