#!/usr/bin/env python3
"""Repeated stratified project-panel holdout validation."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from p4_calibrate_mvp import TOLERANCES, distance_profile
from p4_sim_common import (
    compute_summary_components,
    load_analysis_data,
    network_project_stats,
    pairwise_aitchison,
    pairwise_bray,
    relative_abundance,
)


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
        default="outputs/real_project_holdout_cv",
    )
    parser.add_argument("--partitions", type=int, default=10)
    parser.add_argument("--predictive-draws", type=int, default=30)
    parser.add_argument("--perturbation-sd", type=float, default=0.03)
    return parser.parse_args()


def stratified_panels(project_crop: pd.Series, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    by_crop = {}
    for project, crop in project_crop.items():
        by_crop.setdefault(crop, []).append(project)
    for projects in by_crop.values():
        rng.shuffle(projects)
    panels = [[], [], []]
    for projects in by_crop.values():
        for panel_index in range(3):
            panels[panel_index].extend(
                projects[panel_index * 5 : (panel_index + 1) * 5]
            )
    return [sorted(project for project in panel) for panel in panels]


def scalar_from_components(components) -> dict[str, float]:
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


def fast_panel_summary(
    counts: np.ndarray,
    meta: pd.DataFrame,
    features: list[str],
    rng: np.random.Generator,
) -> dict[str, float]:
    relative = relative_abundance(counts)
    prevalence = (counts > 0).mean(axis=0)
    mean_relative = relative.mean(axis=0)
    depth = counts.sum(axis=1)
    projects = meta["project"].to_numpy()
    log_relative = np.log1p(relative * 1e6)
    ratios = []
    for column in range(counts.shape[1]):
        if prevalence[column] < 0.1:
            continue
        within = []
        means = []
        for project in np.unique(projects):
            rows = projects == project
            if rows.sum() >= 3:
                within.append(np.var(log_relative[rows, column], ddof=1))
                means.append(log_relative[rows, column].mean())
        if len(within) >= 3 and len(means) >= 3:
            between = np.var(means, ddof=1)
            if between > 0:
                ratios.append(np.mean(within) / between)
    sample_count = min(120, len(meta))
    sample_rows = np.sort(rng.choice(len(meta), sample_count, replace=False))
    subset_relative = relative[sample_rows]
    bray = pairwise_bray(subset_relative)
    aitchison = pairwise_aitchison(
        subset_relative,
        depths=depth[sample_rows],
    )
    network = network_project_stats(counts, meta, features)
    return {
        "prevalence_mean": float(prevalence.mean()),
        "rank_abundance_log_median": float(
            np.median(np.log10(np.maximum(mean_relative, 1e-10)))
        ),
        "depth_log_median": float(np.median(np.log1p(depth))),
        "variance_ratio": float(np.median(ratios)) if ratios else np.nan,
        "bray_median": float(np.median(bray)),
        "aitchison_median": float(np.median(aitchison)),
        "zero_fraction": float((counts == 0).mean()),
        "network_density_median": float(network["edge_density"].median()),
        "network_centrality_median": float(network["centrality_cv"].median()),
    }


def correlation_matrix(counts: np.ndarray) -> np.ndarray:
    depth = counts.sum(axis=1, keepdims=True)
    probability = (counts + 0.5) / (depth + 0.5 * counts.shape[1])
    logged = np.log(probability)
    logged -= logged.mean(axis=1, keepdims=True)
    correlation = np.corrcoef(rankdata(logged, axis=0), rowvar=False)
    correlation = np.nan_to_num(correlation)
    correlation = 0.5 * correlation + 0.5 * np.eye(correlation.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    correlation = (eigenvectors * eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.diag(correlation))
    return correlation / scale[:, None] / scale[None, :]


def main() -> None:
    args = parse_args()
    data = load_analysis_data(Path(args.feature_matrix), min_depth=1000)
    design = pd.read_csv(args.design, dtype={"sample": str}).set_index("sample")
    meta = design.loc[data.meta["sample"]].reset_index()
    sample_meta = pd.read_csv(args.sample_meta)
    project_crop = (
        sample_meta[["project", "crop"]]
        .drop_duplicates("project")
        .set_index("project")["crop"]
    )
    meta["crop"] = meta["project"].map(project_crop)
    study = pd.read_csv(args.study_covariates).set_index("project")
    covariate_columns = [
        "depth_median",
        "asv_len_mode",
        "n_asv_median",
        "unassigned_pct",
        "genus_pct",
        "family_pct",
    ]
    covariates = study[covariate_columns].astype(float)
    covariates["depth_median"] = np.log1p(covariates["depth_median"])
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows = []
    distance_rows = []
    for partition in range(args.partitions):
        panel_seed = 8100 + partition
        panels = stratified_panels(project_crop, panel_seed)
        for fold, held_out_projects in enumerate(panels):
            held_out = meta["project"].isin(held_out_projects).to_numpy()
            training = ~held_out
            training_projects = sorted(meta.loc[training, "project"].unique())
            training_by_crop = {
                crop: [
                    project
                    for project in training_projects
                    if project_crop.loc[project] == crop
                ]
                for crop in project_crop.unique()
            }
            train_covariates = covariates.loc[training_projects]
            covariate_mean = train_covariates.mean(axis=0)
            covariate_sd = train_covariates.std(axis=0).replace(0, 1)
            global_profile = counts[training].sum(axis=0).astype(float)
            global_profile /= global_profile.sum()
            training_correlation = correlation_matrix(counts[training])
            real_components = compute_summary_components(
                counts[held_out],
                meta[held_out].reset_index(drop=True),
                panel_features,
            )
            real_summary = scalar_from_components(real_components)
            real_fast_summary = fast_panel_summary(
                counts[held_out],
                meta[held_out].reset_index(drop=True),
                panel_features,
                np.random.default_rng(panel_seed * 100000),
            )
            for generator in [
                "template_multinomial",
                "correlated_perturbation",
            ]:
                simulated_summaries = []
                distances = []
                for draw in range(args.predictive_draws):
                    rng = np.random.default_rng(
                        panel_seed * 10000 + draw
                    )
                    simulated = np.zeros_like(counts[held_out])
                    held_rows = np.flatnonzero(held_out)
                    for held_project in held_out_projects:
                        project_rows = np.flatnonzero(
                            meta["project"].eq(held_project).to_numpy()
                            & held_out
                        )
                        crop = project_crop.loc[held_project]
                        candidates = training_by_crop[crop]
                        target = (
                            covariates.loc[held_project] - covariate_mean
                        ) / covariate_sd
                        candidate_values = (
                            covariates.loc[candidates] - covariate_mean
                        ) / covariate_sd
                        donor_distances = np.sqrt(
                            ((candidate_values - target) ** 2).sum(axis=1)
                        )
                        nearest = np.argsort(donor_distances)[:3]
                        donor_project = candidates[int(rng.choice(nearest))]
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
                            if generator == "correlated_perturbation":
                                latent = rng.multivariate_normal(
                                    np.zeros(len(panel_features)),
                                    training_correlation,
                                )
                                probability = probability * np.exp(
                                    args.perturbation_sd * latent
                                )
                                probability /= probability.sum()
                            simulated[
                                np.flatnonzero(held_rows == global_row)[0]
                            ] = rng.multinomial(depth, probability)
                    simulated_summaries.append(
                        fast_panel_summary(
                            simulated,
                            meta[held_out].reset_index(drop=True),
                            panel_features,
                            np.random.default_rng(
                                panel_seed * 100000 + draw + 1
                            ),
                        )
                    )
                    if draw == 0:
                        simulated_components = compute_summary_components(
                            simulated,
                            meta[held_out].reset_index(drop=True),
                            panel_features,
                        )
                        distances.append(
                            distance_profile(
                                real_components,
                                simulated_components,
                            )
                        )
                simulation_frame = pd.DataFrame(simulated_summaries)
                distance_frame = pd.DataFrame(distances)
                for metric, real_value in real_fast_summary.items():
                    lower, upper = np.percentile(
                        simulation_frame[metric],
                        [2.5, 97.5],
                    )
                    coverage_rows.append(
                        {
                            "partition": partition,
                            "fold": fold,
                            "generator": generator,
                            "metric": metric,
                            "real": real_value,
                            "predictive_2.5": lower,
                            "predictive_97.5": upper,
                            "covered": lower <= real_value <= upper,
                        }
                    )
                distance_rows.append(
                    {
                        "partition": partition,
                        "fold": fold,
                        "generator": generator,
                        **{
                            f"mean_distance_{name}": distance_frame[name].mean()
                            for name in TOLERANCES
                        },
                        "mean_n_pass": np.mean(
                            [
                                sum(
                                    row[name] <= TOLERANCES[name]
                                    for name in TOLERANCES
                                )
                                for row in distances
                            ]
                        ),
                    }
                )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(out_dir / "holdout_coverage_by_fold.csv", index=False)
    coverage_summary = (
        coverage.groupby(["generator", "metric"])
        .agg(folds=("covered", "size"), coverage=("covered", "mean"))
        .reset_index()
    )
    coverage_summary.to_csv(
        out_dir / "holdout_coverage_summary.csv",
        index=False,
    )
    distances = pd.DataFrame(distance_rows)
    distances.to_csv(out_dir / "holdout_distances_by_fold.csv", index=False)
    distance_summary = (
        distances.groupby("generator")
        .agg(
            folds=("fold", "size"),
            mean_n_pass=("mean_n_pass", "mean"),
            **{
                f"mean_distance_{name}": (f"mean_distance_{name}", "mean")
                for name in TOLERANCES
            },
        )
        .reset_index()
    )
    distance_summary.to_csv(
        out_dir / "holdout_distance_summary.csv",
        index=False,
    )
    print(coverage_summary.to_string(index=False))
    print(distance_summary.to_string(index=False))


if __name__ == "__main__":
    main()
