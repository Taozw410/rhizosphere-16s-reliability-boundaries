#!/usr/bin/env python3
"""Train and validate the TARE-v1 project-panel transport model.

The outer holdout excludes all samples from 15 target projects. Study metadata
and observed sample depths are available for prediction; the held-out
community counts are used only for evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

PROJECT_SCRIPTS = (
    Path(__file__).resolve().parents[2] / "scripts"
)
sys.path.insert(0, str(PROJECT_SCRIPTS))

from p4_calibrate_mvp import TOLERANCES, distance_profile
from p4_sim_common import compute_summary_components, load_analysis_data


SUMMARY_NAMES = list(TOLERANCES)
SCALAR_METRICS = {
    "prevalence_distribution": "prevalence_mean",
    "rank_abundance": "rank_abundance_log_median",
    "sample_depth": "depth_log_median",
    "within_between_variance_ratio": "variance_ratio",
    "bray_curtis_distribution": "bray_median",
    "aitchison_distribution": "aitchison_median",
    "zero_fraction": "zero_fraction",
    "network_topology": "network_topology_scalar",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-matrix", default="data/tare_inputs/feature_matrix.csv")
    parser.add_argument(
        "--design",
        default="data/tare_inputs/sample_design.csv",
    )
    parser.add_argument("--sample-meta", default="data/tare_inputs/sample_meta.csv")
    parser.add_argument(
        "--study-covariates",
        default="data/tare_inputs/study_covariates.csv",
    )
    parser.add_argument(
        "--panel",
        default=(
            "data/tare_inputs/feature_panel.csv"
        ),
    )
    parser.add_argument("--out-dir", default="tare/artifacts/tare_v1")
    parser.add_argument("--partitions", type=int, default=10)
    parser.add_argument("--predictive-draws", type=int, default=100)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--top-donors", type=int, default=5)
    parser.add_argument(
        "--temperature-grid",
        type=float,
        nargs="+",
        default=[0.35, 0.70, 1.40],
    )
    parser.add_argument(
        "--covariance-shrinkage-grid",
        type=float,
        nargs="+",
        default=[0.20, 0.50],
    )
    return parser.parse_args()


@dataclass
class ProjectBank:
    profiles: dict[str, np.ndarray]
    sample_residuals: dict[str, np.ndarray]
    clr: dict[str, np.ndarray]


@dataclass
class TareModel:
    scaler: StandardScaler
    ridge: RidgeCV
    bank: ProjectBank
    train_x_raw: pd.DataFrame
    train_x_scaled: np.ndarray
    train_projects: list[str]
    train_crop: pd.Series
    residual_covariance: np.ndarray
    bandwidth: float
    temperature: float
    covariance_shrinkage: float


def stratified_panels(project_crop: pd.Series, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    by_crop: dict[str, list[str]] = {}
    for project, crop in project_crop.items():
        by_crop.setdefault(str(crop), []).append(str(project))
    for projects in by_crop.values():
        rng.shuffle(projects)
    panels: list[list[str]] = [[], [], []]
    for projects in by_crop.values():
        if len(projects) != 15:
            raise ValueError(
                f"Expected 15 projects per crop, found {len(projects)}"
            )
        for panel_index in range(3):
            panels[panel_index].extend(
                projects[panel_index * 5 : (panel_index + 1) * 5]
            )
    return [sorted(panel) for panel in panels]


def grouped_folds(projects: list[str], n_folds: int, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    ordered = list(projects)
    rng.shuffle(ordered)
    return [sorted(ordered[index::n_folds]) for index in range(n_folds)]


def study_feature_table(
    study_covariates: pd.DataFrame,
    project_crop: pd.Series,
) -> pd.DataFrame:
    columns = [
        "n_samples",
        "depth_median",
        "asv_len_mode",
        "n_asv_median",
        "unassigned_pct",
        "genus_pct",
        "family_pct",
    ]
    features = study_covariates.loc[:, columns].astype(float).copy()
    features["log_n_samples"] = np.log1p(features["n_samples"])
    features["log_depth_median"] = np.log1p(features["depth_median"])
    features["log_n_asv_median"] = np.log1p(features["n_asv_median"])
    features = features.drop(columns=["n_samples", "depth_median", "n_asv_median"])
    features.index.name = "project"
    crop = project_crop.reindex(features.index).astype(str)
    crop_dummies = pd.get_dummies(crop, prefix="crop", dtype=float)
    return pd.concat([features, crop_dummies], axis=1)


def clr_profiles(counts: np.ndarray) -> np.ndarray:
    probability = (counts + 0.5) / (
        counts.sum(axis=1, keepdims=True) + 0.5 * counts.shape[1]
    )
    logged = np.log(probability)
    return logged - logged.mean(axis=1, keepdims=True)


def build_project_bank(
    counts: np.ndarray,
    meta: pd.DataFrame,
    projects: list[str],
) -> ProjectBank:
    total = counts.sum(axis=0).astype(float)
    global_profile = total / max(total.sum(), 1.0)
    profiles: dict[str, np.ndarray] = {}
    sample_residuals: dict[str, np.ndarray] = {}
    clr_centroids: dict[str, np.ndarray] = {}
    for project in projects:
        rows = np.flatnonzero(meta["project"].eq(project).to_numpy())
        project_total = counts[rows].sum(axis=0).astype(float)
        prior = max(0.05 * project_total.sum(), 10.0)
        profile = (project_total + prior * global_profile) / (
            project_total.sum() + prior
        )
        profile /= profile.sum()
        profiles[project] = profile
        project_clr = np.log(np.maximum(profile, 1e-12))
        project_clr -= project_clr.mean()
        clr_centroids[project] = project_clr
        sample_clr = clr_profiles(counts[rows])
        sample_residuals[project] = sample_clr - project_clr[None, :]
    return ProjectBank(
        profiles=profiles,
        sample_residuals=sample_residuals,
        clr=clr_centroids,
    )


def fit_model(
    counts: np.ndarray,
    meta: pd.DataFrame,
    feature_table: pd.DataFrame,
    project_crop: pd.Series,
    training_projects: list[str],
    temperature: float,
    covariance_shrinkage: float,
) -> TareModel:
    training_projects = sorted(training_projects)
    bank = build_project_bank(counts, meta, training_projects)
    x_raw = feature_table.loc[training_projects]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw)
    y = np.vstack([bank.clr[project] for project in training_projects])
    ridge = RidgeCV(
        alphas=np.logspace(-2, 3, 16),
        cv=None,
        fit_intercept=True,
        gcv_mode="auto",
    )
    ridge.fit(x_scaled, y)
    predicted = ridge.predict(x_scaled)
    residuals = y - predicted
    empirical_cov = np.cov(residuals, rowvar=False, ddof=1)
    diagonal = np.diag(np.maximum(np.diag(empirical_cov), 1e-6))
    covariance = (
        (1.0 - covariance_shrinkage) * empirical_cov
        + covariance_shrinkage * diagonal
    )
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors * np.maximum(eigenvalues, 1e-8)) @ eigenvectors.T
    pairwise = cdist(x_scaled, x_scaled, metric="euclidean")
    positive = pairwise[np.triu_indices_from(pairwise, k=1)]
    bandwidth = max(float(np.median(positive)), 1e-3)
    return TareModel(
        scaler=scaler,
        ridge=ridge,
        bank=bank,
        train_x_raw=x_raw,
        train_x_scaled=x_scaled,
        train_projects=training_projects,
        train_crop=project_crop.loc[training_projects],
        residual_covariance=covariance,
        bandwidth=bandwidth,
        temperature=temperature,
        covariance_shrinkage=covariance_shrinkage,
    )


def donor_weights(
    model: TareModel,
    target_row: pd.DataFrame,
    target_crop: str,
    top_donors: int,
) -> np.ndarray:
    target_scaled = model.scaler.transform(target_row)
    distances = np.sqrt(
        ((model.train_x_scaled - target_scaled) ** 2).sum(axis=1)
    )
    crop_match = (model.train_crop.to_numpy() == target_crop).astype(float)
    same_crop = np.flatnonzero(crop_match > 0)
    if len(same_crop) >= top_donors:
        candidates = same_crop
    else:
        candidates = np.arange(len(model.train_projects))
    candidate_distances = distances[candidates]
    keep = np.argsort(candidate_distances)[:top_donors]
    selected = candidates[keep]
    scaled = candidate_distances[keep] / (
        model.bandwidth * model.temperature
    )
    weights = np.exp(-0.5 * scaled**2)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(selected), dtype=float)
    return weights / weights.sum()


def support_score(
    model: TareModel,
    target_row: pd.DataFrame,
    target_crop: str,
    top_donors: int,
) -> float:
    target_scaled = model.scaler.transform(target_row)
    distances = np.sqrt(
        ((model.train_x_scaled - target_scaled) ** 2).sum(axis=1)
    )
    same = model.train_crop.to_numpy() == target_crop
    values = distances[same] if same.any() else distances
    nearest = np.sort(values)[:top_donors]
    normalized = nearest / model.bandwidth
    return float(np.mean(np.exp(-0.5 * normalized**2)))


def simulate_tare(
    model: TareModel,
    counts: np.ndarray,
    meta: pd.DataFrame,
    held_projects: list[str],
    feature_table: pd.DataFrame,
    project_crop: pd.Series,
    seed: int,
    correlated: bool,
    top_donors: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    simulated = np.zeros_like(counts)
    for project in sorted(held_projects):
        rows = np.flatnonzero(meta["project"].eq(project).to_numpy())
        target = feature_table.loc[[project]]
        crop = str(project_crop.loc[project])
        donors = donor_weights(model, target, crop, top_donors)
        target_scaled = model.scaler.transform(target)
        distances = np.sqrt(
            ((model.train_x_scaled - target_scaled) ** 2).sum(axis=1)
        )
        donor_projects = [
            model.train_projects[index]
            for index in np.argsort(distances)[:top_donors]
        ]
        same_crop_projects = [
            project_name
            for project_name in model.train_projects
            if model.train_crop.loc[project_name] == crop
        ]
        if len(same_crop_projects) >= top_donors:
            candidate_indices = np.array(
                [
                    model.train_projects.index(project_name)
                    for project_name in same_crop_projects
                ],
                dtype=int,
            )
            keep = candidate_indices[
                np.argsort(distances[candidate_indices])[:top_donors]
            ]
            donor_projects = [model.train_projects[index] for index in keep]
        predicted_clr = model.ridge.predict(target_scaled)[0]
        covariance_factor = (
            np.linalg.cholesky(
                model.residual_covariance
                * model.covariance_shrinkage
                + np.eye(model.residual_covariance.shape[0]) * 1e-8
            )
            if correlated
            else None
        )
        for global_row in rows:
            donor = str(rng.choice(donor_projects, p=donors))
            residual_bank = model.bank.sample_residuals[donor]
            residual = residual_bank[rng.integers(len(residual_bank))]
            logits = predicted_clr + residual
            if covariance_factor is not None:
                latent = covariance_factor @ rng.standard_normal(
                    model.residual_covariance.shape[0]
                )
                logits = logits + latent
            logits -= logits.max()
            probability = np.exp(logits)
            probability /= probability.sum()
            depth = int(counts[global_row].sum())
            simulated[global_row] = rng.multinomial(depth, probability)
    return simulated


def simulate_nearest_three(
    counts: np.ndarray,
    meta: pd.DataFrame,
    feature_table: pd.DataFrame,
    held_projects: list[str],
    training_projects: list[str],
    project_crop: pd.Series,
    seed: int,
    top_donors: int = 3,
) -> np.ndarray:
    training_projects = sorted(training_projects)
    train_x = feature_table.loc[training_projects].to_numpy(dtype=float)
    x_mean = train_x.mean(axis=0)
    x_sd = train_x.std(axis=0)
    x_sd[x_sd == 0] = 1
    train_scaled = (train_x - x_mean) / x_sd
    global_profile = counts[
        meta["project"].isin(training_projects).to_numpy()
    ].sum(axis=0).astype(float)
    global_profile /= global_profile.sum()
    rng = np.random.default_rng(seed)
    simulated = np.zeros_like(counts)
    for project in sorted(held_projects):
        rows = np.flatnonzero(meta["project"].eq(project).to_numpy())
        target = (
            feature_table.loc[[project]].to_numpy(dtype=float) - x_mean
        ) / x_sd
        crop = str(project_crop.loc[project])
        candidates = [
            name
            for name in training_projects
            if str(project_crop.loc[name]) == crop
        ]
        candidate_index = np.array(
            [training_projects.index(name) for name in candidates], dtype=int
        )
        distances = np.sqrt(
            ((train_scaled[candidate_index] - target) ** 2).sum(axis=1)
        )
        nearest_index = candidate_index[
            np.argsort(distances)[: min(top_donors, len(candidate_index))]
        ]
        donor_projects = [training_projects[index] for index in nearest_index]
        for global_row in rows:
            donor_project = str(rng.choice(donor_projects))
            donor_rows = np.flatnonzero(
                meta["project"].eq(donor_project).to_numpy()
            )
            donor = donor_rows[rng.integers(len(donor_rows))]
            depth = int(counts[global_row].sum())
            probability = (
                counts[donor] + 0.5 * global_profile
            ) / (counts[donor].sum() + 0.5)
            simulated[global_row] = rng.multinomial(depth, probability)
    return simulated


def scalar_summary(
    components: dict[str, object],
) -> dict[str, float]:
    network = components["network"]
    if not isinstance(network, pd.DataFrame):
        raise TypeError("network components must be a DataFrame")
    density = float(network["edge_density"].median())
    centrality = float(network["centrality_cv"].median())
    return {
        "prevalence_mean": float(np.mean(components["prevalence"])),
        "rank_abundance_log_median": float(
            np.median(np.log10(np.maximum(components["rank_abundance"], 1e-10)))
        ),
        "depth_log_median": float(
            np.median(np.log1p(components["depth"]))
        ),
        "variance_ratio": float(components["variance_ratio"]),
        "bray_median": float(np.median(components["bray"])),
        "aitchison_median": float(np.median(components["aitchison"])),
        "zero_fraction": float(components["global_zero_fraction"]),
        "network_topology_scalar": 0.5 * density + 0.5 * centrality,
    }


def fixed_pass(
    real_components: dict[str, object],
    simulated_components: dict[str, object],
) -> tuple[int, dict[str, float], dict[str, bool]]:
    distances = distance_profile(real_components, simulated_components)
    passes = {
        name: distances[name] <= TOLERANCES[name] for name in SUMMARY_NAMES
    }
    return int(sum(passes.values())), distances, passes


def select_hyperparameters(
    counts: np.ndarray,
    meta: pd.DataFrame,
    feature_table: pd.DataFrame,
    project_crop: pd.Series,
    training_projects: list[str],
    temperatures: list[float],
    covariance_values: list[float],
    inner_folds: int,
    top_donors: int,
    seed: int,
) -> tuple[float, float, pd.DataFrame]:
    folds = grouped_folds(training_projects, inner_folds, seed)
    rows = []
    for temperature in temperatures:
        for covariance_shrinkage in covariance_values:
            fold_scores = []
            for fold_index, validation_projects in enumerate(folds):
                inner_training = sorted(
                    set(training_projects).difference(validation_projects)
                )
                model = fit_model(
                    counts,
                    meta,
                    feature_table,
                    project_crop,
                    inner_training,
                    temperature,
                    covariance_shrinkage,
                )
                held_rows = meta["project"].isin(validation_projects).to_numpy()
                real_components = compute_summary_components(
                    counts[held_rows],
                    meta.loc[held_rows].reset_index(drop=True),
                    feature_names,
                )
                simulated = simulate_tare(
                    model,
                    counts[held_rows],
                    meta.loc[held_rows].reset_index(drop=True),
                    validation_projects,
                    feature_table,
                    project_crop,
                    seed + 1000 * fold_index,
                    correlated=False,
                    top_donors=top_donors,
                )
                simulated_components = compute_summary_components(
                    simulated,
                    meta.loc[held_rows].reset_index(drop=True),
                    feature_names,
                )
                distances = distance_profile(
                    real_components, simulated_components
                )
                normalized = np.mean(
                    [
                        distances[name] / TOLERANCES[name]
                        for name in SUMMARY_NAMES
                    ]
                )
                fold_scores.append(float(normalized))
            rows.append(
                {
                    "temperature": temperature,
                    "covariance_shrinkage": covariance_shrinkage,
                    "inner_normalized_distance": float(np.mean(fold_scores)),
                }
            )
    table = pd.DataFrame(rows).sort_values(
        ["inner_normalized_distance", "temperature", "covariance_shrinkage"]
    )
    best = table.iloc[0]
    return (
        float(best["temperature"]),
        float(best["covariance_shrinkage"]),
        table.reset_index(drop=True),
    )


def recommendation(pass_probability: float, support: float) -> str:
    if pass_probability >= 0.80 and support >= 0.25:
        return "ALLOW"
    if pass_probability >= 0.50:
        return "CONDITIONAL"
    return "DENY"


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = labels.sum()
    negatives = (~labels).sum()
    if positives == 0 or negatives == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average_rank
        start = end
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def main() -> None:
    global feature_names
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_analysis_data(Path(args.feature_matrix), min_depth=1000)
    design = pd.read_csv(args.design, dtype={"sample": str}).set_index("sample")
    meta = design.loc[data.meta["sample"]].reset_index()
    sample_meta = pd.read_csv(args.sample_meta)
    project_crop = (
        sample_meta[["project", "crop"]]
        .drop_duplicates("project")
        .set_index("project")["crop"]
    )
    study = pd.read_csv(args.study_covariates).set_index("project")
    feature_table = study_feature_table(study, project_crop)
    panel_features = pd.read_csv(args.panel)["feature"].tolist()
    feature_index = {
        name: index for index, name in enumerate(data.feature_names)
    }
    indices = np.array(
        [feature_index[name] for name in panel_features], dtype=int
    )
    counts = data.counts[:, indices]
    valid = counts.sum(axis=1) > 0
    counts = counts[valid]
    meta = meta.loc[valid].reset_index(drop=True)
    meta["feature_depth"] = counts.sum(axis=1)
    feature_names = panel_features

    selection_rows = []
    fold_rows = []
    metric_rows = []
    all_projects = sorted(project_crop.index)
    for partition in range(args.partitions):
        panels = stratified_panels(project_crop, 9100 + partition)
        for fold, held_projects in enumerate(panels):
            training_projects = sorted(set(all_projects).difference(held_projects))
            best_temperature, best_shrinkage, inner_table = select_hyperparameters(
                counts,
                meta,
                feature_table,
                project_crop,
                training_projects,
                args.temperature_grid,
                args.covariance_shrinkage_grid,
                args.inner_folds,
                args.top_donors,
                seed=9200 + 10 * partition + fold,
            )
            inner_table.insert(0, "fold", fold)
            inner_table.insert(0, "partition", partition)
            selection_rows.append(inner_table)
            model = fit_model(
                counts,
                meta,
                feature_table,
                project_crop,
                training_projects,
                best_temperature,
                best_shrinkage,
            )
            held_rows = meta["project"].isin(held_projects).to_numpy()
            held_counts = counts[held_rows]
            held_meta = meta.loc[held_rows].reset_index(drop=True)
            real_components = compute_summary_components(
                held_counts, held_meta, feature_names
            )
            real_scalar = scalar_summary(real_components)
            supports = {
                project: support_score(
                    model,
                    feature_table.loc[[project]],
                    str(project_crop.loc[project]),
                    args.top_donors,
                )
                for project in held_projects
            }
            support = float(np.mean(list(supports.values())))
            generators = {
                "hierarchical_ridge": False,
                "hierarchical_ridge_correlated": True,
                "nearest_three_donor": None,
            }
            for generator, correlated in generators.items():
                simulated_scalars = []
                distances = []
                for draw_index in range(args.predictive_draws):
                    draw_seed = 930000 + 10000 * partition + 100 * fold + draw_index
                    if generator == "nearest_three_donor":
                        simulated_full = simulate_nearest_three(
                            counts,
                            meta,
                            feature_table,
                            held_projects,
                            training_projects,
                            project_crop,
                            draw_seed,
                        )
                        simulated = simulated_full[held_rows]
                    else:
                        simulated = simulate_tare(
                            model,
                            held_counts,
                            held_meta,
                            held_projects,
                            feature_table,
                            project_crop,
                            draw_seed,
                            correlated=bool(correlated),
                            top_donors=args.top_donors,
                        )
                    components = compute_summary_components(
                        simulated, held_meta, feature_names
                    )
                    simulated_scalars.append(scalar_summary(components))
                    distance = distance_profile(real_components, components)
                    distances.append(distance)
                    if draw_index == 0:
                        n_pass = sum(
                            distance[name] <= TOLERANCES[name]
                            for name in SUMMARY_NAMES
                        )
                scalar_frame = pd.DataFrame(simulated_scalars)
                distance_frame = pd.DataFrame(distances)
                fold_rows.append(
                    {
                        "partition": partition,
                        "fold": fold,
                        "generator": generator,
                        "support": support,
                        "temperature": best_temperature,
                        "covariance_shrinkage": best_shrinkage,
                        "n_pass": n_pass,
                        "mean_n_pass": float(
                            np.mean(
                                [
                                    sum(
                                        row[name] <= TOLERANCES[name]
                                        for name in SUMMARY_NAMES
                                    )
                                    for row in distances
                                ]
                            )
                        ),
                        **{
                            f"mean_distance_{name}": float(
                                distance_frame[name].mean()
                            )
                            for name in SUMMARY_NAMES
                        },
                    }
                )
                for summary in SUMMARY_NAMES:
                    scalar_name = SCALAR_METRICS[summary]
                    lower, median, upper = np.quantile(
                        scalar_frame[scalar_name],
                        [0.025, 0.5, 0.975],
                    )
                    predictive_tolerance = TOLERANCES[summary]
                    pass_probability = float(
                        np.mean(
                            np.abs(
                                scalar_frame[scalar_name].to_numpy() - median
                            )
                            <= predictive_tolerance
                        )
                    )
                    actual_pass = bool(
                        distances[0][summary] <= TOLERANCES[summary]
                    )
                    actual_coverage = bool(
                        lower <= real_scalar[scalar_name] <= upper
                    )
                    metric_rows.append(
                        {
                            "partition": partition,
                            "fold": fold,
                            "generator": generator,
                            "summary": summary,
                            "real": real_scalar[scalar_name],
                            "predictive_2.5": lower,
                            "predictive_50": median,
                            "predictive_97.5": upper,
                            "interval_width": upper - lower,
                            "tolerance": predictive_tolerance,
                            "covered": actual_coverage,
                            "pass_probability": pass_probability,
                            "actual_pass": actual_pass,
                            "support": support,
                            "decision": recommendation(pass_probability, support),
                        }
                    )
            print(
                f"partition={partition} fold={fold} temp={best_temperature:.2f} "
                f"shrink={best_shrinkage:.2f} support={support:.3f}"
            )

    selection = pd.concat(selection_rows, ignore_index=True)
    selection.to_csv(out_dir / "inner_hyperparameter_selection.csv", index=False)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out_dir / "tare_fold_metrics.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "tare_predictive_metrics.csv", index=False)

    fold_summary = (
        folds.groupby("generator")
        .agg(
            folds=("fold", "size"),
            mean_n_pass=("mean_n_pass", "mean"),
            sd_n_pass=("mean_n_pass", "std"),
            mean_support=("support", "mean"),
        )
        .reset_index()
    )
    fold_summary.to_csv(out_dir / "tare_generator_summary.csv", index=False)
    coverage = (
        metrics.groupby(["generator", "summary"])
        .agg(
            folds=("covered", "size"),
            coverage=("covered", "mean"),
            mean_interval_width=("interval_width", "mean"),
            pass_auroc=(
                "actual_pass",
                lambda values: np.nan,
            ),
        )
        .reset_index()
    )
    for index, row in coverage.iterrows():
        subset = metrics[
            metrics["generator"].eq(row["generator"])
            & metrics["summary"].eq(row["summary"])
        ]
        coverage.loc[index, "pass_auroc"] = auc_score(
            subset["actual_pass"].to_numpy(dtype=bool),
            subset["pass_probability"].to_numpy(dtype=float),
        )
    coverage.to_csv(out_dir / "tare_coverage_summary.csv", index=False)

    allow = metrics[metrics["decision"].eq("ALLOW")]
    allow_summary = pd.DataFrame(
        [
            {
                "decisions": int(len(allow)),
                "decision_rate": float(len(allow) / len(metrics)),
                "allow_precision": float(allow["actual_pass"].mean())
                if len(allow)
                else math.nan,
                "allow_actual_pass_count": int(allow["actual_pass"].sum())
                if len(allow)
                else 0,
            }
        ]
    )
    allow_summary.to_csv(out_dir / "tare_allow_summary.csv", index=False)

    manifest = {
        "model": "TARE-v1 conformalized hierarchical transport model",
        "partitions": args.partitions,
        "predictive_draws": args.predictive_draws,
        "inner_folds": args.inner_folds,
        "top_donors": args.top_donors,
        "temperature_grid": args.temperature_grid,
        "covariance_shrinkage_grid": args.covariance_shrinkage_grid,
        "summaries": SUMMARY_NAMES,
        "tolerances": TOLERANCES,
        "generators": [
            "hierarchical_ridge",
            "hierarchical_ridge_correlated",
            "nearest_three_donor",
        ],
        "leakage_control": (
            "Outer target projects excluded from model fitting; target counts "
            "used only for held-out evaluation."
        ),
    }
    with (out_dir / "tare_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(fold_summary.to_string(index=False))
    print(coverage.to_string(index=False))
    print(allow_summary.to_string(index=False))


if __name__ == "__main__":
    main()
