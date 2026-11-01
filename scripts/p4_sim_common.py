#!/usr/bin/env python3
"""Shared data loading, summary statistics and simulation for Paper 4."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.special import expit


LABEL_COLUMNS = [
    "project",
    "base",
    "crop",
    "sample",
    "all_reads",
    "bacteria_reads",
    "organelle_reads",
    "genus_reads",
    "family_reads",
    "unassigned_reads",
    "n_asv",
]

STAT_NAMES = [
    "prevalence_distribution",
    "rank_abundance",
    "sample_depth",
    "within_between_variance_ratio",
    "bray_curtis_distribution",
    "aitchison_distribution",
    "zero_fraction",
    "network_topology",
]


@dataclass
class AnalysisData:
    meta: pd.DataFrame
    counts: np.ndarray
    feature_names: list[str]
    project_order: list[str]


def load_analysis_data(feature_matrix: Path, min_depth: int = 1000) -> AnalysisData:
    frame = pd.read_csv(feature_matrix)
    missing = set(LABEL_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")
    feature_names = [column for column in frame.columns if column not in LABEL_COLUMNS]
    if "unassigned" not in feature_names:
        raise ValueError("The unassigned composition bin is required")

    keep = frame["bacteria_reads"].to_numpy(dtype=float) >= min_depth
    frame = frame.loc[keep].reset_index(drop=True)
    counts = frame[feature_names].to_numpy(dtype=np.int64)
    if (counts < 0).any():
        raise ValueError("Negative counts found")
    if (counts.sum(axis=1) == 0).any():
        raise ValueError("Zero-depth sample found after filtering")
    project_order = sorted(frame["project"].unique())
    meta = frame[LABEL_COLUMNS].copy()
    meta["feature_depth"] = counts.sum(axis=1)
    return AnalysisData(
        meta=meta,
        counts=counts,
        feature_names=feature_names,
        project_order=project_order,
    )


def relative_abundance(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    return counts / np.maximum(totals, 1)


def empirical_bayes_profiles(
    counts: np.ndarray,
    projects: np.ndarray,
    prior_strength: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    total = counts.sum(axis=0).astype(float)
    global_profile = total / total.sum()
    profiles: dict[str, np.ndarray] = {}
    for project in sorted(np.unique(projects)):
        project_counts = counts[projects == project].sum(axis=0).astype(float)
        denominator = project_counts.sum() + prior_strength
        profiles[project] = (
            project_counts + prior_strength * global_profile
        ) / denominator
    profile_matrix = np.vstack([profiles[p] for p in sorted(profiles)])
    return global_profile, profile_matrix, profiles


def fit_presence_model(
    counts: np.ndarray,
    meta: pd.DataFrame,
    global_profile: np.ndarray,
    profiles: dict[str, np.ndarray],
) -> dict[str, float]:
    """Fit feature-offset logistic model for presence conditional on abundance/depth."""
    from scipy.optimize import minimize

    projects = meta["project"].to_numpy()
    depths = meta["feature_depth"].to_numpy(dtype=float)
    depth_dev = np.log(depths / np.median(depths))
    prevalence = (counts > 0).mean(axis=0)
    prevalence = np.clip(prevalence, 1e-5, 1 - 1e-5)
    offset = np.log(prevalence / (1 - prevalence))

    profile_matrix = np.vstack(
        [profiles[project] for project in projects]
    )
    abundance_dev = np.log(
        (profile_matrix + 1e-12) / (global_profile[None, :] + 1e-12)
    )
    observed = counts > 0

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, beta_abundance, beta_depth = theta
        eta = (
            offset[None, :]
            + intercept
            + beta_abundance * abundance_dev
            + beta_depth * depth_dev[:, None]
        )
        # Stable Bernoulli negative log-likelihood.
        nll = np.logaddexp(0.0, eta).sum() - (observed * eta).sum()
        residual = expit(eta) - observed
        grad = np.array(
            [
                residual.sum(),
                (residual * abundance_dev).sum(),
                (residual * depth_dev[:, None]).sum(),
            ]
        )
        return float(nll), grad

    result = minimize(
        lambda theta: objective(theta)[0],
        x0=np.array([0.0, 0.5, 0.1]),
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        options={"maxiter": 250, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Presence model failed: {result.message}")
    return {
        "intercept": float(result.x[0]),
        "abundance_slope": float(result.x[1]),
        "depth_slope": float(result.x[2]),
        "negative_log_likelihood": float(result.fun),
    }


def calibrate_presence_intercept(
    meta: pd.DataFrame,
    global_profile: np.ndarray,
    profiles: dict[str, np.ndarray],
    presence_parameters: dict[str, float],
    target_zero_fraction: float,
) -> float:
    """Find an intercept shift matching the target global zero fraction."""
    from scipy.optimize import brentq

    projects = meta["project"].to_numpy()
    depth_dev = np.log(
        meta["feature_depth"].to_numpy(dtype=float)
        / np.median(meta["feature_depth"].to_numpy(dtype=float))
    )
    prevalence = np.clip(presence_parameters["_prevalence"], 1e-5, 1 - 1e-5)
    offset = np.log(prevalence / (1 - prevalence))
    eta_shift_free = np.empty((len(meta), len(global_profile)), dtype=float)
    for project in sorted(np.unique(projects)):
        rows = np.flatnonzero(projects == project)
        abundance_dev = np.log(
            (profiles[project] + 1e-12) / (global_profile + 1e-12)
        )
        eta_shift_free[rows] = (
            offset[None, :]
            + presence_parameters["intercept"]
            + presence_parameters["abundance_slope"] * abundance_dev[None, :]
            + presence_parameters["depth_slope"] * depth_dev[rows, None]
        )
    target_prevalence = 1.0 - target_zero_fraction

    def difference(shift: float) -> float:
        return float(expit(eta_shift_free + shift).mean() - target_prevalence)

    return float(brentq(difference, -20.0, 20.0, maxiter=200))


def simulate_counts(
    meta: pd.DataFrame,
    feature_names: list[str],
    global_profile: np.ndarray,
    profiles: dict[str, np.ndarray],
    presence_parameters: dict[str, float],
    dirichlet_concentration: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    projects = meta["project"].to_numpy()
    depths = meta["feature_depth"].to_numpy(dtype=np.int64)
    depth_dev_all = np.log(depths / np.median(depths))

    # Presence probabilities are anchored to the empirical prevalence encoded in
    # the parameter file. The caller passes it as `_prevalence` in the dict.
    baseline_prevalence = presence_parameters["_prevalence"]
    offset = np.log(baseline_prevalence / (1 - baseline_prevalence))
    simulated = np.zeros((len(meta), len(feature_names)), dtype=np.int64)

    for project in sorted(np.unique(projects)):
        row_idx = np.flatnonzero(projects == project)
        concentration = np.maximum(
            profiles[project] * dirichlet_concentration,
            1e-10,
        )
        sample_profiles = rng.dirichlet(concentration, size=len(row_idx))
        project_abundance_dev = np.log(
            (profiles[project] + 1e-12) / (global_profile + 1e-12)
        )
        eta = (
            offset[None, :]
            + presence_parameters["intercept"]
            + presence_parameters.get("intercept_shift", 0.0)
            + presence_parameters["abundance_slope"]
            * project_abundance_dev[None, :]
            + presence_parameters["depth_slope"] * depth_dev_all[row_idx, None]
        )
        presence = rng.random(eta.shape) < expit(eta)
        # Guarantee at least one observed feature per sample.
        empty = ~presence.any(axis=1)
        if empty.any():
            best = np.argmax(sample_profiles[empty], axis=1)
            presence[np.flatnonzero(empty), best] = True
        probabilities = sample_profiles * presence
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        for local_index, global_index in enumerate(row_idx):
            simulated[global_index] = rng.multinomial(
                depths[global_index], probabilities[local_index]
            )
    return simulated


def simulate_template_multinomial(
    counts: np.ndarray,
    global_profile: np.ndarray,
    sample_prior_reads: float,
    seed: int,
) -> np.ndarray:
    """Parametric bootstrap around each observed sample composition.

    This simulator preserves the observed sample-level co-variation, zero
    structure and project design while adding multinomial sampling noise. It is
    the calibration MVP; the independent hierarchical generator remains a
    separate Phase-2 technical debt.
    """
    rng = np.random.default_rng(seed)
    depths = counts.sum(axis=1)
    simulated = np.zeros_like(counts)
    for index, depth in enumerate(depths):
        probability = (
            counts[index] + sample_prior_reads * global_profile
        ) / (depth + sample_prior_reads)
        simulated[index] = rng.multinomial(depth, probability)
    return simulated


def pairwise_bray(relative: np.ndarray) -> np.ndarray:
    return pdist(relative, metric="cityblock") / 2.0


def clr_transform(
    relative: np.ndarray,
    pseudocount: float | np.ndarray | None = None,
    depths: np.ndarray | None = None,
) -> np.ndarray:
    if depths is not None:
        pseudocount = (0.5 / np.maximum(depths, 1)).reshape(-1, 1)
    if pseudocount is None:
        positive = relative[relative > 0]
        pseudocount = max(float(np.min(positive)) * 0.5, 1e-8)
    transformed = np.log(relative + pseudocount)
    return transformed - transformed.mean(axis=1, keepdims=True)


def pairwise_aitchison(
    relative: np.ndarray, depths: np.ndarray | None = None
) -> np.ndarray:
    return pdist(clr_transform(relative, depths=depths), metric="euclidean")


def network_project_stats(
    counts: np.ndarray,
    meta: pd.DataFrame,
    feature_names: list[str],
    max_features: int = 200,
    prevalence_floor: float = 0.5,
    correlation_cutoff: float = 0.3,
) -> pd.DataFrame:
    relative = relative_abundance(counts)
    feature_array = np.array(feature_names, dtype=object)
    network_features = feature_array != "unassigned"
    rows = []
    for project, index in meta.groupby("project").groups.items():
        index = np.asarray(list(index))
        if len(index) < 5:
            continue
        subset = relative[np.ix_(index, np.flatnonzero(network_features))]
        prevalence = (subset > 0).mean(axis=0)
        mean_abundance = subset.mean(axis=0)
        candidates = np.flatnonzero(prevalence >= prevalence_floor)
        if len(candidates) < 8:
            candidates = np.argsort(mean_abundance)[-min(max_features, len(mean_abundance)):]
        elif len(candidates) > max_features:
            candidates = candidates[np.argsort(mean_abundance[candidates])[-max_features:]]
        if len(candidates) < 8:
            continue
        subset = subset[:, candidates]
        transformed = clr_transform(subset, depths=subset.sum(axis=1))
        ranks = np.argsort(np.argsort(transformed, axis=0), axis=0).astype(float)
        correlation = np.corrcoef(ranks, rowvar=False)
        correlation = np.nan_to_num(correlation)
        adjacency = np.abs(correlation)
        np.fill_diagonal(adjacency, 0.0)
        selected = adjacency >= correlation_cutoff
        n_nodes = adjacency.shape[0]
        possible = n_nodes * (n_nodes - 1) / 2
        density = selected.sum() / (2 * possible) if possible else math.nan
        weighted = adjacency * selected
        if selected.sum() == 0:
            centrality_cv = math.nan
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(weighted)
            leading = np.abs(eigenvectors[:, np.argmax(eigenvalues)])
            centrality_cv = (
                float(leading.std() / leading.mean()) if leading.mean() > 0 else math.nan
            )
        rows.append(
            {
                "project": project,
                "n_nodes": n_nodes,
                "edge_density": float(density),
                "centrality_cv": centrality_cv,
            }
        )
    return pd.DataFrame(rows)


def compute_summary_components(
    counts: np.ndarray,
    meta: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, object]:
    relative = relative_abundance(counts)
    prevalence = (counts > 0).mean(axis=0)
    mean_relative = relative.mean(axis=0)
    depth = counts.sum(axis=1)
    sample_zero_fraction = (counts == 0).mean(axis=1)

    log_relative = np.log1p(relative * 1e6)
    projects = meta["project"].to_numpy()
    variance_ratios = []
    for j in range(counts.shape[1]):
        if prevalence[j] < 0.1:
            continue
        within = []
        project_means = []
        for project in sorted(np.unique(projects)):
            values = log_relative[projects == project, j]
            if len(values) >= 3:
                within.append(np.var(values, ddof=1))
                project_means.append(values.mean())
        if len(within) >= 3 and len(project_means) >= 3:
            between = np.var(project_means, ddof=1)
            if between > 0:
                variance_ratios.append(np.mean(within) / between)

    network = network_project_stats(counts, meta, feature_names)
    return {
        "prevalence": prevalence,
        "rank_abundance": np.sort(mean_relative)[::-1],
        "depth": depth,
        "sample_zero_fraction": sample_zero_fraction,
        "global_zero_fraction": float((counts == 0).mean()),
        "variance_ratio": float(np.median(variance_ratios))
        if variance_ratios
        else math.nan,
        "bray": pairwise_bray(relative),
        "aitchison": pairwise_aitchison(relative, depths=counts.sum(axis=1)),
        "network": network,
    }


def summary_values(components: dict[str, object]) -> dict[str, object]:
    prevalence = np.asarray(components["prevalence"])
    rank_abundance = np.asarray(components["rank_abundance"])
    depth = np.asarray(components["depth"])
    bray = np.asarray(components["bray"])
    aitchison = np.asarray(components["aitchison"])
    sample_zero = np.asarray(components["sample_zero_fraction"])
    network = components["network"]
    if not isinstance(network, pd.DataFrame):
        raise TypeError("network component must be a DataFrame")
    return {
        "prevalence_quantiles": np.quantile(
            prevalence, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        ),
        "rank_abundance_log10_quantiles": np.quantile(
            np.log10(np.maximum(rank_abundance, 1e-10)),
            [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
        ),
        "depth_log_quantiles": np.quantile(
            np.log1p(depth), [0.05, 0.25, 0.5, 0.75, 0.95]
        ),
        "variance_ratio": float(components["variance_ratio"]),
        "bray_quantiles": np.quantile(
            bray, [0.05, 0.25, 0.5, 0.75, 0.95]
        ),
        "aitchison_quantiles": np.quantile(
            aitchison, [0.05, 0.25, 0.5, 0.75, 0.95]
        ),
        "zero_fraction": float(components["global_zero_fraction"]),
        "sample_zero_fraction_quantiles": np.quantile(
            sample_zero, [0.05, 0.25, 0.5, 0.75, 0.95]
        ),
        "network_density_quantiles": np.quantile(
            network["edge_density"].dropna(), [0.1, 0.5, 0.9]
        ),
        "network_centrality_cv_quantiles": np.quantile(
            network["centrality_cv"].dropna(), [0.1, 0.5, 0.9]
        ),
    }

