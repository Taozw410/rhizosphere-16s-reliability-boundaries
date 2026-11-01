#!/usr/bin/env python3
"""Correlated Gaussian-copula occupancy generator for Paper 4 core smoke."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm, rankdata

from p4_core_sim_common import (
    CoreContext,
    CoreSimulation,
    _calibrate_background_offset,
    assign_core_classes,
)


def estimate_latent_correlation(
    context: CoreContext,
    shrinkage: float = 0.5,
) -> np.ndarray:
    """Estimate a regularized feature-correlation matrix from the real panel."""
    relative = context.real_counts / context.real_counts.sum(
        axis=1,
        keepdims=True,
    )
    prior = 0.5 / np.maximum(
        context.real_counts.sum(axis=1, keepdims=True),
        1,
    )
    transformed = np.log(relative + prior)
    transformed -= transformed.mean(axis=1, keepdims=True)
    ranks = rankdata(transformed, axis=0)
    correlation = np.corrcoef(ranks, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    regularized = (
        (1.0 - shrinkage) * correlation
        + shrinkage * np.eye(correlation.shape[0])
    )
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    eigenvalues = np.maximum(eigenvalues, 1e-5)
    regularized = (eigenvectors * eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.diag(regularized))
    regularized = regularized / scale[:, None] / scale[None, :]
    np.fill_diagonal(regularized, 1.0)
    return regularized


def simulate_correlated_core_scenario(
    context: CoreContext,
    scenario: str,
    seed: int,
    correlation: np.ndarray,
) -> CoreSimulation:
    rng = np.random.default_rng(seed)
    classes, class_indices = assign_core_classes(context, scenario, seed)
    projects = sorted(context.meta["project"].unique())
    project_index = {project: index for index, project in enumerate(projects)}
    project_rows = context.meta["project"].map(project_index).to_numpy()
    feature_names = context.feature_names

    project_occupancy = context.background_occupancy.to_numpy().copy()
    for project_position, project in enumerate(projects):
        if len(class_indices["core"]):
            if scenario == "core20_projectshift":
                center = 0.96
                core_sd = 0.70
            elif scenario == "core20_depthconfound":
                center = float(
                    expit(
                        logit(0.985)
                        - 0.35 * context.project_depth_z.loc[project]
                    )
                )
                core_sd = 0.16
            else:
                center = 0.98
                core_sd = 0.18
            project_occupancy[
                project_position,
                class_indices["core"],
            ] = expit(
                logit(center)
                + rng.normal(0, core_sd, size=len(class_indices["core"]))
            )
        if len(class_indices["boundary"]):
            project_occupancy[
                project_position,
                class_indices["boundary"],
            ] = expit(
                logit(0.72)
                + rng.normal(0, 0.35, size=len(class_indices["boundary"]))
            )
        if len(class_indices["rare"]):
            project_occupancy[
                project_position,
                class_indices["rare"],
            ] = expit(
                logit(0.20)
                + rng.normal(0, 0.50, size=len(class_indices["rare"]))
            )
        background = classes == "background"
        project_occupancy[project_position, background] = expit(
            logit(
                np.clip(
                    project_occupancy[project_position, background],
                    1e-5,
                    0.82,
                )
            )
            + rng.normal(0, 0.12, size=background.sum())
        )

    depth = context.meta["feature_depth"].to_numpy(dtype=float)
    depth_z = (np.log(depth) - np.log(depth).mean()) / max(
        float(np.log(depth).std(ddof=0)),
        1e-8,
    )
    depth_slope = 1.0 if scenario == "core20_depthconfound" else 0.25
    detection_probability = expit(
        logit(
            np.clip(
                project_occupancy[project_rows],
                1e-6,
                1 - 1e-6,
            )
        )
        + depth_slope * depth_z[:, None]
    )
    target_presence_per_sample = (context.real_counts > 0).sum(axis=1)
    detection_probability, background_offsets = _calibrate_background_offset(
        detection_probability,
        classes,
        target_presence_per_sample,
    )

    project_core_fraction = []
    project_latent = []
    for project in projects:
        rows = project_rows == project_index[project]
        project_mean = detection_probability[rows].mean(axis=0)
        project_latent.append(project_mean)
        project_core_fraction.append(project_mean >= 0.90)
    project_latent = np.vstack(project_latent)
    project_core_fraction = np.vstack(project_core_fraction)
    core_fraction = project_core_fraction.mean(axis=0)
    is_core = core_fraction >= 0.80
    if scenario == "core_null" and is_core.any():
        raise RuntimeError("core_null accidentally produced a population core")

    truth = pd.DataFrame(
        {
            "feature": feature_names,
            "scenario": scenario,
            "core_class": classes,
            "latent_occupancy_mean": project_latent.mean(axis=0),
            "latent_occupancy_sd": project_latent.std(axis=0, ddof=1),
            "project_core_fraction": core_fraction,
            "is_population_core": is_core,
            "mean_relative_abundance": (
                context.project_weights.mean(axis=0).to_numpy()
            ),
            "truth_core_candidate": np.isin(
                np.arange(len(feature_names)),
                class_indices["core"],
            ),
            "background_offset_logit_mean": float(background_offsets.mean()),
        }
    )
    project_occupancy_frame = pd.DataFrame(
        project_latent,
        index=projects,
        columns=feature_names,
    )

    # Gaussian copula: the marginal presence probabilities are preserved while
    # the real panel correlation matrix controls cross-feature co-occurrence.
    latent = rng.multivariate_normal(
        mean=np.zeros(len(feature_names)),
        cov=correlation,
        size=len(context.meta),
    )
    copula = norm.cdf(latent)
    presence = copula < detection_probability
    empty_rows = ~presence.any(axis=1)
    if empty_rows.any():
        fallback = np.argmax(detection_probability[empty_rows], axis=1)
        presence[np.flatnonzero(empty_rows), fallback] = True

    global_profile = context.real_counts.sum(axis=0).astype(float)
    global_profile /= global_profile.sum()
    weights = (
        context.real_counts + global_profile[None, :]
    ) / (depth[:, None] + 1.0)
    if len(class_indices["core"]):
        core_weight_floor = 3.0 / np.maximum(depth, 1.0)
        weights[:, class_indices["core"]] = np.maximum(
            weights[:, class_indices["core"]],
            core_weight_floor[:, None],
        )
    weights = np.maximum(weights, 1e-12)
    weights *= np.exp(rng.normal(0, 0.12, size=weights.shape))
    probabilities = weights * presence
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    counts = np.zeros_like(context.real_counts)
    for row, sample_depth in enumerate(depth.astype(int)):
        counts[row] = rng.multinomial(sample_depth, probabilities[row])

    return CoreSimulation(
        counts=counts,
        detection_probability=detection_probability,
        truth=truth,
        project_occupancy=project_occupancy_frame,
    )

