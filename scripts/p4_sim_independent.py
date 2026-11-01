#!/usr/bin/env python3
"""Hierarchical hurdle generator estimated independently from training projects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit, softmax
from scipy.stats import norm, rankdata


@dataclass
class IndependentModel:
    features: list[str]
    crops: list[str]
    crop_clr_mean: dict[str, np.ndarray]
    crop_mean_probability: dict[str, np.ndarray]
    between_covariance: dict[str, np.ndarray]
    within_covariance: np.ndarray
    crop_prevalence: dict[str, np.ndarray]
    prevalence_between_sd: np.ndarray
    presence_correlation: np.ndarray
    train_log_depth_mean: float
    train_log_depth_sd: float


def fit_independent_model(
    counts: np.ndarray,
    meta: pd.DataFrame,
    features: list[str],
    crops: np.ndarray,
    between_shrinkage: float = 0.35,
    within_shrinkage: float = 0.20,
    presence_shrinkage: float = 0.50,
) -> IndependentModel:
    depth = counts.sum(axis=1)
    probability = (counts + 0.5) / (depth[:, None] + 0.5 * counts.shape[1])
    transformed = np.log(probability)
    transformed -= transformed.mean(axis=1, keepdims=True)
    presence = counts > 0
    projects = meta["project"].to_numpy()
    unique_projects = sorted(np.unique(projects))
    crop_order = [str(crop) for crop in pd.unique(crops)]
    project_rows = {
        project: projects == project for project in unique_projects
    }
    project_means = {
        project: transformed[project_rows[project]].mean(axis=0)
        for project in unique_projects
    }
    project_crop = {
        project: str(crops[np.flatnonzero(project_rows[project])[0]])
        for project in unique_projects
    }
    global_prevalence = np.clip(
        (presence.mean(axis=0) * len(meta) + 1) / (len(meta) + 2),
        1e-4,
        1 - 1e-4,
    )
    crop_clr_mean = {}
    between_covariance = {}
    crop_prevalence = {}
    crop_mean_probability = {}
    between_variance = []
    residual_parts = []
    for crop in crop_order:
        crop_projects = [
            project
            for project in unique_projects
            if project_crop[project] == crop
        ]
        means = np.vstack([project_means[project] for project in crop_projects])
        center = means.mean(axis=0)
        crop_clr_mean[crop] = center
        covariance = np.cov(means - center, rowvar=False)
        diagonal = np.diag(np.diag(covariance))
        between_covariance[crop] = (
            (1 - between_shrinkage) * covariance
            + between_shrinkage * diagonal
            + 1e-6 * np.eye(covariance.shape[0])
        )
        rows = np.isin(projects, crop_projects)
        residual_parts.append(transformed[rows] - center)
        relative = counts[rows] / counts[rows].sum(axis=1, keepdims=True)
        mean_probability = relative.mean(axis=0)
        mean_probability = (
            mean_probability + 1e-6
        ) / (mean_probability + 1e-6).sum()
        crop_mean_probability[crop] = mean_probability
        project_prevalence = []
        for project in crop_projects:
            rows_p = project_rows[project]
            q = (
                presence[rows_p].sum(axis=0) + 5.0 * global_prevalence
            ) / (rows_p.sum() + 5.0)
            project_prevalence.append(np.clip(q, 1e-4, 1 - 1e-4))
        project_prevalence = np.vstack(project_prevalence)
        crop_prevalence[crop] = project_prevalence.mean(axis=0)
        between_variance.append(
            project_prevalence.var(axis=0, ddof=1)
        )
    residual = np.vstack(residual_parts)
    within_covariance = np.cov(residual, rowvar=False)
    within_diagonal = np.diag(np.diag(within_covariance))
    within_covariance = (
        (1 - within_shrinkage) * within_covariance
        + within_shrinkage * within_diagonal
        + 1e-6 * np.eye(within_covariance.shape[0])
    )
    presence_correlation = np.corrcoef(presence, rowvar=False)
    presence_correlation = np.nan_to_num(presence_correlation)
    presence_correlation = (
        (1 - presence_shrinkage) * presence_correlation
        + presence_shrinkage * np.eye(presence_correlation.shape[0])
    )
    eigenvalues, eigenvectors = np.linalg.eigh(presence_correlation)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    presence_correlation = (eigenvectors * eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.diag(presence_correlation))
    presence_correlation = (
        presence_correlation / scale[:, None] / scale[None, :]
    )
    return IndependentModel(
        features=features,
        crops=crop_order,
        crop_clr_mean=crop_clr_mean,
        crop_mean_probability=crop_mean_probability,
        between_covariance=between_covariance,
        within_covariance=within_covariance,
        crop_prevalence=crop_prevalence,
        prevalence_between_sd=np.sqrt(np.mean(np.vstack(between_variance), axis=0)),
        presence_correlation=presence_correlation,
        train_log_depth_mean=float(np.log(depth).mean()),
        train_log_depth_sd=float(np.log(depth).std(ddof=0)),
    )


def simulate_independent_projects(
    model: IndependentModel,
    meta: pd.DataFrame,
    crops: np.ndarray,
    seed: int,
    depth_slope: float = 0.8,
    dirichlet_concentration: float = 100.0,
    project_effect_sd: float = 0.5,
    presence_offset_logit: float = 0.0,
    presence_between_multiplier: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    depth = meta["feature_depth"].to_numpy(dtype=float)
    depth_z = (
        np.log(depth) - model.train_log_depth_mean
    ) / max(model.train_log_depth_sd, 1e-8)
    counts = np.zeros((len(meta), len(model.features)), dtype=np.int64)
    projects = meta["project"].to_numpy()
    for project in sorted(np.unique(projects)):
        rows = np.flatnonzero(projects == project)
        crop = str(crops[rows[0]])
        project_profile = model.crop_mean_probability[crop] * np.exp(
            rng.normal(0, project_effect_sd, len(model.features))
        )
        project_profile /= project_profile.sum()
        abundance_shape = rng.dirichlet(
            np.maximum(
                dirichlet_concentration * project_profile,
                1e-8,
            ),
            size=len(rows),
        )
        project_q = expit(
            logit(np.clip(model.crop_prevalence[crop], 1e-5, 1 - 1e-5))
            + rng.normal(
                0,
                model.prevalence_between_sd
                * presence_between_multiplier,
            )
            + presence_offset_logit
        )
        sample_q = expit(
            logit(np.clip(project_q, 1e-5, 1 - 1e-5))[None, :]
            + depth_slope * depth_z[rows, None]
        )
        presence_latent = rng.multivariate_normal(
            np.zeros(len(model.features)),
            model.presence_correlation,
            size=len(rows),
        )
        presence = norm.cdf(presence_latent) < sample_q
        empty = ~presence.any(axis=1)
        if empty.any():
            presence[np.flatnonzero(empty), np.argmax(abundance_shape[empty], axis=1)] = True
        probabilities = abundance_shape * presence
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        for local, global_row in enumerate(rows):
            counts[global_row] = rng.multinomial(
                int(depth[global_row]),
                probabilities[local],
            )
            forced = presence[local] & (counts[global_row] == 0)
            if forced.any():
                counts[global_row, forced] = 1
                excess = int(counts[global_row].sum() - depth[global_row])
                while excess > 0:
                    candidates = np.flatnonzero(
                        (counts[global_row] > 1) & ~forced
                    )
                    if not len(candidates):
                        break
                    order = candidates[
                        np.argsort(counts[global_row, candidates])[::-1]
                    ]
                    reduction = np.minimum(
                        counts[global_row, order] - 1,
                        np.ceil(excess / np.arange(1, len(order) + 1)).astype(int),
                    )
                    for index, amount in zip(order, reduction):
                        if excess <= 0:
                            break
                        remove = min(int(amount), excess)
                        counts[global_row, index] -= remove
                        excess -= remove
    return counts

