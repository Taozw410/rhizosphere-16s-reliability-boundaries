#!/usr/bin/env python3
"""Candidate core/occupancy methods for the Paper 4 smoke benchmark."""

from __future__ import annotations

import concurrent.futures

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln, logit, logsumexp
from scipy.stats import beta


METHOD_NAMES = {
    "sample_prevalence": "Sample prevalence threshold",
    "project_balanced_prevalence": "Project-balanced prevalence",
    "beta_binomial_eb": "Beta-binomial empirical Bayes",
    "hierarchical_occupancy": "Hierarchical occupancy-detection",
}


def sample_prevalence_score(
    counts: np.ndarray,
    meta: pd.DataFrame,
) -> np.ndarray:
    return (counts > 0).mean(axis=0)


def project_balanced_prevalence_score(
    counts: np.ndarray,
    meta: pd.DataFrame,
    project_prevalence_cutoff: float = 0.80,
) -> np.ndarray:
    projects = meta["project"].to_numpy()
    scores = []
    for column in range(counts.shape[1]):
        project_prevalence = [
            (counts[projects == project, column] > 0).mean()
            for project in np.unique(projects)
        ]
        scores.append(np.mean(np.asarray(project_prevalence) >= project_prevalence_cutoff))
    return np.asarray(scores)


def _beta_binomial_prior(presence: np.ndarray) -> tuple[float, float]:
    observed = presence.mean(axis=0)
    mean_observed = float(np.clip(observed.mean(), 1e-4, 1 - 1e-4))
    variance_observed = max(float(observed.var(ddof=1)), 1e-8)
    concentration = (
        mean_observed * (1.0 - mean_observed) / variance_observed - 1.0
    )
    concentration = float(np.clip(concentration, 0.05, 1e4))
    return mean_observed * concentration, (1.0 - mean_observed) * concentration


def beta_binomial_eb_score(
    counts: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    presence = counts > 0
    alpha_prior, beta_prior = _beta_binomial_prior(presence)
    successes = presence.sum(axis=0)
    n = presence.shape[0]
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + n - successes
    posterior_mean = alpha_post / (alpha_post + beta_post)
    posterior_lower = beta.ppf(0.05, alpha_post, beta_post)
    return posterior_mean, posterior_lower


def _binomial_logpmf(
    successes: np.ndarray,
    trials: np.ndarray,
    probability: np.ndarray,
) -> np.ndarray:
    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    return (
        gammaln(trials + 1)
        - gammaln(successes + 1)
        - gammaln(trials - successes + 1)
        + successes * np.log(probability)
        + (trials - successes) * np.log1p(-probability)
    )


def fit_hierarchical_occupancy_feature(
    successes: np.ndarray,
    trials: np.ndarray,
    depth_z: np.ndarray,
) -> tuple[float, float, float]:
    nodes, weights = np.polynomial.hermite.hermgauss(9)
    nodes = np.sqrt(2.0) * nodes
    log_weights = np.log(weights)
    prevalence = float(np.clip(successes.sum() / trials.sum(), 1e-5, 1 - 1e-5))

    def objective(theta: np.ndarray) -> float:
        intercept, depth_slope, log_sigma = theta
        sigma = np.exp(log_sigma)
        nll = 0.0
        for project in range(len(successes)):
            random_effect = sigma * nodes
            eta = (
                intercept
                + depth_slope * depth_z[project]
                + random_effect[None, :]
            )
            log_likelihood = _binomial_logpmf(
                np.full_like(random_effect, successes[project]),
                np.full_like(random_effect, trials[project]),
                expit(eta),
            )
            nll -= float(logsumexp(log_likelihood + log_weights))
        return nll

    result = minimize(
        objective,
        x0=np.array([logit(prevalence), 0.0, np.log(0.25)]),
        method="L-BFGS-B",
        bounds=[(-12.0, 12.0), (-5.0, 5.0), (-6.0, 2.0)],
        options={"maxiter": 200, "ftol": 1e-8},
    )
    if not result.success:
        return prevalence, 0.0, 0.25
    intercept, depth_slope, log_sigma = result.x
    population_occupancy = float(
        np.mean(expit(intercept + depth_slope * depth_z))
    )
    return population_occupancy, float(depth_slope), float(np.exp(log_sigma))


def _fit_hierarchical_column(
    payload: tuple[int, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, object]:
    column, successes, trials, depth_z = payload
    population_occupancy, depth_slope, sigma = (
        fit_hierarchical_occupancy_feature(
            successes,
            trials,
            depth_z,
        )
    )
    return {
        "feature_index": column,
        "population_occupancy": population_occupancy,
        "depth_slope": depth_slope,
        "project_sd": sigma,
        "project_successes": successes.tolist(),
        "project_trials": trials.tolist(),
    }


def hierarchical_occupancy_score(
    counts: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    presence = counts > 0
    projects = meta["project"].to_numpy()
    project_names = sorted(np.unique(projects))
    project_index = np.array(
        [project_names.index(project) for project in projects],
        dtype=int,
    )
    depth = counts.sum(axis=1).astype(float)
    depth_z = (np.log(depth) - np.log(depth).mean()) / max(
        float(np.log(depth).std(ddof=0)),
        1e-8,
    )
    project_depth_z = np.array(
        [
            depth_z[project_index == project].mean()
            for project in range(len(project_names))
        ]
    )
    payloads = []
    for column in range(counts.shape[1]):
        per_project_success = np.array(
            [
                presence[project_index == project, column].sum()
                for project in range(len(project_names))
            ],
            dtype=float,
        )
        per_project_trials = np.array(
            [
                (project_index == project).sum()
                for project in range(len(project_names))
            ],
            dtype=float,
        )
        payloads.append(
            (
                column,
                per_project_success,
                per_project_trials,
                project_depth_z,
            )
        )
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        fits = list(executor.map(_fit_hierarchical_column, payloads))
    fit_frame = pd.DataFrame(fits)
    return fit_frame["population_occupancy"].to_numpy(), fit_frame


def score_method(
    method_id: str,
    counts: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    if method_id == "sample_prevalence":
        scores = sample_prevalence_score(counts, meta)
        return scores, pd.DataFrame()
    if method_id == "project_balanced_prevalence":
        scores = project_balanced_prevalence_score(counts, meta)
        return scores, pd.DataFrame()
    if method_id == "beta_binomial_eb":
        scores, lower = beta_binomial_eb_score(counts, meta)
        return scores, pd.DataFrame({"posterior_lower_0.05": lower})
    if method_id == "hierarchical_occupancy":
        return hierarchical_occupancy_score(counts, meta)
    raise ValueError(f"Unknown core method: {method_id}")

