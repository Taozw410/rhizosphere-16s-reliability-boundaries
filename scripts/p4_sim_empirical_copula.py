#!/usr/bin/env python3
"""Empirical marginal Gaussian-copula count generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from p4_core_sim_common import CoreContext
from p4_run_effect_interface import fit_effect_multipliers
from p4_sim_lognormal import clr_from_counts


@dataclass
class EmpiricalCopulaSimulation:
    counts: np.ndarray
    truth: pd.DataFrame


def simulate_empirical_copula_scenario(
    context: CoreContext,
    effects: pd.DataFrame,
    scenario: str,
    correlation_shrinkage: float,
    correlation_scale: float,
    seed: int,
) -> EmpiricalCopulaSimulation:
    rng = np.random.default_rng(seed)
    projects = sorted(context.meta["project"].unique())
    feature_index = {
        name: index for index, name in enumerate(context.feature_names)
    }
    transformed = clr_from_counts(context.real_counts)
    residuals = np.zeros_like(transformed)
    for project in projects:
        rows = context.meta["project"].eq(project).to_numpy()
        residuals[rows] = transformed[rows] - transformed[rows].mean(axis=0)
    ranks = rankdata(residuals, axis=0)
    correlation = np.corrcoef(ranks, rowvar=False)
    correlation = np.nan_to_num(correlation)
    correlation = correlation * correlation_scale
    np.fill_diagonal(correlation, 1.0)
    correlation = (
        (1.0 - correlation_shrinkage) * correlation
        + correlation_shrinkage * np.eye(correlation.shape[0])
    )
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    correlation = (eigenvectors * eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.diag(correlation))
    correlation = correlation / scale[:, None] / scale[None, :]

    relative = context.real_counts / context.real_counts.sum(
        axis=1,
        keepdims=True,
    )
    depth = context.real_counts.sum(axis=1)
    treatment = context.meta["condition"].eq("treatment").to_numpy()
    scenario_effects = effects[effects["scenario"].eq(scenario)]
    target_columns = np.array(
        [
            feature_index[name]
            for name in scenario_effects["feature"]
        ],
        dtype=int,
    )
    requested = scenario_effects["requested_log2_effect"].to_numpy(
        dtype=float
    )
    base_probability = np.zeros_like(context.real_counts, dtype=float)
    realized_probability = np.zeros_like(context.real_counts, dtype=float)
    counts = np.zeros_like(context.real_counts)

    for project in projects:
        rows = np.flatnonzero(context.meta["project"].eq(project).to_numpy())
        latent = rng.multivariate_normal(
            mean=np.zeros(len(context.feature_names)),
            cov=correlation,
            size=len(rows),
        )
        uniforms = np.clip(norm.cdf(latent), 1e-6, 1 - 1e-6)
        empirical = np.sort(relative[rows], axis=0)
        probabilities = np.zeros_like(uniforms)
        for column in range(len(context.feature_names)):
            positions = uniforms[:, column] * (len(rows) - 1)
            lower = np.floor(positions).astype(int)
            upper = np.minimum(lower + 1, len(rows) - 1)
            fraction = positions - lower
            probabilities[:, column] = (
                empirical[lower, column] * (1 - fraction)
                + empirical[upper, column] * fraction
            )
        row_sums = probabilities.sum(axis=1, keepdims=True)
        for row in np.flatnonzero(row_sums[:, 0] <= 0):
            probabilities[row] = relative[rows[row]]
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        base_probability[rows] = probabilities
        treated_rows = treatment[rows]
        if len(target_columns) and treated_rows.any():
            multipliers = fit_effect_multipliers(
                probabilities,
                treated_rows,
                target_columns,
                requested,
            )
            probabilities[np.ix_(treated_rows, target_columns)] *= (
                multipliers[None, :]
            )
            probabilities /= probabilities.sum(axis=1, keepdims=True)
        realized_probability[rows] = probabilities
        for local_row, global_row in enumerate(rows):
            counts[global_row] = rng.multinomial(
                int(depth[global_row]),
                probabilities[local_row],
            )

    control_mean = base_probability[~treatment].mean(axis=0)
    treated_base_mean = base_probability[treatment].mean(axis=0)
    treated_mean = realized_probability[treatment].mean(axis=0)
    observed_log2 = np.log2(
        np.maximum(treated_mean, 1e-15)
        / np.maximum(control_mean, 1e-15)
    )
    within_change = np.log2(
        np.maximum(treated_mean, 1e-15)
        / np.maximum(treated_base_mean, 1e-15)
    )
    requested_all = np.zeros(len(context.feature_names))
    requested_all[target_columns] = requested
    is_target = np.zeros(len(context.feature_names), dtype=bool)
    is_target[target_columns] = True
    truth = pd.DataFrame(
        {
            "scenario": scenario,
            "feature": context.feature_names,
            "is_target": is_target,
            "requested_log2_effect": requested_all,
            "case_control_observational_log2": observed_log2,
            "within_treatment_log2_change": within_change,
            "closure_effect_on_nontarget": np.where(
                is_target,
                0.0,
                within_change,
            ),
        }
    )
    return EmpiricalCopulaSimulation(counts=counts, truth=truth)

