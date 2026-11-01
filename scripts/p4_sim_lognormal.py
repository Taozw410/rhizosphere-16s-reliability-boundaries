#!/usr/bin/env python3
"""Project-mean plus correlated lognormal-CLR generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import softmax

from p4_core_sim_common import CoreContext
from p4_run_effect_interface import fit_effect_multipliers


@dataclass
class LognormalSimulation:
    counts: np.ndarray
    truth: pd.DataFrame


def clr_from_counts(counts: np.ndarray) -> np.ndarray:
    depth = counts.sum(axis=1, keepdims=True)
    probability = (counts + 0.5 / np.maximum(depth, 1)) / (depth + 0.5)
    logged = np.log(probability)
    return logged - logged.mean(axis=1, keepdims=True)


def simulate_lognormal_scenario(
    context: CoreContext,
    effects: pd.DataFrame,
    scenario: str,
    covariance_shrinkage: float,
    seed: int,
) -> LognormalSimulation:
    rng = np.random.default_rng(seed)
    projects = sorted(context.meta["project"].unique())
    feature_index = {
        name: index for index, name in enumerate(context.feature_names)
    }
    transformed = clr_from_counts(context.real_counts)
    project_means = {}
    residual_rows = []
    for project in projects:
        rows = context.meta["project"].eq(project).to_numpy()
        mean = transformed[rows].mean(axis=0)
        project_means[project] = mean
        residual_rows.append(transformed[rows] - mean)
    residuals = np.vstack(residual_rows)
    covariance = np.cov(residuals, rowvar=False)
    diagonal = np.diag(np.diag(covariance))
    covariance = (
        (1.0 - covariance_shrinkage) * covariance
        + covariance_shrinkage * diagonal
        + 1e-6 * np.eye(covariance.shape[0])
    )
    del residual_rows

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
    treatment = context.meta["condition"].eq("treatment").to_numpy()
    depth = context.real_counts.sum(axis=1)
    base_probability = np.zeros_like(context.real_counts, dtype=float)
    realized_probability = np.zeros_like(context.real_counts, dtype=float)
    counts = np.zeros_like(context.real_counts)

    for project in projects:
        rows = np.flatnonzero(context.meta["project"].eq(project).to_numpy())
        latent = rng.multivariate_normal(
            project_means[project],
            covariance,
            size=len(rows),
        )
        probabilities = softmax(latent, axis=1)
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
    return LognormalSimulation(counts=counts, truth=truth)

