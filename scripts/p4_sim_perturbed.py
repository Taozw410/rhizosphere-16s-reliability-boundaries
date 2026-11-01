#!/usr/bin/env python3
"""Correlated lognormal perturbation of the calibrated sample template."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from p4_core_sim_common import CoreContext
from p4_run_effect_interface import fit_effect_multipliers
from p4_sim_lognormal import clr_from_counts


@dataclass
class PerturbedSimulation:
    counts: np.ndarray
    truth: pd.DataFrame


def simulate_perturbed_scenario(
    context: CoreContext,
    effects: pd.DataFrame,
    scenario: str,
    perturbation_sd: float,
    seed: int,
) -> PerturbedSimulation:
    rng = np.random.default_rng(seed)
    feature_index = {
        name: index for index, name in enumerate(context.feature_names)
    }
    transformed = clr_from_counts(context.real_counts)
    residuals = np.zeros_like(transformed)
    for project in sorted(context.meta["project"].unique()):
        rows = context.meta["project"].eq(project).to_numpy()
        residuals[rows] = transformed[rows] - transformed[rows].mean(axis=0)
    correlation = np.corrcoef(rankdata(residuals, axis=0), rowvar=False)
    correlation = np.nan_to_num(correlation)
    correlation = 0.5 * correlation + 0.5 * np.eye(correlation.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    correlation = (eigenvectors * eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.diag(correlation))
    correlation = correlation / scale[:, None] / scale[None, :]

    depth = context.real_counts.sum(axis=1)
    global_profile = context.real_counts.sum(axis=0).astype(float)
    global_profile /= global_profile.sum()
    base_probability = (
        context.real_counts + 0.5 * global_profile[None, :]
    ) / (depth[:, None] + 0.5)
    latent = rng.multivariate_normal(
        np.zeros(len(context.feature_names)),
        correlation,
        size=len(context.meta),
    )
    probabilities = base_probability * np.exp(perturbation_sd * latent)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

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
    if len(target_columns) and treatment.any():
        fit_base = np.vstack(
            [
                probabilities[~treatment].mean(axis=0),
                probabilities[treatment].mean(axis=0),
            ]
        )
        multipliers = fit_effect_multipliers(
            fit_base,
            np.array([False, True]),
            target_columns,
            requested,
            max_iterations=5000,
            tolerance=1e-5,
        )
        probabilities[np.ix_(treatment, target_columns)] *= multipliers[None, :]
        probabilities /= probabilities.sum(axis=1, keepdims=True)

    counts = np.zeros_like(context.real_counts)
    for row, sample_depth in enumerate(depth):
        counts[row] = rng.multinomial(
            int(sample_depth),
            probabilities[row],
        )

    control_mean = base_probability[~treatment].mean(axis=0)
    treated_base_mean = base_probability[treatment].mean(axis=0)
    treated_mean = probabilities[treatment].mean(axis=0)
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
    return PerturbedSimulation(counts=counts, truth=truth)

