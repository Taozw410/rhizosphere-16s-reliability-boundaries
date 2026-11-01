#!/usr/bin/env python3
"""Hierarchical project-profile Dirichlet-multinomial generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from p4_core_sim_common import CoreContext
from p4_run_effect_interface import fit_effect_multipliers


@dataclass
class DirichletSimulation:
    counts: np.ndarray
    truth: pd.DataFrame


def simulate_dirichlet_scenario(
    context: CoreContext,
    effects: pd.DataFrame,
    scenario: str,
    concentration: float,
    seed: int,
) -> DirichletSimulation:
    rng = np.random.default_rng(seed)
    counts = np.zeros_like(context.real_counts)
    projects = sorted(context.meta["project"].unique())
    feature_index = {
        name: index for index, name in enumerate(context.feature_names)
    }
    global_profile = context.real_counts.sum(axis=0).astype(float)
    global_profile /= global_profile.sum()
    depths = context.real_counts.sum(axis=1)
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

    for project in projects:
        rows = np.flatnonzero(context.meta["project"].eq(project).to_numpy())
        project_counts = context.real_counts[rows].sum(axis=0).astype(float)
        project_profile = (
            project_counts + 10.0 * global_profile
        ) / (project_counts.sum() + 10.0)
        alpha = np.maximum(project_profile * concentration, 1e-8)
        draw = rng.dirichlet(alpha, size=len(rows))
        base_probability[rows] = draw
        treated_rows = treatment[rows]
        probabilities = draw.copy()
        if len(target_columns) and treated_rows.any():
            multipliers = fit_effect_multipliers(
                draw,
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
                int(depths[global_row]),
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
    return DirichletSimulation(counts=counts, truth=truth)

