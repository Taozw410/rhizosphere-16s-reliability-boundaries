#!/usr/bin/env python3
"""Pre-generation latent-factor core simulator for Paper 4."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from p4_core_sim_common import CoreContext, CoreSimulation
from p4_core_sim_scaffold import assign_scaffold_classes


def simulate_latent_factor_core_scenario(
    context: CoreContext,
    scenario: str,
    seed: int,
    correlation: np.ndarray,
) -> CoreSimulation:
    """Set occupancy before multinomial sampling; never edit generated counts."""
    rng = np.random.default_rng(seed)
    classes, class_indices = assign_scaffold_classes(context, scenario, seed)
    projects = sorted(context.meta["project"].unique())
    project_index = {project: index for index, project in enumerate(projects)}
    project_rows = context.meta["project"].map(project_index).to_numpy()
    feature_names = context.feature_names

    project_occupancy = context.background_occupancy.to_numpy().copy()
    for project_position, project in enumerate(projects):
        if len(class_indices["core"]):
            if scenario == "core20_projectshift":
                center = 0.93
                core_sd = 0.25
            elif scenario == "core20_depthconfound":
                center = float(
                    expit(
                        logit(0.93)
                        - 0.15 * context.project_depth_z.loc[project]
                    )
                )
                core_sd = 0.03
            else:
                center = 0.92
                core_sd = 0.03
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

    project_latent = []
    project_core_fraction = []
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
        }
    )
    project_occupancy_frame = pd.DataFrame(
        project_latent,
        index=projects,
        columns=feature_names,
    )

    global_profile = context.real_counts.sum(axis=0).astype(float)
    global_profile /= global_profile.sum()
    probabilities = (
        context.real_counts + 0.5 * global_profile[None, :]
    ) / (depth[:, None] + 0.5)

    core_columns = class_indices["core"]
    if len(core_columns):
        target_detection = detection_probability[:, core_columns]
        target_share = -np.log1p(
            -np.clip(target_detection, 1e-5, 1 - 1e-5)
        ) / depth[:, None]
        core_correlation = correlation[np.ix_(core_columns, core_columns)]
        latent = rng.multivariate_normal(
            mean=np.zeros(len(core_columns)),
            cov=core_correlation,
            size=len(context.meta),
        )
        core_weights = np.maximum(
            probabilities[:, core_columns],
            target_share,
        )
        core_weights *= np.exp(0.08 * latent)
        probabilities[:, core_columns] = core_weights

    probabilities = np.maximum(probabilities, 1e-12)
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

