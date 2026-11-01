#!/usr/bin/env python3
"""Real-template scaffold with latent core perturbation for Paper 4."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from p4_core_sim_common import CoreContext, CoreSimulation
from p4_sim_common import simulate_template_multinomial


def _allocate_remaining_counts(
    background_counts: np.ndarray,
    remaining_depth: int,
) -> np.ndarray:
    if remaining_depth <= 0:
        return np.zeros_like(background_counts)
    total = int(background_counts.sum())
    if total <= 0:
        result = np.zeros_like(background_counts)
        result[np.argmax(background_counts)] = remaining_depth
        return result
    raw = background_counts.astype(float) * remaining_depth / total
    result = np.floor(raw).astype(np.int64)
    remainder = remaining_depth - int(result.sum())
    if remainder > 0:
        order = np.argsort(-(raw - result))
        result[order[:remainder]] += 1
    return result


def assign_scaffold_classes(
    context: CoreContext,
    scenario: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Assign core candidates with minimal disruption to the real scaffold."""
    n_features = len(context.feature_names)
    rng = np.random.default_rng(seed + 9100)
    prevalence = context.background_occupancy.mean(axis=0).to_numpy()
    abundance = context.project_weights.mean(axis=0).to_numpy()
    classes = np.array(["background"] * n_features, dtype=object)
    if scenario == "core_null":
        return classes, {
            "core": np.array([], dtype=int),
            "boundary": np.array([], dtype=int),
            "rare": np.array([], dtype=int),
        }

    high_prevalence = np.argsort(prevalence)[-60:]
    if scenario == "core20_lowabundance":
        low_abundance = np.argsort(abundance)[:70]
        candidates = np.intersect1d(high_prevalence, low_abundance)
        core = rng.choice(candidates, size=20, replace=False)
    else:
        order = high_prevalence[np.argsort(abundance[high_prevalence])]
        strata = np.array_split(order, 20)
        core = np.array([rng.choice(stratum) for stratum in strata], dtype=int)
    core = np.sort(core)
    classes[core] = "core"
    remaining = np.setdiff1d(np.arange(n_features), core)
    boundary = rng.choice(remaining, size=15, replace=False)
    remaining = np.setdiff1d(remaining, boundary)
    rare = rng.choice(remaining, size=15, replace=False)
    classes[boundary] = "boundary"
    classes[rare] = "rare"
    return classes, {
        "core": core,
        "boundary": np.sort(boundary),
        "rare": np.sort(rare),
    }


def simulate_scaffold_core_scenario(
    context: CoreContext,
    scenario: str,
    seed: int,
    preserve_richness: bool = False,
) -> CoreSimulation:
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
    counts = simulate_template_multinomial(
        context.real_counts,
        global_profile,
        sample_prior_reads=0.5,
        seed=seed + 131,
    )
    if len(class_indices["core"]):
        core_columns = class_indices["core"]
        core_presence = (
            rng.random(
                (len(context.meta), len(core_columns))
            )
            < detection_probability[:, core_columns]
        )
        mean_abundance = context.project_weights.mean(axis=0).to_numpy()
        removal_priority = np.argsort(mean_abundance)
        for row in range(len(context.meta)):
            adjusted = counts[row].copy()
            core_counts = np.zeros(len(core_columns), dtype=np.int64)
            present = core_presence[row]
            core_counts[present] = 3
            adjusted[core_columns] = core_counts
            background = np.ones(len(feature_names), dtype=bool)
            background[core_columns] = False
            background_present = int((adjusted[background] > 0).sum())
            if preserve_richness:
                desired_background_present = max(
                    int(
                        (context.real_counts[row] > 0).sum()
                        - present.sum()
                    ),
                    0,
                )
                excess = background_present - desired_background_present
                if excess > 0:
                    candidates = [
                        index
                        for index in removal_priority
                        if background[index] and adjusted[index] > 0
                    ][:excess]
                    adjusted[candidates] = 0
            remaining = int(depth[row] - core_counts.sum())
            adjusted[background] = _allocate_remaining_counts(
                adjusted[background],
                remaining,
            )
            counts[row] = adjusted

    return CoreSimulation(
        counts=counts,
        detection_probability=detection_probability,
        truth=truth,
        project_occupancy=project_occupancy_frame,
    )

