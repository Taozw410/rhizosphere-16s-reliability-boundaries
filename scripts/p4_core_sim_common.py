#!/usr/bin/env python3
"""Latent-occupancy simulator for the Paper 4 core benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

from p4_sim_common import AnalysisData, load_analysis_data


@dataclass
class CoreContext:
    data: AnalysisData
    real_counts: np.ndarray
    meta: pd.DataFrame
    feature_names: list[str]
    project_weights: pd.DataFrame
    background_occupancy: pd.DataFrame
    project_depth_z: pd.Series


@dataclass
class CoreSimulation:
    counts: np.ndarray
    detection_probability: np.ndarray
    truth: pd.DataFrame
    project_occupancy: pd.DataFrame


def load_core_context(
    feature_matrix: Path,
    design_path: Path,
    config_path: Path,
    panel_path: Path,
    project_panel_override: list[str] | None = None,
) -> CoreContext:
    data = load_analysis_data(feature_matrix, min_depth=1000)
    config = pd.read_json(config_path, typ="series")
    project_panel = (
        list(project_panel_override)
        if project_panel_override is not None
        else list(config["project_panel"])
    )
    panel = pd.read_csv(panel_path)
    if "feature" not in panel.columns:
        raise ValueError("Core panel file must contain a `feature` column")
    feature_names = panel["feature"].astype(str).tolist()
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Core panel contains duplicated features")

    design = pd.read_csv(design_path, dtype={"sample": str})
    design = design.set_index("sample")
    meta = design.loc[data.meta["sample"]].reset_index()
    keep = meta["project"].isin(project_panel).to_numpy()
    meta = meta.loc[keep].reset_index(drop=True)
    sample_indices = np.flatnonzero(keep)
    feature_index = {
        name: index for index, name in enumerate(data.feature_names)
    }
    missing = [name for name in feature_names if name not in feature_index]
    if missing:
        raise ValueError(f"Panel features missing from matrix: {missing[:5]}")
    feature_indices = np.array(
        [feature_index[name] for name in feature_names],
        dtype=int,
    )
    real_counts = data.counts[np.ix_(sample_indices, feature_indices)].astype(
        np.int64
    )
    valid_samples = real_counts.sum(axis=1) > 0
    if not valid_samples.all():
        meta = meta.loc[valid_samples].reset_index(drop=True)
        real_counts = real_counts[valid_samples]
    meta["feature_depth"] = real_counts.sum(axis=1)
    if (meta["feature_depth"] <= 0).any():
        raise ValueError("Selected core panel contains zero-depth samples")

    relative = real_counts / real_counts.sum(axis=1, keepdims=True)
    project_rows = []
    projects = sorted(meta["project"].unique())
    for project in projects:
        rows = meta["project"].eq(project).to_numpy()
        project_rows.append(relative[rows].mean(axis=0))
    project_weights = pd.DataFrame(
        np.vstack(project_rows),
        index=projects,
        columns=feature_names,
    )

    observed = real_counts > 0
    global_prevalence = np.clip(
        (observed.mean(axis=0) * len(meta) + 1) / (len(meta) + 2),
        0.005,
        0.82,
    )
    background_rows = []
    for project in projects:
        rows = meta["project"].eq(project).to_numpy()
        project_prevalence = (
            observed[rows].sum(axis=0) + 10.0 * global_prevalence
        ) / (rows.sum() + 10.0)
        background_rows.append(np.clip(project_prevalence, 0.005, 0.82))
    background_occupancy = pd.DataFrame(
        np.vstack(background_rows),
        index=projects,
        columns=feature_names,
    )

    project_depth = (
        meta.groupby("project")["feature_depth"]
        .median()
        .reindex(projects)
        .astype(float)
    )
    log_depth = np.log(project_depth)
    project_depth_z = (log_depth - log_depth.mean()) / max(
        float(log_depth.std(ddof=0)),
        1e-8,
    )
    return CoreContext(
        data=data,
        real_counts=real_counts,
        meta=meta,
        feature_names=feature_names,
        project_weights=project_weights,
        background_occupancy=background_occupancy,
        project_depth_z=project_depth_z,
    )


def _balanced_core_features(
    mean_abundance: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    order = np.argsort(mean_abundance)
    strata = np.array_split(order, 20)
    return np.array([rng.choice(stratum) for stratum in strata], dtype=int)


def assign_core_classes(
    context: CoreContext,
    scenario: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n_features = len(context.feature_names)
    mean_abundance = context.project_weights.mean(axis=0).to_numpy()
    rng = np.random.default_rng(seed + 8100)
    background = np.arange(n_features, dtype=int)
    if scenario == "core_null":
        return np.array(["background"] * n_features, dtype=object), {
            "core": np.array([], dtype=int),
            "boundary": np.array([], dtype=int),
            "rare": np.array([], dtype=int),
        }

    balanced_core = _balanced_core_features(mean_abundance, rng)
    if scenario == "core20_lowabundance":
        candidates = np.argsort(mean_abundance)[: min(50, n_features)]
        core = rng.choice(candidates, size=20, replace=False)
    else:
        core = balanced_core
    remaining = np.setdiff1d(background, core, assume_unique=False)
    boundary = rng.choice(remaining, size=15, replace=False)
    remaining = np.setdiff1d(remaining, boundary, assume_unique=False)
    rare_candidates = remaining[
        np.argsort(mean_abundance[remaining])[: min(50, len(remaining))]
    ]
    rare = rng.choice(rare_candidates, size=15, replace=False)

    classes = np.array(["background"] * n_features, dtype=object)
    classes[core] = "core"
    classes[boundary] = "boundary"
    classes[rare] = "rare"
    return classes, {
        "core": np.sort(core),
        "boundary": np.sort(boundary),
        "rare": np.sort(rare),
    }


def _calibrate_background_offset(
    probabilities: np.ndarray,
    classes: np.ndarray,
    target_presence_per_sample: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    background = classes == "background"
    if not background.any():
        return probabilities, np.zeros(len(probabilities))
    result = probabilities.copy()
    background_values = result[:, background]
    background_logit = logit(np.clip(background_values, 1e-6, 1 - 1e-6))
    non_background_expected = probabilities[:, ~background].sum(axis=1)
    offsets = np.zeros(len(probabilities))
    for row in range(len(probabilities)):
        available = max(
            float(target_presence_per_sample[row] - non_background_expected[row]),
            1.0,
        )

        def difference(offset: float) -> float:
            adjusted = np.clip(
                expit(background_logit[row] + offset),
                0.001,
                0.88,
            )
            return float(adjusted.sum() - available)

        lower, upper = -12.0, 8.0
        if difference(lower) > 0 or difference(upper) < 0:
            offset = 0.0
        else:
            offset = float(brentq(difference, lower, upper, maxiter=100))
        offsets[row] = offset
        result[row, background] = np.clip(
            expit(background_logit[row] + offset),
            0.001,
            0.88,
        )
    return result, offsets


def simulate_core_scenario(
    context: CoreContext,
    scenario: str,
    seed: int,
) -> CoreSimulation:
    rng = np.random.default_rng(seed)
    classes, class_indices = assign_core_classes(context, scenario, seed)
    feature_names = context.feature_names
    projects = sorted(context.meta["project"].unique())
    project_index = {project: index for index, project in enumerate(projects)}
    project_rows = context.meta["project"].map(project_index).to_numpy()

    project_occupancy = context.background_occupancy.to_numpy().copy()
    core_sd = {
        "core20_balanced": 0.18,
        "core20_lowabundance": 0.18,
        "core20_projectshift": 0.70,
        "core20_depthconfound": 0.16,
    }
    for project_position, project in enumerate(projects):
        if "core" in class_indices and len(class_indices["core"]):
            if scenario == "core20_projectshift":
                center = 0.96
            elif scenario == "core20_depthconfound":
                center = float(
                    expit(
                        logit(0.985)
                        - 0.35 * context.project_depth_z.loc[project]
                    )
                )
            else:
                center = 0.98
            project_occupancy[
                project_position,
                class_indices["core"],
            ] = expit(
                logit(center)
                + rng.normal(
                    0,
                    core_sd.get(scenario, 0.2),
                    size=len(class_indices["core"]),
                )
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
            logit(np.clip(project_occupancy[project_position, background], 1e-5, 0.82))
            + rng.normal(0, 0.12, size=background.sum())
        )

    depth = context.meta["feature_depth"].to_numpy(dtype=float)
    depth_z = (np.log(depth) - np.log(depth).mean()) / max(
        float(np.log(depth).std(ddof=0)),
        1e-8,
    )
    depth_slope = 1.0 if scenario == "core20_depthconfound" else 0.25
    sample_occupancy = project_occupancy[project_rows]
    detection_probability = expit(
        logit(np.clip(sample_occupancy, 1e-6, 1 - 1e-6))
        + depth_slope * depth_z[:, None]
    )

    target_presence_per_sample = (context.real_counts > 0).sum(axis=1)
    detection_probability, background_offsets = _calibrate_background_offset(
        detection_probability,
        classes,
        target_presence_per_sample,
    )

    project_latent = []
    project_fraction_core = []
    for project in projects:
        rows = project_rows == project_index[project]
        project_latent.append(detection_probability[rows].mean(axis=0))
        project_fraction_core.append(
            detection_probability[rows].mean(axis=0) >= 0.90
        )
    project_latent = np.vstack(project_latent)
    project_fraction_core = np.vstack(project_fraction_core)
    core_fraction = project_fraction_core.mean(axis=0)
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
            "mean_relative_abundance": context.project_weights.mean(axis=0).to_numpy(),
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

    template_presence = context.real_counts > 0
    presence = template_presence.copy()
    if len(class_indices["core"]):
        presence[:, class_indices["core"]] = (
            rng.random(
                (len(context.meta), len(class_indices["core"]))
            )
            < detection_probability[:, class_indices["core"]]
        )

    # Keep the observed sample-level richness while allowing the injected core
    # features to replace some non-core presences. The observed binary template
    # preserves co-occurrence among the background community.
    target_richness = template_presence.sum(axis=1)
    core_mask = classes == "core"
    for row in range(len(context.meta)):
        difference = int(target_richness[row] - presence[row].sum())
        if difference < 0:
            candidates = np.flatnonzero(presence[row] & ~core_mask)
            remove_n = min(-difference, len(candidates))
            if remove_n:
                remove = rng.choice(candidates, size=remove_n, replace=False)
                presence[row, remove] = False
        elif difference > 0:
            candidates = np.flatnonzero(~presence[row] & ~core_mask)
            add_n = min(difference, len(candidates))
            if add_n:
                probabilities_add = detection_probability[row, candidates]
                probabilities_add = probabilities_add / probabilities_add.sum()
                add = rng.choice(
                    candidates,
                    size=add_n,
                    replace=False,
                    p=probabilities_add,
                )
                presence[row, add] = True

    empty_rows = ~presence.any(axis=1)
    if empty_rows.any():
        fallback = np.argmax(
            context.project_weights.to_numpy()[project_rows],
            axis=1,
        )
        empty_index = np.flatnonzero(empty_rows)
        presence[empty_index, fallback[empty_index]] = True

    global_profile = context.real_counts.sum(axis=0).astype(float)
    global_profile /= global_profile.sum()
    weights = (
        context.real_counts
        + global_profile[None, :]
    ) / (depth[:, None] + 1.0)
    if len(class_indices["core"]):
        core_weight_floor = 3.0 / np.maximum(depth, 1.0)
        weights[:, class_indices["core"]] = np.maximum(
            weights[:, class_indices["core"]],
            core_weight_floor[:, None],
        )
    weights = np.maximum(weights, 1e-12)
    weights = weights * np.exp(rng.normal(0, 0.12, size=weights.shape))
    probabilities = weights * presence
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    counts = np.zeros_like(context.real_counts)
    for row, sample_depth in enumerate(depth.astype(int)):
        counts[row] = rng.multinomial(sample_depth, probabilities[row])
    truth["background_offset_logit_mean"] = float(background_offsets.mean())
    return CoreSimulation(
        counts=counts,
        detection_probability=detection_probability,
        truth=truth,
        project_occupancy=project_occupancy_frame,
    )

