#!/usr/bin/env python3
"""Estimate the Paper 4 simulation parameters from the harmonised 45-study data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from p4_sim_common import (
    clr_transform,
    compute_summary_components,
    empirical_bayes_profiles,
    fit_presence_model,
    load_analysis_data,
    relative_abundance,
    summary_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-matrix",
        default="out_analysis/feature_matrix.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/phase1",
    )
    parser.add_argument("--min-depth", type=int, default=1000)
    parser.add_argument("--prior-strength", type=float, default=10000.0)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def project_level_crop_effects(
    profiles: dict[str, np.ndarray],
    meta: pd.DataFrame,
) -> tuple[list[str], np.ndarray]:
    project_crop = (
        meta[["project", "crop"]]
        .drop_duplicates()
        .sort_values("project")
        .set_index("project")["crop"]
    )
    projects = sorted(profiles)
    profile_matrix = np.vstack([profiles[p] for p in projects])
    transformed = clr_transform(profile_matrix)
    crops = ["maize", "rice", "wheat"]
    effects = []
    grand_mean = transformed.mean(axis=0)
    crop_matrix = project_crop.reindex(projects).to_numpy()
    for crop in crops:
        selected = transformed[crop_matrix == crop]
        effects.append(selected.mean(axis=0) - grand_mean)
    return crops, np.vstack(effects)


def bootstrap_summary(
    counts: np.ndarray,
    meta: pd.DataFrame,
    feature_names: list[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    project_indices = {
        project: np.flatnonzero(meta["project"].to_numpy() == project)
        for project in sorted(meta["project"].unique())
    }
    projects = sorted(project_indices)
    rows = []
    for iteration in range(n_bootstrap):
        selected_projects = rng.choice(projects, size=len(projects), replace=True)
        count_parts = []
        meta_parts = []
        for copy_index, project in enumerate(selected_projects):
            sample_index = rng.choice(
                project_indices[project],
                size=len(project_indices[project]),
                replace=True,
            )
            count_parts.append(counts[sample_index])
            part = meta.iloc[sample_index].copy()
            part["project"] = f"B{copy_index:04d}"
            meta_parts.append(part)
        boot_counts = np.vstack(count_parts)
        boot_meta = pd.concat(meta_parts, ignore_index=True)
        components = compute_summary_components(
            boot_counts, boot_meta, feature_names
        )
        values = summary_values(components)
        row = {"bootstrap": iteration}
        row.update(
            {
                "variance_ratio": values["variance_ratio"],
                "zero_fraction": values["zero_fraction"],
            }
        )
        for label, key in [
            ("prevalence", "prevalence_quantiles"),
            ("rank_abundance", "rank_abundance_log10_quantiles"),
            ("depth", "depth_log_quantiles"),
            ("bray", "bray_quantiles"),
            ("aitchison", "aitchison_quantiles"),
            ("sample_zero", "sample_zero_fraction_quantiles"),
            ("network_density", "network_density_quantiles"),
            ("network_centrality_cv", "network_centrality_cv_quantiles"),
        ]:
            for index, value in enumerate(np.asarray(values[key])):
                row[f"{label}_q{index}"] = float(value)
        rows.append(row)
        print(f"bootstrap {iteration + 1}/{n_bootstrap}")
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_analysis_data(Path(args.feature_matrix), min_depth=args.min_depth)
    projects = data.meta["project"].to_numpy()
    global_profile, profile_matrix, profiles = empirical_bayes_profiles(
        data.counts, projects, prior_strength=args.prior_strength
    )
    presence = fit_presence_model(
        data.counts,
        data.meta,
        global_profile,
        profiles,
    )
    prevalence = (data.counts > 0).mean(axis=0)
    crops, crop_effects = project_level_crop_effects(profiles, data.meta)

    np.savez_compressed(
        out_dir / "simulation_parameters_v1.npz",
        feature_names=np.array(data.feature_names, dtype=str),
        project_names=np.array(sorted(profiles), dtype=str),
        prevalence=prevalence,
        sample_profiles=relative_abundance(data.counts).astype(np.float32),
        sample_ids=np.array(data.meta["sample"], dtype=str),
        global_profile=global_profile,
        project_profiles=profile_matrix,
        crop_effects=crop_effects,
        crop_names=np.array(crops, dtype=str),
    )

    components = compute_summary_components(
        data.counts, data.meta, data.feature_names
    )
    values = summary_values(components)
    summary_rows = []
    for key, value in values.items():
        if np.isscalar(value):
            summary_rows.append(
                {"statistic": key, "index": "scalar", "value": float(value)}
            )
        else:
            for index, item in enumerate(np.asarray(value)):
                summary_rows.append(
                    {"statistic": key, "index": index, "value": float(item)}
                )
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "real_summary_statistics.csv", index=False
    )
    components["network"].to_csv(
        out_dir / "real_project_network_stats.csv", index=False
    )

    if args.bootstrap > 0:
        bootstrap = bootstrap_summary(
            data.counts,
            data.meta,
            data.feature_names,
            args.bootstrap,
            args.seed,
        )
        bootstrap.to_csv(
            out_dir / "bootstrap_summary_statistics.csv", index=False
        )
        bootstrap.quantile([0.025, 0.5, 0.975]).to_csv(
            out_dir / "bootstrap_summary_quantiles.csv"
        )

    parameter_json = {
        "version": "v1",
        "source_feature_matrix": str(Path(args.feature_matrix).resolve()),
        "source_sha256": hashlib.sha256(
            Path(args.feature_matrix).read_bytes()
        ).hexdigest(),
        "min_bacteria_depth": args.min_depth,
        "n_projects": int(data.meta["project"].nunique()),
        "n_samples": int(len(data.meta)),
        "n_features": len(data.feature_names),
        "prior_strength": args.prior_strength,
        "simulator_family": "template_multinomial_v1",
        "presence_model": presence,
        "global_zero_fraction": float((data.counts == 0).mean()),
        "feature_prevalence_quantiles": {
            str(q): float(v)
            for q, v in zip(
                [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
                np.quantile(prevalence, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]),
            )
        },
        "project_sample_sizes": {
            str(k): int(v)
            for k, v in data.meta.groupby("project").size().items()
        },
        "depth_quantiles": {
            str(q): float(v)
            for q, v in zip(
                [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1],
                np.quantile(
                    data.meta["feature_depth"],
                    [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1],
                ),
            )
        },
        "crop_effects": {
            "interpretation": "descriptive CLR project-profile contrast; not causal",
            "crops": crops,
        },
        "array_file": "simulation_parameters_v1.npz",
    }
    with (out_dir / "simulation_parameters_v1.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(parameter_json, handle, indent=2)
    print(f"Wrote parameter estimation outputs to {out_dir}")


if __name__ == "__main__":
    main()
