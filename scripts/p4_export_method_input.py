#!/usr/bin/env python3
"""Export a scenario NPZ and design into CSV files for the R DA runners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-npz", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-projects", type=int, default=0)
    parser.add_argument("--projects-per-crop", type=int, default=0)
    parser.add_argument("--min-prevalence", type=float, default=0.10)
    parser.add_argument("--max-features", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = np.load(args.counts_npz, allow_pickle=False)
    counts = payload["counts"]
    features = payload["feature_names"].astype(str)
    samples = payload["sample"].astype(str)
    projects = payload["project"].astype(str)
    design = pd.read_csv(args.design, dtype={"sample": str})
    design = design.set_index("sample").loc[samples].reset_index()
    if not np.array_equal(design["sample"].to_numpy(), samples):
        raise ValueError("Sample order mismatch between counts and design")
    design["project"] = projects

    selected_projects = sorted(design["project"].unique())
    if args.projects_per_crop > 0:
        project_crop = (
            design[["project", "crop"]]
            .drop_duplicates()
            .sort_values(["crop", "project"])
        )
        selected_projects = (
            project_crop.groupby("crop", sort=True)
            .head(args.projects_per_crop)["project"]
            .tolist()
        )
    elif args.max_projects > 0:
        selected_projects = sorted(selected_projects)[: args.max_projects]

    keep_samples = design["project"].isin(selected_projects).to_numpy()
    counts = counts[keep_samples]
    samples = samples[keep_samples]
    design = design.loc[keep_samples].reset_index(drop=True)

    prevalence = (counts > 0).mean(axis=0)
    truth = pd.read_csv(args.truth)
    truth = truth[truth["scenario"].eq(args.scenario)].copy()
    target_features = set(truth.loc[truth["is_target"], "feature"])
    keep_features = (
        (features != "unassigned")
        & (prevalence >= args.min_prevalence)
    )
    if args.max_features > 0 and keep_features.sum() > args.max_features:
        candidate_indices = np.flatnonzero(keep_features)
        target_indices = [
            index
            for index in candidate_indices
            if features[index] in target_features
        ]
        remaining = [
            index
            for index in candidate_indices
            if index not in set(target_indices)
        ]
        remaining = sorted(
            remaining,
            key=lambda index: (-prevalence[index], features[index]),
        )
        selected_indices = target_indices + remaining[
            : max(args.max_features - len(target_indices), 0)
        ]
        keep_features = np.zeros(len(features), dtype=bool)
        keep_features[selected_indices] = True
    counts = counts[:, keep_features]
    features = features[keep_features]
    if len(features) < 20:
        raise ValueError(f"Only {len(features)} features remain after filtering")
    nonzero_samples = counts.sum(axis=1) > 0
    if not nonzero_samples.all():
        counts = counts[nonzero_samples]
        samples = samples[nonzero_samples]
        design = design.loc[nonzero_samples].reset_index(drop=True)
    if (counts.sum(axis=1) == 0).any():
        raise ValueError("Zero-depth samples remain after feature filtering")

    counts_frame = pd.DataFrame(
        counts.T,
        index=features,
        columns=samples,
    )
    counts_frame.index.name = "feature"
    counts_frame.to_csv(out_dir / "counts.csv", index_label="feature")
    design[["sample", "project", "crop", "condition", "block"]].to_csv(
        out_dir / "metadata.csv", index=False
    )

    truth = truth[
        truth["feature"].isin(features)
    ].copy()
    if truth.empty:
        raise ValueError("No truth rows remain after feature filtering")
    truth.to_csv(out_dir / "truth.csv", index=False)

    manifest = {
        "scenario": args.scenario,
        "counts_npz": str(Path(args.counts_npz).resolve()),
        "design": str(Path(args.design).resolve()),
        "truth": str(Path(args.truth).resolve()),
        "projects": len(design["project"].unique()),
        "samples": len(samples),
        "features": len(features),
        "samples_dropped_zero_depth": int((~nonzero_samples).sum()),
        "min_prevalence": args.min_prevalence,
        "max_features": args.max_features,
        "target_features_retained": int(truth["is_target"].sum()),
    }
    with (out_dir / "input_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
