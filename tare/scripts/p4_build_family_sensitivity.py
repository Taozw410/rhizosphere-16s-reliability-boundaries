#!/usr/bin/env python3
"""Build a prespecified family-only DA sensitivity corpus."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TARE_DIR = Path(__file__).resolve().parents[1]
PROJECT_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(TARE_DIR))
sys.path.insert(0, str(PROJECT_SCRIPTS))

from p4_tare_build_da_corpus import simulate_project_effect
from p4_sim_common import load_analysis_data, relative_abundance


ROOT = Path(__file__).resolve().parents[2]
OUT = TARE_DIR / "artifacts" / "family_sensitivity"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_analysis_data(
        ROOT / "out_analysis" / "feature_matrix.csv", min_depth=1000
    )
    design = pd.read_csv(
        ROOT
        / "论文4-多研究仿真"
        / "outputs"
        / "effect_interface"
        / "sample_design.csv",
        dtype={"sample": str},
    ).set_index("sample")
    design = design.loc[data.meta["sample"]].reset_index()
    family_indices = np.array(
        [
            index
            for index, feature in enumerate(data.feature_names)
            if feature.startswith("f__")
        ],
        dtype=int,
    )
    family_names = [data.feature_names[index] for index in family_indices]
    family_counts = data.counts[:, family_indices]
    prevalence = (family_counts > 0).mean(axis=0)
    abundance = relative_abundance(family_counts).mean(axis=0)
    order = np.lexsort(
        (
            np.array(family_names),
            -abundance,
            -prevalence,
        )
    )
    selected = order[: min(120, len(order))]
    panel_features = [family_names[index] for index in selected]
    panel_counts = family_counts[:, selected]
    pd.DataFrame({"feature": panel_features}).to_csv(
        OUT / "family_feature_panel.csv", index=False
    )

    corpus = OUT / "da_corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    manifest = []
    for project_index, project in enumerate(sorted(design["project"].unique())):
        rows = np.flatnonzero(design["project"].eq(project).to_numpy())
        project_design = design.iloc[rows].reset_index(drop=True)
        project_counts = panel_counts[rows]
        project_relative = relative_abundance(project_counts)
        project_prevalence = (project_counts > 0).mean(axis=0)
        project_mean = project_relative.mean(axis=0)
        eligible = [
            index
            for index in range(len(panel_features))
            if 0.25 <= project_prevalence[index] <= 0.95
            and project_mean[index] > 0
        ]
        rng = np.random.default_rng(31000 + project_index)
        target_indices = sorted(
            rng.choice(
                eligible,
                size=min(20, len(eligible)),
                replace=False,
            ).tolist()
        )
        targets = [panel_features[index] for index in target_indices]
        directions = np.where(
            np.arange(len(targets)) % 2 == 0, 1.0, -1.0
        )
        effects = pd.DataFrame(
            {
                "scenario": "sparse20_family",
                "feature": targets,
                "requested_log2_effect": directions,
            }
        )
        simulated, target_truth, full_truth = simulate_project_effect(
            project_counts,
            panel_features,
            project_design,
            effects,
            "sparse20_family",
            5000.0,
            32000 + project_index,
        )
        run_dir = corpus / project / "sparse20_family"
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            simulated.T,
            index=panel_features,
            columns=project_design["sample"].tolist(),
        ).to_csv(run_dir / "counts.csv")
        project_design.to_csv(run_dir / "metadata.csv", index=False)
        full_truth.to_csv(run_dir / "truth.csv", index=False)
        target_truth.to_csv(run_dir / "target_truth.csv", index=False)
        manifest.append(
            {
                "project": project,
                "crop": project_design["crop"].iloc[0],
                "scenario": "sparse20_family",
                "requested_absolute_log2_effect": 1.0,
                "n_targets": len(targets),
                "n_samples": len(project_design),
                "n_features": len(panel_features),
                "input_dir": str(run_dir.resolve()),
                "seed": 32000 + project_index,
            }
        )
    pd.DataFrame(manifest).to_csv(corpus / "manifest.csv", index=False)
    summary = {
        "family_features_available": len(family_names),
        "panel_features": len(panel_features),
        "projects": len(manifest),
        "panel_rule": (
            "Top 120 family fallback features by prevalence, mean relative "
            "abundance and feature name; fixed before method outcomes."
        ),
    }
    (OUT / "family_sensitivity_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
