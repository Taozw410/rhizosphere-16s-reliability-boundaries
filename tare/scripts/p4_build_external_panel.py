#!/usr/bin/env python3
"""Build the frozen six-project external TARE panel and DA corpus."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

TARE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARE_DIR))

from p4_tare_build_da_corpus import simulate_project_effect


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "out_analysis" / "route_c_external"
FEATURE_MATRIX = (
    EXTERNAL / "06_feature_space" / "routeC_feature_matrix.csv"
)
DADA2 = EXTERNAL / "05_dada2"
DEVELOPMENT = ROOT / "out_analysis" / "feature_matrix.csv"
PANEL = (
    ROOT
    / "论文4-多研究仿真"
    / "outputs"
    / "formal_grid"
    / "data"
    / "sparse20_fc2"
    / "feature_panel.csv"
)
FINAL45 = ROOT / "meta" / "final45_selection.csv"
PLATFORMS = EXTERNAL / "00_locked_candidates" / "routeC_platform_clusters.csv"
POST_QC = EXTERNAL / "05_dada2" / "routeC_post_dada2_qc.csv"
OUT = TARE_DIR / "artifacts" / "external_panel"

SELECTED = {
    "PRJEB104640": "maize",
    "PRJNA1062967": "maize",
    "PRJDB13616": "rice",
    "PRJNA1044052": "rice",
    "PRJEB76879": "wheat",
    "PRJNA1095845": "wheat",
}


def asv_length_mode(project: str) -> int:
    path = DADA2 / project / "asv_table.csv"
    lengths: Counter[int] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if row and row[0] and not row[0].upper().startswith("DRR"):
                sequence = row[0].strip().strip('"').upper()
                if sequence and set(sequence) <= set("ACGTN"):
                    lengths[len(sequence)] += 1
    if not lengths:
        return 0
    return int(max(lengths.items(), key=lambda item: (item[1], -item[0]))[0])


def assign_conditions(
    samples: list[str],
    project: str,
    crop: str,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    conditions = ["control", "treatment"] * ((len(shuffled) + 1) // 2)
    conditions = conditions[: len(shuffled)]
    run_order = rng.permutation(len(shuffled))
    return pd.DataFrame(
        {
            "sample": shuffled,
            "project": project,
            "crop": crop,
            "condition": conditions,
            "block": project,
            "experimental_unit": project,
            "run_order": run_order,
        }
    )


def choose_targets(
    counts: np.ndarray,
    feature_names: list[str],
    seed: int,
) -> list[str]:
    relative = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    prevalence = (counts > 0).mean(axis=0)
    mean_relative = relative.mean(axis=0)
    candidates = [
        feature
        for index, feature in enumerate(feature_names)
        if feature != "unassigned"
        and 0.25 <= prevalence[index] <= 0.95
        and mean_relative[index] > 0
    ]
    rng = np.random.default_rng(seed)
    return sorted(
        rng.choice(candidates, size=min(20, len(candidates)), replace=False).tolist()
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(FEATURE_MATRIX)
    matrix = matrix[matrix["project"].isin(SELECTED)].copy()
    panel = pd.read_csv(PANEL)["feature"].tolist()
    missing = sorted(set(panel).difference(matrix.columns))
    if missing:
        raise ValueError(f"External matrix misses panel features: {missing[:10]}")
    final45 = set(pd.read_csv(FINAL45)["project"])
    overlap = sorted(final45.intersection(SELECTED))
    if overlap:
        raise ValueError(f"External panel overlaps development projects: {overlap}")
    observed_projects = sorted(matrix["project"].unique())
    if observed_projects != sorted(SELECTED):
        raise ValueError(
            f"Expected selected projects, found {observed_projects}"
        )

    sample_meta = matrix[
        [
            "project",
            "base",
            "crop",
            "sample",
            "all_reads",
            "bacteria_reads",
            "organelle_reads",
            "genus_reads",
            "family_reads",
            "unassigned_reads",
            "n_asv",
        ]
    ].copy()
    sample_meta.to_csv(OUT / "external_sample_meta.csv", index=False)
    matrix.to_csv(OUT / "external_feature_matrix.csv", index=False)

    covariate_rows = []
    for project, group in sample_meta.groupby("project", sort=True):
        bacteria = group["bacteria_reads"].astype(float)
        covariate_rows.append(
            {
                "project": project,
                "crop": SELECTED[project],
                "n_samples": len(group),
                "depth_median": float(bacteria.median()),
                "depth_mean": float(bacteria.mean()),
                "asv_len_mode": asv_length_mode(project),
                "n_asv_median": float(group["n_asv"].median()),
                "unassigned_pct": float(
                    100
                    * group["unassigned_reads"].sum()
                    / max(group["bacteria_reads"].sum(), 1)
                ),
                "genus_pct": float(
                    100
                    * group["genus_reads"].sum()
                    / max(group["bacteria_reads"].sum(), 1)
                ),
                "family_pct": float(
                    100
                    * group["family_reads"].sum()
                    / max(group["bacteria_reads"].sum(), 1)
                ),
            }
        )
    study = pd.DataFrame(covariate_rows)
    study.to_csv(OUT / "external_study_covariates.csv", index=False)
    platform_map = {}
    for row in pd.read_csv(PLATFORMS).itertuples(index=False):
        for project in str(row.projects).split(";"):
            platform_map[project] = row.platform_cluster
    qc = pd.read_csv(POST_QC).set_index("project")
    panel_projects = study.copy()
    panel_projects["platform_cluster"] = panel_projects["project"].map(
        platform_map
    )
    panel_projects["post_dada2_qc"] = panel_projects["project"].map(
        qc["status"]
    )
    panel_projects["genus_read_rate_pct"] = panel_projects["project"].map(
        qc["genus_read_rate_pct"]
    )
    panel_projects[
        [
            "project",
            "crop",
            "platform_cluster",
            "n_samples",
            "post_dada2_qc",
            "genus_read_rate_pct",
        ]
    ].to_csv(OUT / "external_panel_projects.csv", index=False)

    corpus = OUT / "da_corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    manifest = []
    for project_index, project in enumerate(sorted(SELECTED)):
        project_rows = matrix[matrix["project"].eq(project)].reset_index(drop=True)
        metadata = assign_conditions(
            project_rows["sample"].astype(str).tolist(),
            project,
            SELECTED[project],
            seed=23000 + project_index,
        )
        project_rows = project_rows.set_index("sample").loc[
            metadata["sample"]
        ].reset_index()
        counts = project_rows[panel].to_numpy(dtype=np.int64)
        metadata["feature_depth"] = counts.sum(axis=1)
        for scenario_index, scenario in enumerate(
            ["no_effect", "sparse20_fc2"]
        ):
            targets = (
                []
                if scenario == "no_effect"
                else choose_targets(
                    counts,
                    panel,
                    25000 + 10 * project_index + scenario_index,
                )
            )
            directions = np.where(
                np.arange(len(targets)) % 2 == 0, 1.0, -1.0
            )
            effects = pd.DataFrame(
                {
                    "scenario": scenario,
                    "feature": targets,
                    "requested_log2_effect": directions,
                }
            )
            simulated, target_truth, full_truth = simulate_project_effect(
                counts,
                panel,
                metadata,
                effects,
                scenario,
                5000.0,
                24000 + 100 * project_index + scenario_index,
            )
            run_dir = corpus / project / scenario
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                simulated.T,
                index=panel,
                columns=metadata["sample"].tolist(),
            ).to_csv(run_dir / "counts.csv")
            metadata.to_csv(run_dir / "metadata.csv", index=False)
            full_truth.to_csv(run_dir / "truth.csv", index=False)
            target_truth.to_csv(run_dir / "target_truth.csv", index=False)
            manifest.append(
                {
                    "project": project,
                    "crop": SELECTED[project],
                    "scenario": scenario,
                    "requested_absolute_log2_effect": (
                        0.0 if scenario == "no_effect" else 1.0
                    ),
                    "n_targets": len(targets),
                    "n_samples": len(metadata),
                    "n_features": len(panel),
                    "input_dir": str(run_dir.resolve()),
                    "seed": 24000 + 100 * project_index + scenario_index,
                }
            )
    pd.DataFrame(manifest).to_csv(corpus / "manifest.csv", index=False)
    summary = {
        "projects": sorted(SELECTED),
        "n_projects": len(SELECTED),
        "n_samples": int(len(matrix)),
        "crops": pd.Series(SELECTED).value_counts().to_dict(),
        "development_overlap": overlap,
        "panel_features": len(panel),
        "datasets": len(manifest),
    }
    (OUT / "external_panel_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
