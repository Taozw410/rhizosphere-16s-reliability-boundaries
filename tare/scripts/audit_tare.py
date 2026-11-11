#!/usr/bin/env python3
"""Audit TARE corpus completeness, holdout integrity and reproducibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = root / "artifacts"
    corpus = pd.read_csv(artifact / "da_corpus" / "manifest.csv")
    metrics = pd.read_csv(
        artifact / "da_method_results" / "da_project_metrics.csv"
    )
    status = pd.read_csv(
        artifact / "da_method_results" / "da_run_status.csv"
    )
    predictions = pd.read_csv(
        artifact / "tare_da" / "tare_da_predictions.csv"
    )
    overall = pd.read_csv(
        artifact / "tare_da" / "tare_da_overall.csv"
    ).iloc[0]
    hashes = pd.read_csv(
        artifact / "repro_check" / "reproducibility_hashes.csv"
    )
    model_path = artifact / "tare_da" / "tare_da_final_model.joblib"
    model = joblib.load(model_path)
    external_dir = artifact / "external_panel"
    external_matrix = pd.read_csv(external_dir / "external_feature_matrix.csv")
    external_predictions = pd.read_csv(
        external_dir
        / "tare_external_validation"
        / "tare_external_predictions.csv"
    )
    external_metrics = pd.read_csv(
        external_dir / "da_method_results" / "da_project_metrics.csv"
    )
    family_metrics = pd.read_csv(
        artifact
        / "family_sensitivity"
        / "da_method_results"
        / "da_project_metrics.csv"
    )
    family_summary = json.loads(
        (
            artifact
            / "family_sensitivity"
            / "family_sensitivity_summary.json"
        ).read_text(encoding="utf-8")
    )
    external_overall = pd.read_csv(
        external_dir
        / "tare_external_validation"
        / "tare_external_overall.csv"
    ).iloc[0]
    external_manifest = json.loads(
        (external_dir / "external_panel_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    external_model_manifest = json.loads(
        (
            external_dir
            / "tare_external_validation"
            / "tare_external_manifest.json"
        ).read_text(encoding="utf-8")
    )
    development_projects = set(
        pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "project_manifest" / "final45_selection.csv")[
            "project"
        ]
    )
    external_projects = set(external_predictions["project"].unique())

    checks = {
        "corpus_rows_90": len(corpus) == 90,
        "corpus_projects_45": corpus["project"].nunique() == 45,
        "corpus_two_scenarios_per_project": bool(
            (
                corpus.groupby("project")["scenario"].nunique() == 2
            ).all()
        ),
        "metric_rows_360": len(metrics) == 360,
        "status_rows_360": len(status) == 360,
        "prediction_rows_3600": len(predictions) == 3600,
        "partition_count_10": predictions["partition"].nunique() == 10,
        "each_partition_contains_each_project_once_per_method_scenario": bool(
            (
                predictions.groupby(
                    ["partition", "scenario", "method", "project"]
                ).size()
                == 1
            ).all()
        ),
        "tare_mae_below_baseline": bool(
            overall["tare_mae"] < overall["baseline_mae"]
        ),
        "coverage_at_least_0_80": bool(overall["coverage"] >= 0.80),
        "allow_precision_above_0_80": bool(
            overall["allow_precision"] > 0.80
        ),
        "all_reproducibility_hashes_identical": bool(
            hashes["identical"].all()
        ),
        "final_model_loaded": bool(model.get("model_name") == "TARE-DA"),
        "external_six_projects": len(external_projects) == 6,
        "external_two_per_crop": bool(
            (
                external_predictions[
                    ["project", "crop"]
                ].drop_duplicates().groupby("crop").size()
                == 2
            ).all()
        ),
        "external_zero_development_overlap": not bool(
            external_projects.intersection(development_projects)
        ),
        "external_426_samples": len(external_matrix) == 426,
        "external_48_predictions": len(external_predictions) == 48,
        "external_all_method_rows_ok": bool(
            external_metrics["status"].eq("ok").all()
            and len(external_metrics) == 48
        ),
        "external_model_not_retrained": (
            external_model_manifest["model_sha256"]
            == sha256(model_path)
            and external_model_manifest["retrained_on_external"] is False
        ),
        "external_mae_below_development_prior": bool(
            external_overall["tare_mae"]
            < external_overall["development_prior_mae"]
        ),
        "external_coverage_at_least_0_80": bool(
            external_overall["coverage"] >= 0.80
        ),
        "external_ood_fraction_recorded": bool(
            external_overall["outside_q95_fraction"] == 1.0
        ),
        "family_sensitivity_180_outcomes": len(family_metrics) == 180,
        "family_sensitivity_order_changed": bool(
            family_summary["method_order_changed"]
        ),
        "application_example_present": (
            artifact
            / "application_example"
            / "project_application_summary.csv"
        ).exists(),
        "mismatch_linkage_present": (
            artifact
            / "transport_reliability_link"
            / "mismatch_correlations.csv"
        ).exists(),
        "decision_threshold_curve_present": (
            artifact / "tare_da" / "decision_threshold_curve.csv"
        ).exists(),
    }
    report = {
        "checks": checks,
        "all_pass": all(checks.values()),
        "methods": metrics["method"].value_counts().to_dict(),
        "method_execution_status": (
            status.groupby(["method", "process_status"])
            .size()
            .rename("count")
            .reset_index()
            .to_dict(orient="records")
        ),
        "overall": overall.to_dict(),
        "model_file_sha256": sha256(model_path),
        "external": {
            "projects": external_manifest["projects"],
            "development_overlap": sorted(
                external_projects.intersection(development_projects)
            ),
            "overall": external_overall.to_dict(),
        },
    }
    (artifact / "tare_da" / "tare_audit.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# TARE audit",
        "",
        f"Overall status: {'PASS' if report['all_pass'] else 'FAIL'}",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for name, passed in checks.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "Method execution status:",
            "",
            "| Method | Status | Count |",
            "|---|---|---:|",
        ]
    )
    for row in report["method_execution_status"]:
        lines.append(
            f"| {row['method']} | {row['process_status']} | {row['count']} |"
        )
    (artifact / "tare_da" / "tare_audit.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
