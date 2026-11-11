#!/usr/bin/env python3
"""Apply the frozen TARE-DA model to the independent six-project panel."""

from __future__ import annotations

import json
import math
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

TARE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARE_DIR))

from p4_train_tare_da import (
    auc_score,
    build_training_table,
    decision,
    feature_matrix,
    project_structural_features,
)
from p4_sim_common import load_analysis_data


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = TARE_DIR / "artifacts" / "external_panel"
OUT = EXTERNAL / "tare_external_validation"
MODEL_PATH = TARE_DIR / "artifacts" / "tare_da" / "tare_da_final_model.joblib"
PANEL = (
    ROOT
    / "论文4-多研究仿真"
    / "outputs"
    / "formal_grid"
    / "data"
    / "sparse20_fc2"
    / "feature_panel.csv"
)


def aligned_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    features = feature_matrix(frame)
    return features.reindex(columns=feature_columns, fill_value=0.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODEL_PATH)
    feature_columns = model["feature_columns"]

    external_metrics = pd.read_csv(
        EXTERNAL / "da_method_results" / "da_project_metrics.csv"
    )
    external_status = pd.read_csv(
        EXTERNAL / "da_method_results" / "da_run_status.csv"
    )
    external_study = pd.read_csv(
        EXTERNAL / "external_study_covariates.csv"
    ).set_index("project")
    external_matrix = pd.read_csv(EXTERNAL / "external_feature_matrix.csv")
    panel = pd.read_csv(PANEL)["feature"].tolist()
    external_counts = external_matrix[panel].to_numpy(dtype=np.int64)
    external_meta = external_matrix[
        ["project", "crop", "sample"]
    ].copy()
    external_structural = project_structural_features(
        external_counts,
        external_meta,
        panel,
    )
    external_frame = build_training_table(
        external_metrics,
        external_status,
        external_study,
        external_structural,
    )

    dev_metrics = pd.read_csv(
        TARE_DIR / "artifacts" / "da_method_results" / "da_project_metrics.csv"
    )
    dev_status = pd.read_csv(
        TARE_DIR / "artifacts" / "da_method_results" / "da_run_status.csv"
    )
    dev_study = pd.read_csv(
        ROOT / "out_analysis" / "study_covariates.csv"
    ).set_index("project")
    dev_data = load_analysis_data(
        ROOT / "out_analysis" / "feature_matrix.csv", min_depth=1000
    )
    feature_index = {
        name: index for index, name in enumerate(dev_data.feature_names)
    }
    dev_indices = np.array(
        [feature_index[name] for name in panel], dtype=int
    )
    dev_structural = project_structural_features(
        dev_data.counts[:, dev_indices],
        dev_data.meta.copy(),
        panel,
    )
    dev_frame = build_training_table(
        dev_metrics,
        dev_status,
        dev_study,
        dev_structural,
    )
    baseline = (
        dev_frame.groupby(["scenario", "method"])["utility"]
        .mean()
        .rename("development_prior_utility")
        .reset_index()
    )
    external_frame = external_frame.merge(
        baseline, on=["scenario", "method"], how="left"
    )

    external_x = aligned_features(external_frame, feature_columns)
    dev_x = aligned_features(dev_frame, feature_columns)
    external_scaled = model["scaler"].transform(external_x)
    dev_scaled = model["scaler"].transform(dev_x)
    point = np.clip(
        model["regressor"].predict(external_scaled),
        0.0,
        1.0,
    )
    probability = (
        model["classifier"].predict_proba(external_scaled)[:, 1]
        if model["classifier"] is not None
        else np.full(len(external_frame), float(dev_frame["reliable"].mean()))
    )
    radius = float(model["conformal_radius"])
    lower = np.clip(point - radius, 0.0, 1.0)
    upper = np.clip(point + radius, 0.0, 1.0)

    train_pairwise = cdist(dev_scaled, dev_scaled)
    np.fill_diagonal(train_pairwise, np.inf)
    train_nearest = train_pairwise.min(axis=1)
    support_scale = max(float(np.quantile(train_nearest, 0.95)), 1e-6)
    test_nearest = cdist(external_scaled, dev_scaled).min(axis=1)
    support = np.exp(-0.5 * (test_nearest / support_scale) ** 2)

    predictions = external_frame[
        ["project", "crop", "scenario", "method", "utility", "reliable"]
    ].copy()
    predictions["actual_utility"] = predictions.pop("utility")
    predictions["actual_reliable"] = predictions.pop("reliable")
    predictions["development_prior_utility"] = external_frame[
        "development_prior_utility"
    ]
    predictions["tare_point"] = point
    predictions["tare_lower"] = lower
    predictions["tare_upper"] = upper
    predictions["tare_probability"] = probability
    predictions["covered"] = (
        predictions["tare_lower"]
        <= predictions["actual_utility"]
    ) & (
        predictions["actual_utility"]
        <= predictions["tare_upper"]
    )
    predictions["decision"] = [
        decision(probability_value, lower_value)
        for probability_value, lower_value in zip(probability, lower)
    ]
    predictions["training_nearest_distance"] = test_nearest
    predictions["support_score"] = support
    predictions["outside_training_q95"] = test_nearest > support_scale
    predictions["error_tare"] = (
        predictions["tare_point"] - predictions["actual_utility"]
    ).abs()
    predictions["error_prior"] = (
        predictions["development_prior_utility"]
        - predictions["actual_utility"]
    ).abs()
    predictions.to_csv(
        OUT / "tare_external_predictions.csv", index=False
    )

    summary_rows = []
    for (scenario, method), subset in predictions.groupby(
        ["scenario", "method"]
    ):
        allow = subset[subset["decision"].eq("ALLOW")]
        summary_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "projects": subset["project"].nunique(),
                "predictions": len(subset),
                "tare_mae": float(subset["error_tare"].mean()),
                "development_prior_mae": float(
                    subset["error_prior"].mean()
                ),
                "coverage": float(subset["covered"].mean()),
                "reliable_prevalence": float(
                    subset["actual_reliable"].mean()
                ),
                "auroc_reliability": auc_score(
                    subset["actual_reliable"].to_numpy(),
                    subset["tare_probability"].to_numpy(dtype=float),
                ),
                "allow_count": len(allow),
                "allow_precision": float(
                    allow["actual_reliable"].mean()
                )
                if len(allow)
                else math.nan,
                "mean_support": float(subset["support_score"].mean()),
                "outside_q95_fraction": float(
                    subset["outside_training_q95"].mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "tare_external_summary.csv", index=False)
    allow = predictions[predictions["decision"].eq("ALLOW")]
    strong = predictions[predictions["scenario"].eq("sparse20_fc2")]
    overall = pd.DataFrame(
        [
            {
                "projects": predictions["project"].nunique(),
                "samples": int(external_matrix.shape[0]),
                "predictions": len(predictions),
                "tare_mae": float(predictions["error_tare"].mean()),
                "development_prior_mae": float(
                    predictions["error_prior"].mean()
                ),
                "mae_improvement": float(
                    predictions["error_prior"].mean()
                    - predictions["error_tare"].mean()
                ),
                "coverage": float(predictions["covered"].mean()),
                "strong_effect_auroc": auc_score(
                    strong["actual_reliable"].to_numpy(),
                    strong["tare_probability"].to_numpy(dtype=float),
                ),
                "allow_count": len(allow),
                "allow_precision": float(
                    allow["actual_reliable"].mean()
                )
                if len(allow)
                else math.nan,
                "allow_coverage": float(len(allow) / len(predictions)),
                "mean_support": float(predictions["support_score"].mean()),
                "outside_q95_fraction": float(
                    predictions["outside_training_q95"].mean()
                ),
            }
        ]
    )
    overall.to_csv(OUT / "tare_external_overall.csv", index=False)
    manifest = {
        "model_used": str(MODEL_PATH),
        "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "retrained_on_external": False,
        "external_projects": sorted(predictions["project"].unique().tolist()),
        "external_crops": predictions.groupby("crop")["project"].nunique().to_dict(),
        "external_samples": int(external_matrix.shape[0]),
        "predictions": len(predictions),
        "support_scale_q95": support_scale,
        "conformal_radius": radius,
        "protocol": "TARE_EXTERNAL_PANEL_PROTOCOL.md",
    }
    (OUT / "tare_external_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
