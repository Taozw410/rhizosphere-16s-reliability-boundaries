#!/usr/bin/env python3
"""Train TARE-DA and validate it on held-out projects.

TARE-DA uses target-project design and community descriptors, method identity
and effect-scenario severity to predict a project-specific utility interval and
the probability that a DA method meets its error-control/power criterion.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

PROJECT_SCRIPTS = (
    Path(__file__).resolve().parents[2] / "scripts"
)
sys.path.insert(0, str(PROJECT_SCRIPTS))

from p4_sim_common import (
    compute_summary_components,
    load_analysis_data,
    network_project_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="tare/artifacts/da_method_results/da_project_metrics.csv")
    parser.add_argument("--status", default="tare/artifacts/da_method_results/da_run_status.csv")
    parser.add_argument("--feature-matrix", default="data/tare_inputs/feature_matrix.csv")
    parser.add_argument(
        "--study-covariates",
        default="data/tare_inputs/study_covariates.csv",
    )
    parser.add_argument(
        "--panel",
        default=(
            "data/tare_inputs/feature_panel.csv"
        ),
    )
    parser.add_argument("--out-dir", default="tare/artifacts/tare_da")
    parser.add_argument("--partitions", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    return parser.parse_args()


def project_structural_features(
    counts: np.ndarray,
    meta: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    rows = []
    for project in sorted(meta["project"].unique()):
        index = np.flatnonzero(meta["project"].eq(project).to_numpy())
        subset = counts[index]
        components = compute_summary_components(
            subset,
            meta.iloc[index].reset_index(drop=True),
            feature_names,
        )
        network = network_project_stats(
            subset,
            meta.iloc[index].reset_index(drop=True),
            feature_names,
        )
        rows.append(
            {
                "project": project,
                "tare_n_samples": len(index),
                "tare_depth_log_median": float(
                    np.median(np.log1p(subset.sum(axis=1)))
                ),
                "tare_prevalence_mean": float(
                    np.mean(components["prevalence"])
                ),
                "tare_zero_fraction": float(
                    components["global_zero_fraction"]
                ),
                "tare_variance_ratio": float(components["variance_ratio"]),
                "tare_bray_median": float(np.median(components["bray"])),
                "tare_aitchison_median": float(
                    np.median(components["aitchison"])
                ),
                "tare_network_density_median": float(
                    network["edge_density"].median()
                )
                if len(network)
                else np.nan,
                "tare_network_centrality_median": float(
                    network["centrality_cv"].median()
                )
                if len(network)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("project")


def build_training_table(
    metrics: pd.DataFrame,
    status: pd.DataFrame,
    study: pd.DataFrame,
    structural: pd.DataFrame,
) -> pd.DataFrame:
    frame = metrics.merge(
        status[
            [
                "project",
                "scenario",
                "method",
                "process_status",
                "elapsed_sec",
                "method_version",
            ]
        ],
        on=["project", "scenario", "method"],
        how="left",
    )
    project_table = pd.concat(
        [study.drop(columns=["crop"], errors="ignore"), structural],
        axis=1,
    )
    frame = frame.join(project_table, on="project")
    frame["scenario_severity"] = frame["scenario"].map(
        {"no_effect": 0.0, "sparse20_fc2": 1.0}
    )
    frame["method_failure"] = (
        frame["status"].ne("ok").astype(bool)
        | frame["process_status"].ne("ok").astype(bool)
    )
    frame["fdp_safe"] = frame["fdp"].fillna(1.0).le(0.10)
    frame["strong_reliable"] = (
        frame["scenario"].eq("sparse20_fc2")
        & frame["fdp_safe"]
        & frame["f1"].fillna(0.0).ge(0.50)
    )
    frame["null_reliable"] = (
        frame["scenario"].eq("no_effect")
        & frame["false_positive_rate"].fillna(1.0).le(0.10)
    )
    frame["reliable"] = (
        frame["strong_reliable"] | frame["null_reliable"]
    ).astype(int)
    frame["utility"] = np.where(
        frame["scenario"].eq("no_effect"),
        1.0 - frame["false_positive_rate"].fillna(1.0),
        frame["f1"].fillna(0.0),
    )
    frame.loc[frame["method_failure"], "utility"] = 0.0
    frame["utility"] = frame["utility"].clip(0.0, 1.0)
    return frame


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    number_columns = [
        "n_samples",
        "depth_median",
        "asv_len_mode",
        "n_asv_median",
        "unassigned_pct",
        "genus_pct",
        "family_pct",
        "tare_n_samples",
        "tare_depth_log_median",
        "tare_prevalence_mean",
        "tare_zero_fraction",
        "tare_variance_ratio",
        "tare_bray_median",
        "tare_aitchison_median",
        "tare_network_density_median",
        "tare_network_centrality_median",
        "scenario_severity",
    ]
    features = frame[number_columns].astype(float).copy()
    features["log_n_samples"] = np.log1p(features["n_samples"])
    features["log_depth_median"] = np.log1p(features["depth_median"])
    features["log_n_asv_median"] = np.log1p(features["n_asv_median"])
    features = features.drop(
        columns=["n_samples", "depth_median", "n_asv_median"]
    )
    features = pd.concat(
        [
            features,
            pd.get_dummies(frame["crop"], prefix="crop", dtype=float),
            pd.get_dummies(frame["method"], prefix="method", dtype=float),
        ],
        axis=1,
    )
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.loc[:, features.notna().any(axis=0)]
    features = features.fillna(features.median(numeric_only=True))
    features = features.loc[:, features.nunique(dropna=False) > 1]
    return features


def make_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=12,
        min_samples_leaf=12,
        l2_regularization=2.0,
        random_state=seed,
    )


def make_classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=8,
        min_samples_leaf=15,
        l2_regularization=2.0,
        random_state=seed,
    )


def fit_predict_tare(
    training: pd.DataFrame,
    testing: pd.DataFrame,
    features: pd.DataFrame,
    seed: int,
    inner_folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = features.loc[training.index]
    test_x = features.loc[testing.index]
    group = training["project"].to_numpy()
    n_splits = min(inner_folds, len(np.unique(group)))
    splitter = GroupKFold(n_splits=n_splits)
    oof = np.full(len(training), np.nan)
    for train_index, valid_index in splitter.split(train_x, groups=group):
        model = make_regressor(seed + int(train_index.sum() % 1000))
        model.fit(
            train_x.iloc[train_index].to_numpy(dtype=float),
            training["utility"].iloc[train_index].to_numpy(dtype=float),
        )
        oof[valid_index] = model.predict(
            train_x.iloc[valid_index].to_numpy(dtype=float)
        )
    residual = np.abs(
        training["utility"].to_numpy(dtype=float) - oof
    )
    conformal_q = float(
        np.quantile(residual, min(0.975, (len(residual) + 1) / len(residual) * 0.95))
    )
    regressor = make_regressor(seed + 17)
    regressor.fit(
        train_x.to_numpy(dtype=float),
        training["utility"].to_numpy(dtype=float),
    )
    point = np.clip(
        regressor.predict(test_x.to_numpy(dtype=float)), 0.0, 1.0
    )
    lower = np.clip(point - conformal_q, 0.0, 1.0)
    upper = np.clip(point + conformal_q, 0.0, 1.0)

    labels = training["reliable"].to_numpy(dtype=int)
    if labels.min() == labels.max():
        probability = np.full(len(testing), float(labels.mean()))
    else:
        classifier = make_classifier(seed + 29)
        classifier.fit(
            train_x.to_numpy(dtype=float),
            labels,
        )
        probability = classifier.predict_proba(
            test_x.to_numpy(dtype=float)
        )[:, 1]
    return point, lower, upper, probability


def decision(probability: float, lower: float) -> str:
    if probability >= 0.80 and lower >= 0.50:
        return "ALLOW"
    if probability < 0.30 or lower < 0.20:
        return "DENY"
    return "CONDITIONAL"


def fit_final_artifact(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    inner_folds: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    scaler = StandardScaler()
    x = scaler.fit_transform(features)
    y = frame["utility"].to_numpy(dtype=float)
    group = frame["project"].to_numpy()
    splitter = GroupKFold(n_splits=min(inner_folds, len(np.unique(group))))
    oof = np.full(len(frame), np.nan)
    for train_index, valid_index in splitter.split(x, groups=group):
        model = make_regressor(18001 + int(train_index.sum() % 1000))
        model.fit(x[train_index], y[train_index])
        oof[valid_index] = model.predict(x[valid_index])
    residual = np.abs(y - oof)
    conformal_q = float(
        np.quantile(
            residual,
            min(0.975, (len(residual) + 1) / len(residual) * 0.95),
        )
    )
    regressor = make_regressor(18002)
    regressor.fit(x, y)
    labels = frame["reliable"].to_numpy(dtype=int)
    if labels.min() == labels.max():
        classifier = None
        probability = np.full(len(frame), float(labels.mean()))
    else:
        classifier = make_classifier(18003)
        classifier.fit(x, labels)
        probability = classifier.predict_proba(x)[:, 1]
    point = np.clip(regressor.predict(x), 0.0, 1.0)
    lower = np.clip(point - conformal_q, 0.0, 1.0)
    upper = np.clip(point + conformal_q, 0.0, 1.0)
    envelope = frame[
        ["project", "crop", "scenario", "method", "utility", "reliable"]
    ].copy()
    envelope["tare_point"] = point
    envelope["tare_lower"] = lower
    envelope["tare_upper"] = upper
    envelope["tare_probability"] = probability
    envelope["decision"] = [
        decision(probability_value, lower_value)
        for probability_value, lower_value in zip(probability, lower)
    ]
    artifact = {
        "model_name": "TARE-DA",
        "feature_columns": features.columns.tolist(),
        "scaler": scaler,
        "regressor": regressor,
        "classifier": classifier,
        "conformal_radius": conformal_q,
        "decision_rule": {
            "ALLOW": "probability >= 0.80 and lower >= 0.50",
            "DENY": "probability < 0.30 or lower < 0.20",
            "else": "CONDITIONAL",
        },
    }
    return artifact, envelope


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = labels.sum()
    negatives = (~labels).sum()
    if positives == 0 or negatives == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.metrics)
    status = pd.read_csv(args.status)
    study = pd.read_csv(args.study_covariates).set_index("project")
    data = load_analysis_data(Path(args.feature_matrix), min_depth=1000)
    panel = pd.read_csv(args.panel)["feature"].tolist()
    index = {name: position for position, name in enumerate(data.feature_names)}
    columns = np.array([index[name] for name in panel], dtype=int)
    counts = data.counts[:, columns]
    meta = data.meta.copy()
    structural = project_structural_features(counts, meta, panel)
    frame = build_training_table(metrics, status, study, structural)
    frame = frame.reset_index(drop=True)
    features = feature_matrix(frame)
    projects = np.array(sorted(frame["project"].unique()))
    prediction_rows = []
    for partition in range(args.partitions):
        splitter = GroupKFold(n_splits=args.folds)
        shuffled = np.random.default_rng(14000 + partition).permutation(projects)
        group_for_fold = {
            project: fold_index % args.folds
            for fold_index, project in enumerate(shuffled)
        }
        fold_ids = frame["project"].map(group_for_fold).to_numpy()
        for fold in range(args.folds):
            train_mask = fold_ids != fold
            test_mask = ~train_mask
            training = frame.loc[train_mask].reset_index(drop=True)
            testing = frame.loc[test_mask].reset_index(drop=True)
            full_features = feature_matrix(pd.concat([training, testing], ignore_index=True))
            full_features.index = pd.RangeIndex(len(full_features))
            training_feature_index = np.arange(len(training))
            testing_feature_index = np.arange(
                len(training), len(training) + len(testing)
            )
            # Refit scaling for TARE point/classification heads.
            scaler = StandardScaler()
            scaler.fit(full_features.iloc[training_feature_index])
            testing_for_model = testing.copy()
            testing_for_model.index = testing_feature_index
            point, lower, upper, probability = fit_predict_tare(
                training,
                testing_for_model,
                pd.DataFrame(
                    scaler.transform(full_features),
                    index=full_features.index,
                    columns=full_features.columns,
                ),
                seed=15000 + 100 * partition + fold,
                inner_folds=args.inner_folds,
            )
            baseline = (
                training.groupby(["scenario", "method"])["utility"]
                .mean()
                .rename("baseline_utility")
                .reset_index()
            )
            testing = testing.merge(
                baseline, on=["scenario", "method"], how="left"
            )
            for offset, row in testing.iterrows():
                prediction_rows.append(
                    {
                        "partition": partition,
                        "fold": fold,
                        "project": row["project"],
                        "crop": row["crop"],
                        "scenario": row["scenario"],
                        "method": row["method"],
                        "actual_utility": row["utility"],
                        "actual_reliable": row["reliable"],
                        "baseline_utility": row["baseline_utility"],
                        "tare_point": point[offset],
                        "tare_lower": lower[offset],
                        "tare_upper": upper[offset],
                        "tare_probability": probability[offset],
                        "covered": bool(
                            lower[offset] <= row["utility"] <= upper[offset]
                        ),
                        "decision": decision(probability[offset], lower[offset]),
                    }
                )
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(out_dir / "tare_da_predictions.csv", index=False)
    summary_rows = []
    for (scenario, method), subset in predictions.groupby(
        ["scenario", "method"]
    ):
        tare_mae = float(
            np.mean(np.abs(subset["tare_point"] - subset["actual_utility"]))
        )
        baseline_mae = float(
            np.mean(
                np.abs(subset["baseline_utility"] - subset["actual_utility"])
            )
        )
        allow = subset[subset["decision"].eq("ALLOW")]
        summary_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "predictions": len(subset),
                "tare_mae": tare_mae,
                "baseline_mae": baseline_mae,
                "coverage": float(subset["covered"].mean()),
                "reliable_prevalence": float(subset["actual_reliable"].mean()),
                "auroc_reliability": auc_score(
                    subset["actual_reliable"].to_numpy(),
                    subset["tare_probability"].to_numpy(dtype=float),
                ),
                "allow_count": len(allow),
                "allow_precision": float(allow["actual_reliable"].mean())
                if len(allow)
                else math.nan,
                "allow_coverage": float(len(allow) / len(subset)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "tare_da_summary.csv", index=False)
    overall = pd.DataFrame(
        [
            {
                "predictions": len(predictions),
                "tare_mae": float(
                    np.mean(
                        np.abs(
                            predictions["tare_point"]
                            - predictions["actual_utility"]
                        )
                    )
                ),
                "baseline_mae": float(
                    np.mean(
                        np.abs(
                            predictions["baseline_utility"]
                            - predictions["actual_utility"]
                        )
                    )
                ),
                "coverage": float(predictions["covered"].mean()),
                "auroc_reliability": auc_score(
                    predictions["actual_reliable"].to_numpy(),
                    predictions["tare_probability"].to_numpy(dtype=float),
                ),
                "allow_count": int(
                    predictions["decision"].eq("ALLOW").sum()
                ),
                "allow_precision": float(
                    predictions.loc[
                        predictions["decision"].eq("ALLOW"), "actual_reliable"
                    ].mean()
                ),
                "allow_coverage": float(
                    predictions["decision"].eq("ALLOW").mean()
                ),
            }
        ]
    )
    overall.to_csv(out_dir / "tare_da_overall.csv", index=False)
    final_artifact, final_envelope = fit_final_artifact(
        frame, features, args.inner_folds
    )
    joblib.dump(final_artifact, out_dir / "tare_da_final_model.joblib")
    final_envelope.to_csv(
        out_dir / "tare_da_project_envelopes.csv", index=False
    )
    manifest = {
        "model": "TARE-DA hierarchical gradient-boosted reliability envelope",
        "partitions": args.partitions,
        "outer_folds": args.folds,
        "inner_folds": args.inner_folds,
        "projects": int(frame["project"].nunique()),
        "methods": sorted(frame["method"].unique().tolist()),
        "scenarios": sorted(frame["scenario"].unique().tolist()),
        "features": features.columns.tolist(),
        "decision_rule": {
            "ALLOW": "P(reliable) >= 0.80 and lower 95% conformal bound >= 0.50",
            "DENY": "P(reliable) < 0.30 or lower bound < 0.20",
            "else": "CONDITIONAL",
        },
        "leakage_control": (
            "All rows from a held-out project are excluded from fitting and "
            "conformal calibration."
        ),
        "final_model": "tare_da_final_model.joblib",
        "conformal_radius": final_artifact["conformal_radius"],
    }
    with (out_dir / "tare_da_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)
    print(summary.to_string(index=False))
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
