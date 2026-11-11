#!/usr/bin/env python3
"""Development-only screen of TARE-DA point heads.

This file was added after TARE-v1 failed its frozen point-accuracy target. It
is retained to make model development transparent; its outer split overlaps
the initial development run and is not a confirmatory holdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

TARE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARE_DIR))

from p4_train_tare_da import (
    build_training_table,
    feature_matrix,
    project_structural_features,
)
from p4_sim_common import load_analysis_data


def main() -> None:
    metrics = pd.read_csv("tare/artifacts/da_method_results/da_project_metrics.csv")
    status = pd.read_csv("tare/artifacts/da_method_results/da_run_status.csv")
    study = pd.read_csv("data/tare_inputs/study_covariates.csv").set_index("project")
    data = load_analysis_data(Path("data/tare_inputs/feature_matrix.csv"), min_depth=1000)
    panel = pd.read_csv(
        "data/tare_inputs/feature_panel.csv"
    )["feature"].tolist()
    index = {name: position for position, name in enumerate(data.feature_names)}
    columns = np.array([index[name] for name in panel], dtype=int)
    structural = project_structural_features(
        data.counts[:, columns], data.meta.copy(), panel
    )
    frame = build_training_table(metrics, status, study, structural)
    features = feature_matrix(frame)
    projects = np.array(sorted(frame["project"].unique()))
    models = {
        "ridge_1": lambda seed: Ridge(alpha=1.0),
        "ridge_10": lambda seed: Ridge(alpha=10.0),
        "ridge_100": lambda seed: Ridge(alpha=100.0),
        "random_forest": lambda seed: RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=12,
            max_features=0.7,
            random_state=seed,
            n_jobs=1,
        ),
        "extra_trees": lambda seed: ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=12,
            max_features=0.7,
            random_state=seed,
            n_jobs=1,
        ),
        "shallow_gradient_boosting": lambda seed: GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=15,
            subsample=0.8,
            random_state=seed,
        ),
        "hist_gradient_boosting": lambda seed: HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=220,
            max_leaf_nodes=12,
            min_samples_leaf=12,
            l2_regularization=2.0,
            random_state=seed,
        ),
    }
    rows = []
    for partition in range(10):
        shuffled = np.random.default_rng(14000 + partition).permutation(projects)
        fold_map = {
            project: fold_index % 5
            for fold_index, project in enumerate(shuffled)
        }
        fold_ids = frame["project"].map(fold_map).to_numpy()
        for fold in range(5):
            train = fold_ids != fold
            test = ~train
            scaler = StandardScaler()
            x_train = scaler.fit_transform(features.loc[train])
            x_test = scaler.transform(features.loc[test])
            y_train = frame.loc[train, "utility"].to_numpy(dtype=float)
            y_test = frame.loc[test, "utility"].to_numpy(dtype=float)
            baseline = frame.loc[train].groupby(
                ["scenario", "method"]
            )["utility"].mean()
            baseline_values = (
                frame.loc[test]
                .set_index(["scenario", "method"])
                .index.map(baseline)
                .to_numpy(dtype=float)
            )
            rows.append(
                {
                    "partition": partition,
                    "fold": fold,
                    "model": "training_mean",
                    "mae": float(np.mean(np.abs(baseline_values - y_test))),
                }
            )
            for name, factory in models.items():
                estimator = factory(17000 + 100 * partition + fold)
                estimator.fit(x_train, y_train)
                prediction = np.clip(estimator.predict(x_test), 0.0, 1.0)
                rows.append(
                    {
                        "partition": partition,
                        "fold": fold,
                        "model": name,
                        "mae": float(np.mean(np.abs(prediction - y_test))),
                    }
                )
    result = pd.DataFrame(rows)
    summary = (
        result.groupby("model")
        .agg(mean_mae=("mae", "mean"), sd_mae=("mae", "std"))
        .sort_values("mean_mae")
        .reset_index()
    )
    output = Path("tare/artifacts/model_screen")
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "model_screen_by_fold.csv", index=False)
    summary.to_csv(output / "model_screen_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
