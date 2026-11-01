#!/usr/bin/env python3
"""Summarize the completed Paper 4 core/occupancy formal grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="outputs/core_formal_grid/results",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261027)
    return parser.parse_args()


def bootstrap_ci(values: pd.Series, iterations: int, seed: int) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(clean, size=(iterations, len(clean)), replace=True)
    means = draws.mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def summarize(frame: pd.DataFrame, groups: list[str], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(groups):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ci_low, ci_high = bootstrap_ci(
            group["f1"],
            args.bootstrap_iterations,
            args.bootstrap_seed,
        )
        row = dict(zip(groups, keys, strict=True))
        row.update(
            {
                "n_datasets": group[["stage", "scenario", "seed"]]
                .drop_duplicates()
                .shape[0],
                "mean_f1": group["f1"].mean(),
                "f1_ci_low": ci_low,
                "f1_ci_high": ci_high,
                "mean_recall": group["recall"].mean(),
                "mean_precision": group["precision"].mean(),
                "mean_false_core_rate": group["false_core_rate"].mean(),
                "worst_false_core_rate": group["false_core_rate"].max(),
                "mean_brier": group["brier"].mean(),
                "mean_n_reject": group["n_reject"].mean(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("" if pd.isna(value) else f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    metrics = pd.read_csv(root / "core_metrics_all.csv")
    truth = pd.read_csv(root / "core_truth_all.csv")
    expected_datasets = 5 * 20
    dataset_count = metrics[["stage", "scenario", "seed"]].drop_duplicates().shape[0]
    integrity = {
        "metric_rows": len(metrics),
        "expected_metric_rows": expected_datasets * 4,
        "datasets": dataset_count,
        "expected_datasets": expected_datasets,
        "methods": sorted(metrics["method_id"].unique().tolist()),
        "stages": sorted(metrics["stage"].unique().tolist()),
        "scenarios": sorted(metrics["scenario"].unique().tolist()),
        "truth_rows": len(truth),
    }
    integrity["complete"] = (
        integrity["metric_rows"] == integrity["expected_metric_rows"]
        and integrity["datasets"] == integrity["expected_datasets"]
        and len(integrity["methods"]) == 4
    )

    stage_summary = summarize(metrics, ["stage", "method_id", "method"], args)
    scenario_summary = summarize(
        metrics,
        ["stage", "scenario", "method_id", "method"],
        args,
    )
    combined_summary = summarize(
        metrics,
        ["scenario", "method_id", "method"],
        args,
    )
    ranking = (
        combined_summary[
            ~combined_summary["scenario"].eq("core_null")
        ]
        .groupby(["method_id", "method"])
        .agg(
            mean_f1=("mean_f1", "mean"),
            worst_false_core_rate=("mean_false_core_rate", "max"),
            mean_brier=("mean_brier", "mean"),
        )
        .reset_index()
        .sort_values(
            ["mean_f1", "mean_brier", "worst_false_core_rate"],
            ascending=[False, True, True],
        )
    )
    truth_counts = (
        truth.groupby(["stage", "scenario", "seed"])
        .agg(n_core=("is_population_core", "sum"))
        .reset_index()
        .groupby(["stage", "scenario"])
        .agg(
            truth_core_mean=("n_core", "mean"),
            truth_core_min=("n_core", "min"),
            truth_core_max=("n_core", "max"),
        )
        .reset_index()
    )
    for name, frame in [
        ("core_stage_method_summary.csv", stage_summary),
        ("core_stage_scenario_method_summary.csv", scenario_summary),
        ("core_combined_scenario_method_summary.csv", combined_summary),
        ("core_formal_ranking.csv", ranking),
        ("core_truth_count_summary.csv", truth_counts),
    ]:
        frame.to_csv(root / name, index=False)
    (root / "core_formal_audit.json").write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    stage_view = stage_summary[
        [
            "stage",
            "method",
            "mean_f1",
            "mean_recall",
            "mean_false_core_rate",
            "mean_brier",
        ]
    ]
    scenario_view = combined_summary[
        [
            "scenario",
            "method",
            "mean_f1",
            "mean_recall",
            "mean_false_core_rate",
            "mean_brier",
        ]
    ]
    ranking_view = ranking.copy()
    report = f"""# Paper 4 core/occupancy formal benchmark report

## Integrity

- Datasets: {integrity['datasets']}/{integrity['expected_datasets']}
- Method metric rows: {integrity['metric_rows']}/{integrity['expected_metric_rows']}
- Truth rows: {integrity['truth_rows']}
- Complete: {integrity['complete']}

## Stage-level results

{markdown_table(stage_view, ['stage', 'method', 'mean_f1', 'mean_recall', 'mean_false_core_rate', 'mean_brier'])}

## Combined scenario results

{markdown_table(scenario_view, ['scenario', 'method', 'mean_f1', 'mean_recall', 'mean_false_core_rate', 'mean_brier'])}

## Ranking

{markdown_table(ranking_view, ['method', 'mean_f1', 'worst_false_core_rate', 'mean_brier'])}

## Truth counts

{markdown_table(truth_counts, ['stage', 'scenario', 'truth_core_mean', 'truth_core_min', 'truth_core_max'])}

## Interpretation rule

The ranking is conditional on the frozen simulator family, scenario targets,
feature panel and method thresholds. Depth-confounded performance must be
interpreted as a separate boundary rather than averaged away.
"""
    (root / "core_formal_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(integrity, indent=2, ensure_ascii=False))
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
