#!/usr/bin/env python3
"""Freeze core-method operating thresholds and evaluate held-out seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "|" + "|".join(["---"] * len(frame.columns)) + "|"
    rows = []
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-dir",
        default="outputs/core_method_smoke",
    )
    parser.add_argument(
        "--validation-dir",
        default="outputs/core_method_validation",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/core_occupancy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration_dir = Path(args.calibration_dir)
    validation_dir = Path(args.validation_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration = pd.read_csv(
        calibration_dir / "core_method_metrics_by_seed.csv"
    )
    calibration_summary = (
        calibration.groupby(
            ["method_id", "method", "scenario", "threshold"]
        )
        .agg(
            f1_mean=("f1", "mean"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
            false_core_rate_mean=("false_core_rate", "mean"),
            brier_mean=("brier", "mean"),
        )
        .reset_index()
    )
    non_null = calibration_summary[
        ~calibration_summary["scenario"].eq("core_null")
    ]
    operating_rows = []
    for method_id, group in non_null.groupby("method_id"):
        threshold_rows = []
        for threshold, threshold_group in group.groupby("threshold"):
            max_fcr = float(threshold_group["false_core_rate_mean"].max())
            mean_f1 = float(threshold_group["f1_mean"].mean())
            threshold_rows.append(
                {
                    "method_id": method_id,
                    "method": threshold_group["method"].iloc[0],
                    "threshold": float(threshold),
                    "max_false_core_rate": max_fcr,
                    "mean_f1_non_null": mean_f1,
                    "eligible": max_fcr <= 0.10,
                }
            )
        candidates = [
            row for row in threshold_rows if row["eligible"]
        ]
        if not candidates:
            raise RuntimeError(f"No eligible threshold for {method_id}")
        candidates.sort(
            key=lambda row: (
                -row["mean_f1_non_null"],
                row["threshold"],
            )
        )
        operating_rows.append(candidates[0])
    operating = pd.DataFrame(operating_rows)
    operating.to_csv(out_dir / "core_method_frozen_thresholds.csv", index=False)
    operating_payload = {
        "version": "core_method_thresholds_v1",
        "calibration_seeds": sorted(
            int(seed) for seed in calibration["seed"].unique()
        ),
        "selection_rule": (
            "Across non-null scenarios, require maximum mean false-core rate "
            "<= 0.10, then maximize mean F1 across scenarios."
        ),
        "thresholds": {
            row.method_id: {
                "method": row.method,
                "threshold": row.threshold,
                "max_false_core_rate": row.max_false_core_rate,
                "mean_f1_non_null": row.mean_f1_non_null,
            }
            for row in operating.itertuples(index=False)
        },
    }
    (out_dir / "core_method_thresholds.json").write_text(
        json.dumps(operating_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    validation_path = (
        validation_dir / "core_method_metrics_by_seed.csv"
    )
    if not validation_path.exists():
        print("Thresholds frozen; validation metrics not found")
        return
    validation = pd.read_csv(validation_path)
    selected = validation.merge(
        operating[["method_id", "threshold"]],
        on=["method_id", "threshold"],
        how="inner",
        validate="many_to_one",
    )
    selected_summary = (
        selected.groupby(["scenario", "method_id", "method", "threshold"])
        .agg(
            seeds=("seed", "nunique"),
            f1_mean=("f1", "mean"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
            false_core_rate_mean=("false_core_rate", "mean"),
            brier_mean=("brier", "mean"),
            tp_mean=("tp", "mean"),
            fp_mean=("fp", "mean"),
        )
        .reset_index()
    )
    selected_summary.to_csv(
        out_dir / "core_method_validation_summary.csv",
        index=False,
    )
    selected.to_csv(
        out_dir / "core_method_validation_by_seed.csv",
        index=False,
    )

    nonnull_validation = selected_summary[
        ~selected_summary["scenario"].eq("core_null")
    ]
    ranking = (
        nonnull_validation.groupby(["method_id", "method", "threshold"])
        .agg(
            mean_f1=("f1_mean", "mean"),
            worst_false_core_rate=("false_core_rate_mean", "max"),
            brier_mean=("brier_mean", "mean"),
        )
        .reset_index()
        .sort_values(
            ["mean_f1", "brier_mean", "worst_false_core_rate"],
            ascending=[False, True, True],
        )
    )
    ranking.to_csv(
        out_dir / "core_method_validation_ranking.csv",
        index=False,
    )
    report = [
        "# Paper 4 core/occupancy method smoke and validation",
        "",
        "Thresholds were frozen on calibration seeds before validation.",
        "",
        "## Frozen thresholds",
        "",
        markdown_table(operating[
            [
                "method",
                "threshold",
                "max_false_core_rate",
                "mean_f1_non_null",
            ]
        ]),
        "",
        "## Validation ranking",
        "",
        markdown_table(ranking),
        "",
    ]
    (out_dir / "core_method_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
