#!/usr/bin/env python3
"""Freeze and validate scenario-specific core simulator envelopes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from p4_calibrate_mvp import TOLERANCES


NULL_STATISTICS = [
    "prevalence_distribution",
    "rank_abundance",
    "sample_depth",
    "within_between_variance_ratio",
    "bray_curtis_distribution",
    "aitchison_distribution",
    "zero_fraction",
    "network_topology",
]
INVARIANT_STATISTICS = [
    "sample_depth",
    "within_between_variance_ratio",
    "bray_curtis_distribution",
]
SCENARIO_TARGET_STATISTICS = [
    "prevalence_distribution",
    "rank_abundance",
    "zero_fraction",
    "aitchison_distribution",
    "network_topology",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-dir",
        default="outputs/core_smoke_latent",
    )
    parser.add_argument(
        "--validation-dir",
        default="outputs/core_smoke_latent_validation",
    )
    parser.add_argument(
        "--target-file",
        default=(
            "outputs/core_occupancy/"
            "core_scenario_targets.json"
        ),
    )
    parser.add_argument(
        "--report-file",
        default=(
            "outputs/core_occupancy/"
            "core_target_validation_report.md"
        ),
    )
    return parser.parse_args()


def build_targets(calibration: pd.DataFrame) -> dict[str, object]:
    targets: dict[str, object] = {
        "version": "core_scenario_targets_v1",
        "definition": (
            "Scenario-target envelopes are frozen from independent calibration "
            "seeds before validation and before any core-method output."
        ),
        "calibration_seeds": sorted(
            int(seed) for seed in calibration["seed"].unique()
        ),
        "invariant_statistics": {
            name: TOLERANCES[name] for name in INVARIANT_STATISTICS
        },
        "scenarios": {},
    }
    for scenario in sorted(calibration["scenario"].unique()):
        scenario_rows = calibration[calibration["scenario"].eq(scenario)]
        metric_targets = {}
        for name in SCENARIO_TARGET_STATISTICS:
            values = scenario_rows[name].astype(float)
            center = float(values.mean())
            observed_range = float(values.max() - values.min())
            half_width = max(1.5 * observed_range, 0.05)
            metric_targets[name] = {
                "center": center,
                "calibration_min": float(values.min()),
                "calibration_max": float(values.max()),
                "envelope_low": max(0.0, center - half_width),
                "envelope_high": center + half_width,
            }
        targets["scenarios"][scenario] = metric_targets
    return targets


def validate(
    frame: pd.DataFrame,
    targets: dict[str, object],
) -> pd.DataFrame:
    rows = []
    for scenario, group in frame.groupby("scenario"):
        passed = True
        failed = []
        if scenario == "core_null":
            for name in NULL_STATISTICS:
                metric_pass = float(group[name].mean()) <= TOLERANCES[name]
                if not metric_pass:
                    failed.append(name)
                    passed = False
        else:
            for name in INVARIANT_STATISTICS:
                metric_pass = float(group[name].mean()) <= TOLERANCES[name]
                if not metric_pass:
                    failed.append(f"invariant:{name}")
                    passed = False
            scenario_targets = targets["scenarios"][scenario]
            for name in SCENARIO_TARGET_STATISTICS:
                bounds = scenario_targets[name]
                metric_pass = (
                    bounds["envelope_low"]
                    <= float(group[name].mean())
                    <= bounds["envelope_high"]
                )
                if not metric_pass:
                    failed.append(f"target:{name}")
                    passed = False
        rows.append(
            {
                "scenario": scenario,
                "validation_seeds": int(group["seed"].nunique()),
                "pass": passed,
                "failed_checks": ";".join(failed),
                **{
                    name: float(group[name].mean())
                    for name in NULL_STATISTICS
                },
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    calibration = pd.read_csv(
        Path(args.calibration_dir) / "core_calibration_distances.csv"
    )
    targets = build_targets(calibration)
    target_path = Path(args.target_file)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(targets, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    validation_path = (
        Path(args.validation_dir) / "core_calibration_distances.csv"
    )
    if not validation_path.exists():
        print(f"Targets frozen at {target_path}; validation data not found")
        return
    validation = pd.read_csv(validation_path)
    checked = validate(validation, targets)
    output_dir = target_path.parent
    checked.to_csv(output_dir / "core_target_validation_by_seed.csv", index=False)
    summary = (
        checked.groupby("scenario")
        .agg(
            validation_seeds=("validation_seeds", "max"),
            pass_gate=("pass", "min"),
            failed_checks=("failed_checks", "first"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "core_target_validation_summary.csv", index=False)
    report_lines = [
        "# Paper 4 core scenario-target validation",
        "",
        f"- Calibration seeds: {targets['calibration_seeds']}",
        f"- Validation seeds: {sorted(validation['seed'].unique().tolist())}",
        "",
        "| Scenario | Validation seeds | Gate passed | Failed checks |",
        "|---|---:|---|---|",
    ]
    report_lines.extend(
        f"| {row.scenario} | {int(row.validation_seeds)} | "
        f"{'yes' if row.pass_gate else 'no'} | {row.failed_checks or '—'} |"
        for row in summary.itertuples(index=False)
    )
    report_lines.extend(
        [
            "",
            "Non-null gates combine nuisance-invariant tolerances with "
            "pre-frozen scenario-target envelopes.",
            "",
        ]
    )
    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
