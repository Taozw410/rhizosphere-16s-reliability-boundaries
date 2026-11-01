#!/usr/bin/env python3
"""Compare differential-abundance rankings across two generator families."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def resolve(path: Path) -> Path:
    if path.exists():
        return path
    nested = path.parent / "results" / path.name
    if nested.exists():
        return nested
    raise FileNotFoundError(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator1",
        default="outputs/formal_grid/formal_metrics_all.csv",
    )
    parser.add_argument(
        "--generator2",
        default=(
            "outputs/formal_grid_generator2/"
            "results/formal_metrics_all.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/da_cross_generator",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    first = pd.read_csv(resolve(Path(args.generator1)))
    second = pd.read_csv(resolve(Path(args.generator2)))
    first["generator"] = "template_multinomial"
    second["generator"] = "correlated_perturbation"
    combined = pd.concat([first, second], ignore_index=True)
    summary = (
        combined.groupby(["generator", "scenario", "method"])
        .agg(
            seeds=("seed", "nunique"),
            rejects=("n_reject_bh", "mean"),
            fdp=("fdp_bh", "mean"),
            power=("power_bh", "mean"),
            f1=("f1_bh", "mean"),
        )
        .reset_index()
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "da_cross_generator_summary.csv", index=False)
    ranking = (
        summary[~summary["scenario"].eq("no_effect")]
        .groupby(["generator", "method"])
        .agg(
            mean_fdp=("fdp", "mean"),
            mean_power=("power", "mean"),
            mean_f1=("f1", "mean"),
        )
        .reset_index()
        .sort_values(
            ["generator", "mean_fdp", "mean_power"],
            ascending=[True, True, False],
        )
    )
    ranking.to_csv(out_dir / "da_cross_generator_ranking.csv", index=False)
    pivot = summary.pivot_table(
        index=["scenario", "method"],
        columns="generator",
        values=["fdp", "power", "f1"],
    )
    pivot.to_csv(out_dir / "da_cross_generator_pivot.csv")
    report = [
        "# Differential-abundance cross-generator report",
        "",
        "## Generator-level ranking",
        "",
        ranking.to_csv(index=False).strip(),
        "",
        "## Interpretation boundary",
        "",
        "- Generator 1 uses the calibrated sample-template multinomial.",
        "- Generator 2 uses a correlated multiplicative perturbation of the same",
        "  template. It is a semi-parametric second family, not a fully",
        "  independent ecological generator.",
        "- Seed counts differ and absolute FDP levels must not be pooled across",
        "  generators.",
        "",
    ]
    (out_dir / "da_cross_generator_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
