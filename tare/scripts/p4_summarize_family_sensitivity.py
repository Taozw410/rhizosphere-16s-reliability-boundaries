#!/usr/bin/env python3
"""Summarize the family-level taxonomic sensitivity benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


TARE_DIR = Path(__file__).resolve().parents[1]
MIXED = TARE_DIR / "artifacts" / "da_method_results" / "da_project_metrics.csv"
FAMILY = (
    TARE_DIR
    / "artifacts"
    / "family_sensitivity"
    / "da_method_results"
    / "da_project_metrics.csv"
)
OUT = TARE_DIR / "artifacts" / "family_sensitivity"


def summarize(frame: pd.DataFrame, panel_name: str) -> pd.DataFrame:
    summary = (
        frame.groupby("method")
        .agg(
            datasets=("project", "size"),
            mean_rejections=("n_reject", "mean"),
            mean_fdp=("fdp", "mean"),
            mean_f1=("f1", "mean"),
        )
        .reset_index()
    )
    summary["panel"] = panel_name
    summary["f1_rank"] = summary["mean_f1"].rank(
        ascending=False, method="min"
    )
    return summary


def safe_spearman(left: pd.Series, right: pd.Series) -> float:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def main() -> None:
    mixed = pd.read_csv(MIXED)
    mixed = mixed[mixed["scenario"].eq("sparse20_fc2")].copy()
    family = pd.read_csv(FAMILY).copy()
    comparison = pd.concat(
        [summarize(mixed, "mixed"), summarize(family, "family")],
        ignore_index=True,
    )
    comparison.to_csv(OUT / "family_vs_mixed_summary.csv", index=False)

    paired = mixed[["project", "method", "f1", "fdp"]].merge(
        family[["project", "method", "f1", "fdp"]],
        on=["project", "method"],
        suffixes=("_mixed", "_family"),
    )
    correlations = []
    for method, group in paired.groupby("method"):
        correlations.append(
            {
                "method": method,
                "projects": int(len(group)),
                "f1_spearman": safe_spearman(
                    group["f1_mixed"], group["f1_family"]
                ),
                "fdp_spearman": safe_spearman(
                    group["fdp_mixed"].fillna(0),
                    group["fdp_family"].fillna(0),
                ),
            }
        )
    correlations = pd.DataFrame(correlations)
    correlations.to_csv(OUT / "family_vs_mixed_correlations.csv", index=False)
    summary = {
        "mixed_f1_order": (
            comparison[comparison["panel"].eq("mixed")]
            .sort_values("f1_rank")["method"]
            .tolist()
        ),
        "family_f1_order": (
            comparison[comparison["panel"].eq("family")]
            .sort_values("f1_rank")["method"]
            .tolist()
        ),
        "method_order_changed": (
            comparison[comparison["panel"].eq("mixed")]
            .sort_values("f1_rank")["method"]
            .tolist()
            != comparison[comparison["panel"].eq("family")]
            .sort_values("f1_rank")["method"]
            .tolist()
        ),
    }
    (OUT / "family_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(comparison.to_string(index=False))
    print(correlations.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
