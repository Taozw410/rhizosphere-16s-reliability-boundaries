#!/usr/bin/env python3
"""Build an illustrative project-level DA and TARE decision example."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT = "PRJNA554261"
SCENARIO = "sparse20_fc2"
METHODS = {
    "aldex2": "ALDEx2",
    "ancombc2": "ANCOM-BC2",
    "linda": "LinDA",
    "maaslin2": "MaAsLin2",
}
OUT = ROOT / "artifacts" / "application_example"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result_dir = (
        ROOT
        / "artifacts"
        / "da_method_results"
        / PROJECT
        / SCENARIO
    )
    metrics = pd.read_csv(
        ROOT / "artifacts" / "da_method_results" / "da_project_metrics.csv"
    )
    metrics = metrics[
        metrics["project"].eq(PROJECT)
        & metrics["scenario"].eq(SCENARIO)
    ]
    predictions = pd.read_csv(
        ROOT / "artifacts" / "tare_da" / "tare_da_predictions.csv"
    )
    predictions = predictions[
        predictions["project"].eq(PROJECT)
        & predictions["scenario"].eq(SCENARIO)
    ]
    rejected_sets = {}
    rows = []
    for method, display in METHODS.items():
        result = pd.read_csv(result_dir / f"{method}.csv")
        rejected = set(
            result.loc[
                result["reject"].fillna(False).astype(bool), "feature"
            ].astype(str)
        )
        rejected_sets[method] = rejected
        metric = metrics[metrics["method"].eq(method)].iloc[0]
        prediction = predictions[predictions["method"].eq(method)]
        decisions = sorted(prediction["decision"].unique().tolist())
        rows.append(
            {
                "project": PROJECT,
                "method_id": method,
                "method": display,
                "rejections": len(rejected),
                "fdp": metric["fdp"],
                "f1": metric["f1"],
                "tare_probability_mean": prediction[
                    "tare_probability"
                ].mean(),
                "tare_lower_mean": prediction["tare_lower"].mean(),
                "decisions_across_10_partitions": ";".join(decisions),
                "rejected_features": ";".join(sorted(rejected)),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "project_application_summary.csv", index=False)
    ancom = rejected_sets["ancombc2"]
    maslin = rejected_sets["maaslin2"]
    union = set().union(*rejected_sets.values())
    intersection = set.intersection(*rejected_sets.values())
    summary = {
        "project": PROJECT,
        "scenario": SCENARIO,
        "interpretation": (
            "Real project composition template with a simulator-defined "
            "strong sparse effect; not a real treatment-effect validation."
        ),
        "methods_with_rejections": sum(
            bool(values) for values in rejected_sets.values()
        ),
        "union_rejections": len(union),
        "intersection_all_methods": len(intersection),
        "ancombc2_rejections": len(ancom),
        "ancombc2_unique_rejections": len(
            ancom
            - set().union(
                rejected_sets["aldex2"],
                rejected_sets["linda"],
                maslin,
            )
        ),
        "ancombc2_maaslin2_overlap": len(ancom.intersection(maslin)),
    }
    (OUT / "project_application_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(table[
        [
            "method",
            "rejections",
            "fdp",
            "f1",
            "tare_probability_mean",
            "tare_lower_mean",
            "decisions_across_10_partitions",
        ]
    ].to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
