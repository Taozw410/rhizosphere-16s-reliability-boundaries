#!/usr/bin/env python3
"""Summarize internal TARE risk-acceptance tradeoffs across decision cutoffs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "artifacts" / "tare_da" / "tare_da_predictions.csv"
OUT = ROOT / "artifacts" / "tare_da" / "decision_threshold_curve.csv"


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS)
    reliable_total = int(predictions["actual_reliable"].sum())
    rows = []
    for probability_cutoff in [0.50, 0.60, 0.70, 0.80, 0.90]:
        for lower_cutoff in [0.20, 0.30, 0.40, 0.50]:
            allowed = predictions[
                predictions["tare_probability"].ge(probability_cutoff)
                & predictions["tare_lower"].ge(lower_cutoff)
            ]
            true_allowed = int(allowed["actual_reliable"].sum())
            rows.append(
                {
                    "probability_cutoff": probability_cutoff,
                    "lower_bound_cutoff": lower_cutoff,
                    "allow_count": len(allowed),
                    "allow_fraction": len(allowed) / len(predictions),
                    "allow_precision": (
                        true_allowed / len(allowed) if len(allowed) else None
                    ),
                    "false_allow_count": len(allowed) - true_allowed,
                    "reliable_recall": (
                        true_allowed / reliable_total if reliable_total else None
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
