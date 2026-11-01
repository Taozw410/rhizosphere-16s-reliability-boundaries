#!/usr/bin/env python3
"""Check key structural summaries after removing the unassigned bin."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from p4_calibrate_mvp import TOLERANCES, distance_profile
from p4_sim_common import compute_summary_components, load_analysis_data


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
OUT = PAPER / "outputs/phase1/unassigned_sensitivity.csv"


def main() -> None:
    data = load_analysis_data(ROOT / "out_analysis/feature_matrix.csv", 1000)
    feature_names = np.asarray(data.feature_names)
    keep = feature_names != "unassigned"
    real_excluding = compute_summary_components(
        data.counts[:, keep],
        data.meta,
        feature_names[keep].tolist(),
    )

    synthetic_path = PAPER / "outputs/phase1/synthetic_counts_seed1000.npz"
    with np.load(synthetic_path, allow_pickle=True) as payload:
        synthetic_counts = payload["counts"]
        synthetic_names = payload["feature_names"].astype(str)
    if synthetic_counts.shape != data.counts.shape:
        raise ValueError("Synthetic and observed matrices are not aligned")
    if not np.array_equal(synthetic_names, feature_names):
        raise ValueError("Synthetic and observed feature names differ")

    synthetic_meta = data.meta.copy()
    synthetic_excluding = compute_summary_components(
        synthetic_counts[:, keep],
        synthetic_meta,
        feature_names[keep].tolist(),
    )
    distances = distance_profile(real_excluding, synthetic_excluding)
    rows = [
        {
            "statistic": name,
            "distance_without_unassigned": distances[name],
            "tolerance": TOLERANCES[name],
            "pass": distances[name] <= TOLERANCES[name],
        }
        for name in TOLERANCES
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(OUT)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
