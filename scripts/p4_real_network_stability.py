#!/usr/bin/env python3
"""Cross-panel stability of network estimates on the real 45-project data."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from p4_network_benchmark import METHODS
from p4_real_project_holdout import make_panels
from p4_sim_common import load_analysis_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-matrix",
        default="out_analysis/feature_matrix.csv",
    )
    parser.add_argument(
        "--design",
        default="outputs/effect_interface/sample_design.csv",
    )
    parser.add_argument(
        "--sample-meta",
        default="out_analysis/sample_meta.csv",
    )
    parser.add_argument(
        "--panel",
        default=(
            "outputs/formal_grid/data/"
            "sparse20_fc2/feature_panel.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/real_network_stability",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_analysis_data(Path(args.feature_matrix), min_depth=1000)
    design = pd.read_csv(args.design, dtype={"sample": str}).set_index("sample")
    meta = design.loc[data.meta["sample"]].reset_index()
    sample_meta = pd.read_csv(args.sample_meta)
    panels = make_panels(sample_meta)
    panel_features = pd.read_csv(args.panel)["feature"].tolist()
    feature_index = {
        name: index for index, name in enumerate(data.feature_names)
    }
    feature_indices = np.array(
        [feature_index[name] for name in panel_features],
        dtype=int,
    )
    counts = data.counts[:, feature_indices]
    scores: dict[tuple[str, str], np.ndarray] = {}
    for panel, projects in panels.items():
        rows = meta["project"].isin(projects).to_numpy()
        for method_id, method in METHODS.items():
            scores[(panel, method_id)] = method(counts[rows])
    rows = []
    for method_id in METHODS:
        for left, right in itertools.combinations(panels, 2):
            left_score = scores[(left, method_id)]
            right_score = scores[(right, method_id)]
            upper = np.triu_indices_from(left_score, k=1)
            left_threshold = np.quantile(left_score[upper], 0.90)
            right_threshold = np.quantile(right_score[upper], 0.90)
            n_nodes = left_score.shape[0]
            left_edges = {
                (left, right)
                for left in range(n_nodes)
                for right in range(left + 1, n_nodes)
                if left_score[left, right] >= left_threshold
            }
            right_edges = {
                (left, right)
                for left in range(n_nodes)
                for right in range(left + 1, n_nodes)
                if right_score[left, right] >= right_threshold
            }
            jaccard = len(left_edges & right_edges) / max(
                len(left_edges | right_edges),
                1,
            )
            left_degree = left_score.sum(axis=1)
            right_degree = right_score.sum(axis=1)
            hub = float(spearmanr(left_degree, right_degree).statistic)
            rows.append(
                {
                    "method_id": method_id,
                    "panel_a": left,
                    "panel_b": right,
                    "edge_jaccard": jaccard,
                    "hub_spearman": hub,
                    "mean_edge_similarity": float(
                        np.corrcoef(
                            left_score[upper],
                            right_score[upper],
                        )[0, 1]
                    ),
                }
            )
    result = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "real_network_panel_stability.csv", index=False)
    summary = (
        result.groupby("method_id")
        .agg(
            edge_jaccard_mean=("edge_jaccard", "mean"),
            hub_spearman_mean=("hub_spearman", "mean"),
            edge_similarity_mean=("mean_edge_similarity", "mean"),
        )
        .reset_index()
        .sort_values("edge_jaccard_mean", ascending=False)
    )
    summary.to_csv(out_dir / "real_network_stability_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
