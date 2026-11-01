#!/usr/bin/env python3
"""Run and score the Paper 4 core/occupancy method smoke benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from p4_core_methods import METHOD_NAMES, score_method


THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="outputs/core_smoke_latent",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/core_method_smoke",
    )
    return parser.parse_args()


def load_run(run_dir: Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    archive = np.load(run_dir / "counts.npz", allow_pickle=True)
    counts = archive["counts"].astype(np.int64)
    meta = pd.read_csv(run_dir / "metadata.csv", dtype={"sample": str})
    truth = pd.read_csv(run_dir / "truth.csv")
    return counts, meta, truth


def metric_rows(
    method_id: str,
    scenario: str,
    seed: int,
    scores: np.ndarray,
    truth: pd.DataFrame,
) -> list[dict[str, object]]:
    target = truth["is_population_core"].astype(str).str.upper().eq("TRUE").to_numpy()
    rows = []
    for threshold in THRESHOLDS:
        predicted = scores >= threshold
        tp = int((target & predicted).sum())
        fp = int((~target & predicted).sum())
        fn = int((target & ~predicted).sum())
        tn = int((~target & ~predicted).sum())
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision)
            and np.isfinite(recall)
            and precision + recall > 0
            else np.nan
        )
        false_core_rate = fp / (fp + tn) if fp + tn else np.nan
        probability = np.clip(scores, 0, 1)
        brier = float(np.mean((probability - target.astype(float)) ** 2))
        rows.append(
            {
                "method_id": method_id,
                "method": METHOD_NAMES[method_id],
                "scenario": scenario,
                "seed": seed,
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_core_rate": false_core_rate,
                "brier": brier,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(data_dir / "core_seed_grid.csv")
    rows = []
    prediction_rows = []
    for row in grid.itertuples(index=False):
        run_dir = Path(row.truth_file).parent
        counts, meta, truth = load_run(run_dir)
        for method_id in METHOD_NAMES:
            scores, diagnostics = score_method(method_id, counts, meta)
            rows.extend(
                metric_rows(
                    method_id,
                    row.scenario,
                    int(row.seed),
                    scores,
                    truth,
                )
            )
            prediction = pd.DataFrame(
                {
                    "feature": truth["feature"],
                    "scenario": row.scenario,
                    "seed": int(row.seed),
                    "method_id": method_id,
                    "score": scores,
                    "is_population_core": truth["is_population_core"],
                }
            )
            prediction_rows.append(prediction)
            if not diagnostics.empty:
                diagnostics = diagnostics.copy()
                diagnostics["scenario"] = row.scenario
                diagnostics["seed"] = int(row.seed)
                diagnostics["method_id"] = method_id
                diagnostics.to_csv(
                    out_dir
                    / f"diagnostics_{method_id}_{row.scenario}_{row.seed}.csv",
                    index=False,
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "core_method_metrics_by_seed.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        out_dir / "core_method_predictions_long.csv.gz",
        index=False,
        compression="gzip",
    )
    summary = (
        metrics.groupby(["scenario", "method_id", "method", "threshold"])
        .agg(
            seeds=("seed", "nunique"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            false_core_rate_mean=("false_core_rate", "mean"),
            brier_mean=("brier", "mean"),
            tp_mean=("tp", "mean"),
            fp_mean=("fp", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "core_method_summary.csv", index=False)
    print(summary.sort_values(["scenario", "method_id", "threshold"]).to_string(index=False))


if __name__ == "__main__":
    main()
