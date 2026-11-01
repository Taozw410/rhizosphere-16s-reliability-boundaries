#!/usr/bin/env python3
"""Run and resume the frozen Paper 4 core/occupancy formal grid."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from p4_core_methods import METHOD_NAMES, score_method
from p4_core_sim_common import load_core_context
from p4_core_sim_correlated import estimate_latent_correlation
from p4_core_sim_latent_factor import simulate_latent_factor_core_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid",
        default=(
            "outputs/core_formal_grid/"
            "core_seed_grid.csv"
        ),
    )
    parser.add_argument(
        "--out-root",
        default="outputs/core_formal_grid/results",
    )
    parser.add_argument(
        "--thresholds",
        default=(
            "outputs/core_occupancy/"
            "core_method_thresholds.json"
        ),
    )
    parser.add_argument(
        "--feature-matrix",
        default="out_analysis/feature_matrix.csv",
    )
    parser.add_argument(
        "--design",
        default="outputs/effect_interface/sample_design.csv",
    )
    parser.add_argument(
        "--formal-config",
        default=(
            "outputs/formal_grid/"
            "formal_grid_config.json"
        ),
    )
    parser.add_argument(
        "--panel",
        default=(
            "outputs/formal_grid/data/"
            "sparse20_fc2/feature_panel.csv"
        ),
    )
    parser.add_argument(
        "--stage",
        choices=["primary", "extension", "all"],
        default="primary",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def aggregate_results(out_root: Path) -> None:
    metric_frames = []
    prediction_frames = []
    truth_frames = []
    for path in sorted(out_root.rglob("core_metrics.csv")):
        metric_frames.append(pd.read_csv(path))
    for path in sorted(out_root.rglob("core_predictions.csv.gz")):
        prediction_frames.append(pd.read_csv(path))
    for path in sorted(out_root.rglob("truth.csv")):
        truth_frames.append(pd.read_csv(path))
    if metric_frames:
        pd.concat(metric_frames, ignore_index=True).to_csv(
            out_root / "core_metrics_all.csv",
            index=False,
        )
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            out_root / "core_predictions_all.csv.gz",
            index=False,
            compression="gzip",
        )
    if truth_frames:
        pd.concat(truth_frames, ignore_index=True).to_csv(
            out_root / "core_truth_all.csv",
            index=False,
        )


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    context = load_core_context(
        Path(args.feature_matrix),
        Path(args.design),
        Path(args.formal_config),
        Path(args.panel),
    )
    correlation = estimate_latent_correlation(context)
    with Path(args.thresholds).open("r", encoding="utf-8") as handle:
        threshold_payload = json.load(handle)["thresholds"]

    grid = pd.read_csv(args.grid)
    if args.stage != "all":
        grid = grid[grid["stage"].eq(args.stage)].copy()
    grid = grid.sort_values(["scenario", "seed"]).reset_index(drop=True)
    if args.limit > 0:
        grid = grid.head(args.limit).copy()
    metric_frames = []
    prediction_frames = []
    truth_frames = []
    manifest_rows = []
    for row in grid.itertuples(index=False):
        run_dir = out_root / row.stage / row.scenario / f"seed_{row.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "core_metrics.csv"
        if metrics_path.exists():
            log(f"skip completed {row.stage} {row.scenario} seed={row.seed}")
            metric_frames.append(pd.read_csv(metrics_path))
            truth_frames.append(pd.read_csv(run_dir / "truth.csv"))
            prediction_path = run_dir / "core_predictions.csv.gz"
            if prediction_path.exists():
                prediction_frames.append(pd.read_csv(prediction_path))
            continue

        started = time.time()
        log(f"simulate {row.stage} {row.scenario} seed={row.seed}")
        simulation = simulate_latent_factor_core_scenario(
            context,
            row.scenario,
            int(row.seed),
            correlation,
        )
        np.savez_compressed(
            run_dir / "counts.npz",
            counts=simulation.counts.astype(np.int32),
            feature_names=np.array(context.feature_names, dtype=str),
            sample=np.array(context.meta["sample"], dtype=str),
            project=np.array(context.meta["project"], dtype=str),
            detection_probability=simulation.detection_probability.astype(
                np.float32
            ),
        )
        context.meta.to_csv(run_dir / "metadata.csv", index=False)
        truth = simulation.truth.copy()
        truth["stage"] = row.stage
        truth["seed"] = int(row.seed)
        truth.to_csv(run_dir / "truth.csv", index=False)
        simulation.project_occupancy.to_csv(
            run_dir / "project_latent_occupancy.csv"
        )

        run_metrics = []
        run_predictions = []
        for method_id, method_name in METHOD_NAMES.items():
            log(
                f"method {row.scenario} seed={row.seed} {method_id}"
            )
            scores, diagnostics = score_method(
                method_id,
                simulation.counts,
                context.meta,
            )
            threshold = float(
                threshold_payload[method_id]["threshold"]
            )
            target = truth["is_population_core"].astype(bool).to_numpy()
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
            run_metrics.append(
                {
                    "stage": row.stage,
                    "scenario": row.scenario,
                    "seed": int(row.seed),
                    "method_id": method_id,
                    "method": method_name,
                    "threshold": threshold,
                    "n_target": int(target.sum()),
                    "n_reject": int(predicted.sum()),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "false_core_rate": (
                        fp / (fp + tn) if fp + tn else np.nan
                    ),
                    "brier": float(
                        np.mean((np.clip(scores, 0, 1) - target) ** 2)
                    ),
                }
            )
            run_predictions.append(
                pd.DataFrame(
                    {
                        "stage": row.stage,
                        "scenario": row.scenario,
                        "seed": int(row.seed),
                        "feature": truth["feature"],
                        "method_id": method_id,
                        "score": scores,
                        "predicted_core": predicted,
                        "is_population_core": target,
                    }
                )
            )
            if not diagnostics.empty:
                diagnostics.to_csv(
                    run_dir / f"diagnostics_{method_id}.csv",
                    index=False,
                )
        metrics = pd.DataFrame(run_metrics)
        metrics.to_csv(metrics_path, index=False)
        pd.concat(run_predictions, ignore_index=True).to_csv(
            run_dir / "core_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
        metric_frames.append(metrics)
        prediction_frames.append(pd.concat(run_predictions, ignore_index=True))
        truth_frames.append(truth)
        elapsed = time.time() - started
        manifest_rows.append(
            {
                "stage": row.stage,
                "scenario": row.scenario,
                "seed": int(row.seed),
                "elapsed_sec": elapsed,
                "status": "ok",
            }
        )
        log(
            f"finish {row.stage} {row.scenario} seed={row.seed} "
            f"elapsed={elapsed:.1f}s"
        )

    aggregate_results(out_root)
    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(
            out_root / "core_manifest_all.csv",
            index=False,
        )
    log("core formal grid stage complete")


if __name__ == "__main__":
    main()
