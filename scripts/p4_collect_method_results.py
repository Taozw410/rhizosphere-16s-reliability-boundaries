#!/usr/bin/env python3
"""Collect unified DA outputs and score them against feature truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out-prefix", required=True)
    return parser.parse_args()


def adjust_pvalues(pvalues: np.ndarray, method: str) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(pvalues, np.nan)
    valid = np.isfinite(pvalues)
    p = pvalues[valid]
    if len(p) == 0:
        return adjusted
    order = np.argsort(p)
    ranked = p[order]
    n = len(p)
    factors = n / np.arange(1, n + 1)
    if method == "by":
        factors *= np.sum(1.0 / np.arange(1, n + 1))
    raw = ranked * factors
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    q = np.empty_like(monotone)
    q[order] = np.minimum(monotone, 1.0)
    adjusted[valid] = q
    return adjusted


def main() -> None:
    args = parse_args()
    result_dir = Path(args.results_dir)
    truth = pd.read_csv(args.truth)
    truth = truth[truth["scenario"].eq(args.scenario)].copy()
    if truth.empty:
        raise ValueError("Truth table contains no rows for the scenario")

    frames = []
    run_rows = []
    for path in sorted(result_dir.glob("*.csv")):
        if path.name.startswith(("method_", "unified_")):
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frames.append(frame)
        metadata_path = path.with_suffix(".json")
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run_rows.append(
            {
                "method": frame["method"].iloc[0],
                "result_file": path.name,
                "status": metadata.get("status", "unknown"),
                "elapsed_sec": metadata.get("elapsed_sec", np.nan),
                "method_version": metadata.get("method_version", ""),
                "n_features_output": metadata.get("n_features_output", np.nan),
                "n_reject": metadata.get("n_reject", np.nan),
                "error": metadata.get("error", ""),
            }
        )
    if not frames:
        raise ValueError("No method result CSVs found")

    results = pd.concat(frames, ignore_index=True)
    truth_columns = [
        "feature",
        "is_target",
        "requested_log2_effect",
        "case_control_observational_log2",
    ]
    truth_columns = [column for column in truth_columns if column in truth.columns]
    results = results.merge(
        truth[truth_columns],
        on="feature",
        how="inner",
        validate="many_to_one",
    )
    results["method_reject"] = results["reject"].fillna(False).astype(bool)
    results["padj_bh"] = np.nan
    results["padj_by"] = np.nan
    for method, group in results.groupby("method"):
        results.loc[group.index, "padj_bh"] = adjust_pvalues(
            group["pvalue"].to_numpy(), "bh"
        )
        results.loc[group.index, "padj_by"] = adjust_pvalues(
            group["pvalue"].to_numpy(), "by"
        )
    results["reject_bh"] = results["padj_bh"] < 0.05
    results["reject_by"] = results["padj_by"] < 0.05
    results["direction_correct"] = np.where(
        results["is_target"] & results["reject_bh"],
        np.sign(results["estimate"]) == np.sign(
            results["requested_log2_effect"]
        ),
        np.nan,
    )

    metric_rows = []
    for method, group in results.groupby("method"):
        ok = group[group["status"].eq("ok")].copy()
        target = ok["is_target"].fillna(False).astype(bool)
        reject = ok["reject_bh"].fillna(False).astype(bool)
        reject_by = ok["reject_by"].fillna(False).astype(bool)
        tp = int((target & reject).sum())
        fp = int((~target & reject).sum())
        fn = int((target & ~reject).sum())
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision)
            and np.isfinite(recall)
            and precision + recall > 0
            else np.nan
        )
        fdp = fp / (tp + fp) if tp + fp else np.nan
        metric_rows.append(
            {
                "method": method,
                "status": "failed"
                if (group["status"] == "failed").any()
                else "ok",
                "n_features": len(ok),
                "n_target": int(target.sum()),
                "n_reject_bh": int(reject.sum()),
                "tp_bh": tp,
                "fp_bh": fp,
                "fn_bh": fn,
                "fdp_bh": fdp,
                "power_bh": recall,
                "precision_bh": precision,
                "f1_bh": f1,
                "direction_accuracy_bh": float(
                    ok.loc[target & reject, "direction_correct"].mean()
                )
                if (target & reject).any()
                else np.nan,
                "n_reject_by": int(reject_by.sum()),
                "tp_by": int((target & reject_by).sum()),
                "fp_by": int((~target & reject_by).sum()),
                "fdp_by": (
                    int((~target & reject_by).sum())
                    / int(reject_by.sum())
                    if reject_by.sum()
                    else np.nan
                ),
            }
        )

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(f"{prefix}_long.csv", index=False)
    metrics = pd.DataFrame(metric_rows).sort_values("method")
    metrics.to_csv(f"{prefix}_metrics.csv", index=False)
    manifest = pd.DataFrame(run_rows).sort_values("method")
    manifest.to_csv(f"{prefix}_manifest.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
