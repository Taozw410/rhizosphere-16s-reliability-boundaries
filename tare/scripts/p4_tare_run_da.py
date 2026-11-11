#!/usr/bin/env python3
"""Run four DA methods on the TARE project-held-out corpus and score truth."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ["aldex2", "ancombc2", "linda", "maaslin2"]


def dataset_complete(run_dir: Path) -> bool:
    if not all(
        (run_dir / f"{method}.csv").exists()
        and (run_dir / f"{method}.json").exists()
        for method in METHODS
    ):
        return False
    for method in METHODS:
        try:
            with (run_dir / f"{method}.json").open(
                "r", encoding="utf-8"
            ) as handle:
                if json.load(handle).get("status") != "ok":
                    return False
        except (OSError, json.JSONDecodeError):
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="tare/artifacts/da_corpus")
    parser.add_argument("--out-dir", default="tare/artifacts/da_method_results")
    parser.add_argument(
        "--runner",
        default="tare/scripts/p4_tare_batch_runner.R",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--conda",
        default=os.environ.get("TARE_CONDA", "conda"),
    )
    parser.add_argument(
        "--conda-env",
        default=os.environ.get("TARE_CONDA_ENV", "riz16s"),
    )
    parser.add_argument("--rlib", default=str(Path.cwd() / "Rlib"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_one_dataset(
    conda: str,
    conda_env: str,
    runner: str,
    rlib: str,
    input_dir: Path,
    run_dir: Path,
    seed: int,
    scenario: str,
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["R_LIBS_USER"] = rlib
    command = [
        conda,
        "run",
        "-p",
        conda_env,
        "Rscript",
        runner,
        str(input_dir),
        str(run_dir.resolve()),
        str(seed),
        scenario,
    ]
    completed = subprocess.run(
        command,
        cwd=str(Path.cwd()),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return {
            "input_dir": str(input_dir),
            "status": "process_failed",
            "returncode": completed.returncode,
            "stderr": completed.stderr[-2000:],
        }
    return {
        "input_dir": str(input_dir),
        "status": "completed",
        "returncode": 0,
    }


def score_result(input_dir: Path, method: str, result_file: Path) -> dict[str, object]:
    truth = pd.read_csv(input_dir / "truth.csv")
    result = pd.read_csv(result_file)
    if len(result) == 0 or result["status"].fillna("failed").ne("ok").all():
        return {
            "status": "failed",
            "n_reject": 0,
            "fdp": np.nan,
            "power": np.nan,
            "f1": np.nan,
            "false_positive_rate": np.nan,
        }
    merged = truth.merge(
        result[["feature", "reject", "direction", "padj"]],
        on="feature",
        how="left",
    )
    merged["reject"] = merged["reject"].fillna(False).astype(bool)
    truth_positive = merged["is_target"].astype(bool)
    rejected = merged["reject"]
    true_positive = int((truth_positive & rejected).sum())
    false_positive = int((~truth_positive & rejected).sum())
    false_negative = int((truth_positive & ~rejected).sum())
    true_negative = int((~truth_positive & ~rejected).sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else np.nan
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else np.nan
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision)
        and np.isfinite(recall)
        and precision + recall > 0
        else (0.0 if true_positive + false_negative else np.nan)
    )
    fdp = (
        false_positive / (false_positive + true_positive)
        if false_positive + true_positive
        else 0.0
    )
    false_positive_rate = (
        false_positive / (false_positive + true_negative)
        if false_positive + true_negative
        else np.nan
    )
    return {
        "status": "ok",
        "n_features": int(len(merged)),
        "n_target": int(truth_positive.sum()),
        "n_reject": int(rejected.sum()),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fdp": fdp,
        "false_positive_rate": false_positive_rate,
    }


def main() -> None:
    args = parse_args()
    corpus = Path(args.corpus)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(corpus / "manifest.csv")
    tasks = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for row in manifest.itertuples(index=False):
            input_dir = Path(row.input_dir)
            run_dir = out_dir / row.project / row.scenario
            complete = dataset_complete(run_dir)
            if args.resume and complete:
                continue
            tasks.append(
                executor.submit(
                    run_one_dataset,
                    args.conda,
                    args.conda_env,
                    str(Path(args.runner).resolve()),
                    args.rlib,
                    input_dir,
                    run_dir,
                    int(row.seed),
                    str(row.scenario),
                )
            )
        failures = []
        for completed_count, future in enumerate(as_completed(tasks), start=1):
            outcome = future.result()
            if outcome["status"] != "completed":
                failures.append(outcome)
            if completed_count % 20 == 0 or completed_count == len(tasks):
                print(
                    f"Completed {completed_count}/{len(tasks)} datasets; "
                    f"failures={len(failures)}",
                    flush=True,
                )
    if failures:
        with (out_dir / "process_failures.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(failures, handle, indent=2)

    metric_rows = []
    status_rows = []
    for row in manifest.itertuples(index=False):
        input_dir = Path(row.input_dir)
        for method in METHODS:
            result_file = out_dir / row.project / row.scenario / f"{method}.csv"
            metadata_file = out_dir / row.project / row.scenario / f"{method}.json"
            elapsed = np.nan
            method_version = ""
            process_status = "missing"
            if result_file.exists() and metadata_file.exists():
                try:
                    with metadata_file.open("r", encoding="utf-8") as handle:
                        metadata = json.load(handle)
                    elapsed = float(metadata.get("elapsed_sec", np.nan))
                    method_version = str(metadata.get("method_version", ""))
                    process_status = str(metadata.get("status", "unknown"))
                except Exception as error:  # noqa: BLE001 - audit status
                    process_status = f"metadata_error:{error}"
            if result_file.exists():
                scored = score_result(input_dir, method, result_file)
            else:
                scored = {
                    "status": "missing",
                    "n_reject": np.nan,
                    "fdp": np.nan,
                    "power": np.nan,
                    "f1": np.nan,
                    "false_positive_rate": np.nan,
                }
            metric_rows.append(
                {
                    "project": row.project,
                    "crop": row.crop,
                    "scenario": row.scenario,
                    "method": method,
                    **scored,
                }
            )
            status_rows.append(
                {
                    "project": row.project,
                    "scenario": row.scenario,
                    "method": method,
                    "process_status": process_status,
                    "elapsed_sec": elapsed,
                    "method_version": method_version,
                }
            )
    pd.DataFrame(metric_rows).to_csv(
        out_dir / "da_project_metrics.csv", index=False
    )
    pd.DataFrame(status_rows).to_csv(
        out_dir / "da_run_status.csv", index=False
    )
    print(f"Wrote {len(metric_rows)} project-method metric rows")


if __name__ == "__main__":
    main()
