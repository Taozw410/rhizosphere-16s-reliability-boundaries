#!/usr/bin/env python3
"""Run and resume the frozen formal Paper 4 DA benchmark grid."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd


METHODS = ["aldex2", "ancombc2", "linda", "maaslin2"]
METHOD_TIMEOUT = {
    "aldex2": 600,
    "ancombc2": 1800,
    "linda": 600,
    "maaslin2": 600,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid",
        default="outputs/formal_grid/seed_grid.csv",
    )
    parser.add_argument(
        "--out-root",
        default="outputs/formal_grid/results",
    )
    parser.add_argument(
        "--work-root",
        default="output_work/p4_formal_grid",
    )
    parser.add_argument("--stage", choices=["primary", "extension", "all"], default="primary")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=METHODS,
    )
    parser.add_argument(
        "--conda",
        default=r"D:\ProgramData\miniconda3\Scripts\conda.exe",
    )
    parser.add_argument("--renv", default=r"D:\bioenv\envs\riz16s")
    parser.add_argument(
        "--rscript",
        default=r"D:\bioenv\envs\riz16s\Scripts\Rscript.exe",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
    )
    parser.add_argument("--root", default=str(Path.cwd()))
    return parser.parse_args()


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def run_command(command: list[str], log_path: Path, timeout: int | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\nCOMMAND: " + " ".join(command) + "\n")
        handle.flush()
        try:
            completed = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            handle.write(f"\nEXIT_CODE: {completed.returncode}\n")
            return int(completed.returncode)
        except subprocess.TimeoutExpired:
            handle.write(f"\nTIMEOUT after {timeout} seconds\n")
            return 124


def write_failure(method: str, csv: Path, json_path: Path, error: str) -> None:
    pd.DataFrame(
        [
            {
                "method": method,
                "feature": None,
                "estimate": None,
                "statistic": None,
                "pvalue": None,
                "padj": None,
                "reject": False,
                "direction": None,
                "method_flag": None,
                "status": "failed",
                "error": error,
            }
        ]
    ).to_csv(csv, index=False)
    json_path.write_text(
        json.dumps({"method": method, "status": "failed", "error": error}, indent=2),
        encoding="utf-8",
    )


def dataset_ready(result_dir: Path, methods: list[str]) -> bool:
    path = result_dir / "method_unified_metrics.csv"
    if not path.exists():
        return False
    metrics = pd.read_csv(path)
    return set(metrics["method"]) == set(methods) and metrics["status"].eq("ok").all()


def prepare_input(
    row: pd.Series,
    args: argparse.Namespace,
    work_dir: Path,
) -> Path:
    input_dir = work_dir / "input"
    if (input_dir / "counts.csv").exists() and (input_dir / "truth.csv").exists():
        return input_dir
    command = [
        args.python,
        str(Path(args.root) / "scripts/p4_export_method_input.py"),
        "--counts-npz",
        row["counts_file"],
        "--design",
        row["metadata_file"],
        "--truth",
        row["truth_file"],
        "--scenario",
        row["scenario"],
        "--out-dir",
        str(input_dir),
        "--min-prevalence",
        "0",
    ]
    code = run_command(command, work_dir / "prepare.log", timeout=300)
    if code != 0:
        raise RuntimeError(f"Input preparation failed for {row['scenario']} {row['seed']}")
    return input_dir


def run_method_job(job: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    method = str(job["method"])
    csv = Path(str(job["csv"]))
    json_path = Path(str(job["json"]))
    if csv.exists() and json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if payload.get("status") == "ok":
                return {"method": method, "status": "ok", "cached": True}
        except Exception:
            pass
    command = [
        args.rscript,
        str(job["runner"]),
        method,
        str(job["input_dir"]),
        str(csv),
        str(json_path),
        str(job["seed"]),
        str(job["scenario"]),
    ]
    log(f"start {job['scenario']} seed={job['seed']} {method}")
    code = run_command(
        command,
        Path(str(job["log"])),
        timeout=METHOD_TIMEOUT[method],
    )
    if code != 0 or not csv.exists() or not json_path.exists():
        write_failure(
            method,
            csv,
            json_path,
            f"runner exit code {code}; output missing or incomplete",
        )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    log(
        f"finish {job['scenario']} seed={job['seed']} {method} "
        f"status={payload.get('status')} elapsed={payload.get('elapsed_sec', 'NA')}"
    )
    return {
        "method": method,
        "status": payload.get("status", "unknown"),
        "cached": False,
    }


def collect_dataset(
    row: pd.Series,
    input_dir: Path,
    result_dir: Path,
    args: argparse.Namespace,
) -> None:
    prefix = result_dir / "method_unified"
    command = [
        args.python,
        str(Path(args.root) / "scripts/p4_collect_method_results.py"),
        "--results-dir",
        str(result_dir),
        "--truth",
        str(input_dir / "truth.csv"),
        "--scenario",
        row["scenario"],
        "--out-prefix",
        str(prefix),
    ]
    code = run_command(command, result_dir / "collect.log", timeout=300)
    if code != 0:
        raise RuntimeError(f"Collection failed for {row['scenario']} {row['seed']}")


def aggregate_results(out_root: Path) -> None:
    metrics = []
    manifests = []
    long_frames = []
    for path in out_root.rglob("method_unified_metrics.csv"):
        run = path.parent
        parts = run.parts
        stage = parts[-3] if len(parts) >= 3 else ""
        scenario = parts[-2]
        seed = int(parts[-1].replace("seed_", ""))
        frame = pd.read_csv(path)
        frame.insert(0, "seed", seed)
        frame.insert(0, "scenario", scenario)
        frame.insert(0, "stage", stage)
        metrics.append(frame)
    for path in out_root.rglob("method_unified_manifest.csv"):
        run = path.parent
        parts = run.parts
        frame = pd.read_csv(path)
        frame.insert(0, "seed", int(parts[-1].replace("seed_", "")))
        frame.insert(0, "scenario", parts[-2])
        frame.insert(0, "stage", parts[-3])
        manifests.append(frame)
    if metrics:
        pd.concat(metrics, ignore_index=True).to_csv(
            out_root / "formal_metrics_all.csv", index=False
        )
    if manifests:
        pd.concat(manifests, ignore_index=True).to_csv(
            out_root / "formal_manifest_all.csv", index=False
        )
    for path in out_root.rglob("method_unified_long.csv"):
        run = path.parent
        parts = run.parts
        frame = pd.read_csv(path)
        frame.insert(0, "seed", int(parts[-1].replace("seed_", "")))
        frame.insert(0, "scenario", parts[-2])
        frame.insert(0, "stage", parts[-3])
        long_frames.append(frame)
    if long_frames:
        pd.concat(long_frames, ignore_index=True).to_csv(
            out_root / "formal_long_all.csv.gz",
            index=False,
            compression="gzip",
        )


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    os.environ["R_LIBS_USER"] = str(root / "Rlib")
    os.environ["R_HOME"] = str(Path(args.renv) / "Lib" / "R")
    os.environ["CONDA_PREFIX"] = str(Path(args.renv))
    env_prefixes = [
        str(Path(args.renv)),
        str(Path(args.renv) / "Scripts"),
        str(Path(args.renv) / "Library" / "bin"),
        str(Path(args.renv) / "Library" / "usr" / "bin"),
        str(Path(args.renv) / "Lib" / "R" / "bin" / "x64"),
    ]
    os.environ["PATH"] = os.pathsep.join(
        env_prefixes + [os.environ.get("PATH", "")]
    )
    grid = pd.read_csv(args.grid)
    if args.stage != "all":
        grid = grid[grid["stage"].eq(args.stage)].copy()
    grid = grid.sort_values(["scenario", "seed"]).reset_index(drop=True)
    out_root = Path(args.out_root)
    work_root = Path(args.work_root)
    out_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    runner = work_root / "p4_unified_da_runner.R"
    shutil.copy2(
        root / "scripts/p4_unified_da_runner.R",
        runner,
    )

    jobs = []
    for row in grid.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        stage_dir = out_root / row_series["stage"]
        result_dir = stage_dir / row_series["scenario"] / f"seed_{row_series['seed']}"
        result_dir.mkdir(parents=True, exist_ok=True)
        if dataset_ready(result_dir, args.methods):
            continue
        work_dir = work_root / row_series["stage"] / f"{row_series['scenario']}_{row_series['seed']}"
        work_dir.mkdir(parents=True, exist_ok=True)
        input_dir = prepare_input(row_series, args, work_dir)
        method_result_dir = work_dir / "results"
        method_result_dir.mkdir(parents=True, exist_ok=True)
        for method in args.methods:
            jobs.append(
                {
                    "method": method,
                    "scenario": row_series["scenario"],
                    "seed": int(row_series["seed"]),
                    "input_dir": input_dir,
                    "runner": runner,
                    "csv": method_result_dir / f"{method}.csv",
                    "json": method_result_dir / f"{method}.json",
                    "log": method_result_dir / f"{method}.log",
                    "result_dir": method_result_dir,
                }
            )

    log(f"prepared {len(grid)} datasets; pending method jobs={len(jobs)}")
    if jobs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_method_job, job, args) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    for row in grid.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        result_dir = (
            out_root
            / row_series["stage"]
            / row_series["scenario"]
            / f"seed_{row_series['seed']}"
        )
        if not dataset_ready(result_dir, args.methods):
            work_dir = work_root / row_series["stage"] / f"{row_series['scenario']}_{row_series['seed']}"
            input_dir = work_dir / "input"
            method_result_dir = work_dir / "results"
            collect_dataset(row_series, input_dir, method_result_dir, args)
            shutil.copytree(method_result_dir, result_dir, dirs_exist_ok=True)
    aggregate_results(out_root)
    log("formal grid stage complete")


if __name__ == "__main__":
    main()
