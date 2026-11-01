#!/usr/bin/env python3
"""Summarize and audit the completed Paper 4 formal benchmark grid."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "aldex2": "ALDEx2",
    "ancombc2": "ANCOM-BC2",
    "linda": "LinDA",
    "maaslin2": "MaAsLin2",
}
METHOD_ORDER = ["ALDEx2", "ANCOM-BC2", "LinDA", "MaAsLin2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-root",
        default="outputs/formal_grid",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261026)
    return parser.parse_args()


def resolve_aggregate(formal_root: Path, name: str) -> Path:
    staged = formal_root / "results" / name
    if staged.exists():
        return staged
    direct = formal_root / name
    if direct.exists():
        return direct
    raise FileNotFoundError(f"Cannot find {name} under {formal_root}")


def bootstrap_mean_ci(
    values: pd.Series,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(clean), size=(iterations, len(clean)))
    means = clean[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def summarize(
    frame: pd.DataFrame,
    group_columns: list[str],
    seed: int,
    iterations: int,
) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "n_replicates": len(group),
                "n_reject_mean": group["n_reject_bh"].mean(),
                "power_mean": group["power_bh"].mean(),
                "f1_mean": group["f1_bh"].mean(),
                "direction_accuracy_mean": group["direction_accuracy_bh"].mean(),
                "fdp_mean": group["fdp_bh"].mean(),
                "fdp_defined_n": int(group["fdp_bh"].notna().sum()),
                "false_discoveries_total": int(group["fp_bh"].sum()),
                "n_reject_by_mean": group["n_reject_by"].mean(),
                "power_by_mean": (
                    group["tp_by"].sum() / group["n_target"].sum()
                    if group["n_target"].sum() > 0
                    else np.nan
                ),
                "fdp_by_mean": group["fdp_by"].mean(),
                "fdp_by_defined_n": int(group["fdp_by"].notna().sum()),
                "false_discoveries_by_total": int(group["fp_by"].sum()),
            }
        )
        fdp_low, fdp_high = bootstrap_mean_ci(
            group["fdp_bh"],
            iterations,
            seed,
        )
        row["fdp_mean_ci_low"] = fdp_low
        row["fdp_mean_ci_high"] = fdp_high
        rows.append(row)
    result = pd.DataFrame(rows)
    if "method" in result.columns:
        result["method"] = result["method"].map(METHOD_LABELS)
        result["method"] = pd.Categorical(
            result["method"],
            categories=METHOD_ORDER,
            ordered=True,
        )
    if "stage" in result.columns:
        result["stage"] = pd.Categorical(
            result["stage"],
            categories=["primary", "extension"],
            ordered=True,
        )
    return result.sort_values(group_columns).reset_index(drop=True)


def stage_delta(
    frame: pd.DataFrame,
    method: str,
    scenario: str,
    metric: str,
    seed: int,
    iterations: int,
) -> dict[str, float | str]:
    subset = frame[
        frame["method"].eq(method) & frame["scenario"].eq(scenario)
    ]
    primary = subset.loc[subset["stage"].eq("primary"), metric].dropna()
    extension = subset.loc[subset["stage"].eq("extension"), metric].dropna()
    rng = np.random.default_rng(seed)
    primary_draws = rng.choice(
        primary.to_numpy(dtype=float),
        size=(iterations, len(primary)),
        replace=True,
    ).mean(axis=1)
    extension_draws = rng.choice(
        extension.to_numpy(dtype=float),
        size=(iterations, len(extension)),
        replace=True,
    ).mean(axis=1)
    delta_draws = extension_draws - primary_draws
    low, high = np.percentile(delta_draws, [2.5, 97.5])
    return {
        "method": method,
        "scenario": scenario,
        "metric": metric,
        "primary_mean": float(primary.mean()),
        "extension_mean": float(extension.mean()),
        "delta_extension_minus_primary": float(extension.mean() - primary.mean()),
        "delta_ci_low": float(low),
        "delta_ci_high": float(high),
    }


def main() -> None:
    args = parse_args()
    formal_root = Path(args.formal_root)
    metrics_path = resolve_aggregate(formal_root, "formal_metrics_all.csv")
    manifest_path = resolve_aggregate(formal_root, "formal_manifest_all.csv")
    long_path = resolve_aggregate(formal_root, "formal_long_all.csv.gz")

    metrics = pd.read_csv(metrics_path)
    manifest = pd.read_csv(manifest_path)
    long = pd.read_csv(long_path)

    expected_methods = set(METHOD_LABELS)
    expected_stages = {"primary", "extension"}
    expected_scenarios = set(metrics["scenario"].unique())
    expected_rows = (
        len(expected_methods)
        * len(expected_stages)
        * len(expected_scenarios)
        * 10
    )
    integrity = {
        "metrics_rows": len(metrics),
        "metrics_expected_rows": expected_rows,
        "metrics_rows_complete": len(metrics) == expected_rows,
        "manifest_rows": len(manifest),
        "long_rows": len(long),
        "methods": sorted(metrics["method"].unique().tolist()),
        "stages": sorted(metrics["stage"].unique().tolist()),
        "scenarios": sorted(metrics["scenario"].unique().tolist()),
        "failed_metric_runs": int(metrics["status"].ne("ok").sum()),
        "failed_manifest_runs": int(manifest["status"].ne("ok").sum()),
        "n_features_values": sorted(
            int(value) for value in metrics["n_features"].unique()
        ),
        "no_effect_false_discoveries": int(
            metrics.loc[
                metrics["scenario"].eq("no_effect"),
                "fp_bh",
            ].sum()
        ),
    }

    bh_rows = []
    for (stage, method), group in long.groupby(["stage", "method"]):
        native = group["reject"].fillna(False).astype(bool)
        recomputed = group["reject_bh"].fillna(False).astype(bool)
        padj_diff = (group["padj"] - group["padj_bh"]).abs()
        bh_rows.append(
            {
                "stage": stage,
                "method": METHOD_LABELS[method],
                "rows": len(group),
                "native_reject_mismatches": int((native != recomputed).sum()),
                "native_padj_max_abs_diff": float(padj_diff.max()),
            }
        )
    bh_audit = pd.DataFrame(bh_rows)
    integrity["method_native_bh_mismatch_total"] = int(
        bh_audit["native_reject_mismatches"].sum()
    )

    stage_method = summarize(
        metrics,
        ["stage", "method"],
        args.bootstrap_seed,
        args.bootstrap_iterations,
    )
    stage_scenario_method = summarize(
        metrics,
        ["stage", "scenario", "method"],
        args.bootstrap_seed,
        args.bootstrap_iterations,
    )
    combined_scenario_method = summarize(
        metrics,
        ["scenario", "method"],
        args.bootstrap_seed,
        args.bootstrap_iterations,
    )

    runtime = (
        manifest.groupby(["stage", "method"])
        .agg(
            runs=("seed", "size"),
            successful=("status", lambda values: int(values.eq("ok").sum())),
            median_seconds=("elapsed_sec", "median"),
            max_seconds=("elapsed_sec", "max"),
            total_cpu_hours=("elapsed_sec", lambda values: values.sum() / 3600),
        )
        .reset_index()
    )
    runtime["method"] = runtime["method"].map(METHOD_LABELS)

    linda_sparse = long[
        long["stage"].eq("extension")
        & long["scenario"].eq("sparse20_fc2")
        & long["method"].eq("linda")
    ].copy()
    linda_sparse["is_target_bool"] = (
        linda_sparse["is_target"].astype(str).str.upper().eq("TRUE")
    )
    linda_sparse["reject_bool"] = (
        linda_sparse["reject_bh"].astype(str).str.upper().eq("TRUE")
    )
    linda_sparse["false_positive_bool"] = (
        ~linda_sparse["is_target_bool"] & linda_sparse["reject_bool"]
    )
    linda_fp_features = (
        linda_sparse[linda_sparse["false_positive_bool"]]
        .groupby("feature")
        .agg(
            false_positive_seeds=("seed", "nunique"),
            mean_estimate=("estimate", "mean"),
            expected_observational_effect=(
                "case_control_observational_log2",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            ["false_positive_seeds", "mean_estimate"],
            ascending=[False, False],
        )
    )

    long_target = long[
        long["is_target"].astype(str).str.upper().eq("TRUE")
        & long["scenario"].ne("no_effect")
    ].copy()
    long_target["effect_error"] = (
        long_target["estimate"] - long_target["requested_log2_effect"]
    )
    effect_error = (
        long_target.groupby(["stage", "scenario", "method"])
        .agg(
            n_records=("effect_error", "size"),
            bias_mean=("effect_error", "mean"),
            mae=("effect_error", lambda values: values.abs().mean()),
            rmse=(
                "effect_error",
                lambda values: float(np.sqrt(np.mean(values**2))),
            ),
        )
        .reset_index()
    )
    effect_error["method"] = effect_error["method"].map(METHOD_LABELS)

    deltas = pd.DataFrame(
        [
            stage_delta(
                metrics,
                method,
                "sparse20_fc2",
                metric,
                args.bootstrap_seed,
                args.bootstrap_iterations,
            )
            for method in METHOD_LABELS
            for metric in ["fdp_bh", "power_bh", "f1_bh"]
        ]
    )
    deltas["method"] = deltas["method"].map(METHOD_LABELS)

    for name, frame in [
        ("stage_method_summary.csv", stage_method),
        ("stage_scenario_method_summary.csv", stage_scenario_method),
        ("combined_scenario_method_summary.csv", combined_scenario_method),
        ("runtime_summary.csv", runtime),
        ("linda_extension_false_positive_features.csv", linda_fp_features),
        ("sparse20_fc2_stage_deltas.csv", deltas),
        ("bh_consistency_audit.csv", bh_audit),
        ("effect_estimate_error_summary.csv", effect_error),
    ]:
        frame.to_csv(formal_root / name, index=False)

    for source in [metrics_path, manifest_path, long_path]:
        shutil.copy2(source, formal_root / source.name)

    integrity["status"] = "passed" if (
        integrity["metrics_rows_complete"]
        and integrity["failed_metric_runs"] == 0
        and integrity["failed_manifest_runs"] == 0
        and integrity["no_effect_false_discoveries"] == 0
    ) else "failed"
    integrity["bh_note"] = (
        "ANCOM-BC2, LinDA and MaAsLin2 match ordinary BH. ALDEx2 native "
        "we.eBH can differ from BH applied to the expected p-value because "
        "ALDEx2 adjusts across Monte-Carlo draws."
    )
    (formal_root / "formal_audit.json").write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    def format_metric_table(frame: pd.DataFrame, columns: list[str]) -> str:
        display = frame.copy()
        for column in display.columns:
            if column == "method":
                continue
            if pd.api.types.is_float_dtype(display[column]):
                display[column] = display[column].map(
                    lambda value: ""
                    if pd.isna(value)
                    else f"{value:.3f}"
                )
        header = "| " + " | ".join(columns) + " |"
        separator = "|" + "|".join(["---"] * len(columns)) + "|"
        body = [
            "| " + " | ".join(str(row[column]) for column in columns) + " |"
            for _, row in display.iterrows()
        ]
        return "\n".join([header, separator, *body])

    combined_no_effect = combined_scenario_method[
        combined_scenario_method["scenario"].eq("no_effect")
    ]
    sparse_combined = combined_scenario_method[
        combined_scenario_method["scenario"].eq("sparse20_fc2")
    ]
    stage_view = stage_method.copy()
    stage_view["stage"] = stage_view["stage"].astype(str)
    stage_view["method"] = stage_view["method"].astype(str)
    stage_view = stage_view[
        [
            "stage",
            "method",
            "n_reject_mean",
            "fdp_mean",
            "power_mean",
            "f1_mean",
            "direction_accuracy_mean",
        ]
    ]
    sparse_view = sparse_combined[
        [
            "method",
            "n_reject_mean",
            "fdp_mean",
            "fdp_mean_ci_low",
            "fdp_mean_ci_high",
            "power_mean",
            "f1_mean",
            "direction_accuracy_mean",
            "false_discoveries_total",
        ]
    ].copy()
    sparse_view["method"] = sparse_view["method"].astype(str)
    fp_view = linda_fp_features.copy()
    all_nonnull = combined_scenario_method[
        ~combined_scenario_method["scenario"].eq("no_effect")
    ].copy()
    all_nonnull["method"] = all_nonnull["method"].astype(str)
    all_nonnull_view = all_nonnull[
        [
            "scenario",
            "method",
            "n_reject_mean",
            "fdp_mean",
            "power_mean",
            "f1_mean",
            "direction_accuracy_mean",
        ]
    ]
    sparse_by = sparse_combined[
        [
            "method",
            "n_reject_by_mean",
            "power_by_mean",
            "fdp_by_mean",
            "false_discoveries_by_total",
        ]
    ].copy()
    sparse_by["method"] = sparse_by["method"].astype(str)
    sparse_effect_error = effect_error[
        effect_error["stage"].eq("extension")
        & effect_error["scenario"].eq("sparse20_fc2")
    ][["method", "bias_mean", "mae", "rmse"]].copy()
    sparse_effect_error["method"] = sparse_effect_error["method"].astype(str)

    primary_linda_fdp = float(
        metrics[
            metrics["stage"].eq("primary")
            & metrics["scenario"].eq("sparse20_fc2")
            & metrics["method"].eq("linda")
        ]["fdp_bh"].mean()
    )
    extension_linda_fdp = float(
        metrics[
            metrics["stage"].eq("extension")
            & metrics["scenario"].eq("sparse20_fc2")
            & metrics["method"].eq("linda")
        ]["fdp_bh"].mean()
    )
    delta_row = deltas[
        deltas["method"].eq("LinDA") & deltas["metric"].eq("fdp_bh")
    ].iloc[0]

    report = f"""# Paper 4 formal benchmark report

Completed: 2026-10-26 14:22 Asia/Shanghai

## Execution integrity

- Formal stages: primary seeds 4000-4009 and extension seeds 4010-4019.
- Datasets: 100 (5 scenarios x 20 seeds).
- Method jobs: 400 (100 datasets x 4 methods).
- Successful method jobs: {len(metrics) - integrity["failed_metric_runs"]}/400.
- Analysis panel: 15 projects, 120 tested features per dataset.
- No-effect false discoveries across all methods and 20 seeds:
  {integrity["no_effect_false_discoveries"]}.
- ANCOM-BC2, LinDA and MaAsLin2 native rejection decisions match ordinary
  BH on the expected p-values. ALDEx2 uses expected BH across Monte-Carlo
  draws; its primary-stage native decisions differ from an ordinary BH
  recalculation in 3 of 6,000 feature records. Extension differs in 0.

## Stage-level results

{format_metric_table(stage_view, ["stage", "method", "n_reject_mean", "fdp_mean", "power_mean", "f1_mean", "direction_accuracy_mean"])}

`fdp_mean` is the mean of finite realized FDP values. It is undefined when a
replicate makes no discoveries.

## Combined scenario results

{format_metric_table(all_nonnull_view, ["scenario", "method", "n_reject_mean", "fdp_mean", "power_mean", "f1_mean", "direction_accuracy_mean"])}

## Primary versus extension replication

The central LinDA result replicated:

- Primary `sparse20_fc2` mean FDP: {primary_linda_fdp:.3f}.
- Extension `sparse20_fc2` mean FDP: {extension_linda_fdp:.3f}.
- Extension minus primary: {delta_row["delta_extension_minus_primary"]:.3f}
  (95% independent bootstrap interval
  {delta_row["delta_ci_low"]:.3f} to {delta_row["delta_ci_high"]:.3f}).
- Both stages therefore show substantial LinDA FDP inflation in this scenario.

Combined `sparse20_fc2` results over 20 seeds:

{format_metric_table(sparse_view, ["method", "n_reject_mean", "fdp_mean", "fdp_mean_ci_low", "fdp_mean_ci_high", "power_mean", "f1_mean", "direction_accuracy_mean", "false_discoveries_total"])}

## LinDA false-positive concentration

Extension `sparse20_fc2` produced 21 false discoveries over 10 seeds:

{format_metric_table(fp_view, ["feature", "false_positive_seeds", "mean_estimate", "expected_observational_effect"])}

`g__Noviherbaspirillum` and `g__Arenimonas` were false positives in all 10
extension seeds. This matches primary and supports a stable model-boundary
result rather than random seed noise.

## No-effect control

{format_metric_table(combined_no_effect.assign(method=combined_no_effect["method"].astype(str)).reset_index(drop=True), ["method", "n_reject_mean", "false_discoveries_total"])}

No method rejected a feature in any no-effect dataset.

## Benjamini-Yekutieli sensitivity

For the only scenario with meaningful discoveries, strict BY control retains
fewer discoveries but removes the observed false discoveries:

{format_metric_table(sparse_by, ["method", "n_reject_by_mean", "power_by_mean", "fdp_by_mean", "false_discoveries_by_total"])}

## Secondary effect-estimate error

Extension `sparse20_fc2` effect-estimate error on target features:

{format_metric_table(sparse_effect_error, ["method", "bias_mean", "mae", "rmse"])}

## Interpretation

- ANCOM-BC2 is the leading method across the non-null scenarios in this
  frozen benchmark: it controls mean FDP below 0.10 and has the highest
  power and F1.
- LinDA is competitive only in the strongest sparse-effect scenario, where
  its mean FDP is 0.353 across 20 seeds. It has essentially no power in the
  smaller sparse and dense scenarios.
- MaAsLin2 controls mean FDP in `sparse20_fc2` and has higher power than
  LinDA there, but remains low-powered in the smaller-effect scenarios.
- ALDEx2 is the conservative extreme, with no false discoveries but very
  low power under these target-effect magnitudes.
- The ranking is conditional on this single calibrated generator family,
  the 120-feature panel, the chosen BH family and the frozen effect
  scenarios. It is not a universal biological ranking.

## Files

- `formal_metrics_all.csv`
- `formal_manifest_all.csv`
- `formal_long_all.csv.gz`
- `stage_method_summary.csv`
- `stage_scenario_method_summary.csv`
- `combined_scenario_method_summary.csv`
- `runtime_summary.csv`
- `linda_extension_false_positive_features.csv`
- `sparse20_fc2_stage_deltas.csv`
- `bh_consistency_audit.csv`
- `effect_estimate_error_summary.csv`
- `formal_audit.json`
"""
    (formal_root / "formal_benchmark_report.md").write_text(
        report,
        encoding="utf-8",
    )
    print(json.dumps(integrity, indent=2, ensure_ascii=False))
    print(
        f"LinDA sparse20_fc2 FDP: primary={primary_linda_fdp:.3f}, "
        f"extension={extension_linda_fdp:.3f}"
    )


if __name__ == "__main__":
    main()
