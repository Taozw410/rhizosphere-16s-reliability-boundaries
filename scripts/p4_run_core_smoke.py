#!/usr/bin/env python3
"""Generate and calibrate the Paper 4 core/occupancy simulator smoke grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p4_calibrate_mvp import (
    STAT_NAMES,
    TOLERANCES,
    distance_profile,
)
from p4_core_sim_common import load_core_context, simulate_core_scenario
from p4_core_sim_correlated import (
    estimate_latent_correlation,
    simulate_correlated_core_scenario,
)
from p4_core_sim_scaffold import simulate_scaffold_core_scenario
from p4_core_sim_latent_factor import simulate_latent_factor_core_scenario
from p4_sim_common import compute_summary_components


SCENARIOS = [
    "core_null",
    "core20_balanced",
    "core20_lowabundance",
    "core20_projectshift",
    "core20_depthconfound",
]
STRUCTURAL_STATISTICS = [
    "rank_abundance",
    "sample_depth",
    "within_between_variance_ratio",
    "bray_curtis_distribution",
    "aitchison_distribution",
    "zero_fraction",
    "network_topology",
]


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
        "--config",
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
        "--out-dir",
        default="outputs/core_smoke",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=6000)
    parser.add_argument(
        "--generator",
        choices=["independent", "copula", "scaffold", "latent-factor"],
        default="independent",
    )
    parser.add_argument(
        "--preserve-core-richness",
        action="store_true",
    )
    return parser.parse_args()


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "|" + "|".join(["---"] * len(frame.columns)) + "|"
    rows = []
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    context = load_core_context(
        Path(args.feature_matrix),
        Path(args.design),
        Path(args.config),
        Path(args.panel),
    )
    real_components = compute_summary_components(
        context.real_counts,
        context.meta,
        context.feature_names,
    )
    correlation = None
    if args.generator == "copula":
        correlation = estimate_latent_correlation(context)
        pd.DataFrame(
            correlation,
            index=context.feature_names,
            columns=context.feature_names,
        ).to_csv(out_dir / "core_feature_correlation.csv")

    distance_rows = []
    truth_rows = []
    grid_rows = []
    for scenario in SCENARIOS:
        scenario_dir = out_dir / "data" / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            if args.generator == "copula":
                simulation = simulate_correlated_core_scenario(
                    context,
                    scenario,
                    seed,
                    correlation,
                )
            elif args.generator == "scaffold":
                simulation = simulate_scaffold_core_scenario(
                    context,
                    scenario,
                    seed,
                    preserve_richness=args.preserve_core_richness,
                )
            elif args.generator == "latent-factor":
                if correlation is None:
                    correlation = estimate_latent_correlation(context)
                simulation = simulate_latent_factor_core_scenario(
                    context,
                    scenario,
                    seed,
                    correlation,
                )
            else:
                simulation = simulate_core_scenario(context, scenario, seed)
            components = compute_summary_components(
                simulation.counts,
                context.meta,
                context.feature_names,
            )
            distances = distance_profile(real_components, components)
            expected_statistics = (
                STAT_NAMES if scenario == "core_null" else STRUCTURAL_STATISTICS
            )
            passed = {
                name: distances[name] <= TOLERANCES[name]
                for name in STAT_NAMES
            }
            distance_rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    **distances,
                    **{
                        f"pass_{name}": bool(passed[name])
                        for name in STAT_NAMES
                    },
                    "gate_applicable_n": len(expected_statistics),
                    "gate_pass_n": int(
                        sum(passed[name] for name in expected_statistics)
                    ),
                    "gate_pass": bool(
                        all(passed[name] for name in expected_statistics)
                    ),
                }
            )
            truth = simulation.truth.copy()
            truth["seed"] = seed
            truth_rows.append(truth)
            run_dir = scenario_dir / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
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
            truth.to_csv(run_dir / "truth.csv", index=False)
            simulation.project_occupancy.to_csv(
                run_dir / "project_latent_occupancy.csv"
            )
            grid_rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "counts_file": str((run_dir / "counts.npz").resolve()),
                    "truth_file": str((run_dir / "truth.csv").resolve()),
                    "metadata_file": str(
                        (run_dir / "metadata.csv").resolve()
                    ),
                    "truth_core_n": int(truth["is_population_core"].sum()),
                    "candidate_core_n": int(truth["truth_core_candidate"].sum()),
                }
            )
            print(
                f"{scenario} seed={seed} core={truth['is_population_core'].sum()}"
            )

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(out_dir / "core_seed_grid.csv", index=False)
    pd.concat(truth_rows, ignore_index=True).to_csv(
        out_dir / "core_truth.csv",
        index=False,
    )
    distances = pd.DataFrame(distance_rows)
    distances.to_csv(out_dir / "core_calibration_distances.csv", index=False)
    summary = (
        distances.groupby("scenario")
        .agg(
            seeds=("seed", "size"),
            gate_pass_seeds=("gate_pass", "sum"),
            **{
                f"mean_{name}": (name, "mean")
                for name in STAT_NAMES
            },
            **{
                f"pass_seeds_{name}": (f"pass_{name}", "sum")
                for name in STAT_NAMES
            },
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "core_calibration_summary.csv", index=False)

    truth_summary = (
        pd.concat(truth_rows, ignore_index=True)
        .groupby("scenario")
        .agg(
            seeds=("seed", "nunique"),
            population_core_mean=("is_population_core", "sum"),
            candidate_core_mean=("truth_core_candidate", "sum"),
            latent_occupancy_mean=("latent_occupancy_mean", "mean"),
            latent_occupancy_sd=("latent_occupancy_sd", "mean"),
        )
        .reset_index()
    )
    truth_summary["population_core_mean"] /= args.seeds
    truth_summary["candidate_core_mean"] /= args.seeds
    truth_summary.to_csv(out_dir / "core_truth_summary.csv", index=False)

    config = {
        "version": f"core_smoke_{args.generator}_v1",
        "generator": args.generator,
        "preserve_core_richness": bool(args.preserve_core_richness),
        "scenarios": SCENARIOS,
        "seed_start": args.seed_start,
        "seeds_per_scenario": args.seeds,
        "n_projects": len(context.meta["project"].unique()),
        "n_samples": len(context.meta),
        "n_features": len(context.feature_names),
        "truth_definition": {
            "project_detection_threshold": 0.90,
            "project_fraction_threshold": 0.80,
        },
    }
    (out_dir / "core_smoke_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = [
        f"# Paper 4 core/occupancy simulator smoke test: {args.generator}",
        "",
        f"- Projects: {config['n_projects']}",
        f"- Samples: {config['n_samples']}",
        f"- Features: {config['n_features']}",
        f"- Seeds per scenario: {args.seeds}",
        "",
        "## Core truth",
        "",
        markdown_table(truth_summary),
        "",
        "## Calibration gates",
        "",
        markdown_table(summary),
        "",
        "## Gate rules",
        "",
        "- `core_null` is evaluated on all eight calibration statistics.",
        "- Non-null scenarios intentionally change prevalence and are evaluated "
        "on seven structural statistics; prevalence is not tuned back to the "
        "null distribution.",
        "- Core truth is computed from latent detection probabilities, never "
        "from the realized simulated count matrix.",
        "",
    ]
    (out_dir / "core_smoke_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
