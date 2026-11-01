#!/usr/bin/env python3
"""Build the frozen Paper 4 formal simulation seed grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p4_run_effect_interface import make_scenarios, simulate_scenario
from p4_sim_common import load_analysis_data, relative_abundance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-matrix",
        default="out_analysis/feature_matrix.csv",
    )
    parser.add_argument(
        "--effect-dir",
        default="outputs/effect_interface",
    )
    parser.add_argument(
        "--phase1-dir",
        default="outputs/phase1",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/formal_grid",
    )
    parser.add_argument("--projects-per-crop", type=int, default=5)
    parser.add_argument("--panel-size", type=int, default=120)
    parser.add_argument("--primary-seeds", type=int, default=10)
    parser.add_argument("--extension-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=4000)
    parser.add_argument("--project-seed", type=int, default=20261025)
    return parser.parse_args()


def select_project_panel(
    design: pd.DataFrame, projects_per_crop: int, seed: int
) -> list[str]:
    rng = np.random.default_rng(seed)
    project_crop = (
        design[["project", "crop"]]
        .drop_duplicates()
        .sort_values(["crop", "project"])
    )
    selected = []
    for crop, group in project_crop.groupby("crop", sort=True):
        projects = group["project"].to_numpy()
        if len(projects) < projects_per_crop:
            raise ValueError(f"Not enough projects for crop {crop}")
        chosen = rng.choice(projects, size=projects_per_crop, replace=False)
        selected.extend(sorted(chosen.tolist()))
    return sorted(selected)


def scenario_effects(
    counts: np.ndarray,
    feature_names: list[str],
    scenario: str,
    n_targets: int,
    absolute_log2: float,
    scenario_index: int,
) -> pd.DataFrame:
    if n_targets == 0:
        targets = []
    else:
        from p4_run_effect_interface import choose_target_features

        targets = choose_target_features(
            counts,
            feature_names,
            n_targets,
            seed=7000 + scenario_index,
        )
    directions = np.where(
        np.arange(len(targets)) % 2 == 0,
        absolute_log2,
        -absolute_log2,
    )
    return pd.DataFrame(
        {
            "scenario": scenario,
            "feature": targets,
            "requested_log2_effect": directions,
        }
    )


def select_feature_panel(
    counts: np.ndarray,
    feature_names: list[str],
    targets: list[str],
    panel_size: int,
) -> list[str]:
    prevalence = (counts > 0).mean(axis=0)
    mean_relative = relative_abundance(counts).mean(axis=0)
    feature_array = np.array(feature_names, dtype=object)
    target_set = set(targets)
    background = [
        (name, prevalence[index], mean_relative[index])
        for index, name in enumerate(feature_names)
        if name != "unassigned" and name not in target_set
    ]
    background.sort(key=lambda row: (-row[1], -row[2], row[0]))
    selected = list(targets)
    for name, _, _ in background:
        if len(selected) >= panel_size:
            break
        selected.append(name)
    if len(selected) < panel_size:
        raise ValueError("Insufficient background features")
    return sorted(selected)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    effect_dir = Path(args.effect_dir)
    phase1_dir = Path(args.phase1_dir)

    data = load_analysis_data(Path(args.feature_matrix), min_depth=1000)
    design = pd.read_csv(effect_dir / "sample_design.csv", dtype={"sample": str})
    design = design.set_index("sample").loc[data.meta["sample"]].reset_index()
    project_panel = select_project_panel(
        design, args.projects_per_crop, args.project_seed
    )
    sample_panel = design["project"].isin(project_panel).to_numpy()
    panel_design = design.loc[sample_panel].reset_index(drop=True)
    panel_design.to_csv(out_dir / "project_panel_design.csv", index=False)

    with (phase1_dir / "simulation_parameters_v1.json").open(
        "r", encoding="utf-8"
    ) as handle:
        parameters = json.load(handle)
    sample_prior_reads = float(parameters["sample_prior_reads"])

    specifications = [
        ("no_effect", 0, 0.0),
        ("sparse20_fc1_19", 20, 0.25),
        ("sparse20_fc1_41", 20, 0.50),
        ("sparse20_fc2", 20, 1.00),
        ("dense40_fc1_41", 40, 0.50),
    ]
    seed_total = args.primary_seeds + args.extension_seeds
    seeds = list(range(args.seed_start, args.seed_start + seed_total))
    grid_rows = []

    for scenario_index, (scenario, n_targets, effect_size) in enumerate(
        specifications
    ):
        effects = scenario_effects(
            data.counts,
            data.feature_names,
            scenario,
            n_targets,
            effect_size,
            scenario_index,
        )
        targets = effects["feature"].tolist()
        panel_features = select_feature_panel(
            data.counts,
            data.feature_names,
            targets,
            args.panel_size,
        )
        feature_index = {
            name: index for index, name in enumerate(data.feature_names)
        }
        panel_feature_indices = np.array(
            [feature_index[name] for name in panel_features]
        )
        scenario_dir = data_dir / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        effects.to_csv(scenario_dir / "effects.csv", index=False)
        pd.DataFrame({"feature": panel_features}).to_csv(
            scenario_dir / "feature_panel.csv", index=False
        )

        for seed_index, seed in enumerate(seeds):
            synthetic, target_truth, full_truth = simulate_scenario(
                data.counts,
                data.feature_names,
                design,
                effects,
                scenario,
                sample_prior_reads,
                seed,
            )
            panel_counts = synthetic[np.ix_(sample_panel, panel_feature_indices)]
            panel_truth = full_truth[
                full_truth["feature"].isin(panel_features)
            ].copy()
            run_dir = scenario_dir / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                run_dir / "counts.npz",
                counts=panel_counts.astype(np.int32),
                feature_names=np.array(panel_features, dtype=str),
                sample=np.array(panel_design["sample"], dtype=str),
                project=np.array(panel_design["project"], dtype=str),
            )
            panel_truth.to_csv(run_dir / "truth.csv", index=False)
            panel_design.to_csv(run_dir / "metadata.csv", index=False)
            grid_rows.append(
                {
                    "stage": "primary" if seed_index < args.primary_seeds else "extension",
                    "scenario": scenario,
                    "seed": seed,
                    "n_targets": n_targets,
                    "absolute_log2_effect": effect_size,
                    "n_projects": len(project_panel),
                    "n_samples": len(panel_design),
                    "n_features": len(panel_features),
                    "counts_file": str((run_dir / "counts.npz").resolve()),
                    "truth_file": str((run_dir / "truth.csv").resolve()),
                    "metadata_file": str((run_dir / "metadata.csv").resolve()),
                }
            )
            print(f"{scenario} seed={seed} stage={grid_rows[-1]['stage']}")

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(out_dir / "seed_grid.csv", index=False)
    config = {
        "project_panel": project_panel,
        "projects_per_crop": args.projects_per_crop,
        "panel_size": args.panel_size,
        "seeds": seeds,
        "primary_seeds": seeds[: args.primary_seeds],
        "extension_seeds": seeds[args.primary_seeds :],
        "scenarios": [
            {
                "scenario": scenario,
                "n_targets": n_targets,
                "absolute_log2_effect": effect_size,
            }
            for scenario, n_targets, effect_size in specifications
        ],
    }
    with (out_dir / "formal_grid_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2)
    print(f"Wrote {len(grid)} runs to {out_dir}")


if __name__ == "__main__":
    main()
