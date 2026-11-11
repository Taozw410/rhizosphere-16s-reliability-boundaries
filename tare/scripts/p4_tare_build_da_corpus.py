#!/usr/bin/env python3
"""Build project-held-out known-effect datasets for TARE-DA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_SCRIPTS = (
    Path(__file__).resolve().parents[2] / "scripts"
)
sys.path.insert(0, str(PROJECT_SCRIPTS))

from p4_sim_common import load_analysis_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-matrix", default="data/tare_inputs/feature_matrix.csv")
    parser.add_argument(
        "--design",
        default="data/tare_inputs/sample_design.csv",
    )
    parser.add_argument(
        "--panel",
        default=(
            "data/tare_inputs/feature_panel.csv"
        ),
    )
    parser.add_argument(
        "--phase1-parameters",
        default="data/tare_inputs/simulation_parameters_v1.json",
    )
    parser.add_argument("--out-dir", default="tare/artifacts/da_corpus")
    parser.add_argument("--seed", type=int, default=12000)
    return parser.parse_args()


def make_effects(
    panel_features: list[str],
    scenario: str,
    seed: int,
) -> pd.DataFrame:
    if scenario == "no_effect":
        return pd.DataFrame(
            columns=["scenario", "feature", "requested_log2_effect"]
        )
    rng = np.random.default_rng(seed)
    targets = sorted(
        rng.choice(panel_features, size=min(20, len(panel_features)), replace=False)
    )
    directions = np.where(np.arange(len(targets)) % 2 == 0, 1.0, -1.0)
    return pd.DataFrame(
        {
            "scenario": scenario,
            "feature": targets,
            "requested_log2_effect": directions,
        }
    )


def simulate_project_effect(
    counts: np.ndarray,
    feature_names: list[str],
    design: pd.DataFrame,
    effects: pd.DataFrame,
    scenario: str,
    sample_prior_reads: float,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Inject a sparse treatment shift within one project.

    The requested fold change may not be exactly attainable after
    renormalization in a single project, so the realized observational effect
    is recorded in truth.csv and used as the evaluation truth.
    """
    rng = np.random.default_rng(seed)
    depths = counts.sum(axis=1)
    global_profile = counts.sum(axis=0).astype(float)
    global_profile /= global_profile.sum()
    base_probability = (
        counts + sample_prior_reads * global_profile
    ) / (depths[:, None] + sample_prior_reads)
    probability = base_probability.copy()
    treatment_rows = design["condition"].eq("treatment").to_numpy()
    feature_index = {
        name: index for index, name in enumerate(feature_names)
    }
    target_columns = np.array(
        [feature_index[name] for name in effects["feature"]],
        dtype=int,
    )
    if len(target_columns):
        multipliers = np.exp2(
            effects["requested_log2_effect"].to_numpy(dtype=float)
        )
        probability[np.ix_(treatment_rows, target_columns)] *= multipliers[
            None, :
        ]
        probability /= probability.sum(axis=1, keepdims=True)
    simulated = np.zeros_like(counts)
    for index, depth in enumerate(depths):
        simulated[index] = rng.multinomial(int(depth), probability[index])
    control_mean = base_probability[~treatment_rows].mean(axis=0)
    treated_mean = probability[treatment_rows].mean(axis=0)
    observed_log2 = np.log2(
        np.maximum(treated_mean, 1e-15)
        / np.maximum(control_mean, 1e-15)
    )
    is_target = np.zeros(len(feature_names), dtype=bool)
    is_target[target_columns] = True
    full_truth = pd.DataFrame(
        {
            "scenario": scenario,
            "feature": feature_names,
            "is_target": is_target,
            "requested_log2_effect": pd.Series(
                effects.set_index("feature")["requested_log2_effect"]
            ).reindex(feature_names).fillna(0.0).to_numpy(),
            "expected_log2_effect_after_renormalization": observed_log2,
        }
    )
    target_truth = full_truth[full_truth["is_target"]].copy()
    return simulated, target_truth, full_truth


def main() -> None:
    args = parse_args()
    data = load_analysis_data(Path(args.feature_matrix), min_depth=1000)
    design = (
        pd.read_csv(args.design, dtype={"sample": str})
        .set_index("sample")
        .loc[data.meta["sample"]]
        .reset_index()
    )
    panel_features = pd.read_csv(args.panel)["feature"].tolist()
    feature_index = {
        name: index for index, name in enumerate(data.feature_names)
    }
    indices = np.array(
        [feature_index[name] for name in panel_features], dtype=int
    )
    counts = data.counts[:, indices]
    with Path(args.phase1_parameters).open("r", encoding="utf-8") as handle:
        parameters = json.load(handle)
    sample_prior_reads = float(parameters["sample_prior_reads"])
    scenarios = {
        "no_effect": 0.0,
        "sparse20_fc2": 1.0,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    project_order = sorted(data.meta["project"].unique())
    for project_index, project in enumerate(project_order):
        project_rows = np.flatnonzero(
            data.meta["project"].eq(project).to_numpy()
        )
        project_counts = counts[project_rows]
        project_design = design.iloc[project_rows].reset_index(drop=True)
        if project_design["condition"].nunique() < 2:
            raise ValueError(f"Project lacks two conditions: {project}")
        for scenario_index, (scenario, effect) in enumerate(scenarios.items()):
            effects = make_effects(
                panel_features,
                scenario,
                seed=args.seed + 100 * project_index + scenario_index,
            )
            simulated, target_truth, full_truth = simulate_project_effect(
                project_counts,
                panel_features,
                project_design,
                effects,
                scenario,
                sample_prior_reads,
                args.seed + 1000 * project_index + scenario_index,
            )
            run_dir = out_dir / project / scenario
            run_dir.mkdir(parents=True, exist_ok=True)
            count_frame = pd.DataFrame(
                simulated.T,
                index=panel_features,
                columns=project_design["sample"].tolist(),
            )
            count_frame.to_csv(run_dir / "counts.csv")
            project_design.to_csv(run_dir / "metadata.csv", index=False)
            full_truth.to_csv(run_dir / "truth.csv", index=False)
            target_truth.to_csv(run_dir / "target_truth.csv", index=False)
            manifest.append(
                {
                    "project": project,
                    "crop": project_design["crop"].iloc[0],
                    "scenario": scenario,
                    "requested_absolute_log2_effect": effect,
                    "n_samples": len(project_design),
                    "n_features": len(panel_features),
                    "input_dir": str(run_dir.resolve()),
                    "seed": args.seed + 1000 * project_index + scenario_index,
                }
            )
    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)
    print(f"Built {len(manifest)} project-specific known-effect datasets")


if __name__ == "__main__":
    main()
