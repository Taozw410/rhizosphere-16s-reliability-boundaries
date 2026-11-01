#!/usr/bin/env python3
"""Freeze and validate the Paper 4 known-effect injection interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p4_calibrate_mvp import (
    TOLERANCES,
    distance_profile,
    normalized_score,
)
from p4_sim_common import (
    STAT_NAMES,
    compute_summary_components,
    load_analysis_data,
    relative_abundance,
    summary_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-matrix",
        default="out_analysis/feature_matrix.csv",
    )
    parser.add_argument(
        "--phase1-dir",
        default="outputs/phase1",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/effect_interface",
    )
    parser.add_argument("--design-seed", type=int, default=20261025)
    parser.add_argument("--simulation-seed", type=int, default=3000)
    return parser.parse_args()


def assign_balanced_conditions(meta: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    design = meta[["project", "crop", "sample", "feature_depth"]].copy()
    design["condition"] = ""
    design["block"] = design["project"]
    for _, index in design.groupby("project").groups.items():
        index = np.asarray(list(index))
        shuffled = rng.permutation(index)
        alternating = np.array(
            ["control", "treatment"] * ((len(shuffled) + 1) // 2)
        )[: len(shuffled)]
        design.loc[shuffled, "condition"] = alternating
    counts = design.groupby(["project", "condition"]).size().unstack(fill_value=0)
    if "control" not in counts or "treatment" not in counts:
        raise ValueError("Both conditions must occur in every project")
    imbalance = (counts["control"] - counts["treatment"]).abs()
    if imbalance.max() > 1:
        raise ValueError("Project-level condition imbalance exceeds one sample")
    design["experimental_unit"] = design["project"]
    design["run_order"] = rng.permutation(len(design))
    return design


def choose_target_features(
    counts: np.ndarray,
    feature_names: list[str],
    n_targets: int,
    seed: int,
) -> list[str]:
    prevalence = (counts > 0).mean(axis=0)
    mean_relative = relative_abundance(counts).mean(axis=0)
    candidates = np.array(feature_names, dtype=object)
    eligible = (
        (candidates != "unassigned")
        & (prevalence >= 0.25)
        & (prevalence <= 0.95)
        & (mean_relative > 0)
    )
    eligible_names = candidates[eligible]
    if len(eligible_names) < n_targets:
        raise ValueError(
            f"Requested {n_targets} targets but only {len(eligible_names)} eligible"
        )
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(eligible_names, size=n_targets, replace=False).tolist())


def make_scenarios(
    counts: np.ndarray,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specification = [
        ("no_effect", 0, 0.0),
        ("sparse20_fc1_19", 20, 0.25),
        ("sparse20_fc1_41", 20, 0.50),
        ("sparse20_fc2", 20, 1.00),
        ("dense200_fc1_41", 200, 0.50),
    ]
    rows = []
    summary = []
    for scenario_index, (scenario, n_targets, absolute_log2_effect) in enumerate(
        specification
    ):
        targets = (
            []
            if n_targets == 0
            else choose_target_features(
                counts,
                feature_names,
                n_targets,
                seed=5000 + scenario_index,
            )
        )
        directions = np.where(
            np.arange(len(targets)) % 2 == 0,
            absolute_log2_effect,
            -absolute_log2_effect,
        )
        for feature, log2_effect in zip(targets, directions):
            rows.append(
                {
                    "scenario": scenario,
                    "feature": feature,
                    "requested_log2_effect": float(log2_effect),
                }
            )
        summary.append(
            {
                "scenario": scenario,
                "n_targets": len(targets),
                "absolute_log2_effect": absolute_log2_effect,
                "direction_balance": 0
                if not targets
                else float(np.mean(directions > 0)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary)


def simulate_scenario(
    counts: np.ndarray,
    feature_names: list[str],
    design: pd.DataFrame,
    effects: pd.DataFrame,
    scenario: str,
    sample_prior_reads: float,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    depths = counts.sum(axis=1)
    global_profile = counts.sum(axis=0).astype(float)
    global_profile /= global_profile.sum()
    base_probability = (
        counts + sample_prior_reads * global_profile
    ) / (depths[:, None] + sample_prior_reads)
    probability = base_probability.copy()
    feature_index = {name: index for index, name in enumerate(feature_names)}
    scenario_effects = effects[effects["scenario"].eq(scenario)]
    treatment_rows = design["condition"].eq("treatment").to_numpy()
    target_columns = np.array(
        [feature_index[name] for name in scenario_effects["feature"]],
        dtype=int,
    )
    requested_log2 = scenario_effects["requested_log2_effect"].to_numpy(dtype=float)
    multipliers = fit_effect_multipliers(
        base_probability,
        treatment_rows,
        target_columns,
        requested_log2,
    )
    if len(target_columns):
        probability[np.ix_(treatment_rows, target_columns)] *= multipliers[None, :]
    probability /= probability.sum(axis=1, keepdims=True)
    simulated = np.zeros_like(counts)
    for index, depth in enumerate(depths):
        simulated[index] = rng.multinomial(depth, probability[index])

    target_truth_rows = []
    for position, row in enumerate(scenario_effects.itertuples(index=False)):
        column = feature_index[row.feature]
        control_mean = float(base_probability[~treatment_rows, column].mean())
        treated_mean = float(probability[treatment_rows, column].mean())
        expected_log2 = float(np.log2(treated_mean / max(control_mean, 1e-12)))
        target_truth_rows.append(
            {
                "scenario": scenario,
                "feature": row.feature,
                "requested_log2_effect": row.requested_log2_effect,
                "fitted_multiplier": float(multipliers[position]),
                "expected_log2_effect_after_renormalization": expected_log2,
            }
        )
    control_mean_all = base_probability[~treatment_rows].mean(axis=0)
    treated_base_mean_all = base_probability[treatment_rows].mean(axis=0)
    treated_mean_all = probability[treatment_rows].mean(axis=0)
    case_control_observed = np.log2(
        np.maximum(treated_mean_all, 1e-15)
        / np.maximum(control_mean_all, 1e-15)
    )
    within_treatment_change = np.log2(
        np.maximum(treated_mean_all, 1e-15)
        / np.maximum(treated_base_mean_all, 1e-15)
    )
    requested_all = np.zeros(len(feature_names))
    requested_all[target_columns] = requested_log2
    is_target = np.zeros(len(feature_names), dtype=bool)
    is_target[target_columns] = True
    full_truth = pd.DataFrame(
        {
            "scenario": scenario,
            "feature": feature_names,
            "is_target": is_target,
            "requested_log2_effect": requested_all,
            "case_control_observational_log2": case_control_observed,
            "within_treatment_log2_change": within_treatment_change,
            "closure_effect_on_nontarget": np.where(
                is_target, 0.0, within_treatment_change
            ),
        }
    )
    return simulated, pd.DataFrame(target_truth_rows), full_truth


def fit_effect_multipliers(
    base_probability: np.ndarray,
    treatment_rows: np.ndarray,
    target_columns: np.ndarray,
    requested_log2: np.ndarray,
    max_iterations: int = 500,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Fit multiplicative effects so post-renormalization truth matches the request."""
    if len(target_columns) == 0:
        return np.array([])
    treated_probability = base_probability[treatment_rows]
    control_probability = base_probability[~treatment_rows]
    target_control_mean = control_probability[:, target_columns].mean(axis=0)
    desired_target_mean = np.exp2(requested_log2) * target_control_mean
    non_target_columns = np.setdiff1d(
        np.arange(base_probability.shape[1]), target_columns
    )
    fixed_case_sum = treated_probability[:, non_target_columns].sum(axis=1)
    target_case_probability = treated_probability[:, target_columns]
    multipliers = np.exp2(requested_log2)

    for _ in range(max_iterations):
        weighted = target_case_probability * multipliers[None, :]
        normalizer = fixed_case_sum + weighted.sum(axis=1)
        realized_mean = (weighted / normalizer[:, None]).mean(axis=0)
        ratio = realized_mean / np.maximum(target_control_mean, 1e-15)
        residual = np.log2(np.maximum(ratio, 1e-15)) - requested_log2
        if np.max(np.abs(residual)) < tolerance:
            break
        new_multipliers = multipliers * desired_target_mean / np.maximum(
            realized_mean, 1e-15
        )
        # Damped fixed-point update improves stability for dense target sets.
        multipliers = np.sqrt(multipliers * np.clip(new_multipliers, 1e-12, 1e12))
    else:
        raise RuntimeError("Effect multiplier fitting did not converge")
    return multipliers


def estimate_project_paired_effects(
    counts: np.ndarray,
    feature_names: list[str],
    design: pd.DataFrame,
    sample_prior_reads: float,
) -> pd.DataFrame:
    """Estimate treatment-control log2 effects with project as the paired block."""
    depths = counts.sum(axis=1)
    global_profile = counts.sum(axis=0).astype(float)
    global_profile /= global_profile.sum()
    probability = (
        counts + sample_prior_reads * global_profile
    ) / (depths[:, None] + sample_prior_reads)
    projects = design["project"].to_numpy()
    conditions = design["condition"].to_numpy()
    rows = []
    for feature_index, feature in enumerate(feature_names):
        project_effects = []
        for project in sorted(np.unique(projects)):
            project_rows = projects == project
            control = project_rows & (conditions == "control")
            treatment = project_rows & (conditions == "treatment")
            control_mean = probability[control, feature_index].mean()
            treatment_mean = probability[treatment, feature_index].mean()
            project_effects.append(
                np.log2(
                    np.maximum(treatment_mean, 1e-15)
                    / np.maximum(control_mean, 1e-15)
                )
            )
        rows.append(
            {
                "feature": feature,
                "observed_project_paired_log2": float(np.median(project_effects)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    phase1 = Path(args.phase1_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_analysis_data(Path(args.feature_matrix), min_depth=1000)
    with (phase1 / "simulation_parameters_v1.json").open(
        "r", encoding="utf-8"
    ) as handle:
        parameters = json.load(handle)
    if parameters.get("calibration", {}).get("n_pass") != 8:
        raise ValueError("Phase 1 calibration has not passed 8/8")
    sample_prior_reads = float(parameters["sample_prior_reads"])

    design = assign_balanced_conditions(data.meta, args.design_seed)
    design.to_csv(out_dir / "sample_design.csv", index=False)
    effects, scenario_summary = make_scenarios(
        data.counts, data.feature_names
    )
    effects.to_csv(out_dir / "effect_scenarios_long.csv", index=False)
    scenario_summary.to_csv(out_dir / "effect_scenarios_summary.csv", index=False)

    real_components = compute_summary_components(
        data.counts, data.meta, data.feature_names
    )
    scenario_rows = []
    truth_parts = []
    full_truth_parts = []
    recovery_parts = []
    for scenario_index, scenario in enumerate(scenario_summary["scenario"]):
        synthetic, truth, full_truth = simulate_scenario(
            data.counts,
            data.feature_names,
            design,
            effects,
            scenario,
            sample_prior_reads,
            args.simulation_seed + scenario_index,
        )
        synthetic_components = compute_summary_components(
            synthetic, data.meta, data.feature_names
        )
        distances = distance_profile(real_components, synthetic_components)
        observed_effects = estimate_project_paired_effects(
            synthetic, data.feature_names, design, sample_prior_reads
        )
        recovery = full_truth.merge(
            observed_effects, on="feature", validate="one_to_one"
        )
        recovery["scenario"] = scenario
        recovery_parts.append(recovery)
        target_recovery = recovery[recovery["is_target"]]
        sign_accuracy = (
            float(
                np.mean(
                    np.sign(target_recovery["observed_project_paired_log2"])
                    == np.sign(target_recovery["requested_log2_effect"])
                )
            )
            if len(target_recovery)
            else 1.0
        )
        recovery_correlation = (
            float(
                np.corrcoef(
                    target_recovery["requested_log2_effect"],
                    target_recovery["observed_project_paired_log2"],
                )[0, 1]
            )
            if len(target_recovery) >= 3
            and target_recovery["requested_log2_effect"].nunique() > 1
            else np.nan
        )
        row = {
            "scenario": scenario,
            "n_targets": int(
                scenario_summary.loc[
                    scenario_summary["scenario"].eq(scenario), "n_targets"
                ].iloc[0]
            ),
            "normalized_calibration_score": normalized_score(distances),
            "n_calibration_pass": sum(
                distances[name] <= TOLERANCES[name] for name in STAT_NAMES
            ),
            **{f"distance_{key}": value for key, value in distances.items()},
            "max_abs_requested_effect_error": float(
                (
                    truth["expected_log2_effect_after_renormalization"]
                    - truth["requested_log2_effect"]
                )
                .abs()
                .max()
            )
            if len(truth)
            else 0.0,
            "max_abs_nontarget_closure_effect": float(
                full_truth.loc[
                    ~full_truth["is_target"],
                    "closure_effect_on_nontarget",
                ].abs().max()
            ),
            "target_sign_recovery": sign_accuracy,
            "target_effect_correlation": recovery_correlation,
        }
        scenario_rows.append(row)
        truth_parts.append(truth)
        full_truth_parts.append(full_truth)
        np.savez_compressed(
            out_dir / f"{scenario}_counts.npz",
            counts=synthetic.astype(np.int32),
            feature_names=np.array(data.feature_names, dtype=str),
            sample=np.array(data.meta["sample"], dtype=str),
            project=np.array(data.meta["project"], dtype=str),
        )
        print(
            f"{scenario}: targets={row['n_targets']} "
            f"calibration={row['n_calibration_pass']}/8"
        )

    scenario_metrics = pd.DataFrame(scenario_rows)
    scenario_metrics.to_csv(out_dir / "effect_scenario_metrics.csv", index=False)
    truth = pd.concat(truth_parts, ignore_index=True)
    truth["realized_minus_requested"] = (
        truth["expected_log2_effect_after_renormalization"]
        - truth["requested_log2_effect"]
    )
    truth.to_csv(out_dir / "target_feature_truth.csv", index=False)
    full_truth = pd.concat(full_truth_parts, ignore_index=True)
    full_truth.to_csv(out_dir / "full_feature_truth.csv", index=False)
    recovery = pd.concat(recovery_parts, ignore_index=True)
    recovery.to_csv(out_dir / "simple_effect_recovery.csv", index=False)

    balance = (
        design.groupby(["project", "condition"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    balance.to_csv(out_dir / "condition_balance_by_project.csv", index=False)
    summary = {
        "design_seed": args.design_seed,
        "simulation_seed": args.simulation_seed,
        "experimental_unit": "project",
        "assignment": "balanced control/treatment within each project",
        "sample_prior_reads": sample_prior_reads,
        "max_project_condition_imbalance": int(
            (balance["control"] - balance["treatment"]).abs().max()
        ),
        "scenarios": scenario_metrics.to_dict(orient="records"),
        "max_abs_requested_effect_error": float(
            scenario_metrics["max_abs_requested_effect_error"].max()
        ),
        "no_effect_8_of_8": bool(
            scenario_metrics.loc[
                scenario_metrics["scenario"].eq("no_effect"),
                "n_calibration_pass",
            ].iloc[0]
            == 8
        ),
    }
    with (out_dir / "effect_interface_status.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    report = [
        "# Paper 4 effect-injection interface",
        "",
        "- Experimental unit: project.",
        "- Assignment: balanced control/treatment within project.",
        f"- Design seed: {args.design_seed}.",
        f"- Maximum within-project group imbalance: "
        f"{summary['max_project_condition_imbalance']} sample.",
        f"- No-effect scenario 8/8 calibration: "
        f"{'yes' if summary['no_effect_8_of_8'] else 'no'}.",
        f"- Maximum absolute target-effect error: "
        f"{summary['max_abs_requested_effect_error']:.2e} log2.",
        "",
        "| Scenario | Targets | Calibration pass | Normalized score | Sign recovery | Effect r | Max closure |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scenario_metrics.itertuples(index=False):
        report.append(
            f"| {row.scenario} | {row.n_targets} | "
            f"{row.n_calibration_pass}/8 | "
            f"{row.normalized_calibration_score:.4f} | "
            f"{row.target_sign_recovery:.3f} | "
            f"{row.target_effect_correlation:.3f} | "
            f"{row.max_abs_nontarget_closure_effect:.4f} |"
        )
    report.extend(
        [
            "",
            "## Effect definition",
            "",
            "For treatment samples, feature multipliers are iteratively fitted so "
            "the expected post-renormalization log2 effect equals the requested "
            "value. target_feature_truth.csv records target effects; "
            "full_feature_truth.csv also records closure-induced changes in "
            "non-target features. Effects are injected into the calibrated sample "
            "template; no project or sample is moved across blocks.",
        ]
    )
    (out_dir / "effect_interface_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"Effect interface written to {out_dir}")


if __name__ == "__main__":
    main()
