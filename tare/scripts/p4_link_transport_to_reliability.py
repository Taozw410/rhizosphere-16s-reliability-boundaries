#!/usr/bin/env python3
"""Relate structural simulator mismatch to TARE interval and decision errors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

TARE_DIR = Path(__file__).resolve().parents[1]
PROJECT_SCRIPTS = (
    Path(__file__).resolve().parents[2] / "scripts"
)
sys.path.insert(0, str(TARE_DIR))
sys.path.insert(0, str(PROJECT_SCRIPTS))

from p4_calibrate_mvp import TOLERANCES, distance_profile
from p4_sim_common import compute_summary_components, load_analysis_data
from p4_tare_transport import study_feature_table


ROOT = Path(__file__).resolve().parents[2]
OUT = TARE_DIR / "artifacts" / "transport_reliability_link"


def correlation_matrix(counts: np.ndarray) -> np.ndarray:
    relative = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    depth = counts.sum(axis=1, keepdims=True)
    probability = (counts + 0.5) / (depth + 0.5 * counts.shape[1])
    logged = np.log(probability)
    logged -= logged.mean(axis=1, keepdims=True)
    correlation = np.corrcoef(rankdata(logged, axis=0), rowvar=False)
    correlation = np.nan_to_num(correlation)
    correlation = 0.5 * correlation + 0.5 * np.eye(correlation.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    correlation = (eigenvectors * np.maximum(eigenvalues, 1e-6)) @ eigenvectors.T
    scale = np.sqrt(np.diag(correlation))
    return correlation / scale[:, None] / scale[None, :]


def simulate_panel(
    counts: np.ndarray,
    meta: pd.DataFrame,
    feature_table: pd.DataFrame,
    training_projects: list[str],
    held_projects: list[str],
    project_crop: pd.Series,
    seed: int,
    correlated: bool,
) -> np.ndarray:
    train_x = feature_table.loc[training_projects].to_numpy(dtype=float)
    x_mean = train_x.mean(axis=0)
    x_sd = train_x.std(axis=0)
    x_sd[x_sd == 0] = 1
    train_scaled = (train_x - x_mean) / x_sd
    training_rows = meta["project"].isin(training_projects).to_numpy()
    global_profile = counts[training_rows].sum(axis=0).astype(float)
    global_profile /= global_profile.sum()
    correlation = (
        correlation_matrix(counts[training_rows])
        if correlated
        else None
    )
    rng = np.random.default_rng(seed)
    simulated = np.zeros_like(counts)
    for project in sorted(held_projects):
        rows = np.flatnonzero(meta["project"].eq(project).to_numpy())
        crop = str(project_crop.loc[project])
        candidates = [
            name
            for name in training_projects
            if str(project_crop.loc[name]) == crop
        ]
        candidate_indices = np.array(
            [training_projects.index(name) for name in candidates], dtype=int
        )
        target = (
            feature_table.loc[[project]].to_numpy(dtype=float) - x_mean
        ) / x_sd
        distances = np.sqrt(
            ((train_scaled[candidate_indices] - target) ** 2).sum(axis=1)
        )
        nearest_indices = candidate_indices[
            np.argsort(distances)[: min(3, len(candidate_indices))]
        ]
        donors = [training_projects[index] for index in nearest_indices]
        for row in rows:
            donor = str(rng.choice(donors))
            donor_rows = np.flatnonzero(
                meta["project"].eq(donor).to_numpy()
            )
            donor_row = donor_rows[rng.integers(len(donor_rows))]
            depth = int(counts[row].sum())
            probability = (
                counts[donor_row] + 0.5 * global_profile
            ) / (counts[donor_row].sum() + 0.5)
            if correlation is not None:
                latent = rng.multivariate_normal(
                    np.zeros(counts.shape[1]), correlation
                )
                probability = probability * np.exp(0.03 * latent)
                probability /= probability.sum()
            simulated[row] = rng.multinomial(depth, probability)
    return simulated


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_analysis_data(
        ROOT / "out_analysis" / "feature_matrix.csv", min_depth=1000
    )
    design = pd.read_csv(
        ROOT
        / "论文4-多研究仿真"
        / "outputs"
        / "effect_interface"
        / "sample_design.csv",
        dtype={"sample": str},
    ).set_index("sample")
    meta = design.loc[data.meta["sample"]].reset_index()
    sample_meta = pd.read_csv(ROOT / "out_analysis" / "sample_meta.csv")
    project_crop = (
        sample_meta[["project", "crop"]]
        .drop_duplicates("project")
        .set_index("project")["crop"]
    )
    study = pd.read_csv(
        ROOT / "out_analysis" / "study_covariates.csv"
    ).set_index("project")
    feature_table = study_feature_table(study, project_crop)
    panel = pd.read_csv(
        ROOT
        / "论文4-多研究仿真"
        / "outputs"
        / "formal_grid"
        / "data"
        / "sparse20_fc2"
        / "feature_panel.csv"
    )["feature"].tolist()
    feature_index = {
        name: index for index, name in enumerate(data.feature_names)
    }
    indices = np.array([feature_index[name] for name in panel], dtype=int)
    counts = data.counts[:, indices]
    valid = counts.sum(axis=1) > 0
    counts = counts[valid]
    meta = meta.loc[valid].reset_index(drop=True)
    meta["feature_depth"] = counts.sum(axis=1)
    predictions = pd.read_csv(
        TARE_DIR / "artifacts" / "tare_da" / "tare_da_predictions.csv"
    )
    predictions["abs_error"] = (
        predictions["tare_point"] - predictions["actual_utility"]
    ).abs()
    predictions["interval_width"] = (
        predictions["tare_upper"] - predictions["tare_lower"]
    )
    predictions["decision"] = predictions["decision"].fillna("")
    fold_reliability = (
        predictions.groupby(["partition", "fold"])
        .agg(
            tare_mae=("abs_error", "mean"),
            coverage=("covered", "mean"),
            mean_interval_width=("interval_width", "mean"),
            allow_count=("decision", lambda values: int((values == "ALLOW").sum())),
            false_allow_count=(
                "actual_reliable",
                lambda values: 0,
            ),
        )
        .reset_index()
    )
    false_allow = (
        predictions[predictions["decision"].eq("ALLOW")
                    & predictions["actual_reliable"].eq(0)]
        .groupby(["partition", "fold"])
        .size()
        .rename("false_allow_count")
    )
    fold_reliability = fold_reliability.drop(
        columns=["false_allow_count"]
    ).merge(
        false_allow,
        on=["partition", "fold"],
        how="left",
    )
    fold_reliability["false_allow_count"] = (
        fold_reliability["false_allow_count"].fillna(0).astype(int)
    )
    fold_reliability["false_allow_rate"] = (
        fold_reliability["false_allow_count"]
        / fold_reliability["allow_count"].replace(0, np.nan)
    )

    structural_rows = []
    all_projects = sorted(project_crop.index)
    for (partition, fold), group in predictions.groupby(["partition", "fold"]):
        held_projects = sorted(group["project"].unique().tolist())
        training_projects = sorted(set(all_projects).difference(held_projects))
        held_rows = meta["project"].isin(held_projects).to_numpy()
        real_components = compute_summary_components(
            counts[held_rows],
            meta.loc[held_rows].reset_index(drop=True),
            panel,
        )
        for generator, correlated in [
            ("nearest_three_template", False),
            ("nearest_three_correlated", True),
        ]:
            simulated_full = simulate_panel(
                counts,
                meta,
                feature_table,
                training_projects,
                held_projects,
                project_crop,
                seed=260000 + 1000 * partition + 10 * fold + int(correlated),
                correlated=correlated,
            )
            simulated_components = compute_summary_components(
                simulated_full[held_rows],
                meta.loc[held_rows].reset_index(drop=True),
                panel,
            )
            distances = distance_profile(
                real_components,
                simulated_components,
            )
            structural_rows.append(
                {
                    "partition": partition,
                    "fold": fold,
                    "generator": generator,
                    "structural_n_pass": sum(
                        distances[name] <= TOLERANCES[name]
                        for name in TOLERANCES
                    ),
                    **{
                        f"distance_{name}": distances[name]
                        for name in TOLERANCES
                    },
                }
            )
    structural = pd.DataFrame(structural_rows)
    structural.to_csv(
        OUT / "structural_mismatch_by_tare_fold.csv", index=False
    )
    linked = structural.merge(
        fold_reliability,
        on=["partition", "fold"],
        how="left",
    )
    linked.to_csv(OUT / "linked_fold_metrics.csv", index=False)

    correlation_rows = []
    for generator, subset in linked.groupby("generator"):
        for metric in [
            "tare_mae",
            "coverage",
            "mean_interval_width",
            "false_allow_rate",
        ]:
            valid = subset[["structural_n_pass", metric]].dropna()
            rho, p_value = spearmanr(
                valid["structural_n_pass"], valid[metric]
            )
            correlation_rows.append(
                {
                    "generator": generator,
                    "reliability_metric": metric,
                    "spearman_rho": rho,
                    "descriptive_p": p_value,
                    "folds": len(valid),
                }
            )
        for summary in TOLERANCES:
            column = f"distance_{summary}"
            valid = subset[[column, "tare_mae"]].dropna()
            rho, p_value = spearmanr(
                valid[column], valid["tare_mae"]
            )
            correlation_rows.append(
                {
                    "generator": generator,
                    "reliability_metric": f"tare_mae_vs_{summary}",
                    "spearman_rho": rho,
                    "descriptive_p": p_value,
                    "folds": len(valid),
                }
            )
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(OUT / "mismatch_correlations.csv", index=False)

    strata_rows = []
    for generator, subset in linked.groupby("generator"):
        subset = subset.copy()
        subset["structural_stratum"] = pd.cut(
            subset["structural_n_pass"],
            bins=[-0.5, 4.5, 6.5, 8.5],
            labels=["0-4", "5-6", "7-8"],
        )
        for stratum, stratum_subset in subset.groupby(
            "structural_stratum", observed=True
        ):
            strata_rows.append(
                {
                    "generator": generator,
                    "structural_n_pass": str(stratum),
                    "folds": len(stratum_subset),
                    "tare_mae": stratum_subset["tare_mae"].mean(),
                    "coverage": stratum_subset["coverage"].mean(),
                    "mean_interval_width": stratum_subset[
                        "mean_interval_width"
                    ].mean(),
                    "allow_count": stratum_subset["allow_count"].sum(),
                    "false_allow_count": stratum_subset[
                        "false_allow_count"
                    ].sum(),
                    "false_allow_rate": (
                        stratum_subset["false_allow_count"].sum()
                        / max(stratum_subset["allow_count"].sum(), 1)
                    ),
                }
            )
    strata = pd.DataFrame(strata_rows)
    strata.to_csv(OUT / "mismatch_strata.csv", index=False)
    print(strata.to_string(index=False))
    print(
        correlations[
            correlations["reliability_metric"].isin(
                ["tare_mae", "coverage", "mean_interval_width", "false_allow_rate"]
            )
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
