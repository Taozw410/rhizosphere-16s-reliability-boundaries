#!/usr/bin/env python3
"""Build additional tables requested in manuscript revision."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
OUT = PAPER / "manuscript/tables"
OUT.mkdir(parents=True, exist_ok=True)

METHOD_LABELS = {
    "aldex2": "ALDEx2",
    "ancombc2": "ANCOM-BC2",
    "linda": "LinDA",
    "maaslin2": "MaAsLin2",
}
NETWORK_LABELS = {
    "clr_spearman": "CLR Spearman",
    "clr_pearson": "CLR Pearson",
    "sparcc_like": "SparCC-like",
    "graphical_lasso": "Graphical lasso",
}
CLASSIFICATION_LABELS = {
    "kmer_knn": "k-mer KNN",
    "multinomial_nb": "Multinomial NB",
    "linear_sgd": "Linear SGD",
    "cnn": "CNN",
    "kmer_transformer": "k-mer Transformer",
}


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 5000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def da_summary(path: Path, generator: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rng = np.random.default_rng(20261029)
    rows = []
    for (scenario, method), group in frame.groupby(["scenario", "method"]):
        discoveries = group["n_reject_bh"].astype(float)
        total_rejections = float(discoveries.sum())
        total_fp = float(group["fp_bh"].astype(float).sum())
        row = {
            "generator": generator,
            "scenario": scenario,
            "method": METHOD_LABELS[method],
            "datasets": int(group["seed"].nunique()),
            "discovery_positive_datasets": int(
                group["fdp_bh"].notna().sum()
            ),
            "mean_fdp": float(group["fdp_bh"].mean()),
            "mean_power": float(group["power_bh"].mean()),
            "mean_f1": float(group["f1_bh"].mean()),
            "pooled_fdp": (
                total_fp / total_rejections
                if total_rejections > 0
                else np.nan
            ),
        }
        for metric in ["fdp_bh", "power_bh", "f1_bh"]:
            low, high = bootstrap_ci(group[metric].to_numpy(), rng)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    primary = da_summary(
        PAPER / "outputs/formal_grid/formal_metrics_all.csv",
        "Template multinomial",
    )
    perturbation = da_summary(
        PAPER / "outputs/formal_grid_generator2/results/formal_metrics_all.csv",
        "Correlated perturbation",
    )
    pd.concat([primary, perturbation], ignore_index=True).to_csv(
        OUT / "Supplementary_S3_DA_uncertainty.csv",
        index=False,
    )
    uncertainty = pd.concat([primary, perturbation], ignore_index=True)
    key = uncertainty[uncertainty["scenario"].eq("sparse20_fc2")].copy()
    key["Mean FDP (95% CI)"] = key.apply(
        lambda row: (
            f"{row.mean_fdp:.3f} "
            f"({row.fdp_bh_ci_low:.3f}-{row.fdp_bh_ci_high:.3f})"
        ),
        axis=1,
    )
    key["Power (95% CI)"] = key.apply(
        lambda row: (
            f"{row.mean_power:.3f} "
            f"({row.power_bh_ci_low:.3f}-{row.power_bh_ci_high:.3f})"
        ),
        axis=1,
    )
    key["F1 (95% CI)"] = key.apply(
        lambda row: (
            f"{row.mean_f1:.3f} "
            f"({row.f1_bh_ci_low:.3f}-{row.f1_bh_ci_high:.3f})"
        ),
        axis=1,
    )
    key[
        [
            "generator",
            "method",
            "datasets",
            "discovery_positive_datasets",
            "Mean FDP (95% CI)",
            "pooled_fdp",
            "Power (95% CI)",
            "F1 (95% CI)",
        ]
    ].to_csv(OUT / "Supplementary_S3_DA_key_uncertainty.csv", index=False)

    network = pd.read_csv(
        PAPER / "outputs/network_benchmark/network_metrics_by_seed.csv"
    )
    network = network[
        network["stage"].eq("validation")
        & network["scenario"].ne("network_null")
    ].copy()
    network["true_edge_fraction"] = network["true_edges"] / 780
    rows = []
    for (scenario, method), group in network.groupby(["scenario", "method_id"]):
        rows.append(
            {
                "scenario": scenario,
                "method": NETWORK_LABELS[method],
                "validation_seeds": int(group["seed"].nunique()),
                "candidate_edges": 780,
                "mean_true_edges": float(group["true_edges"].mean()),
                "mean_true_edge_fraction": float(
                    group["true_edge_fraction"].mean()
                ),
                "random_auprc_baseline": float(
                    group["true_edge_fraction"].mean()
                ),
                "mean_observed_auprc": float(group["auprc"].mean()),
                "auprc_minus_random": float(
                    group["auprc"].mean()
                    - group["true_edge_fraction"].mean()
                ),
                "auprc_seed_sd": float(group["auprc"].std(ddof=1)),
            }
        )
    network_display = pd.DataFrame(rows)
    network_display.to_csv(
        OUT / "Supplementary_S4_network_baseline.csv",
        index=False,
    )
    network_display[
        [
            "scenario",
            "method",
            "mean_true_edges",
            "mean_true_edge_fraction",
            "random_auprc_baseline",
            "mean_observed_auprc",
            "auprc_minus_random",
            "auprc_seed_sd",
        ]
    ].to_csv(OUT / "Supplementary_S4_network_baseline_key.csv", index=False)

    classification = pd.read_csv(
        PAPER / "outputs/classification_benchmark/classification_metrics_by_seed.csv"
    )
    closed = classification[
        classification["scenario"].eq("closed_set")
    ].copy()
    rows = []
    for method, group in closed.groupby("method_id"):
        rows.append(
            {
                "method": CLASSIFICATION_LABELS[method],
                "seeds": int(group["seed"].nunique()),
                "balanced_accuracy_mean": float(group["accuracy"].mean()),
                "balanced_accuracy_sd": float(group["accuracy"].std(ddof=1)),
                "macro_f1_mean": float(group["macro_f1"].mean()),
                "macro_f1_sd": float(group["macro_f1"].std(ddof=1)),
                "majority_accuracy_baseline": 0.05,
                "majority_macro_f1_baseline": 0.05 / 20,
                "training_epochs": 4,
                "optimizer_steps_per_epoch": 3,
            }
        )
    classification_display = pd.DataFrame(rows)
    classification_display.to_csv(
        OUT / "Supplementary_S5_classification_diagnostics.csv",
        index=False,
    )
    classification_display[
        [
            "method",
            "balanced_accuracy_mean",
            "balanced_accuracy_sd",
            "macro_f1_mean",
            "macro_f1_sd",
            "majority_accuracy_baseline",
            "training_epochs",
        ]
    ].to_csv(
        OUT / "Supplementary_S5_classification_diagnostics_key.csv",
        index=False,
    )

    truth_rows = [
        {
            "module": "Differential abundance",
            "ground_truth": "Target status and expected signed log2 effect after compositional renormalization",
            "method_target": "Feature-level condition contrast, with project blocking or random effects where supported",
            "primary_metrics": "FDP, power, F1, direction accuracy",
            "evaluation_boundary": "Absolute abundance is unobserved; closure effects are retained in the truth file",
        },
        {
            "module": "Core identification",
            "ground_truth": "Latent detection probability >=0.90 in >=80% of projects",
            "method_target": "Observed prevalence, project-balanced prevalence, or model-estimated occupancy",
            "primary_metrics": "False-core rate, recall, F1, Brier score",
            "evaluation_boundary": "Core is definition-dependent; detection failure is not absence",
        },
        {
            "module": "Network inference",
            "ground_truth": "Non-zero off-diagonal entries of a Gaussian precision matrix",
            "method_target": "Marginal CLR association or conditional partial association",
            "primary_metrics": "Edge AUPRC, AUROC, hub-degree Spearman",
            "evaluation_boundary": "Statistical edges are not ecological interactions",
        },
        {
            "module": "Sequence classification",
            "ground_truth": "SILVA genus label and held-out unseen-genus status",
            "method_target": "Supervised sequence classification and confidence-based rejection",
            "primary_metrics": "Accuracy, macro F1, unseen-genus rejection, known-taxon retention",
            "evaluation_boundary": "Complementary reference-set benchmark; not the same eight-summary simulation",
        },
    ]
    pd.DataFrame(truth_rows).to_csv(
        OUT / "Table5_truth_and_estimands.csv",
        index=False,
    )

    calibration = pd.read_csv(
        PAPER / "outputs/phase1/unassigned_sensitivity.csv"
    )
    calibration.to_csv(
        OUT / "Supplementary_S6_unassigned_sensitivity.csv",
        index=False,
    )
    print("Revision tables written")


if __name__ == "__main__":
    main()
