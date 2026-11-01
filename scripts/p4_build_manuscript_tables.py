#!/usr/bin/env python3
"""Build manuscript tables directly from frozen benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
MANUSCRIPT = PAPER / "manuscript"
TABLES = MANUSCRIPT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

METHOD_LABELS = {
    "aldex2": "ALDEx2",
    "ancombc2": "ANCOM-BC2",
    "linda": "LinDA",
    "maaslin2": "MaAsLin2",
}
DA_METHOD_ORDER = ["ALDEx2", "ANCOM-BC2", "LinDA", "MaAsLin2"]
DA_SCENARIO_ORDER = [
    "no_effect",
    "sparse20_fc1_19",
    "sparse20_fc1_41",
    "sparse20_fc2",
    "dense40_fc1_41",
]
DA_SCENARIO_LABELS = {
    "no_effect": "Null",
    "sparse20_fc1_19": "Sparse, 20 targets, absolute log2FC 0.25",
    "sparse20_fc1_41": "Sparse, 20 targets, absolute log2FC 0.50",
    "sparse20_fc2": "Sparse, 20 targets, absolute log2FC 1.00",
    "dense40_fc1_41": "Dense, 40 targets, absolute log2FC 0.50",
}
CORE_METHOD_ORDER = [
    "Sample prevalence",
    "Project-balanced prevalence",
    "Beta-binomial EB",
    "Hierarchical occupancy",
]
CORE_SCENARIO_ORDER = [
    "core_null",
    "core20_balanced",
    "core20_lowabundance",
    "core20_projectshift",
    "core20_depthconfound",
]
CORE_METHOD_MAP = {
    "sample_prevalence": "Sample prevalence",
    "project_balanced_prevalence": "Project-balanced prevalence",
    "beta_binomial_eb": "Beta-binomial EB",
    "hierarchical_occupancy": "Hierarchical occupancy",
}
CORE_SCENARIO_LABELS = {
    "core_null": "Null",
    "core20_balanced": "Balanced core",
    "core20_lowabundance": "Low-abundance core",
    "core20_projectshift": "Project-shift core",
    "core20_depthconfound": "Depth-confounded core",
}
NETWORK_METHODS = [
    "clr_spearman",
    "clr_pearson",
    "sparcc_like",
    "graphical_lasso",
]
NETWORK_LABELS = {
    "clr_spearman": "CLR Spearman",
    "clr_pearson": "CLR Pearson",
    "sparcc_like": "SparCC-like",
    "graphical_lasso": "Graphical lasso",
}
CLASSIFICATION_METHODS = [
    "kmer_knn",
    "multinomial_nb",
    "linear_sgd",
    "cnn",
    "kmer_transformer",
]
CLASSIFICATION_LABELS = {
    "kmer_knn": "k-mer KNN",
    "multinomial_nb": "Multinomial NB",
    "linear_sgd": "Linear SGD",
    "cnn": "CNN",
    "kmer_transformer": "k-mer Transformer",
}


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def summarize_da(path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path)
    metrics["method_label"] = metrics["method"].map(METHOD_LABELS)
    summary = (
        metrics.groupby(["scenario", "method_label"])
        .agg(
            datasets=("seed", "nunique"),
            fdp=("fdp_bh", "mean"),
            power=("power_bh", "mean"),
            f1=("f1_bh", "mean"),
        )
        .reset_index()
    )
    return summary


def main() -> None:
    # Table 1: observed project structure.
    projects = pd.read_csv(ROOT / "out_analysis/study_covariates.csv")
    projects = projects.sort_values(["crop", "project"]).reset_index(drop=True)
    projects.to_csv(TABLES / "Table1_projects.csv", index=False)
    table1 = projects.rename(
        columns={
            "project": "Project",
            "crop": "Crop",
            "n_samples": "Samples",
            "depth_median": "Median bacterial depth",
            "depth_mean": "Mean bacterial depth",
            "asv_len_mode": "Modal ASV length (bp)",
            "n_asv_median": "Median ASVs",
            "unassigned_pct": "Unassigned (%)",
            "genus_pct": "Genus assigned (%)",
            "family_pct": "Family assigned (%)",
        }
    )
    for column in [
        "Median bacterial depth",
        "Mean bacterial depth",
        "Unassigned (%)",
        "Genus assigned (%)",
        "Family assigned (%)",
    ]:
        table1[column] = table1[column].map(lambda value: fmt(value, 2))

    # Table 2 and Supplementary Table S2: DA results.
    primary_da = summarize_da(PAPER / "outputs/formal_grid/formal_metrics_all.csv")
    secondary_da = summarize_da(
        PAPER / "outputs/formal_grid_generator2/results/formal_metrics_all.csv"
    )
    primary_da["generator"] = "Generator 1"
    secondary_da["generator"] = "Generator 2"
    full_da = pd.concat([primary_da, secondary_da], ignore_index=True)

    key_da = full_da[full_da["scenario"].eq("sparse20_fc2")].copy()
    key_da["method_label"] = pd.Categorical(
        key_da["method_label"], DA_METHOD_ORDER, ordered=True
    )
    key_da = key_da.sort_values(["generator", "method_label"])
    key_da_table = key_da[
        ["generator", "method_label", "datasets", "fdp", "power", "f1"]
    ].rename(
        columns={
            "generator": "Generator",
            "method_label": "Method",
            "datasets": "Datasets",
            "fdp": "Mean FDP",
            "power": "Mean power",
            "f1": "Mean F1",
        }
    )
    for column in ["Mean FDP", "Mean power", "Mean F1"]:
        key_da_table[column] = key_da_table[column].map(lambda value: fmt(value, 3))
    key_da_table.to_csv(TABLES / "Table2_DA_key_scenario.csv", index=False)

    full_da["scenario_label"] = full_da["scenario"].map(DA_SCENARIO_LABELS)
    full_da_table = full_da[
        [
            "generator",
            "scenario_label",
            "method_label",
            "datasets",
            "fdp",
            "power",
            "f1",
        ]
    ].rename(
        columns={
            "generator": "Generator",
            "scenario_label": "Scenario",
            "method_label": "Method",
            "datasets": "Datasets",
            "fdp": "Mean FDP",
            "power": "Mean power",
            "f1": "Mean F1",
        }
    )
    for column in ["Mean FDP", "Mean power", "Mean F1"]:
        full_da_table[column] = full_da_table[column].map(lambda value: fmt(value, 3))
    full_da_table.to_csv(TABLES / "Supplementary_S2_DA_full.csv", index=False)

    # Table 3: core and occupancy.
    core = pd.read_csv(
        PAPER / "outputs/core_formal_grid/results/core_metrics_all.csv"
    )
    core["method_label"] = core["method_id"].map(CORE_METHOD_MAP)
    core_summary = (
        core.groupby(["scenario", "method_label"])
        .agg(
            datasets=("seed", "nunique"),
            f1=("f1", "mean"),
            recall=("recall", "mean"),
            precision=("precision", "mean"),
            false_core_rate=("false_core_rate", "mean"),
            brier=("brier", "mean"),
        )
        .reset_index()
    )
    core_summary["scenario_label"] = core_summary["scenario"].map(CORE_SCENARIO_LABELS)
    core_summary["method_label"] = pd.Categorical(
        core_summary["method_label"], CORE_METHOD_ORDER, ordered=True
    )
    core_summary["scenario"] = pd.Categorical(
        core_summary["scenario"], CORE_SCENARIO_ORDER, ordered=True
    )
    core_summary = core_summary.sort_values(["scenario", "method_label"])
    table3 = core_summary[
        [
            "scenario_label",
            "method_label",
            "datasets",
            "f1",
            "recall",
            "precision",
            "false_core_rate",
            "brier",
        ]
    ].rename(
        columns={
            "scenario_label": "Scenario",
            "method_label": "Method",
            "datasets": "Datasets",
            "f1": "Mean F1",
            "recall": "Mean recall",
            "precision": "Mean precision",
            "false_core_rate": "Mean false-core rate",
            "brier": "Mean Brier score",
        }
    )
    for column in [
        "Mean F1",
        "Mean recall",
        "Mean precision",
        "Mean false-core rate",
        "Mean Brier score",
    ]:
        table3[column] = table3[column].map(lambda value: fmt(value, 3))
    table3.to_csv(TABLES / "Table3_core_scenarios.csv", index=False)

    # Table 4: network and taxonomy classification.
    network = pd.read_csv(
        PAPER / "outputs/network_benchmark/network_metrics_by_seed.csv"
    )
    validation = network[
        network["stage"].eq("validation")
        & network["scenario"].ne("network_null")
    ]
    network_synth = (
        validation.groupby("method_id")
        .agg(
            seeds=("seed", "nunique"),
            auprc=("auprc", "mean"),
            auroc=("auroc", "mean"),
            hub=("hub_spearman", "mean"),
        )
        .reset_index()
    )
    stability = pd.read_csv(
        PAPER / "outputs/real_network_stability/real_network_stability_summary.csv"
    )
    network_table = network_synth.merge(stability, on="method_id", how="left")
    network_table["method"] = network_table["method_id"].map(NETWORK_LABELS)
    network_table = network_table.set_index("method").reindex(
        [NETWORK_LABELS[method] for method in NETWORK_METHODS]
    )
    network_table = network_table.reset_index()
    classification = pd.read_csv(
        PAPER / "outputs/classification_benchmark/classification_summary.csv"
    )
    closed = classification[classification["scenario"].eq("closed_set")].set_index(
        "method_id"
    )
    unseen = classification[
        classification["scenario"].eq("unseen_genus")
    ].set_index("method_id")
    classification_table = pd.DataFrame(
        {
            "Domain": "Classification",
            "Method": [CLASSIFICATION_LABELS[m] for m in CLASSIFICATION_METHODS],
            "Primary metric 1": [
                fmt(closed.loc[m, "accuracy_mean"], 3)
                for m in CLASSIFICATION_METHODS
            ],
            "Primary metric 2": [
                fmt(closed.loc[m, "macro_f1_mean"], 3)
                for m in CLASSIFICATION_METHODS
            ],
            "External metric 1": [
                fmt(unseen.loc[m, "unknown_recall_mean"], 3)
                for m in CLASSIFICATION_METHODS
            ],
            "External metric 2": [
                fmt(unseen.loc[m, "known_retention_mean"], 3)
                for m in CLASSIFICATION_METHODS
            ],
        }
    )
    network_display = pd.DataFrame(
        {
            "Domain": "Network",
            "Method": network_table["method"],
            "Primary metric 1": network_table["auprc"].map(lambda x: fmt(x, 3)),
            "Primary metric 2": network_table["hub"].map(lambda x: fmt(x, 3)),
            "External metric 1": network_table["edge_jaccard_mean"].map(
                lambda x: fmt(x, 3)
            ),
            "External metric 2": network_table["hub_spearman_mean"].map(
                lambda x: fmt(x, 3)
            ),
        }
    )
    table4 = pd.concat([network_display, classification_table], ignore_index=True)
    table4.to_csv(TABLES / "Table4_network_classification.csv", index=False)

    # Supplementary Table S1: frozen design.
    seed_grid = pd.DataFrame(
        [
            (
                "Differential abundance, generator 1",
                "5 scenarios",
                "4000–4019",
                "20 per scenario",
                100,
                "Primary 4000–4009; extension 4010–4019",
            ),
            (
                "Differential abundance, generator 2",
                "5 scenarios",
                "7200–7204",
                "5 per scenario",
                25,
                "Correlated template perturbation",
            ),
            (
                "Core and occupancy",
                "5 scenarios",
                "6200–6219",
                "20 per scenario",
                100,
                "Primary and extension seed blocks",
            ),
            (
                "Network",
                "Null, sparse, dense, project-shift",
                "5000–5004; 5100–5104",
                "5 calibration; 5 validation",
                40,
                "Disjoint calibration and validation blocks",
            ),
            (
                "Taxonomy classification",
                "Closed set and unseen genus",
                "6100–6104",
                "5 split seeds",
                5,
                "Each seed resamples genus holdout and sequence assignment",
            ),
            (
                "Project-panel validation",
                "3 disjoint panels",
                "7300–7302",
                "3",
                18,
                "Two generators; all 45 projects represented",
            ),
            (
                "Repeated project holdout",
                "10 three-fold partitions",
                "8100–8109",
                "30 held-out panels",
                30,
                "Five predictive draws per fold in the completed run",
            ),
        ],
        columns=[
            "Benchmark",
            "Design",
            "Seed block",
            "Replicates",
            "Analysis units",
            "Notes",
        ],
    )
    seed_grid.to_csv(TABLES / "Supplementary_S1_seed_design.csv", index=False)

    truth_table = pd.read_csv(
        TABLES / "Table5_truth_and_estimands.csv"
    ).rename(
        columns={
            "module": "Module",
            "ground_truth": "Ground truth",
            "method_target": "Method target",
            "primary_metrics": "Primary metrics",
            "evaluation_boundary": "Evaluation boundary",
        }
    )
    da_uncertainty_raw = pd.read_csv(
        TABLES / "Supplementary_S3_DA_uncertainty.csv"
    )
    da_uncertainty = da_uncertainty_raw[
        [
            "generator",
            "scenario",
            "method",
            "datasets",
            "discovery_positive_datasets",
            "mean_fdp",
            "fdp_bh_ci_low",
            "fdp_bh_ci_high",
            "pooled_fdp",
            "mean_power",
            "power_bh_ci_low",
            "power_bh_ci_high",
            "mean_f1",
            "f1_bh_ci_low",
            "f1_bh_ci_high",
        ]
    ].copy()
    for prefix, mean_col, low_col, high_col in [
        ("Mean FDP (95% CI)", "mean_fdp", "fdp_bh_ci_low", "fdp_bh_ci_high"),
        ("Power (95% CI)", "mean_power", "power_bh_ci_low", "power_bh_ci_high"),
        ("F1 (95% CI)", "mean_f1", "f1_bh_ci_low", "f1_bh_ci_high"),
    ]:
        da_uncertainty[prefix] = da_uncertainty.apply(
            lambda row: (
                f"{fmt(row[mean_col], 3)} "
                f"({fmt(row[low_col], 3)}-{fmt(row[high_col], 3)})"
            ),
            axis=1,
        )
    da_uncertainty["pooled_fdp"] = da_uncertainty["pooled_fdp"].map(
        lambda value: fmt(value, 3)
    )
    da_uncertainty = da_uncertainty[
        [
            "generator",
            "scenario",
            "method",
            "datasets",
            "discovery_positive_datasets",
            "Mean FDP (95% CI)",
            "pooled_fdp",
            "Power (95% CI)",
            "F1 (95% CI)",
        ]
    ].rename(
        columns={
            "generator": "Generator",
            "scenario": "Scenario",
            "method": "Method",
            "datasets": "Datasets",
            "discovery_positive_datasets": "Discovery-positive",
            "pooled_fdp": "Pooled FDP",
        }
    )
    network_baseline = pd.read_csv(
        TABLES / "Supplementary_S4_network_baseline.csv"
    )
    for column in [
        "mean_true_edges",
        "mean_true_edge_fraction",
        "random_auprc_baseline",
        "mean_observed_auprc",
        "auprc_minus_random",
        "auprc_seed_sd",
    ]:
        network_baseline[column] = network_baseline[column].map(
            lambda value: fmt(value, 3)
        )
    classification_diag = pd.read_csv(
        TABLES / "Supplementary_S5_classification_diagnostics.csv"
    )
    for column in [
        "balanced_accuracy_mean",
        "balanced_accuracy_sd",
        "macro_f1_mean",
        "macro_f1_sd",
        "majority_accuracy_baseline",
        "majority_macro_f1_baseline",
    ]:
        classification_diag[column] = classification_diag[column].map(
            lambda value: fmt(value, 3)
        )
    unassigned_sensitivity = pd.read_csv(
        TABLES / "Supplementary_S6_unassigned_sensitivity.csv"
    )
    for column in ["distance_without_unassigned", "tolerance"]:
        unassigned_sensitivity[column] = unassigned_sensitivity[column].map(
            lambda value: fmt(value, 3)
        )

    lines = [
        "# Tables",
        "",
        "All tabulated values are generated from the frozen result files. "
        "Mean FDP is undefined and shown as an em dash when a method made no "
        "discoveries in a replicate family.",
        "",
        "## Table 1 | Observed structure of the 45 rhizosphere 16S projects",
        "",
        markdown_table(table1),
        "",
        "Counts are sample-level. Median and mean bacterial depths are reads "
        "retained after bacterial-read filtering. Modal ASV length is the "
        "most frequent exact-sequence length in a project. Annotation "
        "percentages are reads assigned at the stated rank.",
        "",
        "## Table 2 | Differential abundance in the strong sparse scenario",
        "",
        markdown_table(key_da_table),
        "",
        "Generator 1 is the calibrated sample-template multinomial. Generator "
        "2 is a correlated multiplicative perturbation of the same template. "
        "The two generators use different seed counts and their absolute "
        "values are not pooled. Mean FDP is a descriptive average of "
        "per-dataset FDP values among datasets with at least one discovery; "
        "it is not an independently estimated FDR. Additional confidence "
        "intervals and pooled FDP values are given in Supplementary Table S3.",
        "",
        "## Table 3 | Recovery of latent core features",
        "",
        markdown_table(table3),
        "",
        "F1, recall and precision are undefined in the null scenario because "
        "there are no true core features. The false-core rate is therefore the "
        "primary null endpoint. All method thresholds were frozen before the "
        "formal grid.",
        "",
        "## Table 4 | Network inference and sequence-classification benchmarks",
        "",
        markdown_table(table4),
        "",
        "For network methods, primary metric 1 is mean synthetic edge AUPRC "
        "across non-null validation scenarios, primary metric 2 is mean "
        "hub-degree Spearman correlation, external metric 1 is cross-panel edge "
        "Jaccard and external metric 2 is cross-panel hub-degree Spearman. For "
        "classifiers, primary metrics are closed-set accuracy and macro F1; "
        "external metrics are unseen-genus rejection and known-taxon retention. "
        "Network true-edge prevalence, random AUPRC baselines and classifier "
        "training diagnostics are provided in Supplementary Tables S4 and S5.",
        "",
        "## Table 5 | Truth definitions, method targets and evaluation boundaries",
        "",
        markdown_table(truth_table),
        "",
        "The four modules use different estimands. They form a shared risk map "
        "rather than a single leaderboard, and the classification benchmark is "
        "a complementary reference-set analysis rather than an eight-summary "
        "simulation module.",
        "",
        "## Supplementary Table S1 | Frozen benchmark design and seed blocks",
        "",
        markdown_table(seed_grid),
        "",
        "## Supplementary Table S2 | Differential abundance across all generators and scenarios",
        "",
        markdown_table(full_da_table),
        "",
        "## Supplementary Table S3 | Differential-abundance uncertainty and pooled FDP",
        "",
        markdown_table(da_uncertainty),
        "",
        "Intervals are 95% percentile bootstrap intervals from 5,000 resamples "
        "of datasets within each generator, scenario and method. Pooled FDP is "
        "the total number of false discoveries divided by the total number of "
        "rejections; it is reported separately from the mean of per-dataset "
        "FDP values.",
        "",
        "## Supplementary Table S4 | Network true-edge prevalence and random AUPRC baselines",
        "",
        markdown_table(network_baseline),
        "",
        "The random AUPRC baseline equals the prevalence of true edges among "
        "780 candidate undirected edges. Difference from random is a "
        "descriptive benchmark and does not validate ecological interactions.",
        "",
        "## Supplementary Table S5 | Classification performance and training diagnostics",
        "",
        markdown_table(classification_diag),
        "",
        "The majority-class baseline corresponds to 20 equally represented "
        "known genera. The neural models were deliberately small, trained for "
        "four epochs without class weighting, and are not representative of "
        "extensively tuned deep-learning systems.",
        "",
        "## Supplementary Table S6 | Structural sensitivity after removing the unassigned bin",
        "",
        markdown_table(unassigned_sensitivity),
        "",
        "Removing the unassigned bin changed the composition denominator but "
        "retained all eight calibration criteria within their operational "
        "tolerances.",
    ]
    (MANUSCRIPT / "Tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("Manuscript tables written")


if __name__ == "__main__":
    main()
