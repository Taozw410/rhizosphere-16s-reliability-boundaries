#!/usr/bin/env python3
"""Known-truth network simulation and method benchmark for Paper 4."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.covariance import graphical_lasso
from sklearn.metrics import average_precision_score, roc_auc_score


SCENARIOS = {
    "network_null": {"edge_probability": 0.0, "project_sd": 0.4},
    "network_sparse": {"edge_probability": 0.04, "project_sd": 0.4},
    "network_dense": {"edge_probability": 0.10, "project_sd": 0.4},
    "network_projectshift": {
        "edge_probability": 0.04,
        "project_sd": 1.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="outputs/network_benchmark",
    )
    parser.add_argument("--n-nodes", type=int, default=40)
    parser.add_argument("--n-projects", type=int, default=15)
    parser.add_argument("--samples-per-project", type=int, default=20)
    parser.add_argument("--calibration-seeds", type=int, default=5)
    parser.add_argument("--validation-seeds", type=int, default=5)
    return parser.parse_args()


def make_graph(n_nodes: int, edge_probability: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=int)
    if edge_probability <= 0:
        return adjacency
    for left, right in itertools.combinations(range(n_nodes), 2):
        if rng.random() < edge_probability:
            adjacency[left, right] = adjacency[right, left] = 1
    return adjacency


def graph_covariance(adjacency: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_nodes = adjacency.shape[0]
    precision = np.eye(n_nodes)
    for left, right in itertools.combinations(range(n_nodes), 2):
        if adjacency[left, right]:
            value = rng.choice([-1.0, 1.0]) * rng.uniform(0.15, 0.35)
            precision[left, right] = precision[right, left] = value
    minimum = float(np.min(np.linalg.eigvalsh(precision)))
    if minimum <= 0.2:
        precision += np.eye(n_nodes) * (0.25 - minimum)
    covariance = np.linalg.inv(precision)
    scale = np.sqrt(np.diag(covariance))
    return covariance / scale[:, None] / scale[None, :]


def simulate_network(
    n_nodes: int,
    n_projects: int,
    samples_per_project: int,
    edge_probability: float,
    project_sd: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    adjacency = make_graph(n_nodes, edge_probability, seed + 701)
    covariance = graph_covariance(adjacency, seed + 911)
    baseline = rng.normal(0, 0.6, size=n_nodes)
    counts = []
    metadata = []
    for project in range(n_projects):
        project_mean = baseline + rng.normal(0, project_sd, size=n_nodes)
        latent = rng.multivariate_normal(
            project_mean,
            covariance,
            size=samples_per_project,
        )
        probabilities = softmax(latent, axis=1)
        depth = rng.lognormal(np.log(10000), 0.45, size=samples_per_project)
        for local, sample_depth in enumerate(depth):
            abundance_rank = probabilities[local] / probabilities[local].max()
            presence_probability = np.clip(0.58 + 0.38 * abundance_rank, 0.05, 0.97)
            present = rng.random(n_nodes) < presence_probability
            if not present.any():
                present[np.argmax(probabilities[local])] = True
            sample_probability = probabilities[local] * present
            sample_probability /= sample_probability.sum()
            counts.append(
                rng.multinomial(int(sample_depth), sample_probability)
            )
            metadata.append(
                {
                    "sample": f"P{project:02d}_S{local:03d}",
                    "project": f"P{project:02d}",
                }
            )
    return np.vstack(counts), adjacency, pd.DataFrame(metadata)


def clr(counts: np.ndarray) -> np.ndarray:
    depth = counts.sum(axis=1, keepdims=True)
    probability = (counts + 0.5) / (depth + 0.5 * counts.shape[1])
    logged = np.log(probability)
    return logged - logged.mean(axis=1, keepdims=True)


def spearman_score(counts: np.ndarray) -> np.ndarray:
    transformed = clr(counts)
    score = spearmanr(transformed).statistic
    score = np.nan_to_num(np.abs(score))
    np.fill_diagonal(score, 0)
    return score


def pearson_score(counts: np.ndarray) -> np.ndarray:
    transformed = clr(counts)
    score = np.nan_to_num(np.abs(np.corrcoef(transformed, rowvar=False)))
    np.fill_diagonal(score, 0)
    return score


def sparcc_score(counts: np.ndarray, iterations: int = 10) -> np.ndarray:
    transformed = clr(counts)
    covariance = np.cov(transformed, rowvar=False)
    variance = np.diag(covariance).copy()
    for _ in range(iterations):
        proportion = covariance / np.sqrt(
            np.maximum(variance[:, None] * variance[None, :], 1e-12)
        )
        basis = np.maximum(-np.min(proportion), 1e-6)
        variance = np.diag(covariance) + basis
        covariance = covariance - basis
    score = -2.0 * covariance / np.sqrt(
        np.maximum(variance[:, None] * variance[None, :], 1e-12)
    )
    score = np.nan_to_num(np.abs(score))
    np.fill_diagonal(score, 0)
    return score


def glasso_score(counts: np.ndarray, alpha: float = 0.12) -> np.ndarray:
    covariance, precision = graphical_lasso(
        np.cov(clr(counts), rowvar=False),
        alpha=alpha,
        max_iter=500,
    )
    diagonal = np.diag(precision)
    partial = -precision / np.sqrt(
        np.maximum(diagonal[:, None] * diagonal[None, :], 1e-12)
    )
    score = np.nan_to_num(np.abs(partial))
    np.fill_diagonal(score, 0)
    return score


METHODS = {
    "clr_spearman": spearman_score,
    "clr_pearson": pearson_score,
    "sparcc_like": sparcc_score,
    "graphical_lasso": glasso_score,
}


def evaluate(
    score: np.ndarray,
    adjacency: np.ndarray,
) -> dict[str, float]:
    upper = np.triu_indices_from(score, k=1)
    labels = adjacency[upper]
    values = score[upper]
    if labels.sum() > 0:
        auprc = float(average_precision_score(labels, values))
        auroc = float(roc_auc_score(labels, values))
    else:
        auprc = np.nan
        auroc = np.nan
    true_degree = adjacency.sum(axis=1)
    estimated_degree = (score >= np.quantile(values, 0.9)).sum(axis=1)
    hub_correlation = (
        float(pd.Series(true_degree).corr(pd.Series(estimated_degree), method="spearman"))
        if labels.sum() > 0
        else np.nan
    )
    return {
        "auprc": auprc,
        "auroc": auroc,
        "hub_spearman": hub_correlation,
        "mean_degree_90pct": float(np.mean(estimated_degree)),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario, settings in SCENARIOS.items():
        stage_specs = [
            ("calibration", 5000, args.calibration_seeds),
            ("validation", 5100, args.validation_seeds),
        ]
        for stage, start, count in stage_specs:
            for seed in range(start, start + count):
                counts, adjacency, metadata = simulate_network(
                    args.n_nodes,
                    args.n_projects,
                    args.samples_per_project,
                    settings["edge_probability"],
                    settings["project_sd"],
                    seed,
                )
                run_dir = out_dir / "data" / stage / scenario / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    run_dir / "data.npz",
                    counts=counts,
                    adjacency=adjacency,
                )
                metadata.to_csv(run_dir / "metadata.csv", index=False)
                for method_id, method in METHODS.items():
                    score = method(counts)
                    metrics = evaluate(score, adjacency)
                    upper = np.triu_indices_from(score, k=1)
                    predictions = pd.DataFrame(
                        {
                            "node_i": upper[0],
                            "node_j": upper[1],
                            "score": score[upper],
                            "is_edge": adjacency[upper],
                        }
                    )
                    predictions.to_csv(
                        run_dir / f"{method_id}_edges.csv.gz",
                        index=False,
                        compression="gzip",
                    )
                    rows.append(
                        {
                            "stage": stage,
                            "scenario": scenario,
                            "seed": seed,
                            "method_id": method_id,
                            **metrics,
                            "true_edges": int(adjacency.sum() / 2),
                        }
                    )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "network_metrics_by_seed.csv", index=False)
    summary = (
        metrics.groupby(["stage", "scenario", "method_id"])
        .agg(
            seeds=("seed", "nunique"),
            auprc_mean=("auprc", "mean"),
            auroc_mean=("auroc", "mean"),
            hub_spearman_mean=("hub_spearman", "mean"),
            mean_degree_90pct=("mean_degree_90pct", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "network_summary.csv", index=False)
    ranking = (
        summary[~summary["scenario"].eq("network_null")]
        .groupby("method_id")
        .agg(
            mean_auprc=("auprc_mean", "mean"),
            mean_hub_spearman=("hub_spearman_mean", "mean"),
        )
        .reset_index()
        .sort_values("mean_auprc", ascending=False)
    )
    ranking.to_csv(out_dir / "network_ranking.csv", index=False)
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
