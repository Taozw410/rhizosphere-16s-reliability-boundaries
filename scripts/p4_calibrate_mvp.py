#!/usr/bin/env python3
"""Calibrate the Paper 4 simulation MVP against eight pre-specified summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from p4_sim_common import (
    STAT_NAMES,
    compute_summary_components,
    load_analysis_data,
    simulate_template_multinomial,
    summary_values,
)


TOLERANCES = {
    "prevalence_distribution": 0.12,
    "rank_abundance": 0.12,
    "sample_depth": 0.05,
    "within_between_variance_ratio": 0.30,
    "bray_curtis_distribution": 0.10,
    "aitchison_distribution": 0.10,
    "zero_fraction": 0.08,
    "network_topology": 0.20,
}

VALUE_KEYS = {
    "prevalence_distribution": "prevalence_quantiles",
    "rank_abundance": "rank_abundance_log10_quantiles",
    "sample_depth": "depth_log_quantiles",
    "within_between_variance_ratio": "variance_ratio",
    "bray_curtis_distribution": "bray_quantiles",
    "aitchison_distribution": "aitchison_quantiles",
    "zero_fraction": "sample_zero_fraction_quantiles",
    "network_topology": "network_density_quantiles",
}


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
    parser.add_argument("--search-seeds", type=int, default=2)
    parser.add_argument("--final-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=1000)
    return parser.parse_args()


def array_distance(real: np.ndarray, synthetic: np.ndarray) -> float:
    return float(ks_2samp(real, synthetic).statistic)


def distance_profile(real: dict[str, object], sim: dict[str, object]) -> dict[str, float]:
    real_network = real["network"]
    sim_network = sim["network"]
    if not isinstance(real_network, pd.DataFrame) or not isinstance(
        sim_network, pd.DataFrame
    ):
        raise TypeError("network components must be DataFrames")
    variance_ratio = abs(
        float(sim["variance_ratio"]) - float(real["variance_ratio"])
    ) / max(float(real["variance_ratio"]), 1e-6)
    zero = max(
        abs(
            float(sim["global_zero_fraction"])
            - float(real["global_zero_fraction"])
        ),
        array_distance(
            real["sample_zero_fraction"], sim["sample_zero_fraction"]
        ),
    )
    density_ks = array_distance(
        real_network["edge_density"].dropna(),
        sim_network["edge_density"].dropna(),
    )
    centrality_ks = array_distance(
        real_network["centrality_cv"].dropna(),
        sim_network["centrality_cv"].dropna(),
    )
    return {
        "prevalence_distribution": array_distance(
            real["prevalence"], sim["prevalence"]
        ),
        "rank_abundance": array_distance(
            np.log10(np.maximum(real["rank_abundance"], 1e-10)),
            np.log10(np.maximum(sim["rank_abundance"], 1e-10)),
        ),
        "sample_depth": array_distance(real["depth"], sim["depth"]),
        "within_between_variance_ratio": variance_ratio,
        "bray_curtis_distribution": array_distance(
            real["bray"], sim["bray"]
        ),
        "aitchison_distribution": array_distance(
            real["aitchison"], sim["aitchison"]
        ),
        "zero_fraction": zero,
        "network_topology": max(density_ks, centrality_ks),
        "network_density_ks": density_ks,
        "network_centrality_cv_ks": centrality_ks,
    }


def normalized_score(distances: dict[str, float]) -> float:
    return float(
        np.mean([distances[name] / TOLERANCES[name] for name in STAT_NAMES])
    )


def summarize_simulation(
    counts: np.ndarray,
    meta: pd.DataFrame,
    feature_names: list[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    components = compute_summary_components(counts, meta, feature_names)
    values = summary_values(components)
    network = components["network"]
    if not isinstance(network, pd.DataFrame):
        raise TypeError("network summary must be a DataFrame")
    return components, network


def flatten_values(values: dict[str, object], seed: int) -> dict[str, object]:
    row: dict[str, object] = {"seed": seed}
    for key, value in values.items():
        if np.isscalar(value):
            row[key] = float(value)
        else:
            for index, item in enumerate(np.asarray(value)):
                row[f"{key}_{index}"] = float(item)
    return row


def calibration_table(
    real_values: dict[str, object],
    synthetic_summary: pd.DataFrame,
    mean_distances: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for statistic in STAT_NAMES:
        key = VALUE_KEYS[statistic]
        real_value = real_values[key]
        if np.isscalar(real_value):
            real_text = f"{float(real_value):.6g}"
            synthetic_text = (
                f"{synthetic_summary[key].mean():.6g} "
                f"[{synthetic_summary[key].quantile(0.05):.6g}, "
                f"{synthetic_summary[key].quantile(0.95):.6g}]"
            )
        else:
            columns = [
                column
                for column in synthetic_summary.columns
                if column.startswith(f"{key}_")
            ]
            real_text = ", ".join(f"{float(x):.5g}" for x in real_value)
            synthetic_text = "; ".join(
                f"{synthetic_summary[column].mean():.5g}"
                for column in columns
            )
        distance = float(mean_distances[statistic])
        rows.append(
            {
                "statistic": statistic,
                "real_summary": real_text,
                "synthetic_mean_summary": synthetic_text,
                "distance": distance,
                "tolerance": TOLERANCES[statistic],
                "pass": distance <= TOLERANCES[statistic],
            }
        )
    return pd.DataFrame(rows)


def update_parameters(
    path: Path,
    sample_prior_reads: float,
    diagnostics: dict[str, object],
) -> None:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["simulator_family"] = "template_multinomial_v1"
    payload["sample_prior_reads"] = sample_prior_reads
    payload["calibration"] = diagnostics
    payload["hierarchical_generator_status"] = (
        "not calibrated; project-mean plus independent presence model "
        "overestimated zero fraction and underestimated co-occurrence"
    )
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    phase1 = Path(args.phase1_dir)
    phase1.mkdir(parents=True, exist_ok=True)
    data = load_analysis_data(Path(args.feature_matrix), min_depth=1000)
    real_components = compute_summary_components(
        data.counts, data.meta, data.feature_names
    )
    real_values = summary_values(real_components)
    real_network = real_components["network"]
    if not isinstance(real_network, pd.DataFrame):
        raise TypeError("real network summary must be a DataFrame")
    real_network.to_csv(phase1 / "real_project_network_stats.csv", index=False)

    global_profile = data.counts.sum(axis=0).astype(float)
    global_profile /= global_profile.sum()
    prior_grid = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    search_rows = []
    for prior in prior_grid:
        seed_distances = []
        for offset in range(args.search_seeds):
            seed = args.seed_start + offset
            synthetic = simulate_template_multinomial(
                data.counts, global_profile, prior, seed
            )
            simulated_components, _ = summarize_simulation(
                synthetic, data.meta, data.feature_names
            )
            seed_distances.append(
                distance_profile(real_components, simulated_components)
            )
        mean_distances = {
            key: float(np.mean([row[key] for row in seed_distances]))
            for key in seed_distances[0]
        }
        row = {
            "sample_prior_reads": prior,
            "normalized_score": normalized_score(mean_distances),
            **{
                f"distance_{key}": value
                for key, value in mean_distances.items()
            },
        }
        row.update(
            {
                f"pass_{key}": mean_distances[key] <= TOLERANCES[key]
                for key in STAT_NAMES
            }
        )
        row["n_pass"] = sum(row[f"pass_{key}"] for key in STAT_NAMES)
        search_rows.append(row)
        print(
            f"prior={prior:g} score={row['normalized_score']:.4f} "
            f"passes={row['n_pass']}/8"
        )

    search = pd.DataFrame(search_rows).sort_values(
        ["n_pass", "normalized_score"], ascending=[False, True]
    )
    search.to_csv(phase1 / "calibration_search.csv", index=False)
    best = search.iloc[0]
    sample_prior_reads = float(best["sample_prior_reads"])

    distance_rows = []
    synthetic_rows = []
    synthetic_networks = []
    first_synthetic = None
    for offset in range(args.final_seeds):
        seed = args.seed_start + offset
        synthetic = simulate_template_multinomial(
            data.counts, global_profile, sample_prior_reads, seed
        )
        components, network = summarize_simulation(
            synthetic, data.meta, data.feature_names
        )
        distances = distance_profile(real_components, components)
        distance_rows.append({"seed": seed, **distances})
        synthetic_rows.append(
            flatten_values(summary_values(components), seed)
        )
        network = network.copy()
        network["seed"] = seed
        synthetic_networks.append(network)
        if first_synthetic is None:
            first_synthetic = synthetic

    distance_frame = pd.DataFrame(distance_rows)
    synthetic_summary = pd.DataFrame(synthetic_rows)
    distance_frame.to_csv(
        phase1 / "calibration_distances_by_seed.csv", index=False
    )
    synthetic_summary.to_csv(
        phase1 / "synthetic_summary_by_seed.csv", index=False
    )
    pd.concat(synthetic_networks, ignore_index=True).to_csv(
        phase1 / "synthetic_project_network_stats.csv", index=False
    )
    np.savez_compressed(
        phase1 / "synthetic_counts_seed1000.npz",
        counts=first_synthetic.astype(np.int32),
        feature_names=np.array(data.feature_names, dtype=str),
        project=np.array(data.meta["project"], dtype=str),
        sample=np.array(data.meta["sample"], dtype=str),
    )

    mean_distances = distance_frame[list(TOLERANCES)].mean(axis=0).to_dict()
    calibration = calibration_table(
        real_values, synthetic_summary, mean_distances
    )
    calibration.to_csv(phase1 / "calibration_table.csv", index=False)
    diagnostics = {
        "simulator_family": "template_multinomial_v1",
        "sample_prior_reads": sample_prior_reads,
        "search_seeds": args.search_seeds,
        "final_seeds": args.final_seeds,
        "n_pass": int(calibration["pass"].sum()),
        "all_pass": bool(calibration["pass"].all()),
        "statistics": calibration.set_index("statistic").to_dict(
            orient="index"
        ),
    }
    parameter_path = phase1 / "simulation_parameters_v1.json"
    if parameter_path.exists():
        update_parameters(parameter_path, sample_prior_reads, diagnostics)
    with (phase1 / "calibration_status.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(diagnostics, handle, indent=2)

    report = [
        "# Paper 4 phase 1 calibration",
        "",
        f"- Simulator: {diagnostics['simulator_family']}",
        f"- Selected sample pseudocount: {sample_prior_reads:g} reads",
        f"- Final seeds: {args.final_seeds}",
        f"- Passed summaries: {diagnostics['n_pass']}/8",
        "",
        "| Statistic | Real | Synthetic mean | Distance | Tolerance | Pass |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in calibration.to_dict(orient="records"):
        passed = bool(row["pass"])
        report.append(
            f"| {row['statistic']} | {row['real_summary']} | "
            f"{row['synthetic_mean_summary']} | {row['distance']:.4f} | "
            f"{row['tolerance']:.3f} | {'yes' if passed else 'no'} |"
        )
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "The selected MVP is a sample-template multinomial parametric "
            "bootstrap. It preserves the observed project design, sample-level "
            "co-variation and zero structure while adding read sampling noise. "
            "It is suitable as a calibrated null/template simulator. It is not "
            "yet an independently generated hierarchical community model.",
            "",
            "The independent project-mean + presence-mask generator was tested "
            "and rejected at this stage: it overestimated zeros and "
            "underestimated network co-occurrence. It remains a Phase-2 task.",
        ]
    )
    (phase1 / "calibration_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(
        f"Calibration complete: {diagnostics['n_pass']}/8 pass; "
        f"sample_prior_reads={sample_prior_reads:g}"
    )


if __name__ == "__main__":
    main()
