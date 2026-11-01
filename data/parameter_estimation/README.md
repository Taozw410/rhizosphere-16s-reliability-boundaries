# Phase 1 outputs

Date: 2026-10-25

## Main parameters

- `simulation_parameters_v1.json`: scalar parameters, software-facing metadata,
  selected simulator family and calibration status.
- `simulation_parameters_v1.npz`: feature names, prevalence, global/project
  profiles, sample profiles, project names and descriptive crop contrasts.

## Real-data evidence

- `real_summary_statistics.csv`: eight pre-specified real-data summaries.
- `real_project_network_stats.csv`: project-level CLR-Spearman network density
  and eigenvector-centrality CV.
- `bootstrap_summary_statistics.csv`: 100 project-cluster bootstrap replicates.
- `bootstrap_summary_quantiles.csv`: 2.5%, 50% and 97.5% intervals.

## Calibration

- `calibration_search.csv`: sample-template pseudocount grid.
- `calibration_distances_by_seed.csv`: five-seed final calibration distances.
- `synthetic_summary_by_seed.csv`: synthetic summaries for the five final seeds.
- `synthetic_project_network_stats.csv`: synthetic network summaries.
- `calibration_table.csv`: the eight-statistic calibration gate.
- `calibration_status.json`: machine-readable pass/fail result.
- `calibration_report.md`: human-readable report.
- `synthetic_counts_seed1000.npz`: first calibrated synthetic count matrix.

## Rejected model

`*_hierarchical_failed.*` preserve the independent project-mean plus presence-mask
generator attempt. It was rejected because it overestimated zeros and
underestimated network co-occurrence. These files are retained as Phase-2
diagnostics and must not be used as calibrated inputs.

