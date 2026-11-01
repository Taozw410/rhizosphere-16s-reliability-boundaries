# Paper 4 integrated multi-domain benchmark report

Date: 2026-10-28

## 1. Completion status

| Benchmark | Status | Main scale |
|---|---|---|
| Differential abundance | complete | 100 primary/extension datasets plus 25 second-generator datasets |
| Core/occupancy | complete | 100 datasets, 400 method metrics |
| Network | complete | 4 methods, 4 scenarios, 5 calibration and 5 validation seeds |
| Taxonomy classification | complete | 5 seed splits, 20 known and 4 unseen genera |
| Three-panel robustness | complete | 3 disjoint 15-project panels covering all 45 projects |
| Real project-panel holdout | complete and still failed | 3 folds, covariate matching and 100 predictive draws |

## 2. Differential abundance

Primary generator, combined primary and extension seeds:

| Scenario | Method | FDP | Power | F1 |
|---|---|---:|---:|---:|
| `sparse20_fc2` | ALDEx2 | 0.000 | 0.150 | 0.261 |
| `sparse20_fc2` | ANCOM-BC2 | 0.008 | 0.878 | 0.930 |
| `sparse20_fc2` | LinDA | 0.353 | 0.220 | 0.326 |
| `sparse20_fc2` | MaAsLin2 | 0.048 | 0.275 | 0.423 |

Second semi-parametric generator:

| Method | FDP | Power | F1 |
|---|---:|---:|---:|
| ALDEx2 | 0.000 | 0.150 | 0.261 |
| ANCOM-BC2 | 0.000 | 0.840 | 0.913 |
| LinDA | 0.257 | 0.350 | 0.474 |
| MaAsLin2 | 0.047 | 0.400 | 0.563 |

Cross-generator conclusion:

- ANCOM-BC2 retains the best FDP-power-F1 balance.
- LinDA FDP remains materially above 0.10 in the strong sparse scenario.
- MaAsLin2 and ALDEx2 control FDP but remain less powerful.
- The second generator is a correlated perturbation of the same template.
  It is not a fully independent ecological generator.

## 3. Core and occupancy

Across 20 primary and extension seeds:

| Method | Mean F1 | Worst scenario mean false-core rate |
|---|---:|---:|
| Project-balanced prevalence | 0.859 | 0.076 |
| Hierarchical occupancy-detection | 0.833 | 0.071 |
| Sample prevalence | 0.804 | 0.054 |
| Beta-binomial empirical Bayes | 0.752 | 0.038 |

- Balanced, low-abundance and project-shift core scenarios had F1 between
  approximately 0.8 and 1.0 for the strongest methods.
- Depth-confounded F1 was only 0.423-0.497.
- Project-balanced prevalence closely mirrors the project-level truth
  definition, so its advantage is partly definitional.

## 4. Network

Synthetic direct-edge AUPRC:

| Method | Mean AUPRC |
|---|---:|
| CLR Spearman | 0.101 |
| CLR Pearson | 0.078 |
| Graphical lasso | 0.076 |
| SparCC-like | 0.064 |

Real three-panel stability:

| Method | Edge Jaccard | Hub Spearman |
|---|---:|---:|
| SparCC-like | 0.188 | 0.617 |
| CLR Pearson | 0.136 | 0.239 |
| CLR Spearman | 0.132 | 0.271 |
| Graphical lasso | 0.085 | 0.431 |

Conclusion:

- Direct-edge recovery is weak for every method.
- SparCC-like is more stable across real project panels but less accurate
  under the synthetic graph truth.
- Synthetic accuracy and cross-panel stability are distinct endpoints.

## 5. Taxonomy classification

SILVA V4 genus benchmark:

| Method | Closed-set accuracy | Closed-set macro F1 | Unseen-genus rejection |
|---|---:|---:|---:|
| Linear SGD | 0.915 | 0.905 | 0.455 |
| k-mer nearest neighbor | 0.870 | 0.847 | 0.295 |
| Multinomial NB | 0.855 | 0.829 | 0.490 |
| CNN | 0.090 | 0.030 | 0.085 |
| k-mer Transformer | 0.075 | 0.031 | 0.060 |

Conclusion:

- Classical k-mer linear classification outperforms CNN and a lightweight
  from-scratch Transformer in this small reference dataset.
- Unknown-genus detection remains difficult.
- Pretrained DNA-BERT was not evaluated because model artifacts were not
  available locally and external download failed. No DNA-BERT claim is made.

## 6. Project sampling and external validation

Three disjoint panels covering all 45 projects:

- primary template generator passed 7.33-7.67 of 8 summaries per panel;
- correlated perturbation passed 7.67-8.0 of 8;
- network topology was the recurring borderline statistic.

True leave-panel-out validation:

| Held-out panel | Mean summaries passed |
|---|---:|
| Panel 1 | 3.6/8 |
| Panel 2 | 2.0/8 |
| Panel 3 | 4.0/8 |

Covariate-matched donor assignment improved the strict point-summary check
from 3.6/2.0/4.0 to 5.0/4.0/4.0 of 8. A broader posterior predictive check
passed 6/9 aggregate summaries in each held-out panel, but prevalence, zero
fraction, Bray-Curtis and network density remained outside the predictive
interval in at least one panel.

This remains a negative external-validity result. The simulator should not be
described as fully generalizable to all 45 projects.

Repeated validation over 10 stratified three-fold partitions gave 30 held-out
panels. Mean fixed-tolerance passes were 3.87/8 for the template generator and
3.40/8 for correlated perturbation. The 95% predictive coverage for network
density was only 0.167 and 0.200, and for network centrality 0.033 in both
generators. The failure is therefore systematic rather than a single unlucky
partition.

The attempted fully independent hurdle generator is documented separately and
was not promoted after failing strict holdout calibration.

## 7. Overall conclusion

The benchmark suite now covers all four requested domains and is technically
complete. The results support:

1. scenario-specific, not universal, method selection;
2. ANCOM-BC2 as the strongest DA candidate in the tested grid;
3. caution against LinDA in sparse, strong-effect settings;
4. weak direct-edge recovery from microbial co-occurrence;
5. preference for established k-mer classifiers over small-data neural models;
6. a clear distinction between real cross-panel stability and synthetic truth
   recovery.

However, external project-panel validation failed. The project therefore has
high value as a transparent stress-test and reproducible benchmark resource,
but not yet as proof of universal method rankings.

## 8. Primary artifacts

- `outputs/formal_grid/formal_benchmark_report.md`
- `outputs/da_cross_generator/da_cross_generator_report.md`
- `outputs/core_formal_grid/results/core_formal_report.md`
- `outputs/network_benchmark/network_report.md`
- `outputs/real_network_stability/report.md`
- `outputs/classification_benchmark/classification_report.md`
- `outputs/project_panel_validation/report.md`
- `outputs/real_project_holdout/report.md`
- `outputs/real_project_holdout_cv/report.md`
- `outputs/independent_generator_holdout/report.md`

