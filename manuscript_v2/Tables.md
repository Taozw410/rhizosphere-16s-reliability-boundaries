# Tables

## Table 1 | Evidence layers and analysis scale

| Layer | Projects | Samples | Unit of evaluation | Primary purpose |
| --- | --- | --- | --- | --- |
| Development resource | 45 | 1,219 | Project and sample | Build the common feature space and observe multi-project heterogeneity |
| Structural calibration | 45 | 1,219 | 15-project panel | Assess agreement with eight observed structural summaries |
| Repeated project holdout | 45 | 1,219 | 30 correlated held-out panels | Test simulator transportability |
| TARE-DA training | 45 | 1,219 | 90 project-specific known-effect datasets | Train and validate project-level reliability envelopes |
| TARE-DA internal holdout | 45 | 1,219 | 3,600 project-method predictions | Evaluate interval coverage and recommendation decisions |
| Family-only sensitivity | 45 | 1,219 | 45 project-level strong-effect datasets | Test dependence on the mixed taxonomic hierarchy |
| Independent external panel | 6 | 426 | 48 project-method predictions | Stress-test the frozen model with zero project overlap |

The 45 development projects comprise 15 maize, 15 rice and 15 wheat studies. The external panel comprises two projects per crop. The external projects are outside the development feature support but independent at the project-accession level.

## Table 2 | Differential-abundance and cross-domain reliability boundaries

| Domain | Setting | Method | Metric 1 | Metric 2 | Reliability interpretation |
| --- | --- | --- | --- | --- | --- |
| Differential abundance | Strong sparse, generator 1 | ANCOM-BC2 | FDP 0.008 | F1 0.930 | Strong recovery but not a universal recommendation |
| Differential abundance | Strong sparse, generator 1 | ALDEx2 | FDP 0.000 | F1 0.261 | Error control with low power |
| Differential abundance | Strong sparse, generator 1 | LinDA | FDP 0.353 | F1 0.326 | Reproducible false-discovery boundary |
| Differential abundance | Strong sparse, generator 1 | MaAsLin2 | FDP 0.048 | F1 0.423 | Intermediate recovery |
| Differential abundance | Strong sparse, generator 2 | ANCOM-BC2 | FDP 0.000 | F1 0.913 | Main ranking direction reproduced |
| Differential abundance | Strong sparse, generator 2 | LinDA | FDP 0.257 | F1 0.474 | Elevated FDP persists |
| Differential abundance | Family-only sensitivity | LinDA | FDP 0.048 | F1 0.650 | Became top-ranked under family features |
| Differential abundance | Family-only sensitivity | MaAsLin2 | FDP 0.086 | F1 0.644 | Similar recovery with higher FDP |
| Differential abundance | Family-only sensitivity | ANCOM-BC2 | FDP 0.087 | F1 0.543 | Mixed-level top rank did not persist |
| Differential abundance | Family-only sensitivity | ALDEx2 | FDP 0.000 | F1 0.374 | Conservative but lower recovery |
| Core taxa | Four non-null scenarios | Project-balanced prevalence | Mean F1 0.859 | Highest false-core rate 0.076 | Best mean but strong depth sensitivity |
| Core taxa | Depth-confounded | All methods | F1 0.423-0.497 | - | Common failure boundary |
| Network | Synthetic non-null | CLR Spearman | AUPRC 0.088 | - | Highest direct-edge recovery |
| Network | Real panels | SparCC-like | Edge Jaccard 0.188 | Hub rho 0.617 | Highest stability |
| Classification | Closed set | Linear SGD | Accuracy 0.915 | Macro F1 0.905 | Best limited-budget classifier |
| Classification | Unseen genus | Multinomial NB | Unknown rejection 0.490 | Known retention 0.950 | Highest unknown rejection |

Exact generator-specific confidence intervals and supplemental scenarios are provided in the analysis package. The table is a boundary summary, not a single cross-domain leaderboard.

## Table 3 | TARE-DA internal project-held-out validation

| Scenario | Method | Predictions | TARE MAE | Development-prior MAE | 95% coverage | Reliability AUROC | ALLOW count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No effect | ALDEx2 | 450 | 0.0103 | 0.0000 | 1.000 | - | 450 |
| No effect | ANCOM-BC2 | 450 | 0.0173 | 0.0132 | 1.000 | 0.153 | 449 |
| No effect | LinDA | 450 | 0.0863 | 0.0869 | 0.953 | 0.226 | 437 |
| No effect | MaAsLin2 | 450 | 0.0094 | 0.0021 | 1.000 | - | 450 |
| Strong sparse | ALDEx2 | 450 | 0.0979 | 0.1227 | 0.929 | 0.868 | 0 |
| Strong sparse | ANCOM-BC2 | 450 | 0.1730 | 0.1988 | 0.880 | 0.900 | 0 |
| Strong sparse | LinDA | 450 | 0.1240 | 0.1570 | 0.922 | 0.853 | 0 |
| Strong sparse | MaAsLin2 | 450 | 0.1137 | 0.1776 | 0.951 | 0.925 | 0 |
| Overall | All methods | 3,600 | 0.0790 | 0.0948 | 0.954 | 0.883 strong effect | 1,786 |

The analysis uses 10 repeated stratified five-fold project holdouts. The development-prior comparator is the mean utility of the same method and scenario in the outer training projects. ALLOW precision across the full set is 0.983.

## Table 4 | Independent external projects

| Crop | Project | Platform cluster | Samples | Post-DADA2 QC | Genus reads (%) |
| --- | --- | --- | --- | --- | --- |
| Maize | PRJEB104640 | XC_ETH_MAIZE_DROUGHT_2025 | 177 | pass | 81.590 |
| Maize | PRJNA1062967 | XC_GXU_MAIZE_DROUGHT_2024 | 12 | pass | 72.824 |
| Rice | PRJDB13616 | XC_UTOKYO_RICE_2022 | 60 | pass | 89.598 |
| Rice | PRJNA1044052 | XC_CAAS_RICE_2023 | 15 | pass | 86.180 |
| Wheat | PRJEB76879 | XC_UMR5557_WHEAT_2025 | 108 | pass | 81.030 |
| Wheat | PRJNA1095845 | XC_NWAF_WHEAT | 54 | pass | 75.249 |

The reviewed rice project PRJDB19884 was excluded before external outcome generation because its non-NA genus-read rate was 24.482%, within the prespecified 20%-60% review stratum. No external accession appears in the 45-project development corpus.

## Table 5 | TARE-DA external stress test

| Scenario | Method | Projects | TARE MAE | Development-prior MAE | Coverage | Reliability AUROC | ALLOW-core |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No effect | ALDEx2 | 6 | 0.0060 | 0.0000 | 1.000 | - | 6 |
| No effect | ANCOM-BC2 | 6 | 0.0143 | 0.0098 | 1.000 | - | 6 |
| No effect | LinDA | 6 | 0.0411 | 0.0465 | 1.000 | - | 6 |
| No effect | MaAsLin2 | 6 | 0.0045 | 0.0011 | 1.000 | - | 6 |
| Strong sparse | ALDEx2 | 6 | 0.2190 | 0.3037 | 0.833 | 1.000 | 0 |
| Strong sparse | ANCOM-BC2 | 6 | 0.1970 | 0.2933 | 0.833 | - | 0 |
| Strong sparse | LinDA | 6 | 0.3479 | 0.4315 | 0.500 | 0.875 | 0 |
| Strong sparse | MaAsLin2 | 6 | 0.3016 | 0.3962 | 0.667 | 1.000 | 0 |
| Overall | All methods | 6 | 0.1414 | 0.1853 | 0.854 | 0.837 strong effect | 24 |

The model was not retrained on external data. ALLOW-core is the underlying utility decision before the applicability gate. All 48 external predictions were outside the 95th percentile of nearest-neighbour distance to the development feature space and therefore received the operational status DEFER_OUT_OF_SUPPORT. The mean support score was 0.062. Method-specific AUROCs are descriptive because each cell contains six projects.

## Table 6 | External project-level TARE results

| Crop | Project | TARE MAE | Development-prior MAE | Coverage | Reliable methods (of 4) | ALLOW-core | Support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Maize | PRJEB104640 | 0.744 | 0.804 | 0.00 | 3 | 0 | outside q95 |
| Maize | PRJNA1062967 | 0.154 | 0.128 | 1.00 | 0 | 0 | outside q95 |
| Rice | PRJDB13616 | 0.329 | 0.234 | 0.50 | 1 | 0 | outside q95 |
| Rice | PRJNA1044052 | 0.029 | 0.151 | 1.00 | 0 | 0 | outside q95 |
| Wheat | PRJEB76879 | 0.142 | 0.448 | 1.00 | 3 | 0 | outside q95 |
| Wheat | PRJNA1095845 | 0.201 | 0.373 | 0.75 | 2 | 0 | outside q95 |

Values summarize the four strong-effect method predictions per external project. Every project is outside the development support envelope and therefore receives the operational status DEFER_OUT_OF_SUPPORT. Project-specific method predictions are provided in the source data.

## Table 7 | Illustrative project-level differential-abundance application

| Method | Rejected features | FDP | F1 | Candidate-set relation | TARE decisions across 10 partitions |
| --- | --- | --- | --- | --- | --- |
| ALDEx2 | 0 | 0.000 | 0.000 | No candidates | DENY |
| ANCOM-BC2 | 16 | 0.188 | 0.722 | 13 candidates unique to this method | DENY |
| LinDA | 0 | 0.000 | 0.000 | No candidates | CONDITIONAL or DENY |
| MaAsLin2 | 3 | 0.000 | 0.261 | All 3 shared with ANCOM-BC2 | CONDITIONAL or DENY |

The illustration uses the observed composition template of PRJNA554261 with a simulator-defined strong sparse effect. It is not a real treatment-effect validation. No rejected taxon was shared by all four methods. TARE withheld recommendation across partitions even though ANCOM-BC2 recovered most targets, because its FDP was 0.188 and the model's reliability evidence remained low. This example shows how method-dependent candidate lists arise and why TARE is designed to abstain rather than convert one method's output into biological evidence.

## Table 8 | Interpretation boundaries

| Claim | Supported evidence | Not supported |
| --- | --- | --- |
| The domains require different reliability criteria | DA, core, network and classification failure boundaries | A universal cross-domain method ranking |
| Project-specific prediction adds information | TARE-DA internal and external MAE improvements | Accuracy on unlabelled real-project biological truth |
| The envelope supports abstention | High ALLOW precision; all strong external utility decisions were DENY | Recovery of every reliable external strong-effect case |
| Applicability is distinct from utility | All external projects were flagged DEFER_OUT_OF_SUPPORT | Conditional coverage guarantee under arbitrary shift |
| The external panel is project-independent | Six projects with zero development overlap | In-distribution external generalization |
| A calibrated simulator can be useful | Eight-summary calibration and scenario comparisons | Reliable structural transfer to arbitrary unseen panels |
| Network estimates capture association | Reproducible co-occurrence rankings | Ecological interactions or causal roles |
