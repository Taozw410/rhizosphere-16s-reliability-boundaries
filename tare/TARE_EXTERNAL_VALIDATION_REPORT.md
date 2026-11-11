# Independent external-panel validation

## Material Passport

- Origin skill: academic-research-suite / experiment-agent
- Origin mode: run + validate
- Origin date: 2026-11-11
- Verification status: ANALYZED + deterministic model reuse
- Panel version: TARE-EXT-1

## Panel

Six projects from the pre-locked Route-C external cohort, two per crop:

| Crop | Projects | Samples |
|---|---|---:|
| Maize | PRJEB104640, PRJNA1062967 | 189 |
| Rice | PRJDB13616, PRJNA1044052 | 75 |
| Wheat | PRJEB76879, PRJNA1095845 | 162 |

Total: 6 projects, 426 samples. None of the six accessions occurs in the
45-project development corpus.

Project inclusion used only pre-existing sequencing and taxonomy QC. One
additional processed rice project, PRJDB19884, was excluded before external
TARE outcomes because non-NA genus reads were 24.482%, which belongs to the
pre-specified 20%–60% review stratum.

## Frozen model

TARE-DA was loaded from the serialized 45-project model without retraining:

- model SHA-256:
  `95b51f36c4fb383ef96edb7e82ebaeeb6ec6565711a3c1ad7b4623a8a55024ab`
- conformal radius: 0.313147118912476
- external datasets: 12 (six null + six strong sparse-effect)
- external project-method predictions: 48

## Results

| Endpoint | TARE-DA | Development mean-only | Result |
|---|---:|---:|---|
| Utility MAE | 0.1414 | 0.1853 | 23.7% lower |
| 95% interval coverage | 0.854 | not applicable | above 0.80 |
| Strong-effect reliability AUROC | 0.837 | 0.500 | above chance |
| ALLOW precision | 1.000 | not applicable | 24/24 correct |
| ALLOW coverage | 0.500 | not applicable | 24/48 |
| Predictions outside training q95 | 1.000 | not applicable | all external |
| Mean support score | 0.062 | not applicable | low |

### Strong-effect method results

| Method | TARE MAE | Development mean-only MAE | Coverage | ALLOW |
|---|---:|---:|---:|---:|
| ALDEx2 | 0.2190 | 0.3037 | 0.833 | 0 |
| ANCOM-BC2 | 0.1970 | 0.2933 | 0.833 | 0 |
| LinDA | 0.3479 | 0.4315 | 0.500 | 0 |
| MaAsLin2 | 0.3016 | 0.3962 | 0.667 | 0 |

All 24 null-scenario predictions received ALLOW and were correct. All 24
strong-effect predictions received DENY, including 12 outcomes that did meet
the reliability definition. The decision rule is therefore conservative on
this external panel: it avoided false ALLOW decisions but did not recover the
reliable strong-effect cases.

## External-validity decision

The external panel supports two directional conclusions:

1. The frozen TARE-DA model still reduced utility MAE and preserved acceptable
   aggregate interval coverage despite no retraining.
2. The model withheld recommendation for every external strong-effect case,
   consistent with its development-data boundary.

It does **not** support a claim of in-distribution external generalization.
All 48 predictions lie beyond the 95th percentile of nearest-neighbour
distance to the development feature rows. The dominant visible extrapolation
is sample count: four external projects contain 54–177 samples, while the
development corpus contains 4–35. Depth and community-structure shifts also
contribute.

Accordingly, the external panel is a genuine project-independent stress test,
but it is out of the TARE-DA calibration support. It should be reported as
evidence of directional robustness and conservative abstention, not as proof
that TARE generalizes to arbitrary large rhizosphere projects.

## Files

- `artifacts/external_panel/external_feature_matrix.csv`
- `artifacts/external_panel/external_sample_meta.csv`
- `artifacts/external_panel/external_study_covariates.csv`
- `artifacts/external_panel/da_method_results/da_project_metrics.csv`
- `artifacts/external_panel/tare_external_validation/tare_external_predictions.csv`
- `artifacts/external_panel/tare_external_validation/tare_external_summary.csv`
- `artifacts/external_panel/tare_external_validation/tare_external_overall.csv`

The four external evaluation outputs, including the manifest, were reproduced
byte-for-byte; hashes are in
`artifacts/external_panel/tare_external_validation/reproducibility_hashes.csv`.
