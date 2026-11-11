# TARE external-panel protocol

## Status

Panel frozen on 2026-11-11 before generating or inspecting any external TARE
performance outcome.

## Independence

The panel was selected from the previously locked Route-C external cohort.
None of its project accessions occurs in the 45-project development corpus.
External projects are never used to fit, tune, conformalize or recalibrate
TARE-DA.

## Included projects

| Crop | Project | Platform cluster | Samples |
|---|---|---|---:|
| Maize | PRJEB104640 | XC_ETH_MAIZE_DROUGHT_2025 | 177 |
| Maize | PRJNA1062967 | XC_GXU_MAIZE_DROUGHT_2024 | 12 |
| Rice | PRJDB13616 | XC_UTOKYO_RICE_2022 | 60 |
| Rice | PRJNA1044052 | XC_CAAS_RICE_2023 | 15 |
| Wheat | PRJEB76879 | XC_UMR5557_WHEAT_2025 | 108 |
| Wheat | PRJNA1095845 | XC_NWAF_WHEAT | 54 |

Total: six projects, 426 samples, two projects per crop.

## Exclusions resolved before outcome generation

- PRJDB19884 was excluded because post-DADA2 non-NA genus reads were 24.482%,
  placing it in the pre-specified 20%–60% review stratum. The balanced panel
  used the six projects that passed the hard post-DADA2 gate.
- PRJEB98904, PRJNA1082480 and PRJNA1089842 were not eligible because DADA2
  outputs were incomplete at panel freeze.
- PRJNA1005591 and PRJNA1015669 had already been excluded for ITS marker
  evidence.

These decisions use sequencing and taxonomy QC only. No external DA or TARE
outcome informed inclusion.

## Feature contract

- Same frozen 2,025-feature development vocabulary.
- Same common 120-feature analysis panel used by TARE-DA.
- Same project-specific null and strong sparse-effect generator.
- Conditions balanced within each project.
- Realized compositional effects after renormalization are recorded as truth.

## Model contract

Use the serialized TARE-DA final model trained on the 45 development projects:

`artifacts/tare_da/tare_da_final_model.joblib`

No model parameter, feature, conformal radius or decision threshold may change.

## Outcomes

- Utility MAE versus the method-by-scenario mean learned from the 45
  development projects.
- 95% conformal interval coverage.
- Reliability AUROC in the strong-effect scenario.
- ALLOW precision and coverage.
- OOD support: nearest-neighbor distance to the 45-project training feature
  distribution.

Because the external panel contains only six projects and 48 project-method
predictions, AUC and decision precision are descriptive and will not be used to
retrain TARE-DA.

## Decision rules

External support is declared only if:

1. TARE utility MAE is lower than the development-only method-by-scenario
   mean;
2. interval coverage is at least 0.80;
3. the direction of the strong-effect reliability ranking is positive;
4. no decision result is interpreted outside the measured support range.

Failure is retained as an external-validity boundary rather than solved by
recalibration on the external panel.
