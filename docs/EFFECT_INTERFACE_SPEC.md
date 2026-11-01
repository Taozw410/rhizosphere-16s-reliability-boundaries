# Effect-injection interface specification v1.0

Date frozen: 2026-10-25

## Purpose

Inject known feature-level effects into the calibrated Paper 4 sample-template
simulator without moving samples across projects or breaking the Phase 1
calibration gate.

## Design contract

- Experimental unit: project.
- Block: project.
- Conditions: `control` and `treatment`.
- Assignment: seeded balanced randomization within each project.
- Maximum allowed imbalance: one sample per project.
- Run order: independently randomized and stored in `sample_design.csv`.
- Analysis must account for project, because assignment is within project.

Design seed: `20261025`.

## Effect contract

`effect_scenarios_long.csv` columns:

| Column | Meaning |
|---|---|
| `scenario` | Scenario identifier |
| `feature` | Feature receiving a biological effect |
| `requested_log2_effect` | Target treatment/control log2 abundance ratio |

For each scenario:

1. Start from each observed sample's calibrated composition template.
2. Fit a positive feature multiplier for treatment samples.
3. Renormalize each treatment composition to sum to one.
4. Iterate multipliers until the expected post-renormalization effect equals
   the requested log2 effect.
5. Generate counts from the resulting sample-specific multinomial.

The maximum accepted target-effect error is `1e-5` log2.

## Compositional closure

Increasing or decreasing target features changes the remaining feature
probabilities after renormalization. This closure effect is part of the known
truth and is recorded for every non-target feature in
`full_feature_truth.csv`.

Do not treat closure-induced shifts as biological target effects.

## Frozen scenarios v1.0

| Scenario | Targets | Requested absolute log2 effect |
|---|---:|---:|
| `no_effect` | 0 | 0 |
| `sparse20_fc1_19` | 20 | 0.25 |
| `sparse20_fc1_41` | 20 | 0.50 |
| `sparse20_fc2` | 20 | 1.00 |
| `dense200_fc1_41` | 200 | 0.50 |

Target directions are balanced between positive and negative effects.

## Required validation gates

1. Every project contains both conditions with imbalance ≤1.
2. Target-effect fitting error ≤`1e-5` log2.
3. The no-effect scenario passes 8/8 Phase 1 calibration summaries.
4. Effect scenarios retain 8/8 summary calibration.
5. Target sign recovery and effect correlation are reported, not optimized
   post hoc.

## Outputs

- `sample_design.csv`: block, condition, depth and run order.
- `effect_scenarios_long.csv`: target-feature effect specification.
- `target_feature_truth.csv`: fitted multiplier and expected target truth.
- `full_feature_truth.csv`: target and non-target truth, including closure.
- `simple_effect_recovery.csv`: project-paired descriptive recovery.
- `<scenario>_counts.npz`: simulated count matrix.
- `effect_scenario_metrics.csv`: calibration and recovery summary.
- `effect_interface_status.json`: machine-readable freeze status.

## Not supported in v1.0

- Prevalence-only effects that change zero probability without changing
  conditional abundance.
- Taxon-taxon interaction effects.
- Treatment assigned only at project level.
- Cross-project case-control confounding by design.
- Time-varying treatment effects.
- Deliberate reference-library or classification-error effects.

