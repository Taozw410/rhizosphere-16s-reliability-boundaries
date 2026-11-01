# Paper 4 three-panel robustness validation

## Design

All 45 projects were divided into three disjoint panels. Each panel contains
five maize, five rice and five wheat projects. Three seeds per panel were run
under both the primary template-multinomial and the correlated-perturbation
generators.

## Results

| Panel | Generator | Mean checks passed | Minimum checks passed |
|---|---|---:|---:|
| Panel 1 | Template multinomial | 7.67/8 | 7/8 |
| Panel 1 | Correlated perturbation | 7.67/8 | 7/8 |
| Panel 2 | Template multinomial | 7.33/8 | 7/8 |
| Panel 2 | Correlated perturbation | 8.00/8 | 8/8 |
| Panel 3 | Template multinomial | 7.67/8 | 7/8 |
| Panel 3 | Correlated perturbation | 7.67/8 | 7/8 |

The recurring borderline statistic is network topology, with KS distances
between 0.178 and 0.222. Other summaries remained well within tolerance.

## Interpretation

- Structural calibration is broadly stable when all 45 projects are
  represented through three disjoint panels.
- The network summary sits close to the threshold in two panels.
- This validates panel-level structural stability, not leave-panel-out
  generalization. The separate leave-panel-out validation failed and must be
  reported independently.

## Artifacts

- `project_panels.csv`
- `panel_calibration_by_seed.csv`
- `panel_calibration_summary.csv`

