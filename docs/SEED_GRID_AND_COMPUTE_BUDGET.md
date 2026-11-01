# Formal seed grid and compute budget v1.0

Date frozen: 2026-10-25

## Frozen grid

- Scenarios: 5
- Seeds per scenario: 20
- Seed range: 4000–4019
- Primary analysis: 4000–4009
- Extension confirmation: 4010–4019
- Total simulated datasets: 100
- Methods: ALDEx2, ANCOM-BC2, LinDA, MaAsLin2
- Method jobs: 400

The same seed is paired across scenarios.

## Frozen analysis panel

- Projects: 15, five per crop.
- Samples: 390 before zero-feature-depth removal.
- Features: 120 per dataset.
- Project panel and target-selection seeds are stored in
  `outputs/formal_grid/formal_grid_config.json`.

## Scenarios

| Scenario | Targets | Absolute log2 effect |
|---|---:|---:|
| `no_effect` | 0 | 0 |
| `sparse20_fc1_19` | 20 | 0.25 |
| `sparse20_fc1_41` | 20 | 0.50 |
| `sparse20_fc2` | 20 | 1.00 |
| `dense40_fc1_41` | 40 | 0.50 |

`dense200_fc1_41` remains available as a supplementary stress test but is not in
the primary compute budget.

## Measured per-run budget

Measured on one formal-size dataset: 15 projects, 388 non-zero-depth samples,
120 features.

| Method | Measured seconds/run | Primary 50 runs CPU hours | Extension 50 runs CPU hours |
|---|---:|---:|---:|
| ALDEx2 | 1.82 | 0.03 | 0.03 |
| ANCOM-BC2 | 598.67 | 8.31 | 8.31 |
| LinDA | 10.32 | 0.14 | 0.14 |
| MaAsLin2 | 10.52 | 0.15 | 0.15 |

Total CPU time:

- primary: about 8.7 hours;
- extension: about 8.7 hours;
- both stages: about 17.4 hours.

With two concurrent method jobs, estimated wall time is about 4.4 hours per stage
and 8.8 hours for both stages. ANCOM-BC2 accounts for over 95% of compute time.

## Execution policy

- Use two concurrent R method processes.
- Do not run more than two ANCOM-BC2 jobs simultaneously.
- Resume from completed method JSON/CSV records.
- Preserve failed and timed-out runs in the result denominator.
- Method timeout: 1,800 seconds for ANCOM-BC2, 600 seconds for the others.
- Primary and extension rankings are reported separately.

## Stop conditions

- More than 10% of primary method jobs fail before collection.
- Result feature IDs do not map one-to-one to the frozen truth table.
- A method systematically filters target features differently across seeds.
- Repeated timeout indicates the compute budget must be revised before extension
  seeds are run.

