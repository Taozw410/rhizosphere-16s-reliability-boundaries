# TARE runbook

## 1. Freeze

The protocol is `TARE_METHOD_SPEC.md`. Tolerances, decision thresholds, outer
splits and training objectives must not be changed after the final holdout run.

## 2. Project-panel structural model

Smoke command:

```powershell
python tare/p4_tare_transport.py `
  --partitions 1 `
  --predictive-draws 5 `
  --inner-folds 2 `
  --temperature-grid 0.70 `
  --covariance-shrinkage-grid 0.20 `
  --out-dir tare/artifacts/tare_v1_smoke
```

Formal command:

```powershell
python tare/p4_tare_transport.py
```

The formal run writes:

- `tare/artifacts/tare_v1/tare_generator_summary.csv`
- `tare/artifacts/tare_v1/tare_coverage_summary.csv`
- `tare/artifacts/tare_v1/tare_allow_summary.csv`
- `tare/artifacts/tare_v1/tare_predictions.csv`

## 3. TARE-DA project-held-out corpus

Build one null and one strong sparse known-effect dataset per project:

```powershell
python tare/p4_tare_build_da_corpus.py
```

The truth file stores the realized effect after compositional renormalization.
The requested fold change is not treated as realized truth when the two differ.

## 4. Run four DA methods

```powershell
python tare/p4_tare_run_da.py --workers 4 --resume
```

The batch runner executes ALDEx2, ANCOM-BC2, LinDA and MaAsLin2 in one R
process for each project-by-scenario dataset. Project is held constant in this
corpus, so DA models use condition as the fixed term; a project random effect
would be unidentifiable. The main multi-project benchmark remains the
appropriate analysis for between-project random effects.

## 5. Train and validate TARE-DA

```powershell
python tare/p4_train_tare_da.py
```

TARE-DA predicts project-specific utility intervals from:

- study design and sequencing descriptors;
- target-project prevalence, zero fraction, beta-diversity and network
  structure;
- DA method;
- effect-scenario severity.

Every held-out project contributes no rows to model fitting or conformal
calibration. The outer partitions are repeated five-fold group splits.

## 6. Interpretation

An `ALLOW` decision means that the project lies within the calibrated support
of TARE-DA and the lower conformal utility bound meets the frozen threshold.
It does not mean that the method recovers biologically causal taxa. A `DENY`
decision means that the evidence is insufficient to recommend that method for
the target project under the tested scenario.
