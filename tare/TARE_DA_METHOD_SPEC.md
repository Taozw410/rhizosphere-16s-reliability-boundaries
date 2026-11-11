# TARE-DA: project-specific reliability envelope for differential abundance

## Status

Frozen on 2026-11-11 before any outer TARE-DA prediction was inspected.

## Question

Given study design, sequencing descriptors, observed community structure and a
pre-specified effect scenario, can a model predict whether ALDEx2, ANCOM-BC2,
LinDA or MaAslin2 will meet an error-control and recovery criterion on a new
rhizosphere project?

## Corpus

- 45 projects: 15 maize, 15 rice and 15 wheat.
- Same frozen 120-feature panel used by the main benchmark.
- Two project-specific scenarios: no effect and a 20-feature strong sparse
  effect.
- Conditions are balanced within each project.
- The realized compositional effect after renormalization is recorded and is
  the evaluation truth.

## Outcomes

For a non-null scenario:

- utility = F1.
- reliable = FDP <= 0.10 and F1 >= 0.50.

For the null scenario:

- utility = 1 - false-positive rate.
- reliable = false-positive rate <= 0.10.

Failed method executions receive utility 0 and reliable 0.

## Model

TARE-DA is a stacked, conformalized reliability model.

1. **Input representation.** Study descriptors, target-project prevalence,
   zero fraction, variance ratio, Bray-Curtis, Aitchison and network topology
   are combined with method and scenario indicators.
2. **Point head.** A shallow histogram gradient-boosted regression tree
   predicts utility. Tree depth and leaf size are constrained for the
   45-project sample size.
3. **Conformal head.** Four-fold inner group cross-fitting produces
   out-of-project absolute residuals. The 95% empirical residual quantile is
   added and subtracted from the point prediction.
4. **Reliability head.** A shallow histogram gradient-boosted classifier
   predicts the probability that the method meets the reliability definition.

Exact fixed hyperparameters are in `p4_train_tare_da.py`; no outer-fold tuning
is allowed.

## Decision

- `ALLOW`: reliability probability >= 0.80 and lower conformal utility bound
  >= 0.50.
- `DENY`: reliability probability < 0.30 or lower conformal bound < 0.20.
- `CONDITIONAL`: all remaining cases.

## Validation

- Ten repeated stratified five-fold partitions.
- All rows from a project are assigned to one fold.
- TARE-DA is fitted from scratch in each outer fold.
- Conformal residuals are generated using only the outer training projects.
- Comparator: method-by-scenario training mean, representing a
  study-agnostic recommendation.

Endpoints are MAE, 95% interval coverage, AUROC for reliability, ALLOW
precision and ALLOW coverage. TARE-DA is considered to add practical value
only if it reduces MAE relative to the study-agnostic comparator while
retaining interval coverage and useful ALLOW precision.

## Interpretation boundary

TARE-DA is calibrated on simulator-defined effects applied to real project
community templates. It predicts performance under those defined scenarios,
not biological truth in an unlabelled real study. The 45 projects are
correlated by crop and public-data provenance; repeated partitions are not
independent datasets.
