# Version 2 restructuring summary

## What changed

The manuscript is no longer organized as a multi-domain leaderboard. The
central contribution is now TARE-DA, a project-specific model that predicts
method utility, produces a conformal interval, estimates reliability and issues
an explicit recommendation decision.

The former benchmark modules now serve three narrower roles:

1. They show why a single method ranking is insufficient.
2. They define the simulator and domain boundaries used to motivate TARE-DA.
3. They prevent the model's differential-abundance result from being generalized
   to core, network or classification tasks.

## New evidence

- 90 project-specific known-effect datasets
- 360 project-method outcomes
- 3,600 internal held-out predictions from 10 repeated five-fold project splits
- 48 frozen-model predictions from six independent external projects
- Explicit support-distance analysis for the external panel

## Primary results retained in the new draft

- Internal utility MAE: 0.0790 versus 0.0948
- Internal 95% interval coverage: 0.954
- Strong-effect reliability AUROC: 0.883
- ALLOW precision: 0.983
- External MAE: 0.1414 versus 0.1853
- External coverage: 0.854
- External strong-effect AUROC: 0.837

## Claims removed or narrowed

- A universal ranking of differential-abundance methods
- General biological accuracy on unlabelled real projects
- Reliable transportability to arbitrary large rhizosphere projects
- Any implication that TARE-DA was trained jointly across all four analysis
  domains
- Any implication that network edges are ecological interactions

## Reviewer-driven revision

- Defined utility mathematically by scenario: F1 for strong effects and one
  minus false-positive rate for null effects, with no weighted composite and a
  fixed range of 0 to 1.
- Defined the interval target as project-method utility, grouped all rows by
  project during calibration, and stated that empirical marginal coverage is
  not a per-project conditional guarantee.
- Separated evidence decisions from applicability: DENY means low utility or
  insufficient utility confidence, whereas DEFER_OUT_OF_SUPPORT means the
  predictor is outside the measured development domain.
- Reported interval width, false-ALLOW risk and a threshold risk-acceptance
  grid in addition to AUROC.
- Added a real project-template application showing nearly disjoint candidate
  sets across DA methods and the resulting conservative TARE decision.
- Quantified the relationship between simulator structural mismatch and TARE
  reliability; the relationship was weak and did not justify using simulator
  failure as evidence for TARE success.
- Added a prespecified family-only sensitivity, which changed the DA method
  ordering and made taxonomic resolution part of the reliability boundary.
- Added per-project external results instead of relying only on aggregate AUC.
- Clarified that TARE was trained and validated on the primary generator only,
  so generator-2 structural robustness is not a TARE cross-generator test.
