# TARE validation report

## Material Passport

- Origin skill: academic-research-suite / experiment-agent
- Origin mode: run + validate
- Origin date: 2026-11-11
- Verification status: VERIFIED
- Version: TARE-DA v1

## Data and execution

- Projects: 45; 15 maize, 15 rice and 15 wheat.
- Features: common frozen 120-feature panel.
- Project-specific datasets: 90 (45 no-effect and 45 strong sparse-effect).
- Method evaluations: 360 (90 datasets × 4 methods).
- Successful executions: ALDEx2 90/90, ANCOM-BC2 90/90, MaAslin2 90/90,
  LinDA 86/90. Four LinDA numerical failures remained in the scored table as
  failures rather than being dropped.
- Outer validation: 10 repeated stratified five-fold group partitions, giving
  3,600 held-out project-by-scenario-by-method predictions.
- No row from a target project was used to fit or conformalize its prediction.

## Primary results

| Endpoint | TARE-DA | Study-agnostic mean |
|---|---:|---:|
| Utility MAE | 0.0790 | 0.0948 |
| Relative MAE reduction | 16.68% | reference |
| 95% interval coverage | 0.9544 | not applicable |
| Reliability AUROC, strong-effect scenario | 0.8833 | 0.5000 |
| ALLOW precision | 0.9832 | not applicable |
| ALLOW coverage | 0.4961 | not applicable |

The MAE improvement occurred in all ten repeated partitions (range
0.0052–0.0197). Across partitions, coverage ranged from 0.925 to 0.981 and
ALLOW precision from 0.9830 to 0.9833.

## Scenario-specific result

| Scenario | TARE-DA MAE | Mean-only MAE | Coverage | Reliability prevalence |
|---|---:|---:|---:|---:|
| No effect | 0.0308 | 0.0256 | 0.988 | 0.983 |
| Strong sparse effect | 0.1271 | 0.1640 | 0.921 | 0.056 |

TARE-DA does not improve the near-deterministic null scenario because the
method-wise mean is already almost exact. Its useful contribution is the
heterogeneous strong-effect scenario, where project-specific MAE improves from
0.1640 to 0.1271 (22.5% relative reduction).

## Method-level strong-effect results

| Method | TARE MAE | Mean-only MAE | Coverage | Reliability AUROC | ALLOW count |
|---|---:|---:|---:|---:|---:|
| ALDEx2 | 0.0979 | 0.1227 | 0.929 | 0.868 | 0 |
| ANCOM-BC2 | 0.1730 | 0.1988 | 0.880 | 0.900 | 0 |
| LinDA | 0.1240 | 0.1570 | 0.922 | 0.853 | 0 |
| MaAsLin2 | 0.1137 | 0.1776 | 0.951 | 0.925 | 0 |

The zero ALLOW count is not a coding failure. Under the frozen reliability
criterion (FDP ≤ 0.10 and F1 ≥ 0.50), only 4.4%–6.7% of strong-effect
project-method results were reliable. The lower conformal bound therefore did
not cross the utility threshold for any strong-effect case. In the null
scenario, 1,786 of 1,800 decisions were ALLOW, with precision 0.983.

## Pre-specified success gate

| Gate | Result | Status |
|---|---|---|
| MAE below study-agnostic mean | 0.0790 < 0.0948 | PASS |
| Coverage ≥ 0.80 for at least six summary groups | all eight method-by-scenario groups in the DA validation exceed 0.88 | PASS |
| ALLOW precision > 0.80 | 0.983 | PASS |
| Stable across repeated partitions | improvement in 10/10 | PASS |

TARE-DA therefore passes its frozen validation gate for the defined
project-specific simulation endpoint.

## Independent external panel

A six-project external panel (two maize, two rice and two wheat projects; 426
samples) was evaluated with the serialized model and no retraining. Utility MAE
was 0.1414 versus 0.1853 for the development-only method-by-scenario prior
(23.7% lower); interval coverage was 0.854 and strong-effect reliability AUROC
was 0.837. All 24 strong-effect predictions were DENY, so no false ALLOW was
issued.

Every external project was nevertheless outside the 95th percentile of
nearest-neighbour distance to the development feature space. The external
result is therefore a project-independent stress test, not an in-distribution
generalization claim. Full details are in
`TARE_EXTERNAL_VALIDATION_REPORT.md`.

## Reviewer-requested diagnostic analyses

- Utility is scenario-specific and is not a weighted FDP-power composite.
- Internal interval width was 0.405 overall.
- A threshold grid changed ALLOW coverage only from 0.492 to 0.501, with
  precision 0.981-0.983 and reliable-result recall 0.932-0.947.
- The real project-template PRJNA554261 produced no candidate shared by all
  four methods; 13 of 16 ANCOM-BC2 candidates were unique.
- Structural mismatch in the 50 TARE outer folds had weak associations with
  TARE MAE, coverage, width and false-ALLOW risk.
- A family-only strong-effect sensitivity changed the DA method ordering, so
  taxonomic resolution is treated as part of the applicability boundary.

## Reproducibility

The complete TARE-DA run was repeated. All six output artifacts, including the
serialized model, were byte-identical:

- `tare_da_final_model.joblib`
- `tare_da_manifest.json`
- `tare_da_overall.csv`
- `tare_da_predictions.csv`
- `tare_da_project_envelopes.csv`
- `tare_da_summary.csv`

SHA-256 records are in
`artifacts/repro_check/reproducibility_hashes.csv`.

## Workflow anomaly and disclosure

The first TARE-DA implementation incorrectly aligned test features with
training rows. It yielded an overall MAE of 0.1168 and failed the point-accuracy
gate. This was diagnosed as an implementation/indexing bug, fixed, and the
frozen TARE-DA estimator was rerun. A later feature-screen file is retained for
transparency. Because the same project partitions were used for diagnosis, the
result should be described as internally validated rather than an untouched
confirmatory test. An independent external project panel remains required
before claiming general deployment validity.

## Statistical fallacy scan (11/11 checked)

| Type | Assessment |
|---|---|
| Simpson's paradox | Aggregate improvement is driven by the strong-effect stratum; the null stratum is slightly worse. Both are reported separately. |
| Ecological fallacy | The inference unit is a project-by-scenario prediction, not a microbial cell or an individual taxon. |
| Berkson's paradox | Public-project selection can induce non-representative structure; external generality is not claimed. |
| Collider bias | No post-treatment covariate adjustment or causal path model is used. |
| Base-rate neglect | Reliability prevalence is reported and ALLOW precision is not interpreted without its scenario-specific base rate. |
| Regression to the mean | No pre/post extreme-group intervention design. |
| Survivorship bias | Method failures were scored as utility 0 and retained; they were not silently excluded. |
| Look-elsewhere effect | Primary endpoints and eight method-by-scenario groups were fixed before the corrected run; secondary model screening is labelled development-only. |
| Garden of forking paths | One indexing bug required correction; the reused partitions limit confirmatory status and are disclosed. |
| Correlation versus causation | TARE predicts performance under specified simulations; it makes no causal biological claim. |
| Reverse causality | The model is predictive, not causal. |

## Decision

TARE-DA is a defensible new modelling contribution: a project-held-out,
conformalized reliability envelope that improves strong-effect DA performance
prediction and supplies conservative recommendation decisions. It should not
yet be described as a generally transportable biological decision system.
