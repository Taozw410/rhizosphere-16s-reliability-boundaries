# Figure 6 contract

Core conclusion: TARE-DA improves project-specific prediction under
heterogeneous sparse effects and withholds method recommendations when the
calibrated reliability evidence is insufficient.

Results-level question: Does the reliability envelope add project-held-out
performance information beyond a study-agnostic method-by-scenario mean while
remaining calibrated and decision-safe?

Figure archetype: quantitative grid.

Target journal/output: double-column manuscript figure; editable SVG and PDF,
600-dpi TIFF, PNG preview, source data and QA reports.

Backend: Python/matplotlib.

Final size: 183 mm by 128 mm.

Panel map:

- a: paired improvement in utility MAE across ten independent project-split
  partitions; hero evidence for transport prediction.
- b: method-level TARE-DA versus mean-only MAE in the strong-effect scenario;
  decomposition of where prediction improves.
- c: empirical coverage of 95% conformal intervals across scenario and method;
  calibration and risk of over-confident intervals.
- d: ALLOW, CONDITIONAL and DENY decisions by scenario, with strong-effect
  reliability AUROC and aggregate ALLOW precision; recommendation boundary.

Evidence hierarchy:

- Hero evidence: panel a.
- Decomposition: panel b.
- Calibration: panel c.
- Decision boundary: panel d.

Statistics needed:

- 10 repeated stratified five-fold project holdouts.
- 3,600 held-out predictions.
- MAE, empirical interval coverage, reliability AUROC and ALLOW precision.
- Correlated repeated partitions are shown individually; no independent-sample
  significance test is applied.

Source data:

- `artifacts/tare_da/tare_da_predictions.csv`
- `artifacts/tare_da/tare_da_summary.csv`
- `artifacts/tare_da/tare_da_overall.csv`

Reviewer risk:

- Do not imply biological ground truth on unlabelled real projects.
- Do not imply that 45 projects are independent.
- The strong-effect denominator and reliability prevalence must be visible.
- The zero ALLOW count for strong effects must be interpreted as abstention,
  not absence of usable ranking information.
