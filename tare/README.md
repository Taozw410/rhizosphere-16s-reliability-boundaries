# TARE

This directory contains the Transportability-Aware Reliability Envelope added
after the Microbiological Research desk rejection.

- `TARE_DA_METHOD_SPEC.md`: frozen project-held-out DA model.
- `TARE_EXTERNAL_PANEL_PROTOCOL.md`: frozen six-project independent panel.
- `TARE_VALIDATION_REPORT.md`: complete validation and statistical audit.
- `TARE_EXTERNAL_VALIDATION_REPORT.md`: independent external stress test.
- `TARE_TAXONOMIC_SENSITIVITY_REPORT.md`: family-only feature sensitivity.
- `TARE_MANUSCRIPT_SECTION.md`: manuscript-ready methods, results and figure
  legend.
- `RUNBOOK.md`: exact reproduction commands.
- `artifacts/tare_da/tare_da_final_model.joblib`: fitted deployable TARE-DA
  model.
- `artifacts/tare_da/tare_da_project_envelopes.csv`: project-by-scenario
  envelopes from the final fit.
- `artifacts/external_panel/`: external matrices, known-effect corpus, four
  DA results and frozen-model predictions.
- `artifacts/application_example/`: project-level DA conflict example.
- `artifacts/family_sensitivity/`: family-only method benchmark.
- `artifacts/transport_reliability_link/`: simulator-mismatch linkage.
- `artifacts/tare_da/decision_threshold_curve.csv`: risk-acceptance grid.

The primary model result is project-held-out and simulator-defined. It should
not be described as biological accuracy on unlabelled real projects.
