# v0.2.0: TARE reliability envelope

This release adds the Transportability-Aware Reliability Envelope and the
restructured manuscript.

## Added

- TARE-DA model, serialized artifact and project-specific envelopes.
- 90 project-level known-effect datasets and 360 DA method outcomes.
- Ten repeated five-fold project holdouts and 3,600 predictions.
- Six-project independent external panel with no development overlap.
- External support-distance and out-of-support decisions.
- Family-level taxonomic sensitivity.
- Project-level candidate-set application example.
- Simulator-mismatch to reliability linkage analysis.
- Decision-threshold risk-acceptance curve.
- Figure 6 and the restructured manuscript DOCX/PDF.

## Key outcomes

- Internal utility MAE: 0.0790 versus 0.0948.
- Internal conformal coverage: 0.954.
- Strong-effect reliability AUROC: 0.883.
- ALLOW precision: 0.983.
- External MAE: 0.1414 versus 0.1853.
- External coverage: 0.854.
- All external projects were outside the development support envelope and
  received DEFER_OUT_OF_SUPPORT.

The previous benchmark package remains in the repository and in release
v0.1.1.
