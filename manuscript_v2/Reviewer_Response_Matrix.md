# Reviewer response matrix

| Reviewer issue | Revision |
| --- | --- |
| TARE-DA scope could be mistaken for all four analysis domains | Abstract, Introduction and Discussion now state that TARE-DA is trained and validated only for DA. Core, network and classification results are context and failure-boundary evidence. |
| Utility and interval target were undefined | Utility is defined by scenario as U = F1 for strong effects or U = 1 - false-positive rate for null effects, range 0-1, with no weighted composite. The interval targets project-method utility under the empirical project distribution. |
| Composite utility could hide false-positive risk | FDP and false-positive rate are reported separately. Reliability labels require FDP <= 0.10 and F1 >= 0.50 for strong effects. False-ALLOW risk and threshold tradeoffs are reported. |
| Simulator failure does not prove the envelope is valid | The manuscript now treats simulator failure only as motivation. A fold-level analysis shows weak relationships between structural mismatch and TARE MAE, coverage, width or false-ALLOW risk. |
| Calibration-versus-transport logic needed strengthening | The eight summaries, tolerances, analysis units, 3.87/8 and 3.40/8 aggregation, SDs and ranges are now explicit. The structural and TARE evidence chains are separated. |
| Project-level leakage needed to be ruled out | Methods state that all rows from a project remain in the same inner and outer fold. Model, scaler, conformal residuals and classifier are refit from outer training projects only. |
| Conformal guarantees needed precise boundaries | The nonconformity score, grouped calibration, nominal radius and empirical marginal coverage are defined. The paper states that project-specific intervals are not per-project conditional guarantees. |
| External shift conflicts with a generic conformal claim | All external projects are reported as outside support and receive DEFER_OUT_OF_SUPPORT. External coverage is described as an empirical result under covariate shift, not a formal guarantee. |
| ALLOW/CONDITIONAL/DENY rules needed auditing | Thresholds, lower-bound requirements, support gating and concrete CONDITIONAL actions are specified. DENY (low utility evidence) and DEFER (outside applicability domain) are separate outputs. |
| AUC alone is insufficient | The revision reports interval width, ALLOW count, ALLOW precision, false-ALLOW count, false-ALLOW rate and a threshold risk-acceptance grid. |
| The paper needed a real rhizosphere application | PRJNA554261 provides a real project-template application. Four methods produce disjoint or partially overlapping candidate sets, and TARE withholds approval. The example is explicitly not a real treatment-effect validation. |
| Mixed genus/family hierarchy could affect comparability | The mutually exclusive hierarchy and cross-project union are defined. A prespecified family-only sensitivity shows that the DA method order changes, so taxonomic resolution is now part of the reliability boundary. |
| External AUC may overstate independent information | Per-project external results are reported. The manuscript states that method-specific AUROCs use six projects per scenario and are descriptive. |
| English terminology was incomplete | Main terminology now uses 16S rRNA gene sequencing, random seeds, simulator-defined ground truth and distinct reliability properties. |
| Figure 1 denominators were ambiguous | The caption now lists denominators for the four bars: 80, 10, 144 and 480 checks. |
| Reference and formatting integrity needed checking | All 24 references are cited, all 6 figures and 8 tables are cited, line numbers and page fields are present, and the DOCX contains no comments or tracked changes. |
