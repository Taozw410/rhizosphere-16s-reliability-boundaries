# TARE-v2 reproduction

## Environment

Use Python 3.12 with NumPy, pandas, SciPy, scikit-learn, joblib and python-docx.
The DA runners require R 4.5.3 with ALDEx2, ANCOMBC, MicrobiomeStat and
Maaslin2. Set `R_LIBS_USER` to a writable R library directory.

Set `TARE_CONDA_ENV` to the conda environment containing R and the four DA
packages. If conda is not used, invoke the bundled R runner directly.

## Inputs

Portable input tables are under `data/tare_inputs/`:

- `feature_matrix.csv`
- `sample_meta.csv`
- `study_covariates.csv`
- `sample_design.csv`
- `feature_panel.csv`
- external-panel matrices and selected family panel

The original public project accessions and raw-read provenance are retained
under `data/project_manifest/`.

## Core commands

```bash
python tare/scripts/p4_build_da_corpus.py
python tare/scripts/p4_tare_run_da.py --workers 4 --resume
python tare/scripts/p4_train_tare_da.py
python tare/scripts/p4_build_external_panel.py
python tare/scripts/p4_evaluate_external_panel.py
python tare/scripts/p4_link_transport_to_reliability.py
python tare/scripts/p4_application_example.py
python tare/scripts/p4_build_family_sensitivity.py
python tare/scripts/p4_summarize_family_sensitivity.py
python tare/scripts/p4_decision_threshold_curve.py
python tare/scripts/audit_tare.py
```

Some upstream path defaults are retained in the analysis scripts as explicit
arguments. Override them with `--feature-matrix`, `--sample-meta`,
`--study-covariates`, `--design`, `--panel` and `--out-dir`.

## Frozen outputs

The release includes the serialized TARE-DA model, internal and external
predictions, family-only sensitivity summaries, mismatch diagnostics,
application example, Figure 6, and the finalized manuscript. Repeated local
execution produced byte-identical model and summary artifacts; hashes are in
`tare/artifacts/tare_da/tare_audit.json`.
