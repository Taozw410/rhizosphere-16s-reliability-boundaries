# A transportability-aware reliability envelope for rhizosphere 16S differential-abundance analysis

This repository contains the code, frozen parameters, summary outputs, source
data and manuscript for a multi-study rhizosphere 16S reliability study. The
central contribution is TARE-DA, a project-specific differential-abundance
reliability envelope that predicts method utility intervals, estimates the
probability of meeting a reliability criterion and distinguishes evidence
decisions from out-of-support applicability.

The development corpus contains 45 public maize, rice and wheat rhizosphere
projects (1,219 samples). The independent stress panel contains six projects
(two per crop; 426 samples) with no project-accession overlap.

## Key results

- Internal TARE-DA utility MAE: 0.0790 versus 0.0948 for the development-only
  method-by-scenario prior.
- Internal 95% conformal coverage: 0.954.
- Strong-effect reliability AUROC: 0.883.
- ALLOW precision: 0.983.
- Independent external MAE: 0.1414 versus 0.1853.
- External coverage: 0.854.
- All external projects were outside the development support envelope and were
  assigned DEFER_OUT_OF_SUPPORT.

## Repository layout

- `tare/`: TARE specifications, scripts, model artifact, predictions, external
  panel, family-level sensitivity, application example, diagnostics and
  Figure 6.
- `manuscript_v2/`: restructured manuscript source, DOCX/PDF, tables, figure
  legends, references, Chinese abstract and reviewer response matrix.
- `data/tare_inputs/`: portable input tables required by the TARE scripts.
- `scripts/`: original benchmark scripts plus portable TARE runners.
- `data/summary/`, `data/project_manifest/`, `figures/`: the earlier benchmark
  release retained for provenance.
- `submission/`: the earlier Microbiological Research package retained as
  historical material.

## Reproduction

Run commands from the repository root. See `tare/REPRODUCING.md`. Raw FASTQ
files and large intermediate outputs are not redistributed.

## Authors

- Ziwei Tao, Guangxi University of Science and Technology
- Jian Xie, Guangxi University of Science and Technology
- Fan Yang, Zhejiang University of Water Resources and Electric Power

Corresponding author: Ziwei Tao, taozw@gxust.edu.cn

## Version

Current release: v0.2.0.

The v0.1.1 benchmark release remains available in Git history and GitHub
releases.
