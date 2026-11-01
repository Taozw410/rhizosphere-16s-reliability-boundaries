# Reproducibility, synthetic accuracy and transportability in multi-study rhizosphere 16S analyses

This repository contains the code, frozen simulation parameters, summary outputs, source data, figure data and submission materials for a multi-study benchmark of rhizosphere 16S analyses.

The study asks whether three reliability dimensions are interchangeable:

1. reproducibility across seeds, splits or real project panels;
2. accuracy against a specified simulation truth;
3. structural transportability of a fitted generator to withheld 15-project panels.

The benchmark integrates 45 public maize, rice and wheat rhizosphere projects comprising 1,219 samples and 2,024 named genus- or family-level features plus one unassigned aggregate. Raw FASTQ files are not redistributed. Public project accessions are listed in `data/project_manifest/project_accessions.csv`.

## Repository contents

- `scripts/`: Python, R and PowerShell analysis code.
- `data/parameter_estimation/`: calibrated simulation parameters and calibration summaries.
- `data/formal_measure/`: a compact measured DA example used to verify the unified method interface.
- `data/summary/`: frozen summary outputs used by the manuscript.
- `data/project_manifest/`: project selection, metadata and sequencing summaries.
- `data/figure_source_data/`: source data for Figures 1-5.
- `data/source_data/`: consolidated workbook and manuscript tables.
- `submission/`: current manuscript, supplementary material, figures and declarations.
- `docs/`: benchmark specifications, seed budget and reproduction notes.
- `environment/`: software versions and package requirements.

## Reproduction

Run commands from the repository root. The scripts use paths relative to the repository and no longer depend on the original workstation path. See `REPRODUCING.md`.

## Authors

- Ziwei Tao, Guangxi University of Science and Technology
- Jian Xie, Guangxi University of Science and Technology
- Fan Yang, Zhejiang University of Water Resources and Electric Power

Corresponding author: Ziwei Tao, taozw@gxust.edu.cn

## Data provenance

The 45 source projects were reprocessed from public FASTQ reads. The repository stores derived feature summaries and analysis inputs that contain no private or controlled-access data.

## Reuse terms

No reuse license has been assigned in this working repository. The authors should choose a code license and a data license before publishing a DOI for the associated article.
