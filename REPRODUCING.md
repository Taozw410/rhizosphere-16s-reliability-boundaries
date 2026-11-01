# Reproduction guide

## Environment

The manuscript analyses used R 4.5.3, Python 3.12 and the package versions recorded in `environment/SOFTWARE_VERSIONS.md`. The compact DA rerun requires the R packages ALDEx2, ANCOMBC, MicrobiomeStat and Maaslin2. Python analyses require NumPy, pandas, SciPy, scikit-learn and PyTorch.

## Path layout

All scripts are intended to run from the repository root. The original workstation-specific absolute path was replaced with repository-relative resolution.

## Compact DA verification

The files in `data/formal_measure` preserve one measured, frozen method-interface run:

- `input/counts.csv`
- `input/metadata.csv`
- `input/truth.csv`
- `input/input_manifest.json`
- `results/*`

The R launcher is `scripts/p4_unified_da_runner.R`.

## Rebuilding tables and figures

The frozen summaries in `data/summary` feed the manuscript tables and figures. Figure source data are duplicated under `data/figure_source_data` for direct inspection. The figure generator is `scripts/p4_make_manuscript_figures.py`.

## Large intermediate files

Generated benchmark datasets, dense per-replicate method outputs and raw FASTQ files are intentionally excluded. They are reproducible from the frozen parameter files, seed grids and scripts; the included aggregate summaries preserve all numerical results reported in the manuscript.
