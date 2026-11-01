# Paper 4 benchmark reproduction

All commands are run from `<repository-root>` on Windows.

## Environment

- Python: `<repository-root>\.venv_deeptaxa\Scripts\python.exe`
- R: `D:\bioenv\envs\riz16s\Scripts\Rscript.exe`
- R library: `<repository-root>\Rlib`
- R packages: ALDEx2 1.40.0, ANCOMBC 2.12.1, MicrobiomeStat 1.4,
  Maaslin2 1.26.0

## Differential abundance

Build the original grid:

```powershell
python scripts/p4_build_formal_grid.py
python scripts/p4_orchestrate_formal_grid.py --stage primary --workers 2
python scripts/p4_orchestrate_formal_grid.py --stage extension --workers 2
python scripts/p4_summarize_formal_results.py
```

Build and run the second semi-parametric generator:

```powershell
python scripts/p4_calibrate_perturbed.py
python scripts/p4_build_perturbed_da_grid.py
python scripts/p4_orchestrate_formal_grid.py `
  --stage primary --workers 4 `
  --grid outputs/formal_grid_generator2/seed_grid.csv `
  --out-root outputs/formal_grid_generator2/results `
  --work-root output_work/p4_formal_grid_generator2
python scripts/p4_summarize_da_cross_generator.py
```

## Core and occupancy

```powershell
python scripts/p4_run_core_smoke.py --generator latent-factor
python scripts/p4_validate_core_targets.py
python scripts/p4_build_core_grid.py
python scripts/p4_orchestrate_core_grid.py --stage primary
python scripts/p4_orchestrate_core_grid.py --stage extension
python scripts/p4_audit_core_formal.py
python scripts/p4_summarize_core_formal.py
```

## Network

```powershell
python scripts/p4_network_benchmark.py
python scripts/p4_real_network_stability.py
```

## Classification

```powershell
python scripts/p4_classification_benchmark.py --seeds 5
```

The pretrained DNA-BERT comparison is not included. The model artifacts were
not available locally and external retrieval failed.

## Project sampling and real holdout

```powershell
python scripts/p4_validate_project_panels.py
python scripts/p4_real_project_holdout.py
```

## Main entry points

- `INTEGRATED_BENCHMARK_REPORT.md`
- `outputs/integrated_benchmark_status.json`
- `outputs/da_cross_generator/da_cross_generator_report.md`
- `outputs/core_formal_grid/results/core_formal_report.md`
- `outputs/network_benchmark/network_report.md`
- `outputs/classification_benchmark/classification_report.md`
- `outputs/project_panel_validation/report.md`
- `outputs/real_project_holdout/report.md`

