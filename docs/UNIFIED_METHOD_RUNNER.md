# Unified differential-abundance runner

## Purpose

Run ALDEx2, ANCOM-BC2, LinDA and MaAsLin2 on one prepared Paper 4 scenario
dataset and return a common feature-level schema.

## Prerequisites

- R environment: `D:\bioenv\envs\riz16s`
- R library: `<repository-root>\Rlib`
- Python: `<repository-root>\.venv_deeptaxa\Scripts\python.exe`
- Installed versions:
  - ALDEx2 1.40.0
  - ANCOMBC 2.12.1
  - MicrobiomeStat 1.4
  - Maaslin2 1.26.0

## One-command run

```powershell
& '<repository-root>\scripts\p4_run_all_methods.ps1' `
  -InputDir '<prepared-input-directory>' `
  -OutputDir '<result-directory>' `
  -Scenario 'sparse20_fc1_41' `
  -Seed 4000 `
  -Methods @('aldex2','ancombc2','linda','maaslin2')
```

The wrapper:

1. stages input to an ASCII working directory;
2. runs each method in a separate R process;
3. captures stdout/stderr to a method-specific log;
4. preserves failed methods in the manifest;
5. computes BH/BY thresholds, TP/FP/FN, FDP, power, F1 and direction accuracy;
6. copies results to `OutputDir`.

## Common output schema

| Column | Meaning |
|---|---|
| `method` | Method identifier |
| `feature` | Feature identifier |
| `estimate` | Model effect estimate; scale is method-specific and recorded |
| `statistic` | Test statistic |
| `pvalue` | Raw p-value |
| `padj` | Method-native multiple-testing result, BH where configurable |
| `reject` | BH significance at 0.05 |
| `direction` | Sign of the method effect |
| `method_flag` | Method-specific diagnostic, such as ANCOM-BC2 `passed_ss` |
| `status` | `ok` or `failed` |
| `error` | Failure reason |

## Method settings

- ALDEx2:
  - project-condition pseudobulk aggregation;
  - 128 Monte-Carlo draws;
  - paired Welch test across projects.
- ANCOM-BC2:
  - `condition` fixed effect;
  - `(1 | project)` random effect;
  - BH adjustment;
  - pseudo-count sensitivity enabled.
- LinDA:
  - CLR model with `condition + (1 | project)`;
  - prevalence filter 0.10;
  - fixed pseudo-count 0.5 with `adaptive = FALSE`;
  - default winsorization at the upper 3% tail;
  - BH adjustment.
- MaAsLin2:
  - TSS + log transform + linear mixed model;
  - `condition` fixed and `project` random;
  - prevalence filter 0.10;
  - BH adjustment.

## Files

- `<method>.csv`: common feature-level results.
- `<method>.json`: version, status, runtime, argument and failure metadata.
- `<method>.log`: full console log.
- `method_unified_long.csv`: merged feature-level results.
- `method_unified_metrics.csv`: method × scenario performance.
- `method_unified_manifest.csv`: execution manifest.
- `method_wrapper_status.csv`: wrapper exit codes and status.

## Current smoke status

All four methods completed with `status = ok` on a 60-feature test subset.
ANCOM-BC2 is much slower because pseudo-count sensitivity analysis is enabled.

