# Paper 4 manuscript figure contract

Date: 2026-10-29  
Backend: Python, matplotlib only  
Output: 183 mm double-column figures, editable SVG and PDF text, TIFF preview

## Figure 1 | A calibrated multi-study benchmark framework

Results-level question: Is the benchmark grounded in the observed 45-project
structure and a documented validation design?

Figure-level claim: The study defines a reproducible hierarchy from real
multi-project data to calibrated simulations, four analysis families, and an
independent project-panel external check.

- a Workflow from observed projects through calibration, benchmarks, and
  external validation.
- b Sample sizes of the 45 projects by crop.
- c Benchmark scale and execution completeness.
- d Null calibration and primary/extension structural gate status.

Archetype: schematic-led composite.

## Figure 2 | Differential-abundance results diverge by method and scenario

Results-level question: Which differential-abundance method preserves FDP and
power under sparse, dense, and project-heterogeneous effect scenarios?

Figure-level claim: Scenario-specific performance is substantial, and the main
ranking is reproducible only in direction across the two generator families.

- a Method by scenario F1 under the primary generator.
- b Method by scenario FDP under the primary generator.
- c Joint FDP-power comparison in the strong sparse scenario across both
  generators.
- d Primary versus extension FDP replication for LinDA.

Archetype: quantitative grid with c as hero panel.

## Figure 3 | Core recovery is strong except under depth confounding

Results-level question: How accurately do prevalence, empirical-Bayes, and
occupancy models recover latent core status?

Figure-level claim: Project-level prevalence is a strong baseline, while depth
confounding is the common failure boundary.

- a Method by scenario F1.
- b Method-level mean F1 with 95% bootstrap interval.
- c Worst scenario false-core rate.
- d Precision-recall boundary in the depth-confounded scenario.

Archetype: quantitative grid with failure boundary as hero panel.

## Figure 4 | Network truth recovery and taxonomic classification are weak

Results-level question: Do correlation and sequence classifiers recover network
edges and 16S genus labels under realistic perturbations?

Figure-level claim: Network edge recovery is weak and unstable, while classical
k-mer classifiers outperform small from-scratch neural models.

- a Synthetic network AUPRC by scenario and method.
- b Real three-panel network stability.
- c Closed-set taxonomy accuracy and macro F1.
- d Unseen-genus rejection against known-taxon retention.

Archetype: asymmetric mixed-metric quantitative grid.

## Figure 5 | External project-panel validation remains the principal boundary

Results-level question: Do simulation conclusions survive project-panel
holdout?

Figure-level claim: Three-panel structural calibration is stable, but repeated
leave-panel-out prediction systematically fails.

- a Passed structural summaries in three-panel calibration.
- b Fixed-tolerance pass count across 30 held-out folds.
- c Posterior predictive coverage by metric and generator.
- d Explicit external-validity decision.

Archetype: quantitative validation envelope with d as interpretive endpoint.

## Shared statistical rules

- Replicate unit: project for data structure; seed for simulation performance;
  fold for external validation.
- Center: mean except where bootstrap intervals are reported explicitly.
- Spread: seed SD or 95% bootstrap interval as stated in each legend.
- Multiple testing: BH at 0.05 within method × scenario × replicate.
- Source data: one CSV per figure under `figures/source_data`.
- Methods use one consistent palette throughout.

