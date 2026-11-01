# Paper 4 network benchmark specification v1.0

Date frozen: 2026-10-27

## Scope

Benchmark network edge recovery from compositional counts with a known direct
edge graph and project structure.

## Simulation

- 40 nodes.
- 15 projects, 20 samples per project.
- Log-normal sequencing depths centered at 10,000 reads.
- Depth-dependent zero generation.
- Feature and node effects, project-level latent shifts and multinomial
  sampling.

Scenarios:

- `network_null`, no graph edges;
- `network_sparse`, edge probability 0.04;
- `network_dense`, edge probability 0.10;
- `network_projectshift`, sparse graph with larger project effects.

True direct edges are encoded in a Gaussian precision matrix. Compositional
closure is introduced by converting latent abundance to probabilities.

## Methods

- CLR Spearman correlation;
- CLR Pearson correlation;
- SparCC-like iterative proportionality;
- graphical lasso on CLR covariance.

## Endpoints

Primary:

- edge AUPRC;
- edge AUROC;
- hub-degree Spearman correlation.

Secondary:

- estimated degree at the top 10% edge-score cutoff;
- null-scenario false edge count.

Calibration and validation use disjoint five-seed blocks.

