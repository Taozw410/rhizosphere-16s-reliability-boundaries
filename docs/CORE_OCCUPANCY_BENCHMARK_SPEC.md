# Paper 4 core / occupancy benchmark specification v1.1

Date frozen: 2026-10-27  
Revision note: v1.1 was made before any core-method output existed. Simulator
smoke showed that requiring non-null core scenarios to preserve every null-data
summary confounds the intended occupancy intervention with generator error.

## 1. Purpose

Test whether prevalence-threshold, empirical-Bayes and hierarchical
occupancy-detection approaches recover a known core microbiome under
project-level heterogeneity and imperfect detection.

This benchmark is separate from the completed differential-abundance grid.
Its methods, truth definition and endpoints are not inferred from the DA
ranking.

## 2. Analysis unit and truth

The analysis unit remains the project. For feature `i`, project `j` and sample
`s`, define the latent detection probability:

`pi_ijs = logit^-1(mu_i + b_ij + gamma * depth_z_js)`

where:

- `mu_i` is the feature-level occupancy;
- `b_ij ~ Normal(0, sigma_project^2)` is project-level occupancy heterogeneity;
- `gamma` is a fixed detection-depth coefficient;
- `depth_z` is standardized log sequencing depth.

A feature is a **population core feature** when, across the sampled projects:

1. its latent detection probability is at least 0.90; and
2. it meets that criterion in at least 80% of projects.

A feature with high abundance but occupancy below this threshold is not core.
A low-abundance feature that nevertheless meets the detection-probability
criterion is core. Core truth therefore does not depend on an observed
prevalence cutoff.

The simulation also exports:

- `latent_occupancy_mean`;
- `latent_occupancy_sd`;
- `project_core_fraction`;
- `is_population_core`;
- `core_class` (`core`, `boundary`, `rare`);
- expected detection probability for every sample-feature pair.

## 3. Scenarios

The following scenarios will be frozen before implementation:

| Scenario | Design | Purpose |
|---|---|---|
| `core_null` | no feature meets the core definition | false-core control |
| `core20_balanced` | 20% core; abundance independent of core status | baseline recovery |
| `core20_lowabundance` | 20% core, including low-abundance core features | abundance-prevalence separation |
| `core20_projectshift` | 20% core with larger between-project occupancy SD | project heterogeneity |
| `core20_depthconfound` | core status associated with lower sequencing depth | detection-bias stress test |

Initial smoke grid:

- 15 projects;
- 10 seeds per scenario;
- 120 candidate features;
- 20 core or boundary features in non-null scenarios;
- the observed depth distribution from the 45-project panel.

The extension grid will use a separate fixed seed block only after the smoke
test passes.

## 4. Candidate methods

### M1: sample prevalence threshold

Declare feature `i` core when its observed prevalence across pooled samples is
at least a fixed threshold.

Threshold grid:

- 0.50;
- 0.60;
- 0.70;
- 0.80;
- 0.90.

The threshold cannot be chosen per simulation seed. The grid is retained to
show operating characteristics and threshold transfer.

### M2: project-balanced prevalence threshold

Declare feature `i` core when it meets a within-project prevalence threshold in
at least 80% of projects. This mirrors the truth definition without borrowing
latent probabilities or depth coefficients.

### M3: empirical-Bayes beta-binomial

For each feature, estimate a beta-binomial posterior mean occupancy from
observed detections. A feature is core when the posterior lower bound exceeds
0.90. Empirical-Bayes priors are estimated across features within a dataset,
not supplied from simulation truth.

### M4: hierarchical occupancy-detection model

Fit a logistic model for detection with:

- feature-specific intercepts;
- project random intercepts;
- log-depth offset or slope;
- optional project-by-feature heterogeneity.

Core status is based on the posterior or penalized-likelihood estimate of the
project-averaged detection probability, with uncertainty propagated to the
core call.

The first implementation will use the installed `glmmTMB` stack if its runtime
and convergence are acceptable. A fallback maximum-likelihood or empirical-Bayes
approximation must be declared before running the grid.

## 5. Endpoints

Primary:

1. execution validity;
2. false-core rate at a pre-specified nominal level;
3. core recall;
4. core F1;
5. probability calibration: Brier score and calibration slope/intercept.

Secondary:

- low-abundance core recall;
- boundary-feature classification;
- project-level core agreement;
- threshold transfer across project count and depth;
- among-project rank correlation of latent and estimated occupancy;
- runtime and convergence failures.

Primary ranking is lexicographic:

1. execution validity;
2. false-core rate no greater than 0.10;
3. higher core F1;
4. better Brier score;
5. higher low-abundance core recall;
6. lower runtime.

## 6. Pooling and calibration rules

- Do not tune a probability cutoff separately for each simulation seed.
- If a method outputs probabilities, report calibration before thresholding.
- A threshold selected on smoke seeds must be frozen before extension seeds.
- Undefined F1 from a method with no positive calls remains undefined and cannot
  be silently replaced by zero.
- Failed fits remain in the denominator for execution validity.

## 7. Required implementation gates

### Gate C1: simulator calibration

The `core_null` simulation is the calibration reference and must match the
corresponding 15-project, 120-feature real panel for:

- prevalence quantiles;
- rank-abundance quantiles;
- sequencing-depth quantiles;
- global zero fraction;
- within/between-project variance ratio;
- Bray-Curtis and Aitchison distance quantiles.

Non-null core scenarios intentionally alter prevalence, rank abundance, zero
fraction, Aitchison distance and network topology. Their calibration therefore
has two parts:

1. **Invariant-nuisance gates**
   - sample depths must match the real panel exactly;
   - no sample may have zero depth;
   - within/between-project variance ratio must remain within the null
     tolerance;
   - Bray-Curtis distance must remain within the null tolerance.
2. **Scenario-target gates**
   - prevalence, rank abundance, zero fraction, Aitchison distance and network
     topology must match a target envelope computed from the frozen occupancy
     intervention before seeing any method result.

The scenario-target envelope must be written to a separate configuration file.
It cannot be chosen to favour one of the candidate core methods.

The latent core truth must not be computed from the realized simulated matrix.

### Gate C2: null calibration

In `core_null`, no method may declare a feature core beyond its declared
false-core operating level across seeds.

### Gate C3: probability calibration

Methods that output probability scores must be assessed by Brier score and
calibration slope before threshold-based F1 is interpreted.

### Gate C4: project leakage check

Any method using a global prevalence threshold must be evaluated separately
from methods that model project heterogeneity. The latter cannot claim an
advantage unless the between-project occupancy SD is nonzero.

## 8. Outputs

The future core benchmark will write:

- `core_seed_grid.csv`;
- `core_truth.csv`;
- `core_predictions_long.csv`;
- `core_metrics_by_replicate.csv`;
- `core_summary_by_scenario.csv`;
- `core_calibration.csv`;
- `core_failures.csv`;
- `core_report.md`;
- `core_audit.json`.
- `core_scenario_targets.json`.

## 9. Stop conditions

- Simulator fails any calibration gate after three documented revisions.
- A non-null scenario is compared directly with the null distribution without
  defining its intervention-specific target envelope.
- A method's fit failure rate exceeds 20% in smoke data.
- Core truth requires a post hoc threshold based on observed prevalence.
- The active method set changes after seeing extension results.

