# Transportability-Aware Reliability Envelope (TARE)

## Material Passport

- Origin skill: academic-research-suite / experiment-agent
- Origin mode: plan
- Origin date: 2026-11-11
- Verification status: protocol frozen before outer holdout evaluation
- Version: TARE-v1

## Scientific question

Can a target-project reliability envelope distinguish (i) structure that is
stable under repeated simulation, (ii) structure that remains accurate against
a simulator-defined target, and (iii) structure that transports to projects
whose data were unavailable during model fitting?

The unit of prediction is a crop-stratified project panel containing five
maize, five rice and five wheat projects. The unit of inference is the panel's
eight prespecified structural summaries. TARE does not estimate differential
abundance, core membership or network accuracy on unlabelled real projects.

## Model

TARE is a conformalized hierarchical transport model with three components.

1. **Project encoder.** Study descriptors are sample count, median sequencing
   depth, modal ASV length, median ASV richness, unassigned percentage, genus
   percentage, family percentage and crop. Continuous descriptors are
   standardized using training projects only.
2. **Hierarchical community predictor.** A crop-aware ridge model predicts the
   centred-log-ratio project centroid from study descriptors. Training-project
   within-project residuals form a donor bank. A project's similarity weights
   combine crop support, descriptor distance and a global prior. A
   project-centroid residual covariance supplies a correlated generator.
3. **Reliability envelope.** For every target panel and generator, 100
   predictive panels are generated at the observed sample depths. TARE reports
   the 2.5th and 97.5th percentile of each structural summary, a Monte Carlo
   pass probability, an out-of-distribution support score, and an
   ALLOW/CONDITIONAL/DENY recommendation.

### Frozen reliability decision

For summary /(m/), let /(q_{0.025}/) and /(q_{0.975}/) be predictive
percentiles, /(T_m/) the existing operational tolerance, and /(p_m/) the
fraction of predictive panels whose summary is within /(T_m/) of the
predictive median.

- `ALLOW`: /(p_m /ge 0.80/) and support score /(/ge 0.25/).
- `CONDITIONAL`: /(0.50 /le p_m < 0.80/), or /(p_m /ge 0.80/) with support
  below 0.25.
- `DENY`: /(p_m < 0.50/).

The support score is the Gaussian-kernel similarity between the target panel
and its five nearest training projects, averaged over target projects.

## Training and leakage control

- Training corpus: all 45 projects; common frozen 120-feature panel.
- Outer validation: ten repeated stratified three-fold partitions. Every outer
  fold holds out 15 projects and fits TARE on the other 30.
- Target descriptors and sample depths are available to the predictor.
- Target community counts are used only to calculate the held-out truth.
- Feature panel, tolerances, model form, hyperparameter grid and decision
  thresholds are fixed before outer validation.
- Hyperparameters are selected only by inner leave-project-out calibration
  within each outer training set.

## Generators

1. `hierarchical_ridge`: descriptor-driven predicted centroid plus donor
   residuals.
2. `hierarchical_ridge_correlated`: the same model plus a shrunk covariance
   perturbation of project centroids.

The existing nearest-three donor model and crop-pool model are comparators.

## Endpoints

Primary endpoints:

1. Mean number of the eight fixed-tolerance summaries passed per held-out
   panel.
2. Empirical 95% predictive-interval coverage for each summary.
3. Discrimination of observed pass/fail by the predicted pass probability
   (AUROC).
4. Precision and coverage of `ALLOW` decisions.

Success is declared only if TARE exceeds the stronger comparator mean pass
count by at least 0.50, achieves at least 0.80 coverage for six of eight
summaries, and has `ALLOW` precision above 0.80. Otherwise the outcome is
reported as partial or negative.

## Interpretation boundaries

- The target truth is simulator-defined structural truth, not DA truth on real
  projects.
- A high TARE decision is permission to use a simulation calibration envelope;
  it is not evidence that a method is biologically correct.
- The 30 outer panels are correlated because every project is held out ten
  times. Results are summarized descriptively across folds.
- Network summaries are statistical co-occurrence topology, not ecological
  interaction networks.
