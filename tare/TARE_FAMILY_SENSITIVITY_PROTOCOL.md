# Family-level feature sensitivity protocol

## Status

Freeze date: 2026-11-11, before generating or inspecting family-only method
results.

## Question

Do the principal differential-abundance conclusions depend on the mixed
genus-primary/family-fallback feature representation?

## Feature rule

- Eligible features: the 444 `f__` family fallback features in the development
  matrix.
- Ranking rule: descending cross-sample prevalence, then descending mean
  relative abundance, then feature name.
- Panel: the first 120 eligible family features.
- No method outcome, TARE prediction or external result contributes to feature
  ranking.

## Data generation

- 45 development projects.
- One strong sparse-effect dataset per project.
- Twenty eligible family features receive alternating positive and negative
  log2 effects.
- Conditions balanced within project.
- The realized effect after compositional renormalization is truth.
- Seeds are fixed independently of the mixed-level analysis.

## Methods

ALDEx2, ANCOM-BC2, LinDA and MaAsLin2 are run with the same project-specific
fixed-effect condition model used for TARE-DA construction.

## Endpoint

- Mean FDP, power and F1 by method.
- Method order by F1 and FDP.
- Directional agreement with the mixed-level strong-effect benchmark.

TARE-DA is not retrained. The analysis tests whether the broader DA benchmark
conclusions are sensitive to the mixed taxonomic hierarchy; it does not
establish TARE performance on a family-only model.
