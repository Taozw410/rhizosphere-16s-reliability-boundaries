# Family-level taxonomic sensitivity

## Design

- 444 family fallback features in the development matrix.
- Fixed panel of the top 120 features by cross-sample prevalence, mean
  relative abundance and feature name.
- One simulator-defined strong sparse-effect dataset per development project.
- Four DA methods run with the same project-level fixed-effect interface.

## Results

| Method | Mixed-level F1 | Mixed rank | Family-only F1 | Family rank |
| --- | ---: | ---: | ---: | ---: |
| ALDEx2 | 0.093 | 4 | 0.374 | 4 |
| ANCOM-BC2 | 0.221 | 1 | 0.543 | 3 |
| LinDA | 0.140 | 3 | 0.650 | 1 |
| MaAsLin2 | 0.155 | 2 | 0.644 | 2 |

The F1 ordering changed from ANCOM-BC2 > MaAsLin2 > LinDA > ALDEx2 under
the mixed-level panel to LinDA > MaAsLin2 > ANCOM-BC2 > ALDEx2 under the
family-only panel.

Project-level F1 agreement between the two representations was weak and
negative for ALDEx2 (rho = -0.22), ANCOM-BC2 (-0.08) and MaAsLin2 (-0.34).
LinDA F1 correlation was not estimable because its mixed-level values were
constant or missing across the paired finite results. FDP agreement was weak
to moderate where estimable (rho = 0.14-0.41).

## Interpretation

The DA ranking is taxonomically conditional. The revised manuscript therefore
does not treat the mixed-level ranking as invariant. TARE-DA was not retrained
or recalibrated on the family-only panel. The result supports a broader
reliability-boundary argument: taxonomic resolution is part of the applicability
domain and should be stated before a method recommendation is transferred.

## Files

- `artifacts/family_sensitivity/family_feature_panel.csv`
- `artifacts/family_sensitivity/da_method_results/da_project_metrics.csv`
- `artifacts/family_sensitivity/family_vs_mixed_summary.csv`
- `artifacts/family_sensitivity/family_vs_mixed_correlations.csv`
