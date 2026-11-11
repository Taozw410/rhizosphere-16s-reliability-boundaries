# TARE audit

Overall status: PASS

| Check | Status |
|---|---|
| corpus_rows_90 | PASS |
| corpus_projects_45 | PASS |
| corpus_two_scenarios_per_project | PASS |
| metric_rows_360 | PASS |
| status_rows_360 | PASS |
| prediction_rows_3600 | PASS |
| partition_count_10 | PASS |
| each_partition_contains_each_project_once_per_method_scenario | PASS |
| tare_mae_below_baseline | PASS |
| coverage_at_least_0_80 | PASS |
| allow_precision_above_0_80 | PASS |
| all_reproducibility_hashes_identical | PASS |
| final_model_loaded | PASS |
| external_six_projects | PASS |
| external_two_per_crop | PASS |
| external_zero_development_overlap | PASS |
| external_426_samples | PASS |
| external_48_predictions | PASS |
| external_all_method_rows_ok | PASS |
| external_model_not_retrained | PASS |
| external_mae_below_development_prior | PASS |
| external_coverage_at_least_0_80 | PASS |
| external_ood_fraction_recorded | PASS |
| family_sensitivity_180_outcomes | PASS |
| family_sensitivity_order_changed | PASS |
| application_example_present | PASS |
| mismatch_linkage_present | PASS |
| decision_threshold_curve_present | PASS |

Method execution status:

| Method | Status | Count |
|---|---|---:|
| aldex2 | ok | 90 |
| ancombc2 | ok | 90 |
| linda | failed | 4 |
| linda | ok | 86 |
| maaslin2 | ok | 90 |
