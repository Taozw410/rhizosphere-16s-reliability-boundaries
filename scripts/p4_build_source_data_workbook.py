#!/usr/bin/env python3
"""Bundle figure and table source data into one Excel workbook."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
SOURCE = PAPER / "figures/source_data"
TABLES = PAPER / "manuscript/tables"
OUTPUT = PAPER / "manuscript/Source_Data.xlsx"

SHEET_NAMES = {
    "Fig1_project_sample_counts.csv": "Fig1_project_samples",
    "Fig1_benchmark_scale.csv": "Fig1_benchmark_scale",
    "Fig1_validation_summary.csv": "Fig1_validation",
    "Fig2_DA_summary.csv": "Fig2_DA_summary",
    "Fig2_LinDA_fdp_by_seed.csv": "Fig2_LinDA_by_seed",
    "Fig3_core_summary.csv": "Fig3_core_summary",
    "Fig3_method_bootstrap.csv": "Fig3_bootstrap_CI",
    "Fig4_network_summary.csv": "Fig4_network_summary",
    "Fig4_real_network_stability.csv": "Fig4_network_stability",
    "Fig4_classification_summary.csv": "Fig4_classification_all",
    "Fig4_classification_closed_set.csv": "Fig4_classification_closed",
    "Fig4_classification_unseen.csv": "Fig4_classification_unseen",
    "Fig5_project_panel_summary.csv": "Fig5_panel_summary",
    "Fig5_holdout_by_fold.csv": "Fig5_holdout_folds",
    "Fig5_holdout_coverage.csv": "Fig5_holdout_coverage",
}


def main() -> None:
    with pd.ExcelWriter(OUTPUT, engine="xlsxwriter") as writer:
        for source_file in sorted(SOURCE.glob("*.csv")):
            sheet = SHEET_NAMES.get(source_file.name, source_file.stem[:31])
            pd.read_csv(source_file).to_excel(writer, sheet_name=sheet, index=False)

        pd.read_csv(TABLES / "Table1_projects.csv").to_excel(
            writer, sheet_name="Table1_projects", index=False
        )
        pd.read_csv(TABLES / "Table2_DA_key_scenario.csv").to_excel(
            writer, sheet_name="Table2_DA_key", index=False
        )
        pd.read_csv(TABLES / "Table3_core_scenarios.csv").to_excel(
            writer, sheet_name="Table3_core", index=False
        )
        pd.read_csv(TABLES / "Table4_network_classification.csv").to_excel(
            writer, sheet_name="Table4_network_class", index=False
        )
        pd.read_csv(TABLES / "Supplementary_S1_seed_design.csv").to_excel(
            writer, sheet_name="TableS1_design", index=False
        )
        pd.read_csv(TABLES / "Supplementary_S2_DA_full.csv").to_excel(
            writer, sheet_name="TableS2_DA_full", index=False
        )
        pd.read_csv(TABLES / "Supplementary_S3_DA_uncertainty.csv").to_excel(
            writer, sheet_name="TableS3_DA_uncertainty", index=False
        )
        pd.read_csv(TABLES / "Supplementary_S4_network_baseline.csv").to_excel(
            writer, sheet_name="TableS4_network_baseline", index=False
        )
        pd.read_csv(
            TABLES / "Supplementary_S5_classification_diagnostics.csv"
        ).to_excel(writer, sheet_name="TableS5_class_diagnostics", index=False)
        pd.read_csv(TABLES / "Supplementary_S6_unassigned_sensitivity.csv").to_excel(
            writer, sheet_name="TableS6_unassigned_check", index=False
        )
        pd.read_csv(TABLES / "Table5_truth_and_estimands.csv").to_excel(
            writer, sheet_name="Table5_truth_estimands", index=False
        )

        workbook = writer.book
        for worksheet in workbook.worksheets():
            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(0, 0, worksheet.dim_rowmax, worksheet.dim_colmax)
            worksheet.set_column(0, worksheet.dim_colmax, 18)

    print(OUTPUT)


if __name__ == "__main__":
    main()
