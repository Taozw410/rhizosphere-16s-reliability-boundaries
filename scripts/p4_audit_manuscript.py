#!/usr/bin/env python3
"""Audit manuscript text, references, tables and figure-QA artefacts."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
MS = PAPER / "manuscript"
FIG = PAPER / "figures"

REQUIRED = [
    MS / "Manuscript.md",
    MS / "Figure_legends.md",
    MS / "Abstract_zh.md",
    MS / "Tables.md",
    MS / "References.md",
    MS / "Submission_metadata.md",
    MS / "README.md",
    MS / "Source_Data.xlsx",
    MS / "通用版_修订版.docx",
    MS / "tables/Table1_projects.csv",
    MS / "tables/Table2_DA_key_scenario.csv",
    MS / "tables/Table3_core_scenarios.csv",
    MS / "tables/Table4_network_classification.csv",
    MS / "tables/Table5_truth_and_estimands.csv",
    MS / "tables/Supplementary_S1_seed_design.csv",
    MS / "tables/Supplementary_S2_DA_full.csv",
    MS / "tables/Supplementary_S3_DA_uncertainty.csv",
    MS / "tables/Supplementary_S4_network_baseline.csv",
    MS / "tables/Supplementary_S5_classification_diagnostics.csv",
    MS / "tables/Supplementary_S6_unassigned_sensitivity.csv",
]

INTERNAL_PATTERNS = [
    (r"[A-Za-z]:\\", "absolute Windows path"),
    (r"paper4", "internal project name"),
    (r"\bscripts?[\\/]", "internal script path"),
    (r"\boutputs?[\\/]", "internal output path"),
    (r"\bAGENTS\.md\b", "internal governance file"),
    (r"\bAUTHOR_INPUT_NEEDED\b", "unresolved metadata marker"),
    (r"\b(?:TBD|TODO|FIXME)\b", "unresolved work marker"),
]


def expand_citations(text: str) -> set[int]:
    cited: set[int] = set()
    for group in re.findall(r"\[([0-9,\u2013\- ]+)\]", text):
        for part in group.split(","):
            part = part.strip()
            if not part:
                continue
            if re.fullmatch(r"\d+", part):
                cited.add(int(part))
            else:
                bounds = re.split(r"[\u2013\-]", part)
                if len(bounds) == 2 and all(item.strip().isdigit() for item in bounds):
                    start, end = map(int, bounds)
                    cited.update(range(start, end + 1))
    return cited


def main() -> None:
    findings: list[str] = []
    passes: list[str] = []

    for path in REQUIRED:
        if path.exists():
            passes.append(f"Required file exists: {path.relative_to(MS)}")
        else:
            findings.append(f"Missing required file: {path.relative_to(MS)}")

    source_workbook = MS / "Source_Data.xlsx"
    if source_workbook.exists():
        try:
            with zipfile.ZipFile(source_workbook) as archive:
                if "xl/workbook.xml" in archive.namelist():
                    worksheet_count = sum(
                        name.startswith("xl/worksheets/sheet")
                        for name in archive.namelist()
                    )
                    passes.append(
                        "Source_Data.xlsx has a valid workbook container "
                        f"with {worksheet_count} worksheets."
                    )
                else:
                    findings.append("Source_Data.xlsx does not contain xl/workbook.xml.")
        except zipfile.BadZipFile:
            findings.append("Source_Data.xlsx is not a valid XLSX archive.")

    manuscript = (MS / "Manuscript.md").read_text(encoding="utf-8")
    legends = (MS / "Figure_legends.md").read_text(encoding="utf-8")
    tables = (MS / "Tables.md").read_text(encoding="utf-8")
    references = (MS / "References.md").read_text(encoding="utf-8")

    abstract_match = re.search(
        r"## Abstract\s+(.*?)\*\*Keywords:",
        manuscript,
        flags=re.S,
    )
    if not abstract_match:
        findings.append("Could not locate the English abstract.")
        abstract_words = 0
    else:
        abstract_words = len(re.findall(r"\b[\w–-]+\b", abstract_match.group(1)))
        if 200 <= abstract_words <= 350:
            passes.append(f"English abstract word count is {abstract_words}.")
        else:
            findings.append(
                f"English abstract word count is outside 200–350: {abstract_words}."
            )

    required_sections = [
        "## Abstract",
        "## Introduction",
        "## Results",
        "## Discussion",
        "## Limitations",
        "## Conclusions",
        "## Methods",
        "### Data and code availability",
        "### Ethics declaration",
        "### Author contributions",
        "### Competing interests",
        "### Funding",
    ]
    for section in required_sections:
        if section in manuscript:
            passes.append(f"Required section present: {section}")
        else:
            findings.append(f"Required section missing: {section}")

    citation_numbers = expand_citations(manuscript)
    reference_numbers = {
        int(match.group(1))
        for match in re.finditer(r"^(\d+)\.\s", references, flags=re.M)
    }
    if citation_numbers == reference_numbers:
        passes.append(
            f"Citation-reference set matches exactly (n={len(reference_numbers)})."
        )
    else:
        findings.append(
            "Citation-reference mismatch: "
            f"uncited={sorted(reference_numbers - citation_numbers)}, "
            f"missing={sorted(citation_numbers - reference_numbers)}."
        )

    for figure_number in range(1, 6):
        if re.search(rf"\bFig\.?\s*{figure_number}(?![0-9])", manuscript):
            passes.append(f"Figure {figure_number} is cited in the manuscript.")
        else:
            findings.append(f"Figure {figure_number} is not cited in the manuscript.")
        if f"**Fig. {figure_number} |" in legends:
            passes.append(f"Figure {figure_number} has a legend.")
        else:
            findings.append(f"Figure {figure_number} legend missing.")

    for table_number in range(1, 6):
        if re.search(rf"\bTable\s+{table_number}\b", manuscript):
            passes.append(f"Table {table_number} is cited in the manuscript.")
        else:
            findings.append(f"Table {table_number} is not cited in the manuscript.")

    for filename, text in [
        ("Manuscript.md", manuscript),
        ("Figure_legends.md", legends),
        ("Tables.md", tables),
    ]:
        for pattern, label in INTERNAL_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                findings.append(f"{filename} contains {label}: {pattern}")
    if not any(item.startswith(("Manuscript.md", "Figure_legends.md", "Tables.md"))
               for item in findings):
        passes.append("Submission-facing text contains no known internal-path markers.")

    malformed_table_lines = []
    table_blocks = re.split(r"\n(?=## )", tables)
    for block in table_blocks:
        lines = [line for line in block.splitlines() if line.startswith("|")]
        if not lines:
            continue
        expected = lines[0].count("|")
        for line in lines:
            if line.count("|") != expected:
                malformed_table_lines.append(line[:120])
    if malformed_table_lines:
        findings.append(
            "Markdown table column mismatch in: "
            + " | ".join(malformed_table_lines[:5])
        )
    else:
        passes.append("All Markdown tables have consistent column counts.")

    figure_qa = []
    stems = list(FIG.glob("Fig*_*.alignment.json"))
    for alignment_path in sorted(stems):
        stem = alignment_path.name.removesuffix(".alignment.json")
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
        collision_path = FIG / f"{stem}.collision-audit.json"
        collision = (
            json.loads(collision_path.read_text(encoding="utf-8"))
            if collision_path.exists()
            else None
        )
        collision_verdict = collision["verdict"] if collision else "MISSING"
        figure_qa.append(
            (
                stem,
                alignment.get("verdict", "MISSING"),
                collision_verdict,
            )
        )
        if alignment.get("verdict") != "PASS":
            findings.append(f"{stem}: alignment verdict is not PASS.")
        if collision_verdict != "PASS":
            findings.append(f"{stem}: collision verdict is not PASS.")

    qa_lines = [
        "# Manuscript QA",
        "",
        f"Generated from the current manuscript and figure artefacts.",
        "",
        "## Summary",
        "",
        f"- English abstract words: {abstract_words}",
        f"- Manuscript words: {len(re.findall(r'\b[\w–-]+\b', manuscript))}",
        f"- References: {len(reference_numbers)}",
        f"- Findings: {len(findings)}",
        f"- Verdict: {'PASS' if not findings else 'FIX BEFORE SUBMISSION'}",
        "",
        "## Figure QA",
        "",
        "| Figure | Panel alignment | Rendered collisions |",
        "| --- | --- | --- |",
    ]
    for stem, alignment_verdict, collision_verdict in figure_qa:
        qa_lines.append(
            f"| {stem} | {alignment_verdict} | {collision_verdict} |"
        )
    qa_lines.extend(["", "## Passed checks", ""])
    qa_lines.extend(f"- {item}" for item in passes)
    qa_lines.extend(["", "## Findings", ""])
    if findings:
        qa_lines.extend(f"- FAIL: {item}" for item in findings)
    else:
        qa_lines.append("- None.")
    (MS / "Manuscript_QA.md").write_text("\n".join(qa_lines) + "\n", encoding="utf-8")

    print(f"findings={len(findings)} verdict={'PASS' if not findings else 'FAIL'}")
    for item in findings:
        print(f"FAIL: {item}")


if __name__ == "__main__":
    main()
