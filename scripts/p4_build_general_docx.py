#!/usr/bin/env python3
"""Build the consolidated general-purpose DOCX for Paper 4."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
MS = PAPER / "manuscript"
FIG = PAPER / "figures"
TABLES = MS / "tables"
OUTPUT = MS / "通用版_修订版.docx"

BLACK = RGBColor(0, 0, 0)
HEADER_BLUE = "1F4E79"
ALT_BLUE = "EAF2F8"
WHITE = "FFFFFF"
LIGHT_BORDER = "D9D9D9"

FIGURE_FILES = [
    FIG / "Fig1_benchmark_framework.png",
    FIG / "Fig2_DA_cross_generator.png",
    FIG / "Fig3_core_occupancy.png",
    FIG / "Fig4_network_classification.png",
    FIG / "Fig5_project_external_validation.png",
]

FIGURE_ALT = {
    1: "Four-panel benchmark overview showing the 45-project workflow, sample-size distribution, executed benchmark scale and validation status.",
    2: "Four-panel differential-abundance comparison showing method-by-scenario F1 and false-discovery proportion, the sparse-effect trade-off, and LinDA seed-level replication.",
    3: "Four-panel core-recovery benchmark showing scenario F1, method-level confidence intervals, false-core rate and the depth-confounded precision-recall boundary.",
    4: "Four-panel network and classification benchmark showing synthetic edge recovery, real panel stability, closed-set classification and unseen-genus trade-offs.",
    5: "Four-panel structural transportability summary showing three-panel calibration, repeated project holdout, empirical interval coverage and the evidence decision.",
}


def set_cell_margins(cell, top=70, start=80, bottom=70, end=80) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin_name, value in [
        ("top", top),
        ("start", start),
        ("bottom", bottom),
        ("end", end),
    ]:
        node = tc_mar.find(qn(f"w:{margin_name}"))
        if node is None:
            node = OxmlElement(f"w:{margin_name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = tc_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), fill)


def set_table_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), LIGHT_BORDER)


def repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def set_section_line_numbers(section, restart: str = "continuous") -> None:
    sect_pr = section._sectPr
    existing = sect_pr.find(qn("w:lnNumType"))
    if existing is not None:
        sect_pr.remove(existing)
    line_numbers = OxmlElement("w:lnNumType")
    line_numbers.set(qn("w:countBy"), "1")
    line_numbers.set(qn("w:start"), "1")
    line_numbers.set(qn("w:restart"), restart)
    anchor = sect_pr.find(qn("w:cols"))
    if anchor is None:
        anchor = sect_pr.find(qn("w:docGrid"))
    if anchor is not None:
        anchor.addprevious(line_numbers)
    else:
        sect_pr.append(line_numbers)


def add_page_field(paragraph, instruction: str) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, placeholder, end])


def configure_styles(document: Document) -> None:
    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = BLACK
    normal.paragraph_format.line_spacing = 1.25
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.space_before = Pt(0)

    title = styles["Title"]
    title.font.name = "Arial"
    title._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    title.font.size = Pt(18)
    title.font.bold = True
    title.font.color.rgb = BLACK
    title.paragraph_format.space_before = Pt(24)
    title.paragraph_format.space_after = Pt(10)
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p_pr = title._element.get_or_add_pPr()
    for border in title_p_pr.findall(qn("w:pBdr")):
        title_p_pr.remove(border)

    subtitle = styles["Subtitle"]
    subtitle.font.name = "Arial"
    subtitle._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    subtitle.font.size = Pt(11)
    subtitle.font.color.rgb = BLACK
    subtitle.paragraph_format.space_after = Pt(18)
    subtitle.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for style_name, size, bold, italic, before, after in [
        ("Heading 1", 14, True, False, 16, 7),
        ("Heading 2", 12, True, False, 12, 5),
        ("Heading 3", 10.8, True, True, 10, 4),
    ]:
        style = styles[style_name]
        style.font.name = "Arial"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.bold = bold
        style.font.italic = italic
        style.font.color.rgb = BLACK
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        p_pr = style._element.get_or_add_pPr()
        for border_name in ["pBdr", "bdr"]:
            for node in p_pr.findall(qn(f"w:{border_name}")):
                p_pr.remove(node)

    caption = styles["Caption"]
    caption.font.name = "Arial"
    caption._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    caption.font.size = Pt(8.5)
    caption.font.italic = True
    caption.font.color.rgb = BLACK
    caption.paragraph_format.space_before = Pt(4)
    caption.paragraph_format.space_after = Pt(8)


def add_inline_markdown(paragraph, text: str) -> None:
    tokens = re.split(r"(\*\*.*?\*\*|\*.*?\*)", text)
    for token in tokens:
        if not token:
            continue
        bold = token.startswith("**") and token.endswith("**")
        italic = (
            token.startswith("*")
            and token.endswith("*")
            and not token.startswith("**")
        )
        clean = token[2:-2] if bold else token[1:-1] if italic else token
        run = paragraph.add_run(clean)
        run.bold = bold
        run.italic = italic


def add_heading(document: Document, text: str, level: int) -> None:
    paragraph = document.add_paragraph(style=f"Heading {level}")
    paragraph.add_run(text)


def add_body_paragraph(document: Document, text: str, first_indent: bool = False) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if first_indent:
        paragraph.paragraph_format.first_line_indent = Inches(0.2)
    add_inline_markdown(paragraph, text)


def add_markdown_section(document: Document, markdown: str, skip_title: bool) -> None:
    lines = markdown.splitlines()
    paragraph_buffer: list[str] = []

    def flush() -> None:
        if not paragraph_buffer:
            return
        text = " ".join(item.strip() for item in paragraph_buffer)
        paragraph_buffer.clear()
        if text.startswith("logit(p_ijs)"):
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run(text)
            run.italic = True
        else:
            add_body_paragraph(document, text)

    for index, raw_line in enumerate(lines):
        line = raw_line.rstrip()
        if not line:
            flush()
            continue
        if line.startswith("# "):
            flush()
            if skip_title and index == 0:
                continue
            add_heading(document, line[2:].strip(), 1)
        elif line.startswith("## "):
            flush()
            add_heading(document, line[3:].strip(), 1)
        elif line.startswith("### "):
            flush()
            add_heading(document, line[4:].strip(), 2)
        elif line.startswith("#### "):
            flush()
            add_heading(document, line[5:].strip(), 3)
        else:
            paragraph_buffer.append(line)
    flush()


def add_cover(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(0.9)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)
    section.different_first_page_header_footer = True

    for _ in range(4):
        document.add_paragraph()
    title = document.add_paragraph(style="Title")
    title.add_run(
        "An empirically anchored multi-study rhizosphere 16S rRNA benchmark "
        "reveals method-specific failure boundaries and limits to cross-"
        "project generalization"
    )
    subtitle = document.add_paragraph(style="Subtitle")
    subtitle.add_run(
        "Research manuscript with figures tables and supplementary information"
    )
    meta = document.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = meta.add_run("Revised manuscript | 29 October 2026")
    run.font.size = Pt(9.5)
    run.font.color.rgb = BLACK

    document.add_paragraph()
    scope = document.add_paragraph()
    scope.alignment = WD_ALIGN_PARAGRAPH.CENTER
    scope.paragraph_format.left_indent = Inches(0.55)
    scope.paragraph_format.right_indent = Inches(0.55)
    scope_run = scope.add_run(
        "This study evaluates common rhizosphere 16S analysis methods against "
        "explicit simulation truth, distinguishes method performance from "
        "structural transportability, and reports where the evidence does not "
        "support cross-project generalization."
    )
    scope_run.font.size = Pt(10)

    body_section = document.add_section(WD_SECTION.NEW_PAGE)
    body_section.orientation = WD_ORIENT.PORTRAIT
    body_section.page_width = Inches(8.5)
    body_section.page_height = Inches(11)
    body_section.top_margin = Inches(0.85)
    body_section.bottom_margin = Inches(0.8)
    body_section.left_margin = Inches(0.9)
    body_section.right_margin = Inches(0.9)


def setup_page_footer(document: Document) -> None:
    for index, section in enumerate(document.sections):
        section.footer.is_linked_to_previous = False
        footer = section.footer
        paragraph = footer.paragraphs[0]
        paragraph.clear()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        prefix = paragraph.add_run("Page ")
        prefix.font.size = Pt(8)
        prefix.font.color.rgb = BLACK
        add_page_field(paragraph, "PAGE")
        separator = paragraph.add_run(" of ")
        separator.font.size = Pt(8)
        add_page_field(paragraph, "NUMPAGES")
        for run in paragraph.runs:
            run.font.name = "Arial"
            run.font.size = Pt(8)
            run.font.color.rgb = BLACK
        if index > 0:
            set_section_line_numbers(section)


def add_alt_text(inline_shape, description: str) -> None:
    doc_pr = inline_shape._inline.docPr
    doc_pr.set("descr", description)
    doc_pr.set("title", description[:120])


def add_figure(document: Document, number: int, image_path: Path, legend: str) -> None:
    document.add_page_break()
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    with Image.open(image_path) as image:
        aspect = image.height / image.width
    width = Inches(6.45)
    height = width * aspect
    shape = run.add_picture(str(image_path), width=width, height=height)
    add_alt_text(shape, FIGURE_ALT[number])

    caption = document.add_paragraph(style="Caption")
    caption.alignment = WD_ALIGN_PARAGRAPH.LEFT
    caption.paragraph_format.keep_together = True
    add_inline_markdown(caption, legend)


def add_csv_table(
    document: Document,
    csv_path: Path,
    caption: str,
    note: str,
    font_size: float = 8.2,
    widths: list[float] | None = None,
    header_labels: list[str] | None = None,
) -> None:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    caption_paragraph = document.add_paragraph(style="Caption")
    caption_paragraph.paragraph_format.keep_with_next = True
    add_inline_markdown(caption_paragraph, caption)

    table = document.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)

    for row_index, values in enumerate(rows):
        row = table.rows[row_index]
        prevent_row_split(row)
        for column_index, value in enumerate(values):
            cell = row.cells[column_index]
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if widths:
                cell.width = Inches(widths[column_index])
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT
                if column_index == 0 or row_index == 0
                else WD_ALIGN_PARAGRAPH.CENTER
            )
            display_value = (
                header_labels[column_index]
                if row_index == 0 and header_labels
                else value
            )
            run = paragraph.add_run(display_value)
            run.font.name = "Arial"
            run.font.size = Pt(font_size)
            run.font.color.rgb = BLACK
            if row_index == 0:
                run.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                set_cell_shading(cell, HEADER_BLUE)
            elif row_index % 2 == 0:
                set_cell_shading(cell, ALT_BLUE)
    repeat_header(table.rows[0])
    note_paragraph = document.add_paragraph()
    note_paragraph.paragraph_format.space_before = Pt(4)
    note_paragraph.paragraph_format.space_after = Pt(10)
    run = note_paragraph.add_run(note)
    run.font.size = Pt(8)
    run.italic = True


def add_figures(document: Document) -> None:
    add_heading(document, "Figures", 1)
    add_body_paragraph(
        document,
        "Each figure is reproduced at its designed two-column width. The "
        "legend below each image states sample counts, uncertainty definitions "
        "and the interpretation boundary relevant to that figure.",
    )
    legends = extract_figure_legends()
    for number, image_path in enumerate(FIGURE_FILES, start=1):
        add_figure(document, number, image_path, legends[number])


def extract_figure_legends() -> dict[int, str]:
    text = (MS / "Figure_legends.md").read_text(encoding="utf-8")
    matches = re.findall(
        r"\*\*Fig\.\s*([1-5])\s*\|\s*(.*?)\*\*\s*(.*?)(?=\n\n\*\*Fig\.|\Z)",
        text,
        flags=re.S,
    )
    legends: dict[int, str] = {}
    for number, title, body in matches:
        clean_title = re.sub(r"\s+", " ", title).strip()
        clean_body = re.sub(r"\s*\n\s*", " ", body).strip()
        legends[int(number)] = f"**Fig. {number} | {clean_title}.** {clean_body}"
    return legends


def add_tables(document: Document) -> None:
    # Table 1 is wide and benefits from a landscape page.
    landscape = document.add_section(WD_SECTION.NEW_PAGE)
    landscape.orientation = WD_ORIENT.LANDSCAPE
    landscape.page_width = Inches(11)
    landscape.page_height = Inches(8.5)
    landscape.top_margin = Inches(0.55)
    landscape.bottom_margin = Inches(0.55)
    landscape.left_margin = Inches(0.55)
    landscape.right_margin = Inches(0.55)
    add_heading(document, "Main tables", 1)
    add_csv_table(
        document,
        TABLES / "Table1_projects.csv",
        "**Table 1 | Observed structure of the 45 rhizosphere 16S projects.**",
        "Depth is the median or mean number of retained bacterial reads. ASV "
        "length is the modal exact-sequence length. Annotation percentages "
        "are read-weighted assignments at the stated rank.",
        font_size=6.8,
        widths=[1.35, 0.55, 0.55, 0.88, 0.88, 0.78, 0.72, 0.78, 0.72, 0.72],
        header_labels=[
            "Project",
            "Crop",
            "Samples",
            "Median depth",
            "Mean depth",
            "ASV length (bp)",
            "Median ASVs",
            "Unassigned (%)",
            "Genus (%)",
            "Family (%)",
        ],
    )

    portrait = document.add_section(WD_SECTION.NEW_PAGE)
    portrait.orientation = WD_ORIENT.PORTRAIT
    portrait.page_width = Inches(8.5)
    portrait.page_height = Inches(11)
    portrait.top_margin = Inches(0.85)
    portrait.bottom_margin = Inches(0.8)
    portrait.left_margin = Inches(0.9)
    portrait.right_margin = Inches(0.9)

    add_csv_table(
        document,
        TABLES / "Table2_DA_key_scenario.csv",
        "**Table 2 | Differential abundance in the strong sparse scenario.**",
        "Generator 1 uses 20 datasets per method; generator 2 uses five. The "
        "generators are not pooled because the second family is a correlated "
        "perturbation of the same template.",
        font_size=8.2,
        widths=[1.0, 1.15, 0.65, 0.72, 0.72, 0.72],
    )

    document.add_page_break()
    add_csv_table(
        document,
        TABLES / "Table3_core_scenarios.csv",
        "**Table 3 | Recovery of latent core features.**",
        "F1, recall and precision are undefined in the null scenario because "
        "there are no true core features; the false-core rate is therefore the "
        "primary null endpoint.",
        font_size=7.4,
        widths=[1.05, 1.28, 0.58, 0.54, 0.54, 0.54, 0.62, 0.58],
    )

    document.add_page_break()
    add_csv_table(
        document,
        TABLES / "Table4_network_classification.csv",
        "**Table 4 | Network inference and sequence-classification benchmarks.**",
        "For networks, Primary metric 1 is mean synthetic edge AUPRC and "
        "Primary metric 2 is mean hub-degree Spearman correlation; External "
        "metrics are cross-panel edge Jaccard and hub-degree Spearman. For "
        "classification, Primary metrics are closed-set accuracy and macro F1; "
        "external metrics are unseen-genus rejection and known-taxon retention.",
        font_size=7.8,
        widths=[0.85, 1.05, 0.83, 0.83, 0.83, 0.83],
    )

    document.add_page_break()
    add_csv_table(
        document,
        TABLES / "Table5_truth_and_estimands.csv",
        "**Table 5 | Truth definitions, method targets and evaluation boundaries.**",
        "The four modules use related but non-identical estimands. Classification "
        "is a complementary reference-set analysis rather than an eight-summary "
        "project-structure simulation module.",
        font_size=7.1,
        widths=[0.9, 1.4, 1.45, 1.2, 1.35],
    )

    document.add_page_break()
    add_heading(document, "Supplementary tables", 1)
    add_csv_table(
        document,
        TABLES / "Supplementary_S1_seed_design.csv",
        "**Supplementary Table S1 | Frozen benchmark design and seed blocks.**",
        "Seeds and analysis units were fixed before the formal benchmark. "
        "Primary and extension blocks remain separate.",
        font_size=7.5,
        widths=[1.3, 1.15, 0.8, 0.8, 0.65, 1.0],
    )
    document.add_page_break()
    add_csv_table(
        document,
        TABLES / "Supplementary_S2_DA_full.csv",
        "**Supplementary Table S2 | Differential abundance across all generators and scenarios.**",
        "Mean FDP is undefined when a method made no discoveries in a replicate "
        "family. Absolute levels are not pooled across generators.",
        font_size=7.2,
        widths=[0.9, 1.45, 1.05, 0.55, 0.58, 0.58, 0.58],
    )
    document.add_page_break()
    add_csv_table(
        document,
        TABLES / "Supplementary_S3_DA_key_uncertainty.csv",
        "**Supplementary Table S3 | Differential-abundance uncertainty and pooled FDP in the strong sparse scenario.**",
        "Intervals are 95% percentile bootstrap intervals from 5,000 resamples "
        "of datasets. Pooled FDP is total false discoveries divided by total "
        "discoveries and is distinct from the mean of per-dataset FDP.",
        font_size=7.1,
        widths=[
            0.9,
            0.86,
            0.42,
            0.64,
            1.05,
            0.55,
            0.95,
            0.95,
        ],
    )
    document.add_page_break()
    add_csv_table(
        document,
        TABLES / "Supplementary_S4_network_baseline_key.csv",
        "**Supplementary Table S4 | Network true-edge prevalence and random AUPRC baselines.**",
        "The random baseline equals true-edge prevalence among 780 candidate "
        "undirected edges. Conditional-dependence and marginal-correlation "
        "targets are not identical.",
        font_size=7.1,
        widths=[0.95, 0.92, 0.72, 0.82, 0.82, 0.82, 0.82, 0.65],
    )
    document.add_page_break()
    add_csv_table(
        document,
        TABLES / "Supplementary_S5_classification_diagnostics_key.csv",
        "**Supplementary Table S5 | Classification performance and training diagnostics.**",
        "The balanced majority-class baseline is 0.05 accuracy. Neural models "
        "used four epochs without class weighting and should not be interpreted "
        "as modern deep-learning benchmarks.",
        font_size=7.5,
        widths=[1.15, 0.85, 0.75, 0.75, 0.72, 0.88, 0.65],
    )
    document.add_page_break()
    add_csv_table(
        document,
        TABLES / "Supplementary_S6_unassigned_sensitivity.csv",
        "**Supplementary Table S6 | Structural sensitivity after removing the unassigned bin.**",
        "All eight operational calibration criteria remained within tolerance "
        "after recomputing observed and simulated summaries without the "
        "unassigned bin.",
        font_size=8.0,
        widths=[1.55, 1.45, 0.8, 0.7],
    )


def add_references(document: Document) -> None:
    document.add_page_break()
    add_heading(document, "References", 1)
    text = (MS / "References.md").read_text(encoding="utf-8")
    for line in text.splitlines():
        if not re.match(r"^\d+\.\s", line):
            continue
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.25)
        paragraph.paragraph_format.first_line_indent = Inches(-0.25)
        paragraph.paragraph_format.space_after = Pt(4)
        add_inline_markdown(paragraph, line)
        for run in paragraph.runs:
            run.font.size = Pt(8.5)


def main() -> None:
    document = Document()
    configure_styles(document)
    document.core_properties.title = (
        "An empirically anchored multi-study rhizosphere 16S rRNA benchmark"
    )
    document.core_properties.subject = (
        "Consolidated manuscript with integrated figures tables and references"
    )
    document.core_properties.author = ""
    document.core_properties.last_modified_by = ""

    add_cover(document)

    add_heading(document, "Abstract", 1)
    manuscript = (MS / "Manuscript.md").read_text(encoding="utf-8")
    abstract = re.search(
        r"## Abstract\s+(.*?)\*\*Keywords:",
        manuscript,
        flags=re.S,
    )
    if abstract:
        for paragraph in re.split(r"\n\s*\n", abstract.group(1).strip()):
            add_body_paragraph(document, paragraph)
    keywords_match = re.search(r"\*\*Keywords:\*\*\s*(.+)", manuscript)
    if keywords_match:
        paragraph = document.add_paragraph()
        lead = paragraph.add_run("Keywords: ")
        lead.bold = True
        paragraph.add_run(keywords_match.group(1).strip())

    zh_text = (MS / "Abstract_zh.md").read_text(encoding="utf-8")
    add_heading(document, "中文摘要", 1)
    for block in re.split(r"\n\s*\n", zh_text):
        cleaned = block.strip()
        if not cleaned or cleaned == "# 中文摘要":
            continue
        add_body_paragraph(document, cleaned)

    document.add_page_break()
    body_start = manuscript.find("## Introduction")
    add_markdown_section(
        document,
        manuscript[body_start:],
        skip_title=False,
    )

    add_figures(document)
    add_tables(document)
    add_references(document)
    setup_page_footer(document)

    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
