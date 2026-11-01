#!/usr/bin/env python3
"""Generate Paper 4 manuscript figures and source-data tables."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import ScaledTranslation

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
OUT = PAPER / "figures"
SOURCE = OUT / "source_data"
PAPER.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)

SKILL_SCRIPTS = Path(r"C:\Users\hthome\.codex\skills\nature-figure\scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


METHOD_LABELS = {
    "aldex2": "ALDEx2",
    "ancombc2": "ANCOM-BC2",
    "linda": "LinDA",
    "maaslin2": "MaAsLin2",
}
METHOD_ORDER = ["ALDEx2", "ANCOM-BC2", "LinDA", "MaAsLin2"]
METHOD_COLORS = {
    "ALDEx2": "#767676",
    "ANCOM-BC2": "#0F4D92",
    "LinDA": "#B64342",
    "MaAsLin2": "#42949E",
}
CROP_COLORS = {
    "maize": "#E69F00",
    "rice": "#0072B2",
    "wheat": "#009E73",
}

mpl.rcParams.update(
    {
        "backend": "Agg",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def add_panel_label(ax, label: str) -> None:
    offset = ScaledTranslation(
        -8 / 72,
        4 / 72,
        ax.figure.dpi_scale_trans,
    )
    ax.text(
        0,
        1,
        label,
        transform=ax.transAxes + offset,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_figure(
    figure: plt.Figure,
    stem: str,
    axes: list[plt.Axes],
    panel_ids: list[str],
    row_groups: list[list[str]],
    column_groups: list[list[str]],
) -> None:
    figure.tight_layout(pad=0.8)
    require_matplotlib_panel_alignment(
        figure,
        axes=axes,
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        json_out=str(OUT / f"{stem}.alignment.json"),
        overlay_svg=str(OUT / f"{stem}.alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    figure.savefig(OUT / f"{stem}.svg")
    figure.savefig(OUT / f"{stem}.pdf")
    figure.savefig(OUT / f"{stem}.png", dpi=300)
    figure.savefig(OUT / f"{stem}.tiff", dpi=600)
    plt.close(figure)


def style_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    cmap: str,
    value_format: str = "{:.2f}",
    vmin: float | None = None,
    vmax: float | None = None,
    rotate_labels: bool = True,
) -> None:
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(np.arange(len(column_labels)))
    if rotate_labels:
        ax.set_xticklabels(
            column_labels,
            rotation=28,
            ha="right",
            rotation_mode="anchor",
        )
    else:
        ax.set_xticklabels(column_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, loc="left", pad=8)
    ax.set_frame_on(False)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                rgba = image.cmap(
                    image.norm(value)
                )
                luminance = (
                    0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                )
                ax.text(
                    column,
                    row,
                    value_format.format(value),
                    ha="center",
                    va="center",
                    color="white" if luminance < 0.55 else "#222222",
                    fontsize=6.2,
                )
    # Keep the heatmap plot-area rectangle identical to the axes below it.
    # A figure-level colorbar steals width from the parent axes; an inset
    # colorbar lives inside the allocated panel envelope and preserves the
    # strict two-column alignment contract.
    colorbar_ax = ax.inset_axes([1.025, 0.0, 0.032, 1.0])
    colorbar = ax.figure.colorbar(image, cax=colorbar_ax)
    colorbar.ax.tick_params(labelsize=6)


def figure1() -> None:
    sample_meta = pd.read_csv(ROOT / "out_analysis/sample_meta.csv")
    project_counts = (
        sample_meta.groupby(["project", "crop"])
        .size()
        .rename("samples")
        .reset_index()
        .sort_values(["crop", "project"])
    )
    project_counts.to_csv(SOURCE / "Fig1_project_sample_counts.csv", index=False)

    benchmark_scale = pd.DataFrame(
        [
            ("DA generator 1", 100),
            ("DA generator 2", 25),
            ("Core/occupancy", 100),
            ("Network", 40),
            ("Classification seeds", 5),
        ],
        columns=["benchmark", "datasets"],
    )
    benchmark_scale.to_csv(SOURCE / "Fig1_benchmark_scale.csv", index=False)

    panel_validation = pd.read_csv(
        PAPER / "outputs/project_panel_validation/panel_calibration_summary.csv"
    )
    holdout = pd.read_csv(
        PAPER / "outputs/real_project_holdout_cv/holdout_distance_summary.csv"
    )
    validation = pd.DataFrame(
        [
            ("No-effect DA", 1.0),
            ("Core structural", 1.0),
            (
                "Three-panel structure",
                float((panel_validation["mean_n_pass"] / 8).mean()),
            ),
            (
                "Held-out structure",
                float((holdout["mean_n_pass"] / 8).mean()),
            ),
        ],
        columns=["check", "pass_fraction"],
    )
    validation.to_csv(SOURCE / "Fig1_validation_summary.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    ax = axes[0, 0]
    ax.axis("off")
    boxes = [
        (0.03, 0.74, 0.28, 0.16, "45 public\nrhizosphere\nprojects"),
        (0.37, 0.74, 0.26, 0.16, "8-summary\ncalibration"),
        (0.69, 0.74, 0.28, 0.16, "Project-level\nsimulation"),
        (0.03, 0.36, 0.28, 0.16, "DA\nbenchmark"),
        (0.37, 0.36, 0.26, 0.16, "Core\nbenchmark"),
        (0.69, 0.36, 0.28, 0.16, "Network and\nclassification"),
        (0.20, 0.05, 0.60, 0.15, "Repeated project-panel\nexternal validation"),
    ]
    for x, y, width, height, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.018",
            facecolor="#F2F5F9" if y > 0.5 else "#E8F1F7",
            edgecolor="#4D4D4D",
            linewidth=0.8,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=6.8,
        )
    arrows = [
        ((0.31, 0.82), (0.37, 0.82)),
        ((0.63, 0.82), (0.69, 0.82)),
        ((0.82, 0.74), (0.82, 0.52)),
        ((0.17, 0.74), (0.17, 0.52)),
        ((0.50, 0.36), (0.50, 0.20)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.8,
                color="#4D4D4D",
            )
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Benchmark architecture", loc="left", pad=8)
    add_panel_label(ax, "a")

    ax = axes[0, 1]
    positions = np.arange(len(project_counts))
    ax.bar(
        positions,
        project_counts["samples"],
        color=[CROP_COLORS[crop] for crop in project_counts["crop"]],
        width=0.82,
        edgecolor="white",
        linewidth=0.25,
    )
    ax.set_ylabel("Samples per project")
    ax.set_xlabel("45 projects ordered by crop")
    ax.set_xticks([])
    ax.set_ylim(0, 40)
    ax.set_title("Observed project sample sizes", loc="left", pad=8)
    for crop, color in CROP_COLORS.items():
        selected = project_counts["crop"].eq(crop)
        count = int(selected.sum())
        center = float(positions[selected].mean())
        display = {"maize": "Maize", "rice": "Rice", "wheat": "Wheat"}[crop]
        ax.text(
            center,
            37.2,
            f"{display} (n={count})",
            color=color,
            ha="center",
            va="center",
            fontsize=6.2,
            fontweight="bold",
        )
    add_panel_label(ax, "b")

    ax = axes[1, 0]
    y_positions = np.arange(len(benchmark_scale))
    ax.barh(
        y_positions,
        benchmark_scale["datasets"],
        color=["#0F4D92", "#3775BA", "#42949E", "#9A4D8E", "#E69F00"],
        edgecolor="white",
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(benchmark_scale["benchmark"])
    ax.invert_yaxis()
    ax.set_xlabel("Independent datasets or seed splits")
    for y, value in zip(y_positions, benchmark_scale["datasets"]):
        if value >= 50:
            ax.text(
                value - 2,
                y,
                str(value),
                va="center",
                ha="right",
                color="white",
                fontsize=6.2,
            )
        else:
            ax.text(value + 1.5, y, str(value), va="center", fontsize=6.2)
    ax.set_title("Executed benchmark scale", loc="left", pad=8)
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    colors = ["#4D4D4D", "#42949E", "#3775BA", "#B64342"]
    bars = ax.barh(
        validation["check"],
        validation["pass_fraction"],
        color=colors,
        edgecolor="white",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of checks passed")
    for bar, value in zip(bars, validation["pass_fraction"]):
        if value >= 0.85:
            ax.text(
                value - 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="right",
                color="white",
                fontsize=6.2,
            )
        else:
            ax.text(
                value + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                fontsize=6.2,
            )
    ax.set_title("Calibration and structural transportability", loc="left", pad=8)
    add_panel_label(ax, "d")
    save_figure(
        figure,
        "Fig1_benchmark_framework",
        list(axes.ravel()),
        ["a", "b", "c", "d"],
        [["a", "b"], ["c", "d"]],
        [["a", "c"], ["b", "d"]],
    )


def figure2() -> None:
    primary = pd.read_csv(PAPER / "outputs/formal_grid/formal_metrics_all.csv")
    secondary = pd.read_csv(
        PAPER / "outputs/formal_grid_generator2/results/formal_metrics_all.csv"
    )
    scenario_order = [
        "no_effect",
        "sparse20_fc1_19",
        "sparse20_fc1_41",
        "sparse20_fc2",
        "dense40_fc1_41",
    ]
    scenario_labels = [
        "Null",
        "Sparse 0.25",
        "Sparse 0.50",
        "Sparse 1.00",
        "Dense 0.50",
    ]
    method_order_ids = ["aldex2", "ancombc2", "linda", "maaslin2"]

    def summarize(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["method_label"] = frame["method"].map(METHOD_LABELS)
        return (
            frame.groupby(["scenario", "method_label"])
            .agg(
                seeds=("seed", "nunique"),
                fdp=("fdp_bh", "mean"),
                power=("power_bh", "mean"),
                f1=("f1_bh", "mean"),
            )
            .reset_index()
        )

    primary_summary = summarize(primary)
    secondary_summary = summarize(secondary)
    pd.concat(
        [
            primary_summary.assign(generator="Generator 1"),
            secondary_summary.assign(generator="Generator 2"),
        ],
        ignore_index=True,
    ).to_csv(SOURCE / "Fig2_DA_summary.csv", index=False)

    def matrix(frame, metric):
        pivot = frame.pivot(index="method_label", columns="scenario", values=metric)
        pivot = pivot.reindex(index=METHOD_ORDER, columns=scenario_order)
        return pivot.to_numpy(dtype=float)

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    style_heatmap(
        axes[0, 0],
        matrix(primary_summary, "f1"),
        METHOD_ORDER,
        scenario_labels,
        "Generator 1: F1",
        "Blues",
        vmin=0,
        vmax=1,
    )
    add_panel_label(axes[0, 0], "a")
    style_heatmap(
        axes[0, 1],
        matrix(primary_summary, "fdp"),
        METHOD_ORDER,
        scenario_labels,
        "Generator 1: realized FDP",
        "OrRd",
        vmin=0,
        vmax=0.5,
    )
    add_panel_label(axes[0, 1], "b")

    ax = axes[1, 0]
    markers = {"Generator 1": "o", "Generator 2": "s"}
    for generator, summary, offset in [
        ("Generator 1", primary_summary, -0.012),
        ("Generator 2", secondary_summary, 0.012),
    ]:
        rows = summary[summary["scenario"] == "sparse20_fc2"]
        for method in METHOD_ORDER:
            row = rows[rows["method_label"] == method].iloc[0]
            ax.scatter(
                row["fdp"],
                row["power"],
                s=35 + 120 * row["f1"],
                color=METHOD_COLORS[method],
                marker=markers[generator],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.9,
            )
    ax.axvline(0.1, color="#767676", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean realized FDP")
    ax.set_ylabel("Mean power")
    ax.set_xlim(-0.02, 0.42)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Strong sparse effect: markers show generators", loc="left", pad=8)
    method_handles = [
        plt.Line2D(
            [],
            [],
            color=METHOD_COLORS[method],
            marker="o",
            linestyle="None",
            markersize=5,
            label=method,
        )
        for method in METHOD_ORDER
    ]
    generator_handles = [
        plt.Line2D(
            [],
            [],
            color="#4D4D4D",
            marker=marker,
            linestyle="None",
            markersize=5,
            label=generator,
        )
        for generator, marker in markers.items()
    ]
    ax.legend(
        handles=method_handles + generator_handles,
        loc="upper right",
        ncol=2,
        frameon=False,
        fontsize=6,
    )
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    linda = primary[
        primary["method"].eq("linda")
        & primary["scenario"].eq("sparse20_fc2")
    ]
    linda[
        ["stage", "seed", "fdp_bh", "power_bh", "f1_bh"]
    ].to_csv(SOURCE / "Fig2_LinDA_fdp_by_seed.csv", index=False)
    stages = ["primary", "extension"]
    data = [
        np.asarray(
            linda.loc[linda["stage"].eq(stage), "fdp_bh"],
            dtype=float,
        )[
            np.isfinite(
                np.asarray(
                    linda.loc[linda["stage"].eq(stage), "fdp_bh"],
                    dtype=float,
                )
            )
        ]
        for stage in stages
    ]
    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "black", "linewidth": 1},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#F0C0CC")
        patch.set_edgecolor("#B64342")
    for index, values in enumerate(data, start=1):
        jitter = np.linspace(-0.11, 0.11, len(values))
        ax.scatter(
            index + jitter,
            values,
            color="#B64342",
            s=16,
            zorder=3,
        )
    ax.axhline(0.1, color="#767676", linestyle="--", linewidth=0.8)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Primary\nn=10", "Extension\nn=10"])
    ax.set_ylabel("LinDA realized FDP")
    ax.set_title("LinDA FDP replicates across seed stages", loc="left", pad=8)
    add_panel_label(ax, "d")
    save_figure(
        figure,
        "Fig2_DA_cross_generator",
        list(axes.ravel()),
        ["a", "b", "c", "d"],
        [["a", "b"], ["c", "d"]],
        [["a", "c"], ["b", "d"]],
    )


def figure3() -> None:
    metrics = pd.read_csv(
        PAPER / "outputs/core_formal_grid/results/core_metrics_all.csv"
    )
    method_map = {
        "sample_prevalence": "Sample prevalence",
        "project_balanced_prevalence": "Project-balanced",
        "beta_binomial_eb": "Beta-binomial EB",
        "hierarchical_occupancy": "Hierarchical occupancy",
    }
    method_order = [
        "Sample prevalence",
        "Project-balanced",
        "Beta-binomial EB",
        "Hierarchical occupancy",
    ]
    scenario_order = [
        "core_null",
        "core20_balanced",
        "core20_lowabundance",
        "core20_projectshift",
        "core20_depthconfound",
    ]
    scenario_labels = [
        "Null",
        "Balanced",
        "Low\nabundance",
        "Project\nshift",
        "Depth\nconfounded",
    ]
    metrics["method_label"] = metrics["method_id"].map(method_map)
    summary = (
        metrics.groupby(["scenario", "method_label"])
        .agg(
            seeds=("seed", "nunique"),
            f1=("f1", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            fcr=("false_core_rate", "mean"),
            brier=("brier", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(SOURCE / "Fig3_core_summary.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.7))
    f1_matrix = (
        summary.pivot(index="method_label", columns="scenario", values="f1")
        .reindex(index=method_order, columns=scenario_order)
        .to_numpy(dtype=float)
    )
    style_heatmap(
        axes[0, 0],
        f1_matrix,
        method_order,
        scenario_labels,
        "Core F1",
        "Blues",
        vmin=0,
        vmax=1,
        rotate_labels=False,
    )
    add_panel_label(axes[0, 0], "a")

    ax = axes[0, 1]
    method_stats = pd.read_csv(
        SOURCE / "Fig3_method_bootstrap.csv"
    ).set_index("method_label")
    method_stats = method_stats.rename(columns={"f1_mean": "f1"}).reindex(method_order)
    y = np.arange(len(method_stats))
    ax.barh(
        y,
        method_stats["f1"],
        color=["#E69F00", "#0F4D92", "#767676", "#B64342"],
        edgecolor="white",
        xerr=[
            method_stats["f1"] - method_stats["f1_ci_low"],
            method_stats["f1_ci_high"] - method_stats["f1"],
        ],
        error_kw={"elinewidth": 0.7, "capsize": 1.7},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(method_stats.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Mean F1 across non-null scenarios")
    for yi, row in zip(y, method_stats.itertuples()):
        ax.text(
            row.f1_ci_high + 0.012,
            yi,
            f"{row.f1:.2f}",
            va="center",
            fontsize=6.2,
        )
    ax.set_title("Method-level core performance", loc="left", pad=8)
    add_panel_label(ax, "b")

    ax = axes[1, 0]
    fcr_matrix = (
        summary.pivot(index="method_label", columns="scenario", values="fcr")
        .reindex(index=method_order, columns=scenario_order)
        .to_numpy(dtype=float)
    )
    style_heatmap(
        ax,
        fcr_matrix,
        method_order,
        scenario_labels,
        "False-core rate",
        "OrRd",
        vmin=0,
        vmax=0.1,
        rotate_labels=False,
    )
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    depth = summary[summary["scenario"] == "core20_depthconfound"]
    method_colors = {
        "Sample prevalence": "#E69F00",
        "Project-balanced": "#0F4D92",
        "Beta-binomial EB": "#767676",
        "Hierarchical occupancy": "#B64342",
    }
    for method in method_order:
        row = depth[depth["method_label"] == method].iloc[0]
        ax.scatter(
            row["recall"],
            row["precision"],
            s=55,
            color=method_colors[method],
            edgecolor="black",
            linewidth=0.5,
            label=method,
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Depth-confounded failure boundary", loc="left", pad=8)
    ax.legend(loc="upper left", frameon=False, fontsize=5.8)
    add_panel_label(ax, "d")
    save_figure(
        figure,
        "Fig3_core_occupancy",
        list(axes.ravel()),
        ["a", "b", "c", "d"],
        [["a", "b"], ["c", "d"]],
        [["a", "c"], ["b", "d"]],
    )


def figure4() -> None:
    network = pd.read_csv(
        PAPER / "outputs/network_benchmark/network_metrics_by_seed.csv"
    )
    network_summary = pd.read_csv(
        PAPER / "outputs/network_benchmark/network_summary.csv"
    )
    network_stability = pd.read_csv(
        PAPER / "outputs/real_network_stability/real_network_stability_summary.csv"
    )
    classification = pd.read_csv(
        PAPER / "outputs/classification_benchmark/classification_metrics_by_seed.csv"
    )
    classification_summary = pd.read_csv(
        PAPER / "outputs/classification_benchmark/classification_summary.csv"
    )
    network_summary.to_csv(SOURCE / "Fig4_network_summary.csv", index=False)
    network_stability.to_csv(
        SOURCE / "Fig4_real_network_stability.csv",
        index=False,
    )
    classification_summary.to_csv(
        SOURCE / "Fig4_classification_summary.csv",
        index=False,
    )

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    network_scenarios = [
        "network_sparse",
        "network_dense",
        "network_projectshift",
    ]
    network_labels = ["Sparse", "Dense", "Project-shift"]
    methods = ["clr_spearman", "clr_pearson", "sparcc_like", "graphical_lasso"]
    method_labels = ["CLR Spearman", "CLR Pearson", "SparCC-like", "Graphical lasso"]
    colors = ["#0F4D92", "#42949E", "#E69F00", "#767676"]
    x = np.arange(len(network_scenarios))
    width = 0.18
    for index, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        values = []
        errors = []
        for scenario in network_scenarios:
            rows = network[
                network["stage"].eq("validation")
                & network["scenario"].eq(scenario)
                & network["method_id"].eq(method)
            ]
            values.append(rows["auprc"].mean())
            errors.append(rows["auprc"].std(ddof=1))
        axes[0, 0].bar(
            x + (index - 1.5) * width,
            values,
            width=width,
            color=color,
            yerr=errors,
            error_kw={"elinewidth": 0.7, "capsize": 1.5},
            label=label,
        )
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(network_labels)
    axes[0, 0].set_ylabel("Edge AUPRC")
    axes[0, 0].set_ylim(0, 0.19)
    axes[0, 0].set_title("Synthetic direct-edge recovery", loc="left", pad=8)
    axes[0, 0].legend(
        ncol=2,
        loc="upper right",
        handletextpad=1.8,
        columnspacing=1.6,
    )
    add_panel_label(axes[0, 0], "a")

    ax = axes[0, 1]
    for _, row in network_stability.iterrows():
        color = colors[methods.index(row["method_id"])]
        is_sparcc = row["method_id"] == "sparcc_like"
        ax.scatter(
            row["edge_jaccard_mean"],
            row["hub_spearman_mean"],
            s=52,
            color=color,
            edgecolor="black",
            linewidth=0.45,
        )
        ax.annotate(
            method_labels[methods.index(row["method_id"])],
            (row["edge_jaccard_mean"], row["hub_spearman_mean"]),
            xytext=(-5, -12) if is_sparcc else (4, 4),
            textcoords="offset points",
            fontsize=6,
            ha="right" if is_sparcc else "left",
            va="top" if is_sparcc else "baseline",
        )
    ax.set_xlabel("Cross-panel edge Jaccard")
    ax.set_ylabel("Hub-degree Spearman")
    ax.set_title("Real three-panel stability", loc="left", pad=8)
    add_panel_label(ax, "b")

    ax = axes[1, 0]
    classification_methods = [
        "kmer_knn",
        "multinomial_nb",
        "linear_sgd",
        "cnn",
        "kmer_transformer",
    ]
    classification_labels = [
        "k-mer KNN",
        "Multinomial NB",
        "Linear SGD",
        "CNN",
        "k-mer Transformer",
    ]
    classification_tick_labels = [
        "k-mer\nKNN",
        "Multinomial\nNB",
        "Linear\nSGD",
        "CNN",
        "k-mer\nTransformer",
    ]
    colors_class = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#767676"]
    closed = classification_summary[classification_summary["scenario"].eq("closed_set")]
    closed_seed = classification[classification["scenario"].eq("closed_set")]
    x = np.arange(len(classification_methods))
    width = 0.34
    accuracy = [
        float(closed[closed["method_id"].eq(method)]["accuracy_mean"].iloc[0])
        for method in classification_methods
    ]
    f1 = [
        float(closed[closed["method_id"].eq(method)]["macro_f1_mean"].iloc[0])
        for method in classification_methods
    ]
    accuracy_sd = [
        float(
            closed_seed[
                closed_seed["method_id"].eq(method)
            ]["accuracy"].std(ddof=1)
        )
        for method in classification_methods
    ]
    f1_sd = [
        float(
            closed_seed[
                closed_seed["method_id"].eq(method)
            ]["macro_f1"].std(ddof=1)
        )
        for method in classification_methods
    ]
    closed_export = (
        closed_seed.groupby("method_id")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", lambda x: x.std(ddof=1)),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", lambda x: x.std(ddof=1)),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    closed_export.to_csv(SOURCE / "Fig4_classification_closed_set.csv", index=False)
    ax.bar(
        x - width / 2,
        accuracy,
        width,
        yerr=accuracy_sd,
        error_kw={"elinewidth": 0.65, "capsize": 1.4},
        color="#B4C0E4",
        edgecolor="#484878",
        label="Accuracy",
    )
    ax.bar(
        x + width / 2,
        f1,
        width,
        yerr=f1_sd,
        error_kw={"elinewidth": 0.65, "capsize": 1.4},
        color="#F0C0CC",
        edgecolor="#B64342",
        label="Macro F1",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(classification_tick_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Closed-set genus classification", loc="left", pad=8)
    ax.legend()
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    unseen = classification_summary[classification_summary["scenario"].eq("unseen_genus")]
    unseen_seed = classification[classification["scenario"].eq("unseen_genus")]
    unseen_export = []
    for method, label, color in zip(
        classification_methods,
        classification_labels,
        colors_class,
    ):
        row = unseen[unseen["method_id"].eq(method)].iloc[0]
        seed_rows = unseen_seed[unseen_seed["method_id"].eq(method)]
        unknown_sd = float(seed_rows["unknown_recall"].std(ddof=1))
        retention_sd = float(seed_rows["known_retention"].std(ddof=1))
        unseen_export.append(
            {
                "method_id": method,
                "unknown_recall_mean": row["unknown_recall_mean"],
                "unknown_recall_sd": unknown_sd,
                "known_retention_mean": row["known_retention_mean"],
                "known_retention_sd": retention_sd,
                "seeds": int(seed_rows["seed"].nunique()),
            }
        )
        unknown_mean = float(row["unknown_recall_mean"])
        retention_mean = float(row["known_retention_mean"])
        unknown_low = max(0.0, unknown_mean - unknown_sd)
        unknown_high = min(0.65, unknown_mean + unknown_sd)
        retention_low = max(0.8, retention_mean - retention_sd)
        retention_high = min(1.01, retention_mean + retention_sd)
        ax.errorbar(
            unknown_mean,
            retention_mean,
            xerr=[[unknown_mean - unknown_low], [unknown_high - unknown_mean]],
            yerr=[
                [retention_mean - retention_low],
                [retention_high - retention_mean],
            ],
            color=color,
            marker="o",
            markersize=5.5,
            markeredgecolor="black",
            markeredgewidth=0.45,
            elinewidth=0.65,
            capsize=1.5,
            label=label,
        )
    pd.DataFrame(unseen_export).to_csv(
        SOURCE / "Fig4_classification_unseen.csv",
        index=False,
    )
    ax.set_xlabel("Unseen-genus rejection")
    ax.set_ylabel("Known-taxon retention")
    ax.set_xlim(0, 0.65)
    ax.set_ylim(0.8, 1.01)
    ax.set_title("Reference-shift trade-off", loc="left", pad=8)
    ax.legend(loc="lower left", frameon=False, fontsize=5.6)
    add_panel_label(ax, "d")
    save_figure(
        figure,
        "Fig4_network_classification",
        list(axes.ravel()),
        ["a", "b", "c", "d"],
        [["a", "b"], ["c", "d"]],
        [["a", "c"], ["b", "d"]],
    )


def figure5() -> None:
    panel = pd.read_csv(
        PAPER / "outputs/project_panel_validation/panel_calibration_summary.csv"
    )
    holdout_fold = pd.read_csv(
        PAPER / "outputs/real_project_holdout_cv/holdout_distances_by_fold.csv"
    )
    coverage = pd.read_csv(
        PAPER / "outputs/real_project_holdout_cv/holdout_coverage_summary.csv"
    )
    panel.to_csv(SOURCE / "Fig5_project_panel_summary.csv", index=False)
    holdout_fold.to_csv(SOURCE / "Fig5_holdout_by_fold.csv", index=False)
    coverage.to_csv(SOURCE / "Fig5_holdout_coverage.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    ax = axes[0, 0]
    panels = ["panel_1", "panel_2", "panel_3"]
    generators_panel = ["template_multinomial", "correlated_perturbation"]
    generator_panel_labels = ["Template", "Correlated perturbation"]
    generator_panel_colors = ["#B4C0E4", "#F0C0CC"]
    x = np.arange(len(panels))
    bar_width = 0.34
    for index, (generator, label, color) in enumerate(
        zip(generators_panel, generator_panel_labels, generator_panel_colors)
    ):
        values = [
            float(
                panel.loc[
                    panel["panel"].eq(panel_id)
                    & panel["generator"].eq(generator),
                    "mean_n_pass",
                ].iloc[0]
            )
            for panel_id in panels
        ]
        bars = ax.bar(
            x + (index - 0.5) * bar_width,
            values,
            width=bar_width,
            color=color,
            edgecolor="#4D4D4D",
            linewidth=0.6,
            label=label,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value - 0.28,
                f"{value:.2f}",
                ha="center",
                va="top",
                color="white",
                fontsize=5.8,
            )
    ax.axhline(8, color="#4D4D4D", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Panel 1", "Panel 2", "Panel 3"])
    ax.set_ylim(0, 9.5)
    ax.set_ylabel("Structural checks passed")
    ax.set_title("Three disjoint project panels", loc="left", pad=8)
    ax.legend(loc="upper right", ncol=1, frameon=False)
    add_panel_label(ax, "a")

    ax = axes[0, 1]
    generators = ["template_multinomial", "correlated_perturbation"]
    generator_labels = ["Template", "Correlated perturbation"]
    data = [
        holdout_fold.loc[
            holdout_fold["generator"].eq(generator),
            "mean_n_pass",
        ].to_numpy()
        for generator in generators
    ]
    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "black", "linewidth": 1},
    )
    for patch, color in zip(box["boxes"], ["#B4C0E4", "#F0C0CC"]):
        patch.set_facecolor(color)
        patch.set_edgecolor("#4D4D4D")
    for index, values in enumerate(data, start=1):
        jitter = np.linspace(-0.09, 0.09, len(values))
        ax.scatter(index + jitter, values, s=10, color="#4D4D4D", alpha=0.5)
    ax.axhline(8, color="#4D4D4D", linestyle="--", linewidth=0.8)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(generator_labels)
    ax.set_ylim(0, 8.4)
    ax.set_ylabel("Fixed-tolerance checks passed")
    ax.set_title("Repeated project holdout, n=30 panels", loc="left", pad=8)
    add_panel_label(ax, "b")

    metric_order = [
        "depth_log_median",
        "prevalence_mean",
        "rank_abundance_log_median",
        "variance_ratio",
        "bray_median",
        "aitchison_median",
        "zero_fraction",
        "network_density_median",
        "network_centrality_median",
    ]
    metric_labels = [
        "Depth",
        "Prevalence",
        "Rank abundance",
        "Variance ratio",
        "Bray-Curtis",
        "Aitchison",
        "Zero fraction",
        "Network density",
        "Network centrality",
    ]
    pivot = (
        coverage.pivot(index="metric", columns="generator", values="coverage")
        .reindex(index=metric_order, columns=generators)
        .to_numpy(dtype=float)
    )
    style_heatmap(
        axes[1, 0],
        pivot,
        metric_labels,
        generator_labels,
        "Empirical interval coverage (5 draws/fold)",
        "Blues",
        vmin=0,
        vmax=1,
    )
    add_panel_label(axes[1, 0], "c")

    ax = axes[1, 1]
    ax.axis("off")
    decision_rows = [
        ("Internal benchmark execution", "PASS", "#8BCF8B"),
        ("Three-panel structural calibration", "PASS", "#8BCF8B"),
        ("Structural project-panel holdout", "LIMITED SUPPORT", "#E9A6A1"),
        ("Cross-project method generalization", "NOT SUPPORTED", "#E9A6A1"),
    ]
    y_positions = np.linspace(0.78, 0.22, len(decision_rows))
    for (claim, status, color), y in zip(decision_rows, y_positions):
        ax.add_patch(
            FancyBboxPatch(
                (0.02, y - 0.08),
                0.96,
                0.14,
                boxstyle="round,pad=0.012",
                facecolor=color,
                edgecolor="#4D4D4D",
                linewidth=0.6,
            )
        )
        ax.text(0.05, y, claim, va="center", fontsize=7)
        ax.text(0.95, y, status, va="center", ha="right", fontsize=7, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Structural transportability decision", loc="left", pad=8)
    add_panel_label(ax, "d")
    save_figure(
        figure,
        "Fig5_project_external_validation",
        list(axes.ravel()),
        ["a", "b", "c", "d"],
        [["a", "b"], ["c", "d"]],
        [["a", "c"], ["b", "d"]],
    )


def main() -> None:
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    print("Figures and source data written")


if __name__ == "__main__":
    main()
