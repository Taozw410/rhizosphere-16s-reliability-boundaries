#!/usr/bin/env python3
"""Create the TARE-DA project-held-out validation figure."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

SKILL_SCRIPTS = Path(os.environ["NATURE_FIGURE_SCRIPTS"])
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)

BLUE = "#0F4D92"
NEUTRAL = "#A8A8A8"
TEAL = "#42949E"
RED = "#B64342"
GOLD = "#C59A27"
INK = "#272727"


def add_panel_label(ax, label: str) -> None:
    offset = ScaledTranslation(
        -7 / 72,
        5 / 72,
        ax.figure.dpi_scale_trans,
    )
    ax.text(
        0,
        1,
        label,
        transform=ax.transAxes + offset,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = root / "artifacts" / "tare_da"
    output_dir = root / "figures"
    source_dir = output_dir / "source_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(artifact / "tare_da_predictions.csv")
    summary = pd.read_csv(artifact / "tare_da_summary.csv")
    predictions["error_tare"] = (
        predictions["tare_point"] - predictions["actual_utility"]
    ).abs()
    predictions["error_baseline"] = (
        predictions["baseline_utility"] - predictions["actual_utility"]
    ).abs()

    partition = (
        predictions[predictions["scenario"].eq("sparse20_fc2")]
        .groupby("partition")
        .agg(
            tare_mae=("error_tare", "mean"),
            baseline_mae=("error_baseline", "mean"),
        )
        .reset_index()
    )
    partition["mae_improvement"] = (
        partition["baseline_mae"] - partition["tare_mae"]
    )
    partition.to_csv(
        source_dir / "Fig6a_paired_strong_effect_mae.csv", index=False
    )

    method_mae = (
        predictions[predictions["scenario"].eq("sparse20_fc2")]
        .groupby(["partition", "method"])
        .agg(
            tare_mae=("error_tare", "mean"),
            baseline_mae=("error_baseline", "mean"),
        )
        .reset_index()
    )
    method_mae_summary = (
        method_mae.groupby("method")
        .agg(
            tare_mean=("tare_mae", "mean"),
            tare_sd=("tare_mae", "std"),
            baseline_mean=("baseline_mae", "mean"),
            baseline_sd=("baseline_mae", "std"),
        )
        .reset_index()
    )
    method_order = ["aldex2", "ancombc2", "linda", "maaslin2"]
    display = {
        "aldex2": "ALDEx2",
        "ancombc2": "ANCOM-BC2",
        "linda": "LinDA",
        "maaslin2": "MaAsLin2",
    }
    method_mae_summary["order"] = method_mae_summary["method"].map(
        {value: index for index, value in enumerate(method_order)}
    )
    method_mae_summary = method_mae_summary.sort_values("order")
    method_mae_summary.to_csv(
        source_dir / "Fig6b_method_strong_effect_mae.csv", index=False
    )

    coverage = (
        predictions.groupby(["scenario", "method", "partition"])
        .agg(coverage=("covered", "mean"))
        .reset_index()
    )
    coverage_summary = (
        coverage.groupby(["scenario", "method"])
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_min=("coverage", "min"),
            coverage_max=("coverage", "max"),
        )
        .reset_index()
    )
    coverage_summary.to_csv(
        source_dir / "Fig6c_conformal_coverage.csv", index=False
    )

    decisions = (
        predictions.groupby(["scenario", "decision"])
        .size()
        .rename("count")
        .reset_index()
    )
    decisions.to_csv(source_dir / "Fig6d_decisions.csv", index=False)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2047, 5.0394),
        gridspec_kw={
            "left": 0.085,
            "right": 0.985,
            "bottom": 0.105,
            "top": 0.965,
            "wspace": 0.34,
            "hspace": 0.47,
        },
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # a: paired improvement across repeated project partitions.
    x = partition["partition"].to_numpy() + 1
    for row in partition.itertuples(index=False):
        ax_a.plot(
            [row.partition + 1, row.partition + 1],
            [row.baseline_mae, row.tare_mae],
            color="#D0D0D0",
            linewidth=0.9,
            zorder=1,
        )
    ax_a.scatter(
        x,
        partition["baseline_mae"],
        s=19,
        facecolor="white",
        edgecolor=NEUTRAL,
        linewidth=0.9,
        zorder=2,
    )
    ax_a.scatter(
        x,
        partition["tare_mae"],
        s=22,
        color=BLUE,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    ax_a.set(
        xlabel="Repeated project-holdout partition",
        ylabel="Utility MAE",
        xlim=(0.5, 10.9),
    )
    ax_a.set_xticks(range(1, 11))
    mean_reduction = (
        partition["mae_improvement"].mean()
        / partition["baseline_mae"].mean()
        * 100
    )
    ax_a.text(
        0.02,
        1.035,
        f"Mean reduction {mean_reduction:.1f}%",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color=INK,
        clip_on=False,
    )
    ax_a.text(
        0.98,
        1.035,
        "○ Mean-only    ● TARE-DA",
        transform=ax_a.transAxes,
        fontsize=6.5,
        ha="right",
        va="bottom",
        color=INK,
        clip_on=False,
    )
    add_panel_label(ax_a, "a")

    # b: method-level strong-effect decomposition.
    positions = np.arange(len(method_mae_summary))
    width = 0.33
    ax_b.bar(
        positions - width / 2,
        method_mae_summary["baseline_mean"],
        width,
        color=NEUTRAL,
        edgecolor="white",
        linewidth=0.4,
        yerr=method_mae_summary["baseline_sd"],
        error_kw={"elinewidth": 0.7, "capsize": 2, "capthick": 0.7},
    )
    ax_b.bar(
        positions + width / 2,
        method_mae_summary["tare_mean"],
        width,
        color=BLUE,
        edgecolor="white",
        linewidth=0.4,
        yerr=method_mae_summary["tare_sd"],
        error_kw={"elinewidth": 0.7, "capsize": 2, "capthick": 0.7},
    )
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels(
        [display[value] for value in method_mae_summary["method"]]
    )
    ax_b.set_ylabel("Strong-effect utility MAE")
    ax_b.set_xlabel("")
    ax_b.set_ylim(0, 0.24)
    ax_b.legend(
        handles=[
            Line2D([0], [0], marker="s", linestyle="", color=NEUTRAL, label="Mean-only"),
            Line2D([0], [0], marker="s", linestyle="", color=BLUE, label="TARE-DA"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=6.5,
        handlelength=0.9,
        borderaxespad=0,
    )
    add_panel_label(ax_b, "b")

    # c: conformal coverage with partition range.
    coverage_method = coverage_summary[
        coverage_summary["scenario"].eq("sparse20_fc2")
    ].copy()
    coverage_method["order"] = coverage_method["method"].map(
        {value: index for index, value in enumerate(method_order)}
    )
    coverage_method = coverage_method.sort_values("order")
    y_positions = np.arange(len(coverage_method))
    ax_c.scatter(
        coverage_method["coverage_mean"],
        y_positions,
        s=28,
        color=TEAL,
        zorder=3,
    )
    ax_c.axvline(0.95, color=RED, linestyle=(0, (3, 2)), linewidth=0.9)
    ax_c.set_yticks(y_positions)
    ax_c.set_yticklabels(
        [display[value] for value in coverage_method["method"]]
    )
    ax_c.invert_yaxis()
    ax_c.set_xlim(0.86, 1.005)
    ax_c.set_xlabel("Empirical 95% interval coverage (nominal 0.95)")
    ax_c.set_ylabel("")
    add_panel_label(ax_c, "c")

    # d: recommendation decisions.
    scenario_order = ["no_effect", "sparse20_fc2"]
    decision_order = ["ALLOW", "CONDITIONAL", "DENY"]
    colors = {"ALLOW": BLUE, "CONDITIONAL": GOLD, "DENY": RED}
    bottoms = np.zeros(len(scenario_order))
    totals = (
        decisions.groupby("scenario")["count"].sum().reindex(scenario_order)
    )
    for decision_name in decision_order:
        values = (
            decisions[decisions["decision"].eq(decision_name)]
            .set_index("scenario")["count"]
            .reindex(scenario_order)
            .fillna(0)
        )
        proportions = values / totals * 100
        ax_d.bar(
            np.arange(len(scenario_order)),
            proportions,
            bottom=bottoms,
            width=0.55,
            color=colors[decision_name],
            edgecolor="white",
            linewidth=0.6,
            label=decision_name,
        )
        bottoms += proportions.to_numpy()
    ax_d.set_xticks(np.arange(len(scenario_order)))
    ax_d.set_xticklabels(["No effect", "Strong sparse"])
    ax_d.set_ylabel("Held-out predictions (%)")
    ax_d.set_ylim(0, 112)
    ax_d.set_xlabel("")
    overall = pd.read_csv(artifact / "tare_da_overall.csv").iloc[0]
    ax_d.text(
        0.03,
        0.96,
        f"ALLOW precision {overall['allow_precision']:.3f}",
        transform=ax_d.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    allow_values = (
        decisions[
            decisions["decision"].eq("ALLOW")
            & decisions["scenario"].eq("no_effect")
        ]["count"].sum()
        / totals.loc["no_effect"]
        * 100
    )
    deny_values = (
        decisions[
            decisions["decision"].eq("DENY")
            & decisions["scenario"].eq("sparse20_fc2")
        ]["count"].sum()
        / totals.loc["sparse20_fc2"]
        * 100
    )
    ax_d.text(
        0,
        50,
        f"ALLOW\n{allow_values:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontsize=7,
        fontweight="bold",
    )
    ax_d.text(
        1,
        50,
        f"DENY\n{deny_values:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontsize=7,
        fontweight="bold",
    )
    add_panel_label(ax_d, "d")

    for axis in axes.ravel():
        axis.tick_params(width=0.7, length=2.7, pad=2.5)
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontsize(6.5)

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        axes=[ax_a, ax_b, ax_c, ax_d],
        panel_ids=["a", "b", "c", "d"],
        row_groups=[["a", "b"], ["c", "d"]],
        column_groups=[["a", "c"], ["b", "d"]],
        json_out=str(output_dir / "Fig6_TARE.alignment.json"),
        overlay_svg=str(output_dir / "Fig6_TARE.alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    base = output_dir / "Fig6_TARE"
    fig.savefig(base.with_suffix(".svg"), dpi=600)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".tiff"), dpi=600)
    fig.savefig(base.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Wrote {base}.svg/.pdf/.tiff/.png")


if __name__ == "__main__":
    main()
