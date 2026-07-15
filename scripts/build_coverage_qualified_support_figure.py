#!/usr/bin/env python3
"""Build the Gate-0 and Layer-1 coverage-qualified support figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


ROOT = Path(__file__).resolve().parents[1]
SCALE = ROOT / "outputs" / "scale" / "box_scale_records.parquet"
COVERAGE = ROOT / "outputs" / "coverage" / "source_common_intersection.csv"
AI_CURVES = ROOT / "outputs" / "coverage_qualification" / "aitod_support_curves.csv"
OUT = ROOT / "figures" / "fig_coverage_qualified_support.pdf"
SOURCES = ["visdrone", "uavdt", "aitod"]
LABEL = {"visdrone": "VisDrone", "uavdt": "UAVDT", "aitod": "AI-TOD"}
COLOR = {"visdrone": "#0072B2", "uavdt": "#D55E00", "aitod": "#009E73"}
STYLE = {"visdrone": "-", "uavdt": "--", "aitod": "-."}
ABS = np.asarray([16, 20, 24, 28, 32, 40, 48, 64], dtype=float)
NORM = np.asarray([.005, .0075, .010, .015, .020, .030], dtype=float)


def retained(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.asarray([100 * np.mean(values >= threshold) for threshold in thresholds])


def main() -> None:
    scale = pd.read_parquet(SCALE)
    coverage = pd.read_csv(COVERAGE).set_index("source")
    ai = pd.read_csv(AI_CURVES)
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.65), gridspec_kw={"width_ratios": [.82, 1.18], "hspace": .58, "wspace": .34})

    ax = axes[0, 0]
    y = np.arange(3)[::-1]
    den = np.asarray([coverage.loc[source, "source_evaluation_images"] for source in SOURCES], dtype=float)
    num = np.asarray([coverage.loc[source, "common_prediction_covered_images"] for source in SOURCES], dtype=float)
    fraction = 100 * num / den
    ax.barh(y, 100, color=".93", edgecolor=".72", lw=.45, height=.56)
    bars = ax.barh(y, fraction, color=[COLOR[source] for source in SOURCES], edgecolor=".25", lw=.35, height=.56)
    bars[2].set_hatch("//")
    for yi, n, d, f in zip(y, num, den, fraction):
        ax.text(min(f - 1.5, 97.5), yi, f"{int(n):,}/{int(d):,}", ha="right", va="center", fontsize=6.4,
                color="white" if f > 48 else ".15")
    ax.set(yticks=y, yticklabels=[LABEL[source] for source in SOURCES], xlim=(0, 103), xlabel="Raw prediction coverage (%)")
    ax.set_title("(a) Coverage qualification", fontsize=7.8, loc="left", pad=3)
    ax.grid(axis="x", color=".88", lw=.45)

    for ax, mode, thresholds, field, headline, title in (
        (axes[0, 1], "absolute", ABS, "max_side_px", 24., "(b) Absolute support"),
        (axes[1, 1], "normalized", NORM, "normalized_side", .015, "(d) Normalized support"),
    ):
        for source in SOURCES:
            values = scale.loc[scale.source.eq(source), field].to_numpy(float)
            curve = retained(values, thresholds)
            ax.plot(thresholds, curve, marker="o", ms=2.8, lw=1.05, ls=STYLE[source], color=COLOR[source])
        ax.axvline(headline, color=".25", lw=.75, ls=":")
        ax.set_ylim(-2, 104); ax.set_ylabel("Retained valid GT (%)")
        ax.set_xlabel("Threshold (px)" if mode == "absolute" else "Normalized threshold")
        ax.set_title(title, fontsize=7.8, loc="left", pad=3)
        ax.grid(color=".89", lw=.45)
        if mode == "absolute":
            ax.set_xticks(thresholds, [f"{value:g}" for value in thresholds], rotation=30)
        else:
            ax.set_xticks(thresholds, [f"{value:g}" for value in thresholds], rotation=30)

    ax = axes[1, 0]
    headline = ai[((ai.coverage_group.isin(["full", "covered"])) &
                   (((ai["mode"] == "absolute") & np.isclose(ai.threshold, 24)) |
                    ((ai["mode"] == "normalized") & np.isclose(ai.threshold, .015))))].copy()
    order = [("absolute", 24.), ("normalized", .015)]
    x = np.arange(2)
    full = [headline[(headline.coverage_group == "full") & (headline["mode"] == mode) & np.isclose(headline.threshold, threshold)].iloc[0].retained_percent for mode, threshold in order]
    covered = [headline[(headline.coverage_group == "covered") & (headline["mode"] == mode) & np.isclose(headline.threshold, threshold)].iloc[0].retained_percent for mode, threshold in order]
    width = .34
    bars1 = ax.bar(x - width / 2, full, width, color=".70", edgecolor=".25", lw=.4, label="Full AI-TOD")
    bars2 = ax.bar(x + width / 2, covered, width, color=COLOR["aitod"], edgecolor=".25", lw=.4, hatch="//", label="Covered subset")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=6.1)
    ax.set_xticks(x, ["24 px", ".015"])
    ax.set_ylim(0, 100); ax.set_ylabel("Retained valid GT (%)")
    ax.set_title("(c) AI-TOD conditioning", fontsize=7.8, loc="left", pad=3)
    ax.grid(axis="y", color=".89", lw=.45)
    ax.legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1.2)

    legend = [Line2D([0], [0], color=COLOR[source], ls=STYLE[source], marker="o", ms=2.8, lw=1.05, label=LABEL[source]) for source in SOURCES]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False, fontsize=6.2, bbox_to_anchor=(.68, 1.01),
               handlelength=1.8, columnspacing=1.1)
    for ax in axes.flat:
        ax.tick_params(labelsize=6.1, length=2.2, pad=1.5)
        for spine in ax.spines.values(): spine.set_linewidth(.55)
    fig.subplots_adjust(left=.09, right=.985, top=.92, bottom=.12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
