#!/usr/bin/env python3
"""Build the Layer-1 support-mismatch figure for the compressed manuscript."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "outputs" / "scale" / "box_scale_records.parquet"
COVERAGE = ROOT / "outputs" / "coverage" / "source_common_intersection.csv"
OUT = ROOT / "figures" / "fig_support_mismatch.pdf"

SOURCES = ["visdrone", "uavdt", "aitod"]
LABEL = {"visdrone": "VisDrone", "uavdt": "Local UAVDT", "aitod": "AI-TOD*"}
COLORS = {"visdrone": "#0072B2", "uavdt": "#D55E00", "aitod": "#009E73"}
STYLES = {"visdrone": "-", "uavdt": "--", "aitod": "-."}
ABS = np.array([16, 20, 24, 28, 32, 40, 48, 64], dtype=float)
NORM = np.array([.005, .0075, .010, .015, .020, .030], dtype=float)


def main() -> None:
    records = pd.read_parquet(RECORDS)
    coverage = pd.read_csv(COVERAGE).set_index("source")
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.3), gridspec_kw={"width_ratios": [.83, 1.10, 1.10]})

    # (a) The evaluation denominator must be visible before later metric or
    # ranking layers are interpreted.
    ax = axs[0]
    y = np.arange(len(SOURCES))[::-1]
    den = np.array([coverage.loc[s, "source_evaluation_images"] for s in SOURCES], float)
    num = np.array([coverage.loc[s, "common_prediction_covered_images"] for s in SOURCES], float)
    frac = 100 * num / den
    ax.barh(y, 100, color=".92", height=.58, edgecolor=".75", lw=.45)
    bars = ax.barh(y, frac, color=[COLORS[s] for s in SOURCES], height=.58, edgecolor=".25", lw=.35)
    # AI-TOD's incomplete raw coverage is an explicit conditional-analysis flag.
    bars[2].set_hatch("//")
    for yi, n, d, f in zip(y, num, den, frac):
        ax.text(min(f - 1.3, 97), yi, f"{int(n):,}/{int(d):,}", ha="right", va="center", fontsize=6.7,
                color="white" if f > 50 else ".15")
    ax.set(yticks=y, yticklabels=[LABEL[s] for s in SOURCES], xlim=(0, 103), xlabel="Raw prediction coverage (%)")
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", alpha=.22, lw=.45)
    ax.text(-.17, 1.03, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=8)

    for ax, measure, thresholds, headline, panel, xlabel in [
        (axs[1], "max_side_px", ABS, 24., "(b)", "Absolute support threshold (px)"),
        (axs[2], "normalized_side", NORM, .015, "(c)", "Normalized support threshold"),
    ]:
        for source in SOURCES:
            values = records.loc[records.source.eq(source), measure].to_numpy(float)
            retained = 100 * np.array([(values >= t).mean() for t in thresholds])
            ax.plot(thresholds, retained, color=COLORS[source], ls=STYLES[source], marker="o", ms=2.6,
                    lw=1.25, label=LABEL[source])
            keep = 100 * (values >= headline).mean()
            # Carefully positioned direct labels avoid a large external legend.
            x = thresholds[-1] if source == "uavdt" and measure == "max_side_px" else headline
            yoff = {"visdrone": 3.5, "uavdt": 4.5, "aitod": -7.0}[source]
            if measure == "normalized_side":
                x = {"visdrone": .017, "uavdt": .017, "aitod": .017}[source]
                yoff = {"visdrone": -8.5, "uavdt": 2.5, "aitod": -4.5}[source]
            ax.annotate(f"{keep:.1f}%", xy=(headline, keep), xytext=(3, yoff), textcoords="offset points",
                        fontsize=6.5, color=COLORS[source], ha="left")
        ax.axvline(headline, color=".15", lw=.75, ls=":")
        ax.set(xlabel=xlabel, ylabel="Retained GT (%)", ylim=(-3, 103))
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.25, lw=.45)
        ax.text(-.15, 1.03, panel, transform=ax.transAxes, fontweight="bold", fontsize=8)
    axs[1].set_xticks(ABS)
    axs[1].tick_params(axis="x", labelrotation=35)
    axs[2].set_xticks(NORM)
    axs[2].set_xticklabels([".005", ".0075", ".01", ".015", ".02", ".03"], rotation=35, ha="right")
    handles = [Line2D([0], [0], color=COLORS[s], ls=STYLES[s], marker="o", ms=3, lw=1.2, label=LABEL[s]) for s in SOURCES]
    fig.legend(handles=handles, ncol=3, loc="upper center", frameon=False, fontsize=7, bbox_to_anchor=(.58, 1.045), columnspacing=1.1)
    fig.text(.50, -.015, "*AI-TOD support is structural; metric/rank layers exclude its incomplete prediction coverage.", ha="center", fontsize=6.1)
    fig.subplots_adjust(left=.065, right=.995, top=.80, bottom=.28, wspace=.44)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
