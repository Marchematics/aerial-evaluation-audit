#!/usr/bin/env python3
"""Build the cross-metric rank trajectory and sequence-bootstrap forest plot."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
POINTS = ROOT / "outputs" / "metric_qualified_rank" / "metric_rank_points.csv"
BOOT = ROOT / "outputs" / "metric_qualified_rank" / "paired_bootstrap.csv"
OUT = ROOT / "figures" / "fig_metric_cluster_qualified_rank.pdf"
ORDER = ["B-640", "B-1280", "Y11m", "ASD", "Y11n-P2"]
COLORS = {"B-640": "#0072B2", "B-1280": "#56B4E9", "Y11m": "#009E73", "ASD": "#D55E00", "Y11n-P2": "#CC79A7"}
MARKERS = {"B-640": "o", "B-1280": "s", "Y11m": "D", "ASD": "^", "Y11n-P2": "v"}


def main() -> None:
    points = pd.read_csv(POINTS)
    boot = pd.read_csv(BOOT)
    boot = boot[boot.resampling_unit.eq("sequence")].copy()
    fig, (ax_rank, ax_ci) = plt.subplots(1, 2, figsize=(7.16, 3.05), gridspec_kw={"width_ratios": [1.22, 1], "wspace": .36})

    columns = [
        ("absolute", 24., "f1_at_025", "24/F1"),
        ("absolute", 24., "max_f1", "24/max"),
        ("absolute", 24., "ap50", "24/AP50"),
        ("normalized", .015, "f1_at_025", ".015/F1"),
        ("normalized", .015, "max_f1", ".015/max"),
        ("normalized", .015, "ap50", ".015/AP50"),
    ]
    for candidate in ORDER:
        ranks, values = [], []
        for mode, threshold, metric, _ in columns:
            row = points[(points.short_candidate == candidate) & (points["mode"] == mode) & np.isclose(points.threshold, threshold)].iloc[0]
            ranks.append(int(row[f"rank_{metric}"])); values.append(float(row[metric]))
        ax_rank.plot(range(len(columns)), ranks, marker=MARKERS[candidate], ms=3.5, lw=1.0,
                     color=COLORS[candidate], label=candidate)
        if candidate in {"Y11m", "ASD"}:
            for x, (rank, value) in enumerate(zip(ranks, values)):
                offset = -7 if candidate == "Y11m" else 6
                ax_rank.annotate(f"{value:.3f}", (x, rank), xytext=(0, offset), textcoords="offset points",
                                 ha="center", va="top" if offset < 0 else "bottom", fontsize=5.1, color=COLORS[candidate])
    ax_rank.set_xticks(range(len(columns)), [value[3] for value in columns], rotation=28, ha="right")
    ax_rank.set_yticks(range(1, 6)); ax_rank.set_ylim(5.35, .65)
    ax_rank.set_ylabel("Rank (1 = best)")
    ax_rank.set_title("(a) Metric-conditioned ranks", fontsize=7.7, loc="left", pad=3)
    ax_rank.grid(color=".89", lw=.45)
    ax_rank.axvline(2.5, color=".45", lw=.65, ls=":")
    handles, labels = ax_rank.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=5.7, ncol=5, loc="upper left",
               bbox_to_anchor=(.09, .985), handlelength=1.2, columnspacing=.8)

    order = [
        ("absolute", "F1@.25", "24 px / F1@.25"),
        ("absolute", "max-F1", "24 px / max-F1"),
        ("absolute", "AP50", "24 px / AP50"),
        ("normalized", "F1@.25", ".015 / F1@.25"),
        ("normalized", "max-F1", ".015 / max-F1"),
        ("normalized", "AP50", ".015 / AP50"),
    ]
    y = np.arange(len(order))[::-1]
    for yi, (mode, metric, _) in zip(y, order):
        row = boot[(boot["mode"] == mode) & (boot.metric == metric)].iloc[0]
        color = "#009E73" if row.ci95_low > 0 else ("#D55E00" if row.ci95_high < 0 else ".40")
        ax_ci.errorbar(row.point_difference, yi,
                       xerr=[[row.point_difference - row.ci95_low], [row.ci95_high - row.point_difference]],
                       fmt="o", ms=3.2, color=color, ecolor=color, elinewidth=1.0, capsize=2.0)
        ax_ci.text(row.ci95_high + .0012, yi, f"{row.point_difference:+.3f}", va="center", ha="left", fontsize=5.4, color=color)
    ax_ci.axvline(0, color=".18", lw=.75, ls="--")
    ax_ci.set_yticks(y, [value[2] for value in order])
    ax_ci.set_xlabel("Y11m $-$ ASD")
    ax_ci.set_title("(b) Sequence-cluster 95% intervals", fontsize=7.7, loc="left", pad=3)
    ax_ci.grid(axis="x", color=".89", lw=.45)
    ax_ci.text(.02, .985, "76 clusters; 10,000 paired resamples", transform=ax_ci.transAxes,
               ha="left", va="top", fontsize=5.5, color=".25")
    for ax in (ax_rank, ax_ci):
        ax.tick_params(labelsize=5.9, length=2.0, pad=1.5)
        for spine in ax.spines.values(): spine.set_linewidth(.55)
    fig.subplots_adjust(left=.085, right=.98, top=.79, bottom=.19)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
