#!/usr/bin/env python3
"""Build the main-text controlled-support-perturbation figure.

The figure has one contract panel and two headline policy-band panels.  It is
derived exclusively from the final V4 full-coverage VisDrone grid.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "resubmission_controlled_pool" / "visdrone_grid_crowd_v4" / "metrics_long.parquet"
OUT = ROOT / "figures" / "fig_controlled_support_perturbation.pdf"
DERIVED = ROOT / "outputs" / "resubmission_controlled_pool" / "visdrone_grid_crowd_v4" / "headline_policy_bands.csv"

CANDIDATES = [
    "visdrone_control_baseline640",
    "visdrone_control_baseline1280",
    "visdrone_control_yolo11m_hf640",
    "visdrone_control_asd1280",
    "visdrone_control_yolo11n_p2_1280",
]
SHORT = {
    "visdrone_control_baseline640": "B-640",
    "visdrone_control_baseline1280": "B-1280",
    "visdrone_control_yolo11m_hf640": "Y11m",
    "visdrone_control_asd1280": "ASD",
    "visdrone_control_yolo11n_p2_1280": "Y11n-P2",
}
POLICIES = ["include_all", "exclude_source_ignore_off", "exclude_source_ignore_on"]
POLICY_SHORT = {"include_all": "I", "exclude_source_ignore_off": "O", "exclude_source_ignore_on": "N"}
COLORS = {"include_all": "#4C78A8", "exclude_source_ignore_off": "#E45756", "exclude_source_ignore_on": "#54A24B"}


def headline_values(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, threshold in [("absolute", 24.0), ("normalized", .015)]:
        q = frame[
            frame.scale_mode.eq(mode)
            & np.isclose(frame.scale_threshold, threshold)
            & np.isclose(frame.confidence, .25)
            & np.isclose(frame.iou, .25)
            & frame.small_object_policy.isin(POLICIES)
        ]
        for candidate in CANDIDATES:
            g = q[q.candidate.eq(candidate)]
            if len(g) != len(POLICIES):
                raise RuntimeError(f"Incomplete headline policy record for {candidate}, {mode}")
            values = dict(zip(g.small_object_policy, g.f1))
            rows.append({
                "scale_mode": mode, "threshold": threshold, "candidate": candidate,
                "candidate_short": SHORT[candidate], **values,
                "F1_policy_band": max(values.values()) - min(values.values()),
            })
    out = pd.DataFrame(rows)
    return out


def contract_panel(ax) -> None:
    """Show the categorical evaluator contract without spatial arrows."""
    ax.set(xlim=(0, 30), ylim=(0, 6.05))
    ax.axis("off")
    ax.text(0, 5.92, "(a)", fontsize=8.2, fontweight="bold", va="top")
    ax.text(1.18, 5.92, "Target and ignore contract", fontsize=7.55, va="top")

    # The three policies differ categorically, not geometrically.  A compact
    # comparison table therefore states each changed evaluator component
    # directly and avoids arrows that could be mistaken for matching steps.
    left, col_w = 4.65, 8.30
    col_x = [left + i * col_w for i in range(3)]
    centers = [x + col_w / 2 for x in col_x]
    row_y = [3.30, 2.05, .80]
    row_h = 1.05
    headers = [
        ("I: include all", COLORS["include_all"]),
        ("O: exclude / ignore off", COLORS["exclude_source_ignore_off"]),
        ("N: exclude / ignore on", COLORS["exclude_source_ignore_on"]),
    ]
    for x, center, (title, color) in zip(col_x, centers, headers):
        ax.add_patch(Rectangle((x + .08, .68), col_w - .16, 4.47,
                               facecolor=color, edgecolor=color, alpha=.055, lw=.75))
        ax.text(center, 4.86, title, ha="center", va="center", fontsize=6.55,
                fontweight="bold", color=color)

    row_labels = ["Scored valid GT", "Source-ignore regions", "Below-threshold valid GT"]
    for y, label in zip(row_y, row_labels):
        ax.text(left - .28, y + row_h / 2, label, ha="right", va="center",
                fontsize=6.05, color=".20", fontweight="bold")
        ax.plot([left, 29.55], [y, y], color=".83", lw=.55)

    cells = [
        ["all mapped boxes\n(small boxes retained)", "above-threshold\nboxes only", "above-threshold\nboxes only"],
        ["enabled", "disabled", "enabled after\nvalid-GT matching"],
        ["not removed", "dropped; not ignore", "dropped; not ignore"],
    ]
    for y, row in zip(row_y, cells):
        for center, text in zip(centers, row):
            ax.text(center, y + row_h / 2, text, ha="center", va="center",
                    fontsize=5.95, color=".12", linespacing=1.12)

    for x in [left, left + col_w, left + 2 * col_w, left + 3 * col_w]:
        ax.plot([x, x], [.68, 5.15], color=".80", lw=.55)
    ax.plot([left, 29.55], [5.15, 5.15], color=".70", lw=.65)
    ax.plot([left, 29.55], [4.48, 4.48], color=".78", lw=.55)
    ax.plot([left, 29.55], [.68, .68], color=".70", lw=.65)
    ax.text(17.10, .18,
            "Under O/N, predictions on removed valid boxes remain FP-eligible; removed boxes never become ignore regions.",
            ha="center", va="center", fontsize=5.65, color=".22")


def band_panel(ax, values: pd.DataFrame, mode: str, title: str, band_range: str) -> None:
    sub = values[values.scale_mode.eq(mode)].set_index("candidate").loc[CANDIDATES].reset_index()
    xs = np.arange(len(sub))
    for x, r in zip(xs, sub.itertuples()):
        ys = [getattr(r, p) for p in POLICIES]
        ax.vlines(x, min(ys), max(ys), color=".45", lw=1.15, zorder=1)
        for off, p in zip((-.18, 0, .18), POLICIES):
            ax.scatter(x + off, getattr(r, p), color=COLORS[p], s=22 if p != "exclude_source_ignore_on" else 29,
                       zorder=3, edgecolor="white", linewidth=.35)
        ax.text(x, min(ys) - .011, f"{r.F1_policy_band:.3f}", ha="center", va="top", fontsize=5.7, color=".30")
    ax.set_xlim(-.55, len(sub) - .45)
    ax.set_ylim(.67, .90)
    ax.set_xticks(xs, sub.candidate_short)
    ax.set_ylabel("Micro-$F_1$")
    ax.set_title(title, fontsize=8.0)
    ax.grid(axis="y", color=".86", lw=.6)
    ax.tick_params(labelsize=6.7)
    ax.text(.02, .96, f"$B^\\pi_{{F_1}}$: {band_range}", transform=ax.transAxes, va="top", fontsize=6.2)


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    values = headline_values(pd.read_parquet(DATA))
    DERIVED.parent.mkdir(parents=True, exist_ok=True)
    values.to_csv(DERIVED, index=False)
    abs_range = values[values.scale_mode.eq("absolute")].F1_policy_band.agg(["min", "max"])
    norm_range = values[values.scale_mode.eq("normalized")].F1_policy_band.agg(["min", "max"])

    fig = plt.figure(figsize=(7.16, 3.12))
    grid = fig.add_gridspec(2, 2, height_ratios=[.95, 1.23], hspace=.51, wspace=.27)
    ax0 = fig.add_subplot(grid[0, :]); contract_panel(ax0)
    ax1 = fig.add_subplot(grid[1, 0]); band_panel(ax1, values, "absolute", "(b) Absolute support: 24 px", f"{abs_range['min']:.3f}--{abs_range['max']:.3f}")
    ax2 = fig.add_subplot(grid[1, 1]); band_panel(ax2, values, "normalized", "(c) Normalized support: .015", f"{norm_range['min']:.3f}--{norm_range['max']:.3f}")
    handles = [Line2D([0], [0], marker="o", lw=0, color=COLORS[p], label={
        "include_all": "I: include all",
        "exclude_source_ignore_off": "O: exclude / ignore off",
        "exclude_source_ignore_on": "N: exclude / ignore on",
    }[p], markersize=5.2) for p in POLICIES]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.50, -.012), ncol=3, frameon=False,
               fontsize=6.15, handletextpad=.35, columnspacing=1.2)
    fig.subplots_adjust(left=.070, right=.988, top=.965, bottom=.15)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(values.to_csv(index=False))


if __name__ == "__main__":
    main()
