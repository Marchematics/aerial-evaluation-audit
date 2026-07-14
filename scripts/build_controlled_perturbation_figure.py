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


def rect(ax, x, y, w, h, color, label, *, dashed=False, label_y=None):
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec=color, alpha=.18, lw=1.05,
                           ls="--" if dashed else "-"))
    if label:
        ax.text(x + w / 2, y + h / 2 if label_y is None else label_y, label, ha="center", va="center", fontsize=6.05)


def contract_panel(ax) -> None:
    valid, removed, ignore, pred = "#0072B2", "#E69F00", "#777777", "#D55E00"
    ax.set(xlim=(0, 30), ylim=(0, 6.05))
    ax.axis("off")
    ax.text(0, 5.92, "(a)", fontsize=8.2, fontweight="bold", va="top")
    ax.text(1.15, 5.92, "Target and ignore contract", fontsize=7.55, va="top")
    for x in (10, 20):
        ax.plot([x, x], [.45, 4.70], color=".84", lw=.65)

    # I: full target.
    ax.text(5, 4.72, "I: include all", ha="center", fontsize=6.85, fontweight="bold", color=COLORS["include_all"])
    rect(ax, 1.20, 2.05, 2.4, 1.55, valid, "valid GT")
    rect(ax, 5.20, 2.05, 2.4, 1.55, removed, "small valid GT")
    ax.annotate("both scored", xy=(8.65, 2.85), xytext=(7.70, 4.05), fontsize=6.0, ha="center",
                arrowprops=dict(arrowstyle="->", lw=.65, color=".25"))
    ax.text(5, .82, "Full target", ha="center", fontsize=6.15)

    # O: filtered target without source-ignore protection.
    ax.text(15, 4.72, "O: exclude / ignore off", ha="center", fontsize=6.85, fontweight="bold", color=COLORS["exclude_source_ignore_off"])
    rect(ax, 11.05, 2.05, 2.4, 1.55, valid, "retained GT")
    rect(ax, 14.60, 2.05, 2.4, 1.55, removed, "", dashed=True)
    rect(ax, 15.05, 2.38, 2.4, 1.55, pred, "")
    ax.text(15.80, 1.74, "removed GT", ha="center", fontsize=5.55, color=removed)
    ax.text(16.24, 4.18, "prediction", ha="center", fontsize=5.55, color=pred)
    ax.annotate("FP", xy=(17.82, 3.15), xytext=(18.75, 3.15), fontsize=7.4, color="#B2182B", fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=.65, color=".25"))
    ax.text(15, .82, "Filtered target; source ignores disabled", ha="center", fontsize=6.15)

    # N: same filtered target, valid GT first then source ignore.
    ax.text(25, 4.72, "N: exclude / ignore on", ha="center", fontsize=6.85, fontweight="bold", color=COLORS["exclude_source_ignore_on"])
    rect(ax, 21.10, 2.05, 2.2, 1.55, valid, "retained GT")
    rect(ax, 24.20, 2.05, 2.2, 1.55, ignore, "", dashed=True)
    rect(ax, 25.20, 2.38, 2.2, 1.55, pred, "")
    ax.text(25.30, 1.74, "source ignore", ha="center", fontsize=5.45, color=ignore)
    ax.text(26.30, 4.18, "prediction", ha="center", fontsize=5.55, color=pred)
    ax.text(29.20, 2.02, "unmatched $\\Rightarrow$\nneutralized", fontsize=5.30, ha="right", va="center", color=ignore)
    ax.text(25, .82, "Filtered target; valid GT matched first", ha="center", fontsize=6.15)
    ax.text(15, .20, "In I/O/N, a removed valid box is never converted to an ignore region.", ha="center", fontsize=5.85, color=".20")


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
    ax.set_ylim(.68, .90)
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
