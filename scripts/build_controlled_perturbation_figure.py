#!/usr/bin/env python3
"""Build the two-panel main-text policy-band figure from the final V4 grid."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.10))
    band_panel(ax1, values, "absolute", "(a) Absolute support: 24 px", f"{abs_range['min']:.3f}--{abs_range['max']:.3f}")
    band_panel(ax2, values, "normalized", "(b) Normalized support: .015", f"{norm_range['min']:.3f}--{norm_range['max']:.3f}")
    handles = [Line2D([0], [0], marker="o", lw=0, color=COLORS[p], label={
        "include_all": "I: include all",
        "exclude_source_ignore_off": "O: exclude / ignore off",
        "exclude_source_ignore_on": "N: exclude / ignore on",
    }[p], markersize=5.2) for p in POLICIES]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.50, -.012), ncol=3, frameon=False,
               fontsize=6.15, handletextpad=.35, columnspacing=1.2)
    fig.subplots_adjust(left=.070, right=.988, top=.88, bottom=.24, wspace=.27)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(values.to_csv(index=False))


if __name__ == "__main__":
    main()
