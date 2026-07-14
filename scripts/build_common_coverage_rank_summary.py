#!/usr/bin/env python3
"""Build the main-text common-coverage rank summary.

The main manuscript deliberately shows only the third link of the audit
chain: the two declared headline rankings and the paired uncertainty records.
The full threshold--confidence rank surfaces remain a supplementary
sensitivity diagnostic.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "outputs" / "resubmission_controlled_pool" / "visdrone_grid_crowd_v4" / "metrics_long.parquet"
BOOT = ROOT / "outputs" / "statistics" / "bootstrap_controlled_visdrone_rank_pairs.csv"
OUT = ROOT / "figures" / "fig_common_coverage_rank_summary.pdf"

POLICY = "exclude_source_ignore_on"
SHORT = {
    "visdrone_control_baseline640": "B-640",
    "visdrone_control_baseline1280": "B-1280",
    "visdrone_control_yolo11m_hf640": "Y11m",
    "visdrone_control_asd1280": "ASD",
    "visdrone_control_yolo11n_p2_1280": "Y11n-P2",
}
ORDER = [
    "visdrone_control_yolo11m_hf640",
    "visdrone_control_asd1280",
    "visdrone_control_yolo11n_p2_1280",
    "visdrone_control_baseline640",
    "visdrone_control_baseline1280",
]
ABS_COLOR = "#0072B2"
NORM_COLOR = "#D55E00"
BOUNDARY_COLOR = "#CC79A7"


def headline_values(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, threshold, label in [("absolute", 24.0, "24 px"), ("normalized", 0.015, ".015")]:
        subset = frame[
            frame.small_object_policy.eq(POLICY)
            & np.isclose(frame.iou, 0.25)
            & np.isclose(frame.confidence, 0.25)
            & frame.scale_mode.eq(mode)
            & np.isclose(frame.scale_threshold, threshold)
        ]
        if set(subset.candidate) != set(ORDER):
            raise RuntimeError(f"Incomplete headline values for {mode} support")
        for r in subset.itertuples():
            rows.append({"candidate": r.candidate, "rule": label, "f1": r.f1})
    return pd.DataFrame(rows)


def main() -> None:
    grid = pd.read_parquet(GRID)
    values = headline_values(grid).pivot(index="candidate", columns="rule", values="f1").loc[ORDER]
    boot = pd.read_csv(BOOT).set_index("id")

    fig = plt.figure(figsize=(3.45, 3.10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.02, 1.00], hspace=0.74)

    # (a) The two primary conclusions are made on exactly the same images.
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(ORDER))
    for yy, candidate in zip(y, ORDER):
        a, n = values.loc[candidate, "24 px"], values.loc[candidate, ".015"]
        ax.plot([a, n], [yy, yy], color=".63", lw=1.0, zorder=1)
        ax.scatter(a, yy, s=27, color=ABS_COLOR, marker="o", zorder=3, edgecolor="white", linewidth=.35)
        ax.scatter(n, yy, s=27, color=NORM_COLOR, marker="s", zorder=3, edgecolor="white", linewidth=.35)
    ax.set_yticks(y, [SHORT[c] for c in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(.69, .89)
    ax.set_xticks([.70, .75, .80, .85])
    ax.set_xlabel("Micro-$F_1$ under policy $N$")
    ax.set_title("(a) Headline ranks (policy N; confidence/IoU=.25)", fontsize=7.20, pad=3.0)
    ax.grid(axis="x", color=".87", lw=.52)
    ax.tick_params(labelsize=6.35, length=2)

    # (b) Headline evidence and intentionally difficult boundary cases are
    # separated by color; their full-grid origin is documented in the supplement.
    ax = fig.add_subplot(gs[1])
    ids = [
        "headline_abs24",
        "headline_norm015",
        "observed_boundary_norm010_c030",
        "observed_boundary_abs64_c035",
    ]
    labels = [
        "H-a  24 px  Y11m$-$ASD",
        "H-n  .015  Y11m$-$ASD",
        "B-n  .010/.30  Y11m$-$ASD",
        "B-a  64/.35  B-640$-$Y11m",
    ]
    colors = [ABS_COLOR, NORM_COLOR, BOUNDARY_COLOR, BOUNDARY_COLOR]
    for yy, (rid, color) in enumerate(zip(ids, colors)):
        r = boot.loc[rid]
        value = float(r.point_difference_a_minus_b)
        ax.errorbar(
            value, yy,
            xerr=[[value - float(r.ci95_low)], [float(r.ci95_high) - value]],
            fmt="o", color=color, markersize=3.8, lw=.95, capsize=1.7, zorder=3,
        )
    ax.axvline(0, color=".20", lw=.78)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(-.007, .054)
    ax.set_xlabel(r"Paired $F_1$ difference (95\% interval)")
    ax.set_title("(b) Paired bootstrap intervals", fontsize=7.45, pad=3.0)
    ax.grid(axis="x", color=".87", lw=.52)
    ax.tick_params(labelsize=5.65, length=2)

    fig.subplots_adjust(left=.262, right=.985, top=.94, bottom=.115)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
