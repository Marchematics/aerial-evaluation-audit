#!/usr/bin/env python3
"""Combine ancillary cap and structural diagnostics for the supplement."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "outputs" / "cap_sufficiency" / "cap_sufficiency_table.csv"
SCALE = ROOT / "outputs" / "scale" / "box_scale_records.parquet"
OUT = ROOT / "figures" / "fig_supplement_controls_structural.pdf"
SHORT = {
    "visdrone_control_baseline640": "B-640",
    "visdrone_control_baseline1280": "B-1280",
    "visdrone_control_yolo11m_hf640": "Y11m",
    "visdrone_control_asd1280": "ASD",
    "visdrone_control_yolo11n_p2_1280": "Y11n-P2",
}
ORDER = list(SHORT)
SOURCES = ["visdrone", "uavdt", "aitod", "dota_v2_val"]
LABELS = ["VisDrone", "UAVDT", "AI-TOD", "DOTA-v2"]


def draw_cap(ax: plt.Axes) -> None:
    table = pd.read_csv(CAP)
    q = table[table.cohort.eq("resubmission_controlled_pool")].set_index("candidate").loc[ORDER]
    y = np.arange(len(ORDER))
    specs = [(100, -.22, "#E45756", "100"), (300, 0, "#E69F00", "300"), (2000, .22, "#54A24B", "2000")]
    for cap, off, color, label in specs:
        val = 100 * q[f"fraction_truncated_at_{cap}"].to_numpy(float)
        ax.barh(y + off, val, height=.20, color=color, edgecolor="white", lw=.3, label=f"{label}")
        if cap == 2000:
            ax.scatter(np.full(len(y), .45), y + off, marker="|", s=42, color=color, linewidths=1.05, zorder=3)
    ax.set_yticks(y, [SHORT[c] for c in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(r"covered images truncated (\%)", fontsize=6.2, labelpad=1)
    ax.set_title("(a) AP cap contract", fontsize=7.6, pad=3)
    ax.grid(axis="x", color=".86", lw=.45)
    ax.tick_params(labelsize=6.0, length=1.8)
    ax.legend(title="maxDets", title_fontsize=5.8, loc="lower right", ncol=3, fontsize=5.5,
              frameon=False, handlelength=.9, columnspacing=.55, handletextpad=.25)


def draw_structural(ax: plt.Axes) -> None:
    records = pd.read_parquet(SCALE)
    values = {source: records.loc[records.source.eq(source), "normalized_side"].to_numpy(float) for source in SOURCES}
    m = np.array([[wasserstein_distance(values[a], values[b]) for b in SOURCES] for a in SOURCES])
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max())
    for i in range(len(SOURCES)):
        for j in range(len(SOURCES)):
            color = "white" if m[i, j] > .65 * m.max() else ".18"
            ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=5.75, color=color)
    ax.set_xticks(range(4), LABELS, rotation=25, ha="right")
    ax.set_yticks(range(4), LABELS)
    ax.tick_params(labelsize=5.55, length=0)
    ax.set_title("(b) Structural support distance", fontsize=7.6, pad=3)
    cbar = plt.colorbar(im, ax=ax, fraction=.046, pad=.035)
    cbar.ax.tick_params(labelsize=5.0, length=1.5)
    cbar.set_label("Wasserstein", fontsize=5.35, labelpad=1)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45), gridspec_kw={"width_ratios": [1.18, .94], "wspace": .36})
    draw_cap(axes[0])
    draw_structural(axes[1])
    fig.tight_layout(pad=.35)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
