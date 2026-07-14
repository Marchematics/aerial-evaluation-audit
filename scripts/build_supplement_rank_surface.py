#!/usr/bin/env python3
"""Build the supplementary full rule-sensitivity surface for the V4 pool."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "resubmission_controlled_pool" / "visdrone_grid_crowd_v4" / "metrics_long.parquet"
OUT = ROOT / "figures" / "fig_supplement_rank_surface.pdf"
POLICY = "exclude_source_ignore_on"
CANDIDATES = [
    "visdrone_control_baseline640",
    "visdrone_control_baseline1280",
    "visdrone_control_yolo11m_hf640",
    "visdrone_control_asd1280",
    "visdrone_control_yolo11n_p2_1280",
]
SHORT = {
    CANDIDATES[0]: "B-640", CANDIDATES[1]: "B-1280", CANDIDATES[2]: "Y11m",
    CANDIDATES[3]: "ASD", CANDIDATES[4]: "Y11n-P2",
}
COLORS = {
    CANDIDATES[0]: "#4C78A8", CANDIDATES[1]: "#9ECAE1", CANDIDATES[2]: "#E45756",
    CANDIDATES[3]: "#54A24B", CANDIDATES[4]: "#79706E",
}


def rank_cells(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subset = frame[(frame.small_object_policy == POLICY) & np.isclose(frame.iou, .25)]
    for keys, g in subset.groupby(["scale_mode", "scale_threshold", "confidence"], sort=True):
        g = g.sort_values(["f1", "candidate"], ascending=[False, True], kind="stable").reset_index(drop=True)
        rows.append({
            "mode": keys[0], "threshold": float(keys[1]), "confidence": float(keys[2]),
            "winner": g.candidate.iloc[0], "margin": float(g.f1.iloc[0] - g.f1.iloc[1]),
        })
    return pd.DataFrame(rows)


def draw(ax, cells: pd.DataFrame, mode: str, panel: str, headline: float) -> None:
    subset = cells[cells["mode"].eq(mode)]
    xs = sorted(subset.threshold.unique())
    ys = sorted(subset.confidence.unique())
    index = {candidate: i for i, candidate in enumerate(CANDIDATES)}
    value = np.full((len(ys), len(xs)), np.nan)
    margin = np.full_like(value, np.nan, dtype=float)
    for r in subset.itertuples():
        value[ys.index(r.confidence), xs.index(r.threshold)] = index[r.winner]
        margin[ys.index(r.confidence), xs.index(r.threshold)] = r.margin
    ax.imshow(value, origin="lower", interpolation="nearest", aspect="auto",
              cmap=ListedColormap([COLORS[c] for c in CANDIDATES]), vmin=-.5, vmax=len(CANDIDATES)-.5)
    yy, xx = np.where(margin < .01)
    ax.scatter(xx, yy, marker="x", color="black", s=23, linewidths=.75, zorder=3)
    ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
    ax.set_yticks(range(len(ys)), [f"{y:.2f}" for y in ys])
    ax.set_xlabel("Absolute support threshold (px)" if mode == "absolute" else "Normalized support threshold")
    ax.set_ylabel("Confidence threshold")
    ax.set_title(panel, fontsize=8.2, pad=3)
    ax.tick_params(labelsize=6.7, length=2)
    if mode == "normalized":
        for label in ax.get_xticklabels():
            label.set_rotation(28)
            label.set_ha("right")
    ax.axvline(xs.index(headline), color="black", ls=":", lw=.9)
    ax.set_xticks(np.arange(-.5, len(xs), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ys), 1), minor=True)
    ax.grid(which="minor", color="white", lw=.62)
    ax.tick_params(which="minor", bottom=False, left=False)


def main() -> None:
    cells = rank_cells(pd.read_parquet(DATA))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.40), gridspec_kw={"wspace": .28})
    draw(axes[0], cells, "absolute", "(a) Absolute support", 24.)
    draw(axes[1], cells, "normalized", "(b) Normalized support", .015)
    handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS[c], markeredgecolor="none",
                      label=SHORT[c], markersize=6) for c in CANDIDATES]
    handles.append(Line2D([0], [0], marker="x", color="black", lw=0, label="top-two gap < .01", markersize=5))
    fig.legend(handles=handles, ncol=6, frameon=False, fontsize=6.25, loc="lower center",
               bbox_to_anchor=(.5, -.055), columnspacing=.72, handletextpad=.30)
    fig.subplots_adjust(left=.075, right=.995, top=.86, bottom=.27)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
