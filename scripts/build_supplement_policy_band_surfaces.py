#!/usr/bin/env python3
"""Build full fixed-rule F1 policy-band surfaces for the supplement."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "resubmission_controlled_pool" / "visdrone_grid_crowd_v4" / "metrics_long.parquet"
OUT = ROOT / "figures" / "fig_supplement_policy_band_surfaces.pdf"
POLICIES = ["include_all", "exclude_source_ignore_off", "exclude_source_ignore_on"]
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


def bands(frame: pd.DataFrame) -> pd.DataFrame:
    q = frame[np.isclose(frame.iou, .25) & frame.small_object_policy.isin(POLICIES)]
    out = (q.groupby(["candidate", "scale_mode", "scale_threshold", "confidence"], as_index=False)
             .agg(policy_band=("f1", lambda x: float(x.max() - x.min()))))
    return out


def panel(ax, table: pd.DataFrame, candidate: str, mode: str, *, show_y: bool, headline: float) -> plt.AxesImage:
    q = table[table.candidate.eq(candidate) & table.scale_mode.eq(mode)]
    xs = sorted(q.scale_threshold.unique())
    ys = sorted(q.confidence.unique())
    grid = np.full((len(ys), len(xs)), np.nan)
    for r in q.itertuples():
        grid[ys.index(r.confidence), xs.index(r.scale_threshold)] = r.policy_band
    im = ax.imshow(grid, origin="lower", interpolation="nearest", aspect="auto", cmap="cividis", vmin=0, vmax=.45)
    ax.axvline(xs.index(headline), color="white", ls=":", lw=.85)
    xt = [0, 2, 5, len(xs) - 1] if mode == "absolute" else [0, 3, len(xs) - 1]
    xt = sorted(set(xt))
    ax.set_xticks(xt, [f"{xs[i]:g}" for i in xt])
    if show_y:
        ax.set_yticks(range(len(ys)), [f"{y:.2f}" for y in ys])
        ax.set_ylabel("confidence", fontsize=6.0, labelpad=1)
    else:
        ax.set_yticks([])
    ax.tick_params(labelsize=5.25, length=1.8, pad=1)
    ax.set_xticks(np.arange(-.5, len(xs), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ys), 1), minor=True)
    ax.grid(which="minor", color="white", lw=.35, alpha=.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def main() -> None:
    table = bands(pd.read_parquet(DATA))
    fig = plt.figure(figsize=(7.16, 3.48))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, .045],
                          left=.085, right=.975, top=.91, bottom=.14, wspace=.19, hspace=.34)
    axes = np.empty((2, 5), dtype=object)
    for col, candidate in enumerate(CANDIDATES):
        axes[0, col] = fig.add_subplot(gs[0, col])
        axes[1, col] = fig.add_subplot(gs[1, col])
        axes[0, col].set_title(SHORT[candidate], fontsize=6.8, pad=2.2)
        panel(axes[0, col], table, candidate, "absolute", show_y=(col == 0), headline=24.)
        im = panel(axes[1, col], table, candidate, "normalized", show_y=(col == 0), headline=.015)
    fig.text(.012, .745, "(a) absolute", rotation=90, va="center", ha="center", fontsize=6.4)
    fig.text(.012, .288, "(b) normalized", rotation=90, va="center", ha="center", fontsize=6.4)
    cbar = fig.colorbar(im, cax=fig.add_subplot(gs[:, 5]))
    cbar.ax.tick_params(labelsize=5.6, length=1.8)
    cbar.set_label(r"$B_{F_1}^{\pi}$", fontsize=6.1, labelpad=2)
    fig.text(.50, .012, "support threshold (dotted line = headline)", ha="center", fontsize=6.15)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
