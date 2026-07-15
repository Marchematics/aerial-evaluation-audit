#!/usr/bin/env python3
"""Build the three-source A→F→R→S contract-path decomposition figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


ROOT = Path(__file__).resolve().parents[1]
ENDPOINTS = ROOT / "outputs" / "contract_path" / "contract_endpoints.csv"
LEDGER = ROOT / "outputs" / "contract_path" / "contract_transition_ledger.csv"
OUT = ROOT / "figures" / "fig_contract_path_decomposition.pdf"
SOURCES = ["VisDrone", "UAVDT", "AI-TOD"]
RULES = [("absolute", 24., "24 px"), ("normalized", .015, ".015")]
CONTRACTS = ["A", "F", "R", "S"]
COLORS = {"A": "#0072B2", "F": "#D55E00", "R": "#E69F00", "S": "#009E73"}


def main() -> None:
    endpoints = pd.read_csv(ENDPOINTS)
    ledger = pd.read_csv(LEDGER)
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 3.75), sharex=True, gridspec_kw={"hspace": .39, "wspace": .20})
    x = np.arange(4)
    for row, source in enumerate(SOURCES):
        source_values = endpoints[endpoints.source.eq(source)]
        row_min, row_max = source_values.f1.min(), source_values.f1.max()
        pad = max(.018, .16 * (row_max - row_min))
        for col, (mode, threshold, title) in enumerate(RULES):
            ax = axes[row, col]
            q = source_values[(source_values["mode"] == mode) & np.isclose(source_values.threshold, threshold)].set_index("contract").loc[CONTRACTS]
            y = q.f1.to_numpy(float)
            ax.plot(x, y, color=".30", lw=.9, zorder=1)
            for xi, contract, value in zip(x, CONTRACTS, y):
                ax.scatter(xi, value, s=24, color=COLORS[contract], edgecolor="white", lw=.45, zorder=2)
                vertical = 5 if np.isclose(value, y.min()) else (-9 if np.isclose(value, y.max()) else 5)
                ax.annotate(f"{value:.3f}", (xi, value), xytext=(0, vertical), textcoords="offset points",
                            ha="center", va="bottom" if vertical > 0 else "top", fontsize=5.7, color=".15")
            transitions = ledger[(ledger.source == source) & (ledger["mode"] == mode) & np.isclose(ledger.threshold, threshold)]
            for xi, transition in enumerate(["A->F", "F->R", "R->S"]):
                delta = float(transitions[transitions.transition.eq(transition)].iloc[0].delta_f1)
                mid = (y[xi] + y[xi + 1]) / 2
                ax.text(xi + .5, mid, f"{delta:+.3f}", ha="center", va="center", fontsize=5.4,
                        color=".20", bbox={"fc": "white", "ec": "none", "pad": .12, "alpha": .88})
            ax.set_ylim(row_min - pad, row_max + pad)
            ax.grid(axis="y", color=".89", lw=.45)
            if row == 0:
                ax.set_title(f"{title} support", fontsize=7.5, pad=3)
            if col == 0:
                ax.set_ylabel(f"{source}\nMicro-$F_1$", fontsize=6.5)
            if row == 2:
                ax.set_xticks(x, CONTRACTS)
                ax.set_xlabel("Registered contract path", fontsize=6.4)
            ax.tick_params(labelsize=5.9, length=2.0, pad=1.5)
            for spine in ax.spines.values(): spine.set_linewidth(.55)
    axes[0, 0].text(-.16, 1.13, "(a)", transform=axes[0, 0].transAxes, fontsize=7.2, fontweight="bold")
    axes[0, 1].text(-.10, 1.13, "(b)", transform=axes[0, 1].transAxes, fontsize=7.2, fontweight="bold")
    fig.text(.5, .005, "A: all valid; F: filtered/background; R: removed-object ignore; S: R + source ignore",
             ha="center", va="bottom", fontsize=6.2)
    fig.subplots_adjust(left=.10, right=.985, top=.94, bottom=.12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
