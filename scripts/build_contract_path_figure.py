#!/usr/bin/env python3
"""Build the five-candidate VisDrone A→F→R→S path figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

ROOT = Path(__file__).resolve().parents[1]
ENDPOINTS = ROOT / "outputs" / "visdrone_path_support_robustness" / "path_endpoints.csv"
OUT = ROOT / "figures" / "fig_contract_path_decomposition.pdf"
RULES = [("absolute_max_side", "24-pixel support"), ("normalized_max_side", "Normalized .015 support")]
CONTRACTS = ["A", "F", "R", "S"]
ORDER = ["B-640", "B-1280", "Y11m", "ASD", "Y11n-P2"]
COLORS = {"B-640": "#0072B2", "B-1280": "#56B4E9", "Y11m": "#009E73", "ASD": "#D55E00", "Y11n-P2": "#CC79A7"}
MARKERS = {"B-640": "o", "B-1280": "s", "Y11m": "D", "ASD": "^", "Y11n-P2": "v"}


def main() -> None:
    endpoints = pd.read_csv(ENDPOINTS)
    endpoints = endpoints[endpoints.primary.astype(bool)]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45), sharex=True, sharey=True,
                             gridspec_kw={"wspace": .14})
    x = np.arange(len(CONTRACTS))
    for panel, (ax, (rule_id, title)) in enumerate(zip(axes, RULES)):
        for candidate in ORDER:
            q = endpoints[
                endpoints.rule_id.eq(rule_id) & endpoints.short_candidate.eq(candidate)
            ].set_index("contract").loc[CONTRACTS]
            ax.plot(x, q.f1, color=COLORS[candidate], marker=MARKERS[candidate],
                    ms=4.0, lw=1.15, label=candidate)
        ax.set_title(f"({chr(97 + panel)}) {title}", fontsize=7.8, loc="left", pad=3)
        ax.set_xticks(x, CONTRACTS)
        ax.set_xlabel("Declared contract path")
        ax.grid(color=".89", lw=.45)
        ax.tick_params(labelsize=6.2, length=2.0, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(.55)
    axes[0].set_ylabel("Micro-$F_1$ at confidence and IoU .25")
    axes[0].set_ylim(.68, .915)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.1, ncol=5,
               loc="upper center", bbox_to_anchor=(.52, 1.01), handlelength=1.6,
               columnspacing=1.1)
    fig.text(.5, .01,
             "A: all valid; F: filtered/background; R: removed-object neutral; S: R + source ignore",
             ha="center", va="bottom", fontsize=6.2)
    fig.subplots_adjust(left=.09, right=.985, top=.82, bottom=.22)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
