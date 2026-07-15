#!/usr/bin/env python3
"""Build supplementary coverage-selection, cap, and structural-distance controls."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "outputs" / "coverage_qualification" / "aitod_image_features.parquet"
SCALE = ROOT / "outputs" / "scale" / "box_scale_records.parquet"
CAP = ROOT / "outputs" / "cap_sufficiency" / "cap_sufficiency_table.csv"
OUT_CONTROL = ROOT / "figures" / "fig_supplement_coverage_cap.pdf"
OUT_DISTANCE = ROOT / "figures" / "fig_supplement_structural_distance.pdf"
OUT_CONTRACT = ROOT / "figures" / "fig_supplement_contract_summary.pdf"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    return values, np.arange(1, len(values) + 1, dtype=float) / len(values)


def finish_axis(ax: plt.Axes) -> None:
    ax.grid(color=".89", lw=.45)
    ax.tick_params(labelsize=5.8, length=2.0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(.55)


def build_coverage_cap() -> None:
    images = pd.read_parquet(FEATURES)
    scale = pd.read_parquet(SCALE)
    ai_boxes = scale[scale.source.eq("aitod")].copy()
    group_map = images.set_index("image_name").coverage_group
    ai_boxes["coverage_group"] = ai_boxes.file_name.map(group_map)
    if ai_boxes.coverage_group.isna().any():
        raise RuntimeError("AI-TOD box-to-coverage mapping is incomplete")

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.35), gridspec_kw={"wspace": .39})
    labels = [("covered", "Covered", GREEN, "-"), ("not_covered", "Noncovered", ORANGE, "--")]

    ax = axes[0]
    for key, label, color, ls in labels:
        x, y = ecdf(images.loc[images.coverage_group.eq(key), "vehicle_boxes"].to_numpy())
        ax.plot(x, y, color=color, ls=ls, lw=1.0, label=label)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 10, 100, 300], ["0", "1", "10", "100", "300"])
    ax.set_xlabel("Vehicle boxes/image")
    ax.set_ylabel("ECDF")
    ax.set_title("(a) Image density", fontsize=7.4, loc="left", pad=3)
    ax.legend(frameon=False, fontsize=5.4, loc="lower right", handlelength=1.4)
    finish_axis(ax)

    for ax, field, headline, title, xlabel in (
        (axes[1], "max_side_px", 24., "(b) Absolute support", "Max side (px)"),
        (axes[2], "normalized_side", .015, "(c) Normalized support", "Normalized max side"),
    ):
        for key, label, color, ls in labels:
            x, y = ecdf(ai_boxes.loc[ai_boxes.coverage_group.eq(key), field].to_numpy())
            ax.plot(x, y, color=color, ls=ls, lw=1.0, label=label)
        ax.axvline(headline, color=".20", lw=.7, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=7.4, loc="left", pad=3)
        finish_axis(ax)

    cap = pd.read_csv(CAP)
    cap = cap[(cap.cohort == "resubmission_controlled_pool") & cap.source.eq("visdrone")].copy()
    short = {
        "visdrone_control_baseline640": "B-640",
        "visdrone_control_baseline1280": "B-1280",
        "visdrone_control_yolo11m_hf640": "Y11m",
        "visdrone_control_asd1280": "ASD",
        "visdrone_control_yolo11n_p2_1280": "Y11n-P2",
    }
    cap["short"] = cap.candidate.map(short)
    cap = cap.set_index("short").loc[list(short.values())].reset_index()
    ax = axes[3]
    y = np.arange(len(cap))
    h = .32
    ax.barh(y - h / 2, 100 * cap.fraction_truncated_at_100, height=h, color="#E6615F", label="100")
    ax.barh(y + h / 2, 100 * cap.fraction_truncated_at_300, height=h, color="#E6A300", label="300")
    ax.set_yticks(y, cap.short)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Images truncated (%)")
    ax.set_title("(d) AP-cap truncation", fontsize=7.4, loc="left", pad=3)
    ax.legend(frameon=False, fontsize=5.2, loc="upper right", bbox_to_anchor=(1.0, 1.035),
              ncol=2, handlelength=1.0, columnspacing=.7)
    finish_axis(ax)

    fig.subplots_adjust(left=.065, right=.995, top=.88, bottom=.24)
    OUT_CONTROL.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_CONTROL, bbox_inches="tight")
    fig.savefig(OUT_CONTROL.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def build_structural_distance() -> None:
    scale = pd.read_parquet(SCALE, columns=["source", "max_side_px", "normalized_side"])
    sources = ["visdrone", "uavdt", "aitod", "dota_v2_val"]
    labels = ["VisDrone", "UAVDT", "AI-TOD", "DOTA-v2"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55), gridspec_kw={"wspace": .31})
    for ax, field, title, fmt in (
        (axes[0], "max_side_px", "(a) Absolute max-side $W_1$ (px)", ".1f"),
        (axes[1], "normalized_side", "(b) Normalized max-side $W_1$", ".3f"),
    ):
        matrix = np.zeros((len(sources), len(sources)), dtype=float)
        for i, left in enumerate(sources):
            a = scale.loc[scale.source.eq(left), field].to_numpy(float)
            for j, right in enumerate(sources):
                b = scale.loc[scale.source.eq(right), field].to_numpy(float)
                matrix[i, j] = wasserstein_distance(a, b)
        image = ax.imshow(matrix, cmap="Blues", vmin=0)
        cutoff = matrix.max() * .57
        for i in range(len(sources)):
            for j in range(len(sources)):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center",
                        fontsize=6.0, color="white" if matrix[i, j] > cutoff else ".18")
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(title, fontsize=7.5, loc="left", pad=3)
        ax.tick_params(labelsize=6.0, length=0, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(.55)
        cb = fig.colorbar(image, ax=ax, fraction=.046, pad=.025)
        cb.ax.tick_params(labelsize=5.4, length=2)
    fig.subplots_adjust(left=.08, right=.98, top=.91, bottom=.22)
    fig.savefig(OUT_DISTANCE, bbox_inches="tight")
    fig.savefig(OUT_DISTANCE.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def build_contract_summary() -> None:
    ranges = pd.read_csv(ROOT / "outputs" / "contract_path" / "contract_ranges.csv")
    ledger = pd.read_csv(ROOT / "outputs" / "contract_path" / "contract_transition_ledger.csv")
    sources = ["VisDrone", "UAVDT", "AI-TOD"]
    fig, (ax_range, ax_step) = plt.subplots(1, 2, figsize=(7.16, 2.25),
                                            gridspec_kw={"width_ratios": [.78, 1.42], "wspace": .28})

    x = np.arange(len(sources))
    width = .34
    absolute = [ranges[(ranges.source == source) & ranges["mode"].eq("absolute")].iloc[0].contract_range
                for source in sources]
    normalized = [ranges[(ranges.source == source) & ranges["mode"].eq("normalized")].iloc[0].contract_range
                  for source in sources]
    bars_a = ax_range.bar(x - width / 2, absolute, width, color=BLUE, label="24 px")
    bars_n = ax_range.bar(x + width / 2, normalized, width, color=ORANGE, hatch="//", label=".015")
    for bars in (bars_a, bars_n):
        for bar in bars:
            ax_range.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .007,
                          f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=5.2)
    ax_range.set_xticks(x, sources)
    ax_range.set_ylabel("Four-contract $F_1$ range")
    ax_range.set_ylim(0, .38)
    ax_range.set_title("(a) Contract sensitivity", fontsize=7.4, loc="left", pad=3)
    ax_range.legend(frameon=False, fontsize=5.3, ncol=2, loc="upper left")
    finish_axis(ax_range)

    settings = [(source, mode) for source in sources for mode in ("absolute", "normalized")]
    labels = [f"{source}\n{'24 px' if mode == 'absolute' else '.015'}" for source, mode in settings]
    steps = [("A->F", "Target", BLUE), ("F->R", "Removed", ORANGE), ("R->S", "Source", GREEN)]
    x = np.arange(len(settings))
    width = .23
    for offset, (transition, label, color) in zip((-width, 0, width), steps):
        values = []
        for source, mode in settings:
            row = ledger[(ledger.source == source) & ledger["mode"].eq(mode) & ledger.transition.eq(transition)].iloc[0]
            values.append(float(row.delta_f1))
        ax_step.bar(x + offset, values, width, color=color, label=label)
    ax_step.axhline(0, color=".25", lw=.65)
    ax_step.set_xticks(x, labels)
    ax_step.set_ylabel(r"Adjacent $\Delta F_1$")
    ax_step.set_title("(b) Ordered path components", fontsize=7.4, loc="left", pad=3)
    ax_step.legend(frameon=False, fontsize=5.3, ncol=3, loc="lower left",
                   handlelength=1.0, columnspacing=.8)
    finish_axis(ax_step)

    fig.subplots_adjust(left=.075, right=.99, top=.88, bottom=.25)
    fig.savefig(OUT_CONTRACT, bbox_inches="tight")
    fig.savefig(OUT_CONTRACT.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    build_coverage_cap()
    build_structural_distance()
    build_contract_summary()


if __name__ == "__main__":
    main()
