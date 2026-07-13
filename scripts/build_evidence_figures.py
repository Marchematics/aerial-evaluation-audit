"""Build the evidence-centered figures used by the GRSL paper and supplement.

Every plotted quantity is read from a released derived artifact.  The script does
not read licensed imagery or prediction caches and does not recompute metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

COLORS = {
    "visdrone": "#0072B2",
    "uavdt": "#D55E00",
    "aitod": "#009E73",
    "dota_v2_val": "#6F6F6F",
}
LABELS = {
    "visdrone": "VisDrone",
    "uavdt": "Local UAVDT",
    "aitod": "AI-TOD",
    "dota_v2_val": "DOTA-v2 HBB",
}
MARKERS = {"visdrone": "o", "uavdt": "s", "aitod": "^"}
LINESTYLES = {"visdrone": "-", "uavdt": "--", "aitod": "-.", "dota_v2_val": ":"}
POLICIES = [
    "include_all",
    "exclude_without_ignore_protection",
    "exclude_with_ignore_protection",
]
POLICY_SHORT = {
    "include_all": "I",
    "exclude_without_ignore_protection": "O",
    "exclude_with_ignore_protection": "N",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
        }
    )


def clean_axis(ax: mpl.axes.Axes, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#D7D7D7", linewidth=0.45, alpha=0.75, zorder=0)
    ax.tick_params(direction="out", pad=1.5)


def panel_label(ax: mpl.axes.Axes, text: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8.0, fontweight="bold", va="bottom")


def save(fig: mpl.figure.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=400)
    plt.close(fig)


def ecdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(list(values), dtype=float))
    return x, np.arange(1, len(x) + 1, dtype=float) / len(x)


def fig2_dataset_support() -> None:
    records = pd.read_parquet(ROOT / "outputs/scale/box_scale_records.parquet")
    coverage = pd.read_csv(ROOT / "outputs/coverage/source_common_intersection.csv").set_index("source")
    source_cards = pd.read_csv(ROOT / "outputs/structural/source_cards.csv").set_index("dataset_name")
    order = ["visdrone", "uavdt", "aitod", "dota_v2_val"]

    fig = plt.figure(figsize=(7.15, 2.42))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.90, 1.16, 1.16, 1.22], wspace=0.48)
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]

    # (a) Coverage denominator. DOTA-v2 has no prediction artifact by design.
    sources = ["VisDrone", "Local UAVDT", "AI-TOD", "DOTA-v2"]
    numerators = np.array([
        coverage.loc["visdrone", "common_prediction_covered_images"],
        coverage.loc["uavdt", "common_prediction_covered_images"],
        coverage.loc["aitod", "common_prediction_covered_images"],
        source_cards.loc["dota_v2_val", "image_count"],
    ], dtype=float)
    denominators = np.array([
        coverage.loc["visdrone", "source_evaluation_images"],
        coverage.loc["uavdt", "source_evaluation_images"],
        coverage.loc["aitod", "source_evaluation_images"],
        source_cards.loc["dota_v2_val", "image_count"],
    ], dtype=float)
    fractions = 100 * numerators / denominators
    colors = [COLORS[x] for x in order]
    y = np.arange(4)[::-1]
    axs[0].barh(y, np.full(4, 100.0), color="#ECECEC", edgecolor="#B8B8B8", linewidth=0.5, zorder=1)
    bars = axs[0].barh(y, fractions, color=colors, edgecolor="#3A3A3A", linewidth=0.4, zorder=2)
    bars[-1].set_facecolor("white")
    bars[-1].set_hatch("///")
    bars[-1].set_edgecolor(COLORS["dota_v2_val"])
    coverage_text = [
        f"{int(numerators[0]):,}/{int(denominators[0]):,}",
        f"{int(numerators[1]):,}/{int(denominators[1]):,}",
        f"{int(numerators[2]):,}/{int(denominators[2]):,}",
        f"{int(numerators[3]):,}; structural-only",
    ]
    for index, (bar, text, frac) in enumerate(zip(bars, coverage_text, fractions)):
        xpos = min(frac - 2.0, 97.0)
        text_color = "white" if index < 3 and frac > 45 else "#333333"
        axs[0].text(xpos, bar.get_y() + bar.get_height() / 2, text,
                    ha="right", va="center", fontsize=5.5, color=text_color)
    axs[0].set_yticks(y, sources)
    axs[0].set_xlim(0, 103)
    axs[0].set_xlabel("Covered / declared images (%)")
    clean_axis(axs[0], "x")
    panel_label(axs[0], "(a)", -0.18)

    retained_abs = {
        source: 100 * np.mean(records.loc[records.source.eq(source), "max_side_px"].to_numpy(float) >= 24)
        for source in ["visdrone", "uavdt", "aitod"]
    }
    retained_norm = {
        source: 100 * np.mean(records.loc[records.source.eq(source), "normalized_side"].to_numpy(float) >= 0.015)
        for source in ["visdrone", "uavdt", "aitod"]
    }
    for source in order:
        q = records.loc[records.source.eq(source), "max_side_px"]
        ex, ey = ecdf(q)
        axs[1].plot(ex, ey, color=COLORS[source], ls=LINESTYLES[source], label=LABELS[source])
    axs[1].axvline(24, color="#222222", ls="--", lw=0.8)
    axs[1].annotate("24 px", xy=(24, 0.985), xycoords=("data", "axes fraction"),
                    xytext=(-3, -2), textcoords="offset points", ha="right", va="top", fontsize=5.8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.45))
    abs_positions = {"visdrone": (68, 0.31), "uavdt": (30, 0.76), "aitod": (13, 0.42)}
    for source, (tx, ty) in abs_positions.items():
        axs[1].text(tx, ty, f"{LABELS[source]} {retained_abs[source]:.1f}%",
                    color=COLORS[source], fontsize=5.5, ha="left", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.4))
    axs[1].set(xscale="log", xlim=(2.5, 650), ylim=(0, 1.01), xlabel="Absolute max side (px)", ylabel="ECDF")
    clean_axis(axs[1], "both")
    panel_label(axs[1], "(b)")

    for source in order:
        q = records.loc[records.source.eq(source), "normalized_side"]
        ex, ey = ecdf(q)
        axs[2].plot(ex, ey, color=COLORS[source], ls=LINESTYLES[source], label=LABELS[source])
    axs[2].axvline(0.015, color="#222222", ls="--", lw=0.8)
    axs[2].annotate(".015", xy=(0.015, 0.985), xycoords=("data", "axes fraction"),
                    xytext=(-3, -2), textcoords="offset points", ha="right", va="top", fontsize=5.8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.45))
    norm_positions = {"visdrone": (0.052, 0.25), "uavdt": (0.027, 0.55), "aitod": (0.0062, 0.72)}
    for source, (tx, ty) in norm_positions.items():
        axs[2].text(tx, ty, f"{LABELS[source]} {retained_norm[source]:.1f}%",
                    color=COLORS[source], fontsize=5.5, ha="left", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.4))
    axs[2].set(xscale="log", xlim=(0.001, 0.5), ylim=(0, 1.01), xlabel="Normalized max side", ylabel="ECDF")
    clean_axis(axs[2], "both")
    panel_label(axs[2], "(c)")

    # (d) Retention is a direct empirical survival curve, independent of evaluator policy.
    abs_thresholds = np.array([16, 20, 24, 28, 32, 40, 48, 64], dtype=float)
    for source in ["visdrone", "uavdt", "aitod"]:
        values = records.loc[records.source.eq(source), "max_side_px"].to_numpy(float)
        retention = [100 * np.mean(values >= threshold) for threshold in abs_thresholds]
        axs[3].plot(abs_thresholds, retention, marker=MARKERS[source], ms=2.8,
                    color=COLORS[source], ls=LINESTYLES[source], label=LABELS[source])
    axs[3].axvline(24, color="#222222", ls="--", lw=0.7)
    axs[3].set(xlabel="Absolute threshold (px)", ylabel="Retained GT (%)", ylim=(-2, 103), xticks=abs_thresholds)
    axs[3].tick_params(axis="x", labelrotation=35)
    clean_axis(axs[3], "both")
    panel_label(axs[3], "(d)")

    handles = [Line2D([0], [0], color=COLORS[s], ls=LINESTYLES[s], lw=1.2, label=LABELS[s]) for s in order]
    fig.legend(handles=handles, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01),
               columnspacing=1.15, handlelength=2.0)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.19, top=0.86)
    save(fig, "fig2_dataset_support_coverage")


def policy_band_surface(mode: str, source: str, candidate: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    threshold_col = "scale_threshold_px" if mode == "absolute" else "scale_threshold_norm"
    data = pd.read_parquet(ROOT / f"outputs/coverage_corrected_grid/{mode}/metrics_long.parquet")
    data = data[
        data.source.eq(source)
        & data.candidate.eq(candidate)
        & np.isclose(data.iou, 0.25)
        & data.small_object_policy.isin(POLICIES)
    ].copy()
    grouped = (
        data.groupby(["confidence", threshold_col], as_index=False)["f1"]
        .agg(lambda values: float(np.max(values) - np.min(values)))
        .rename(columns={"f1": "band"})
    )
    thresholds = np.sort(grouped[threshold_col].unique())
    confidences = np.sort(grouped.confidence.unique())
    surface = grouped.pivot(index="confidence", columns=threshold_col, values="band").reindex(index=confidences, columns=thresholds)
    if surface.isna().any().any():
        raise ValueError(f"Incomplete surface for {mode}/{source}/{candidate}")
    return thresholds, confidences, surface.to_numpy(float)


def fig3_f1_surfaces() -> None:
    columns = [
        ("visdrone", "visdrone_sahi640", "VisDrone--SAHI"),
        ("uavdt", "uavdt_tiling", "Local UAVDT--Tiling"),
        ("aitod", "aitod_baseline640", "AI-TOD--Base"),
    ]
    modes = [("absolute", "Absolute support"), ("normalized", "Normalized support")]
    cached = {(mode, source): policy_band_surface(mode, source, candidate)
              for mode, _ in modes for source, candidate, _ in columns}
    vmax = max(np.max(values[2]) for values in cached.values())
    fig, axs = plt.subplots(2, 3, figsize=(7.15, 3.02), sharey=True)
    image = None
    for row, (mode, row_label) in enumerate(modes):
        for col, (source, _candidate, col_label) in enumerate(columns):
            ax = axs[row, col]
            thresholds, confidences, values = cached[(mode, source)]
            image = ax.imshow(values, origin="lower", aspect="auto", vmin=0, vmax=vmax,
                              cmap="magma_r", interpolation="nearest")
            ax.set_xticks(np.arange(len(thresholds)))
            if mode == "absolute":
                ax.set_xticklabels([f"{int(x)}" for x in thresholds], rotation=35, ha="right")
                xlabel = "Threshold (px)"
            else:
                ax.set_xticklabels([f"{x:.4f}".rstrip("0") for x in thresholds], rotation=35, ha="right")
                xlabel = "Normalized threshold"
            ax.set_yticks(np.arange(len(confidences)))
            ax.set_yticklabels([f"{x:.2f}" for x in confidences])
            ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(f"{row_label}\nConfidence")
            ax.tick_params(length=0, pad=1.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.45)
                spine.set_color("#555555")
            if row == 0:
                ax.text(0.5, 1.06, col_label, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.0)
            panel_label(ax, f"({chr(ord('a') + row * 3 + col)})", -0.10, 1.02)
    cax = fig.add_axes([0.935, 0.18, 0.012, 0.68])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(r"$F_1$ policy band")
    cbar.ax.tick_params(labelsize=6.2, width=0.5, length=2)
    fig.subplots_adjust(left=0.078, right=0.91, bottom=0.14, top=0.91, wspace=0.24, hspace=0.47)
    save(fig, "fig3_f1_policy_surfaces")


def _complete_uavdt_ap_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Show six nominal policies; N duplicates O for the registered UAVDT artifact."""
    data = data.copy()
    for mode in ["absolute", "normalized"]:
        if not ((data.source.eq("uavdt")) & data["mode"].eq(mode) & data.policy.eq(POLICIES[2])).any():
            duplicate = data[data.source.eq("uavdt") & data["mode"].eq(mode) & data.policy.eq(POLICIES[1])].copy()
            duplicate["policy"] = POLICIES[2]
            data = pd.concat([data, duplicate], ignore_index=True)
    return data


def _forest(ax: mpl.axes.Axes, data: pd.DataFrame, metric: str, source: str, letter: str) -> None:
    if metric == "AP50":
        point = "point_ap50_difference"
        data = _complete_uavdt_ap_rows(data)
    else:
        point = "point_f1_difference"
        data = data[np.isclose(data.iou, 0.25)].copy()
    data = data[data.source.eq(source)].copy()
    mode_order = {"absolute": 0, "normalized": 1}
    policy_order = {p: i for i, p in enumerate(POLICIES)}
    data["_mode"] = data["mode"].map(mode_order)
    data["_policy"] = data.policy.map(policy_order)
    data = data.sort_values(["_mode", "_policy"])
    labels = [f"{'Abs' if m == 'absolute' else 'Norm'}-{POLICY_SHORT[p]}" for m, p in zip(data["mode"], data.policy)]
    ypos = np.arange(len(data))[::-1]
    vals = data[point].to_numpy(float)
    low = data.ci95_low.to_numpy(float)
    high = data.ci95_high.to_numpy(float)
    color = COLORS[source]
    ax.errorbar(vals, ypos, xerr=np.vstack([vals - low, high - vals]), fmt=MARKERS[source], ms=3.0,
                color=color, ecolor=color, elinewidth=0.8, capsize=1.7, markeredgecolor="#222222", markeredgewidth=0.25)
    ax.axvline(0, color="#333333", ls="--", lw=0.75)
    ax.set_yticks(ypos, labels)
    ax.set_ylim(-0.7, len(data) - 0.3)
    ax.set_xlabel("Winner $-$ runner-up")
    clean_axis(ax, "x")
    metric_label = r"AP$_{50}$" if metric == "AP50" else r"$F_1$"
    ax.text(0.02, 0.98, f"({letter}) {LABELS[source]} {metric_label}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.6, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.8))


def fig4_rank_uncertainty() -> None:
    robust = pd.read_csv(ROOT / "outputs/statistics/pairwise_policy_robustness.csv")
    ap = pd.read_csv(ROOT / "outputs/statistics/bootstrap_headline_ap50_common_coverage.csv")
    f1 = pd.read_csv(ROOT / "outputs/statistics/bootstrap_headline_f1_common_coverage.csv")

    fig = plt.figure(figsize=(7.15, 3.25))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.18, 1, 1], hspace=0.55, wspace=0.55)
    ax0 = fig.add_subplot(gs[:, 0])
    metric_markers = {"AP50": "o", "F1": "s"}
    for _, row in robust.iterrows():
        ax0.scatter(row.reference_margin, row.differential_radius, s=30,
                    marker=metric_markers[row.metric], color=COLORS[row.source],
                    edgecolor="#222222", linewidth=0.4, zorder=3)
        text = f"{LABELS[row.source].replace('Local ', '')} {row.metric.replace('AP50', 'AP50')}"
        if row.source == "uavdt" and row.metric == "AP50":
            offset, align = (-5, 5), "right"
        elif row.source == "visdrone" and row.metric == "AP50":
            offset, align = (-5, 5), "right"
        else:
            offset, align = (5, 2), "left"
        ax0.annotate(text, (row.reference_margin, row.differential_radius), xytext=offset,
                     textcoords="offset points", fontsize=5.6, ha=align, va="bottom")
    lower, upper = 5e-5, 0.5
    ax0.plot([lower, upper], [lower, upper], color="#333333", ls="--", lw=0.8)
    ax0.text(0.11, 0.13, r"$Gamma=\Delta_0$", rotation=45, fontsize=5.8, color="#333333")
    ax0.set(xscale="log", yscale="log", xlim=(lower, upper), ylim=(lower, upper),
            xlabel=r"Reference margin $\Delta_0$", ylabel=r"Differential radius $\Gamma$")
    ax0.set_aspect("equal", adjustable="box")
    clean_axis(ax0, "both")
    panel_label(ax0, "(a)", -0.14, 1.01)

    _forest(fig.add_subplot(gs[0, 1]), ap, "AP50", "visdrone", "b")
    _forest(fig.add_subplot(gs[0, 2]), ap, "AP50", "uavdt", "c")
    _forest(fig.add_subplot(gs[1, 1]), f1, "F1", "visdrone", "d")
    _forest(fig.add_subplot(gs[1, 2]), f1, "F1", "uavdt", "e")
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.14, top=0.97)
    save(fig, "fig4_rank_robustness_uncertainty")


def fig5_controls() -> None:
    matching = pd.read_csv(ROOT / "outputs/matching_headline/greedy_hungarian_agreement.csv")
    trunc = pd.read_csv(ROOT / "outputs/statistics/ap_maxdets_truncation_audit.csv")
    boot100 = pd.read_csv(ROOT / "outputs/statistics/bootstrap_headline_ap50_common_coverage.csv")
    cap300 = pd.read_csv(ROOT / "outputs/statistics/ap50_maxdets_300_sensitivity.csv")
    cap2000 = pd.read_csv(ROOT / "outputs/statistics/ap50_maxdets_2000_sensitivity.csv")
    selection = pd.read_csv(ROOT / "outputs/coverage/aitod_coverage_selection_audit.csv").set_index("group")

    fig = plt.figure(figsize=(7.15, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.18, 1.05], wspace=0.47)

    # (a) Marker shape encodes source; fill encodes IoU.
    ax = fig.add_subplot(gs[0, 0])
    for source in ["visdrone", "uavdt", "aitod"]:
        for iou, fill in [(0.25, COLORS[source]), (0.50, "none")]:
            q = matching[matching.source.eq(source) & np.isclose(matching.iou, iou)]
            ax.scatter(q.f1_greedy, q.f1_hungarian, s=14, marker=MARKERS[source],
                       facecolors=fill, edgecolors=COLORS[source], linewidth=0.65, alpha=0.8)
    lo = min(matching.f1_greedy.min(), matching.f1_hungarian.min()) - 0.01
    hi = max(matching.f1_greedy.max(), matching.f1_hungarian.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], color="#333333", ls="--", lw=0.75)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Greedy $F_1$", ylabel="Hungarian $F_1$")
    ax.text(0.03, 0.97, "max $|\\Delta F_1|$ = .00546\nwinner agreement = 24/24",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.8,
            bbox=dict(facecolor="white", edgecolor="#BBBBBB", linewidth=0.35, alpha=0.88, pad=1.2))
    clean_axis(ax, "both")
    panel_label(ax, "(a)", -0.16)
    source_handles = [Line2D([0], [0], marker=MARKERS[s], color="none", markerfacecolor=COLORS[s],
                             markeredgecolor=COLORS[s], label=LABELS[s], markersize=4.5) for s in ["visdrone", "uavdt", "aitod"]]
    iou_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markeredgecolor="#555555", label="IoU .25", markersize=4.5),
                   Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#555555", label="IoU .50", markersize=4.5)]
    ax.legend(handles=source_handles + iou_handles, frameon=False, ncol=2, loc="lower right",
              handletextpad=0.25, columnspacing=0.6, borderaxespad=0.2, fontsize=5.2)

    # (b) Include-all, absolute-24-px reference: cap changes both truncation and AP gap.
    ax = fig.add_subplot(gs[0, 1])
    ax2 = ax.twinx()
    cap_files = {100: boot100, 300: cap300, 2000: cap2000}
    winner = {"visdrone": "visdrone_sahi640", "uavdt": "uavdt_tiling"}
    caps = np.array([100, 300, 2000])
    for source in ["visdrone", "uavdt"]:
        gaps = []
        for cap in caps:
            q = cap_files[int(cap)]
            q = q[q.source.eq(source) & q["mode"].eq("absolute") & np.isclose(q.threshold, 24)
                  & q.policy.eq("include_all")]
            gaps.append(float(q.point_ap50_difference.iloc[0]))
        tq = trunc[trunc.source.eq(source) & trunc.candidate.eq(winner[source]) & trunc.policy.eq("include_all")]
        tq = tq.set_index("max_dets").reindex(caps)
        truncated_pct = 100 * tq.images_truncated.to_numpy(float) / tq.images.to_numpy(float)
        ax.plot(caps, gaps, color=COLORS[source], marker=MARKERS[source], ms=3.3, label=LABELS[source])
        ax2.plot(caps, truncated_pct, color=COLORS[source], marker=MARKERS[source], ms=3.0,
                 ls=":", alpha=0.82)
    ax.set_xscale("log")
    ax.set_xticks(caps, ["100", "300", "2000"])
    ax.set_xlabel("maxDets")
    ax.set_ylabel(r"Pairwise AP$_{50}$ gap")
    ax2.set_ylabel("Winner images truncated (%)")
    ax.set_ylim(0, 0.39)
    ax2.set_ylim(-3, 98)
    clean_axis(ax, "both")
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(direction="out", pad=1.5, labelsize=6.3, width=0.6, length=2.5)
    panel_label(ax, "(b)", -0.16)
    style_handles = [Line2D([0], [0], color="#333333", lw=1.0, ls="-", label="AP gap (left)"),
                     Line2D([0], [0], color="#333333", lw=1.0, ls=":", label="truncated (right)")]
    ax.legend(handles=[Line2D([0], [0], color=COLORS[s], marker=MARKERS[s], label=LABELS[s], ms=3.5) for s in ["visdrone", "uavdt"]] + style_handles,
              frameon=False, loc="lower center", bbox_to_anchor=(0.52, 1.01), ncol=2,
              fontsize=5.1, handlelength=1.8, borderaxespad=0.1, columnspacing=0.8)

    # (c) A compact evidence table avoids implying comparability of unlike quantities.
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    covered = selection.loc["covered"]
    noncovered = selection.loc["not_covered"]
    cell_text = [
        ["Images", f"{int(covered.images):,}", f"{int(noncovered.images):,}"],
        ["Vehicle boxes", f"{int(covered.vehicle_boxes):,}", f"{int(noncovered.vehicle_boxes):,}"],
        ["Boxes/image", f"{covered.boxes_per_image:.1f}", f"{noncovered.boxes_per_image:.1f}"],
        ["Median side", f"{covered.median_max_side_px:.0f} px", f"{noncovered.median_max_side_px:.0f} px"],
    ]
    table = ax.table(cellText=cell_text, colLabels=["AI-TOD", "Covered", "Noncovered"],
                     cellLoc="center", colLoc="center", loc="center", colWidths=[0.44, 0.31, 0.34])
    table.auto_set_font_size(False)
    table.set_fontsize(6.2)
    table.scale(1.0, 1.45)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#AFAFAF")
        if row == 0:
            cell.set_facecolor("#DDEFE9" if col > 0 else "#E7E7E7")
            cell.set_text_props(fontweight="bold")
        elif col == 0:
            cell.set_facecolor("#F1F1F1")
            cell.set_text_props(ha="left")
        elif col == 1:
            cell.set_facecolor("#F1F8F6")
        else:
            cell.set_facecolor("#FFF5EC")
    ax.text(0.5, 0.12, "Coverage fraction: 66.7%", transform=ax.transAxes,
            ha="center", va="center", fontsize=6.2, color="#333333")
    panel_label(ax, "(c)", -0.08, 0.99)

    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.17, top=0.96)
    save(fig, "fig5_controls_boundary_conditions")


def fig_s1_wasserstein() -> None:
    data = pd.read_csv(ROOT / "outputs/structural/pairwise_compatibility.csv")
    order = ["visdrone", "uavdt", "aitod", "dota_v2_val"]
    matrix = np.zeros((4, 4), dtype=float)
    for i, source_a in enumerate(order):
        for j, source_b in enumerate(order):
            q = data[data.source_a.eq(source_a) & data.source_b.eq(source_b)]
            matrix[i, j] = float(q.vehicle_scale_wasserstein.iloc[0])
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    labels = [LABELS[s] for s in order]
    ax.set_xticks(range(4), labels, rotation=30, ha="right")
    ax.set_yticks(range(4), labels)
    for i in range(4):
        for j in range(4):
            color = "white" if matrix[i, j] > 0.55 * matrix.max() else "#111111"
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=6.4, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(r"Empirical $W_1$")
    cbar.ax.tick_params(labelsize=6.1, width=0.5, length=2)
    ax.tick_params(length=0)
    fig.subplots_adjust(left=0.28, right=0.88, bottom=0.25, top=0.98)
    save(fig, "figS1_wasserstein_matrix")


def fig_s3_ap_support_diagnostics() -> None:
    ap = pd.read_csv(ROOT / "outputs/ap_headline/ap_policy_band.csv")
    records = pd.read_parquet(ROOT / "outputs/scale/box_scale_records.parquet")
    f1_parts = []
    for mode, threshold, column in [
        ("absolute", 24.0, "scale_threshold_px"),
        ("normalized", 0.015, "scale_threshold_norm"),
    ]:
        data = pd.read_parquet(ROOT / f"outputs/coverage_corrected_grid/{mode}/metrics_long.parquet")
        data = data[np.isclose(data.confidence, 0.25) & np.isclose(data.iou, 0.25) & np.isclose(data[column], threshold)]
        grouped = data.groupby(["source", "candidate"], as_index=False).f1.agg(lambda x: float(np.max(x) - np.min(x)))
        grouped["mode"] = mode
        grouped["threshold"] = threshold
        grouped = grouped.rename(columns={"f1": "F1_policy_band"})
        f1_parts.append(grouped)
    f1 = pd.concat(f1_parts, ignore_index=True)
    joined = ap.merge(f1, on=["source", "candidate", "mode", "threshold"], validate="one_to_one")

    short = {
        "visdrone_baseline640": "VD-Base",
        "visdrone_sahi640": "VD-SAHI",
        "uavdt_baseline640": "UAV-Base",
        "uavdt_tiling": "UAV-Tile",
        "uavdt_yolo11n640": "UAV-Y11n",
        "uavdt_yolo11m704": "UAV-Y11m",
        "aitod_baseline640": "AI-Base",
    }
    candidate_order = [
        "visdrone_baseline640", "visdrone_sahi640", "uavdt_baseline640", "uavdt_tiling",
        "uavdt_yolo11n640", "uavdt_yolo11m704", "aitod_baseline640",
    ]
    fig, axs = plt.subplots(1, 3, figsize=(7.15, 2.25), gridspec_kw={"width_ratios": [1.22, 1.0, 1.05]})

    ypos = np.arange(len(candidate_order))
    for mode, offset, hatch in [("absolute", -0.17, ""), ("normalized", 0.17, "///")]:
        q = ap[ap["mode"].eq(mode)].set_index("candidate").reindex(candidate_order)
        axs[0].barh(ypos + offset, q.AP50_policy_band, 0.32, color=[COLORS[s] for s in q.source],
                    edgecolor="#333333", linewidth=0.35, hatch=hatch, label=mode.capitalize())
    axs[0].set_yticks(ypos, [short[c] for c in candidate_order])
    axs[0].invert_yaxis()
    axs[0].set_xlabel(r"Headline AP$_{50}$ policy band")
    axs[0].set_xlim(0, 0.105)
    clean_axis(axs[0], "x")
    axs[0].legend(frameon=False, loc="lower right")
    panel_label(axs[0], "(a)", -0.15)

    for source, q in joined.groupby("source"):
        for mode, fill in [("absolute", COLORS[source]), ("normalized", "none")]:
            z = q[q["mode"].eq(mode)]
            axs[1].scatter(z.AP50_policy_band, z.F1_policy_band, s=24, marker=MARKERS[source],
                           facecolors=fill, edgecolors=COLORS[source], linewidth=0.75)
    for candidate in ["visdrone_sahi640", "uavdt_tiling"]:
        q = joined[joined.candidate.eq(candidate)]
        row = q.loc[q.F1_policy_band.idxmax()]
        offset = (-4, 3) if candidate == "uavdt_tiling" else (4, 3)
        align = "right" if candidate == "uavdt_tiling" else "left"
        axs[1].annotate(short[candidate], (row.AP50_policy_band, row.F1_policy_band), xytext=offset,
                        textcoords="offset points", fontsize=5.6, ha=align)
    axs[1].set(xlabel=r"AP$_{50}$ policy band", ylabel=r"$F_1$ policy band")
    clean_axis(axs[1], "both")
    panel_label(axs[1], "(b)", -0.16)
    source_handles = [Line2D([0], [0], marker=MARKERS[s], color="none", markerfacecolor=COLORS[s],
                             markeredgecolor=COLORS[s], label=LABELS[s], markersize=4.5)
                      for s in ["visdrone", "uavdt", "aitod"]]
    mode_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markeredgecolor="#555555", label="absolute", markersize=4.2),
                    Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#555555", label="normalized", markersize=4.2)]
    axs[1].legend(handles=source_handles + mode_handles, frameon=False, loc="upper left", fontsize=5.2,
                  handletextpad=0.25, borderaxespad=0.2)

    norm_thresholds = np.sort(pd.read_parquet(ROOT / "outputs/coverage_corrected_grid/normalized/metrics_long.parquet").scale_threshold_norm.unique())
    for source in ["visdrone", "uavdt", "aitod"]:
        values = records.loc[records.source.eq(source), "normalized_side"].to_numpy(float)
        retained = [100 * np.mean(values >= threshold) for threshold in norm_thresholds]
        axs[2].plot(norm_thresholds, retained, color=COLORS[source], ls=LINESTYLES[source],
                    marker=MARKERS[source], ms=2.8, label=LABELS[source])
    axs[2].axvline(0.015, color="#222222", ls="--", lw=0.7)
    axs[2].set(xlabel="Normalized threshold", ylabel="Retained GT (%)", ylim=(-2, 103), xticks=norm_thresholds)
    axs[2].tick_params(axis="x", labelrotation=35)
    clean_axis(axs[2], "both")
    axs[2].legend(frameon=False, loc="upper right", fontsize=5.4)
    panel_label(axs[2], "(c)", -0.16)

    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.20, top=0.96, wspace=0.47)
    save(fig, "figS3_ap50_support_diagnostics")


def main() -> None:
    set_style()
    fig2_dataset_support()
    fig3_f1_surfaces()
    fig4_rank_uncertainty()
    fig5_controls()
    fig_s1_wasserstein()
    fig_s3_ap_support_diagnostics()


if __name__ == "__main__":
    main()
