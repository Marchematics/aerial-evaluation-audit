#!/usr/bin/env python3
"""Audit whether a declared AP maxDets cap can truncate mapped predictions.

The audit is intentionally conservative: a cap is called sufficient only when
it is at least the maximum per-image *mapped vehicle* prediction count in the
frozen artifact.  Consequently every lower score threshold and every
ground-truth support policy has no more detections than the audited count.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_cached_vehicle_grid import CANDIDATES, gt_map  # noqa: E402

OUT = ROOT / "outputs" / "cap_sufficiency"
FIG = ROOT / "figures" / "fig_cap_sufficiency.pdf"


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    return x, np.arange(1, len(x) + 1) / len(x)


def legacy_candidates() -> list[dict]:
    rows = []
    for cid, (source, pred_path, classes) in CANDIDATES.items():
        rows.append({"cohort": "legacy_frozen_pool", "candidate": cid, "source": source,
                     "path": Path(pred_path), "classes": set(classes)})
    return rows


def controlled_candidates() -> list[dict]:
    config = yaml.safe_load((ROOT / "configs" / "resubmission_visdrone_controlled_pool.yaml").read_text())
    rows = []
    for item in config["candidates"]:
        path = ROOT / "outputs" / "resubmission_controlled_pool" / "visdrone" / item["id"] / "pred_rows.parquet"
        if path.exists():
            rows.append({"cohort": "resubmission_controlled_pool", "candidate": item["id"], "source": "visdrone",
                         "path": path, "classes": set(config["shared_vehicle_native_classes"])})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    rows, series = [], []
    for item in legacy_candidates() + controlled_candidates():
        if not item["path"].exists():
            continue
        gt_names = set(gt_map(item["source"]))
        raw = pd.read_parquet(item["path"])
        coverage = sorted(set(raw.img_name.astype(str)) & gt_names)
        vehicle = raw[raw.cls.isin(item["classes"])].copy()
        counts = vehicle.groupby(vehicle.img_name.astype(str)).size().reindex(coverage, fill_value=0).to_numpy(int)
        if not len(counts):
            continue
        record = {
            "cohort": item["cohort"], "source": item["source"], "candidate": item["candidate"],
            "artifact": str(item["path"]), "covered_images": int(len(coverage)),
            "mapped_vehicle_predictions": int(len(vehicle[vehicle.img_name.astype(str).isin(coverage)])),
            "max_vehicle_predictions_per_image": int(counts.max()),
            "q90": float(np.quantile(counts, .90)), "q95": float(np.quantile(counts, .95)),
            "q99": float(np.quantile(counts, .99)),
            "images_truncated_at_100": int((counts > 100).sum()),
            "fraction_truncated_at_100": float((counts > 100).mean()),
            "images_truncated_at_300": int((counts > 300).sum()),
            "fraction_truncated_at_300": float((counts > 300).mean()),
            "images_truncated_at_2000": int((counts > 2000).sum()),
            "fraction_truncated_at_2000": float((counts > 2000).mean()),
            "cap_2000_sufficient": bool(counts.max() <= 2000),
        }
        rows.append(record)
        series.append((record, counts))
    table = pd.DataFrame(rows).sort_values(["cohort", "source", "candidate"])
    table.to_csv(OUT / "cap_sufficiency_table.csv", index=False)
    table.to_parquet(OUT / "cap_sufficiency_table.parquet", index=False)

    sources = [s for s in ["visdrone", "uavdt", "aitod"] if (table.source == s).any()]
    fig, axes = plt.subplots(1, len(sources), figsize=(3.42 * len(sources), 2.45), squeeze=False)
    colors = plt.cm.tab10.colors
    for ax, source in zip(axes[0], sources):
        current = [(r, c) for r, c in series if r["source"] == source]
        for idx, (r, counts) in enumerate(current):
            x, y = ecdf(counts)
            label = r["candidate"].replace("visdrone_control_", "").replace("uavdt_", "UAV-").replace("visdrone_", "VD-").replace("aitod_", "AI-")
            ax.step(x, y, where="post", lw=1.0, color=colors[idx % len(colors)], label=label)
        for cap, ls, text in [(100, "--", "100"), (2000, "-", "2000")]:
            ax.axvline(cap, color="0.25", lw=.8, ls=ls)
            ax.text(cap, .025, text, rotation=90, va="bottom", ha="right", fontsize=6.5, color="0.2")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("Vehicle predictions / image")
        ax.set_title({"visdrone": "VisDrone", "uavdt": "UAVDT", "aitod": "AI-TOD"}[source], fontsize=8)
        ax.grid(True, axis="y", alpha=.25, lw=.5)
        if source == sources[0]:
            ax.set_ylabel("ECDF")
        ax.tick_params(labelsize=6.5)
        if source == "visdrone":
            ax.legend(fontsize=5.2, ncol=2, frameon=False, loc="lower right", handlelength=1.5, columnspacing=.6)
    fig.tight_layout(pad=.35, w_pad=.55)
    fig.savefig(FIG, bbox_inches="tight")
    fig.savefig(FIG.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "contract": "A terminal cap is sufficient only if it exceeds every mapped vehicle prediction count per covered image.",
        "cap": 2000,
        "all_audited_artifacts_cap_sufficient": bool(table.cap_2000_sufficient.all()),
        "artifacts": int(len(table)),
        "max_observed": int(table.max_vehicle_predictions_per_image.max()),
        "legacy_max_observed": int(table.loc[table.cohort.eq("legacy_frozen_pool"), "max_vehicle_predictions_per_image"].max()),
        "controlled_max_observed": int(table.loc[table.cohort.eq("resubmission_controlled_pool"), "max_vehicle_predictions_per_image"].max()) if (table.cohort == "resubmission_controlled_pool").any() else None,
    }
    (OUT / "README.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
