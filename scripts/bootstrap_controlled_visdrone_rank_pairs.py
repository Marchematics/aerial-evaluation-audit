#!/usr/bin/env python3
"""Paired image bootstrap for decisive and boundary cells of the controlled pool.

The pool has identical raw image coverage (548/548 for every configuration),
so image-paired resampling preserves the common evaluation denominator.  The
reported boundary cells are not selected as a universal winner claim: they
show why a score ordering alone is insufficient near an observed rank switch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_cached_vehicle_grid import gt_map  # noqa: E402
from label_reliability.matching import greedy_match_with_ignores  # noqa: E402

CONFIG = ROOT / "configs" / "resubmission_visdrone_controlled_pool.yaml"
CACHE_ROOT = ROOT / "outputs" / "resubmission_controlled_pool" / "visdrone"
OUT = ROOT / "outputs" / "statistics" / "bootstrap_controlled_visdrone_rank_pairs.csv"

Y11M = "visdrone_control_yolo11m_hf640"
ASD = "visdrone_control_asd1280"
B640 = "visdrone_control_baseline640"
CONDITIONS = [
    # Declared headline cells: both use the same source-ignore-on primary policy.
    {"id": "headline_abs24", "mode": "absolute", "threshold": 24., "confidence": .25, "iou": .25, "a": Y11M, "b": ASD},
    {"id": "headline_norm015", "mode": "normalized", "threshold": .015, "confidence": .25, "iou": .25, "a": Y11M, "b": ASD},
    # Observed boundary diagnostics from the finite atlas; they are explicitly
    # reported as descriptive boundary cells rather than headline tests.
    {"id": "observed_boundary_norm010_c030", "mode": "normalized", "threshold": .010, "confidence": .30, "iou": .25, "a": Y11M, "b": ASD},
    {"id": "observed_boundary_abs64_c035", "mode": "absolute", "threshold": 64., "confidence": .35, "iou": .25, "a": B640, "b": Y11M},
]


def f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    den = 2 * tp + fp + fn
    return np.divide(2 * tp, den, out=np.zeros_like(den, dtype=float), where=den > 0)


def support(valid: np.ndarray, width: float, height: float, mode: str) -> np.ndarray:
    if mode == "absolute":
        return np.maximum(valid[:, 2] - valid[:, 0], valid[:, 3] - valid[:, 1])
    return np.maximum((valid[:, 2] - valid[:, 0]) / width, (valid[:, 3] - valid[:, 1]) / height)


def per_image_counts(candidate: str, condition: dict, names: list[str], gt: dict, vehicle_classes: set[int]) -> np.ndarray:
    raw = pd.read_parquet(CACHE_ROOT / candidate / "pred_rows.parquet")
    groups = {str(n): g for n, g in raw[raw.cls.isin(vehicle_classes)].groupby("img_name", sort=False)}
    out = []
    for name in names:
        record = gt[name]
        valid = np.asarray([b[:4] for b in record["boxes"]], float).reshape((-1, 4))
        keep = support(valid, record["W"], record["H"], condition["mode"]) >= condition["threshold"]
        retained = valid[keep]
        ignore = np.asarray([b[:4] for b in record["ignore"]], float).reshape((-1, 4))
        pg = groups.get(name, pd.DataFrame())
        pred = pg[["x1", "y1", "x2", "y2"]].to_numpy(float) if len(pg) else np.empty((0, 4))
        scores = pg.score.to_numpy(float) if len(pg) else np.empty(0)
        chosen = scores >= condition["confidence"]
        tp, fp, fn, neutralized = greedy_match_with_ignores(
            pred[chosen], retained, ignore, condition["iou"], scores=scores[chosen], ignore_overlap="crowd_iop",
        )
        out.append((tp, fp, fn, neutralized))
    return np.asarray(out, dtype=np.int64)


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    classes = set(cfg["shared_vehicle_native_classes"])
    gt = gt_map("visdrone")
    names = sorted(gt)
    if len(names) != 548:
        raise RuntimeError("Expected all VisDrone validation images")
    rng = np.random.default_rng(20260714)
    replicates = 10_000
    rows = []
    for condition in CONDITIONS:
        a = per_image_counts(condition["a"], condition, names, gt, classes)
        b = per_image_counts(condition["b"], condition, names, gt, classes)
        if a.shape != b.shape:
            raise AssertionError("paired common image universe is required")
        # Multinomial count weights are equivalent to resampling image IDs with
        # replacement but avoid storing a 10k x 548 integer-index matrix.
        weights = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=replicates)
        af = f1(weights @ a[:, 0], weights @ a[:, 1], weights @ a[:, 2])
        bf = f1(weights @ b[:, 0], weights @ b[:, 1], weights @ b[:, 2])
        diff = af - bf
        point_a = f1(np.array([a[:, 0].sum()]), np.array([a[:, 1].sum()]), np.array([a[:, 2].sum()]))[0]
        point_b = f1(np.array([b[:, 0].sum()]), np.array([b[:, 1].sum()]), np.array([b[:, 2].sum()]))[0]
        row = {
            **condition, "policy": "exclude_source_ignore_on", "common_images": len(names),
            "resampling_unit": "image", "replicates": replicates,
            "matching": "valid_gt_first_coco_crowd", "point_f1_a": float(point_a), "point_f1_b": float(point_b),
            "point_difference_a_minus_b": float(point_a - point_b), "ci95_low": float(np.quantile(diff, .025)),
            "ci95_high": float(np.quantile(diff, .975)), "probability_a_gt_b": float((diff > 0).mean()),
            "total_neutralized_a": int(a[:, 3].sum()), "total_neutralized_b": int(b[:, 3].sum()),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(json.dumps({"out": str(OUT), "rows": len(rows), "replicates": replicates}, indent=2))


if __name__ == "__main__":
    main()
