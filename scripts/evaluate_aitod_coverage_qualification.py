#!/usr/bin/env python3
"""Materialize the Gate-0 coverage qualification record for AI-TOD.

Coverage is established from raw prediction image names before class filtering.
Covered and noncovered images are compared with image-level support, density,
shape, and crowding descriptors; pooled box-support distributions are reported
separately.  The record diagnoses selection and does not extrapolate metrics to
the uncovered images.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_cached_vehicle_grid import gt_map  # noqa: E402


PRED = Path("/root/zjh_UAV_detection/experiments/aitod/oracle_route/cache_val_baseline640/pred_rows.parquet")
SCALE = ROOT / "outputs" / "scale" / "box_scale_records.parquet"
OUT = ROOT / "outputs" / "coverage_qualification"
ABS = np.asarray([16, 20, 24, 28, 32, 40, 48, 64], dtype=float)
NORM = np.asarray([.005, .0075, .010, .015, .020, .030], dtype=float)


def finite(values: pd.Series | np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=float)
    return value[np.isfinite(value)]


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    a, b = finite(a), finite(b)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled == 0:
        return 0.0 if np.isclose(a.mean(), b.mean()) else float("inf")
    return float((a.mean() - b.mean()) / pooled)


def describe(group: pd.DataFrame, feature: str) -> dict:
    values = finite(group[feature])
    return {"n": int(len(values)), "mean": float(values.mean()), "std": float(values.std(ddof=1)),
            "median": float(np.median(values)), "q1": float(np.quantile(values, .25)),
            "q3": float(np.quantile(values, .75))}


def image_feature_row(image_name: str, record: dict, covered: bool) -> dict:
    boxes = np.asarray([box[:4] for box in record["boxes"]], dtype=float).reshape((-1, 4))
    if len(boxes):
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        absolute = np.maximum(width, height)
        normalized = np.maximum(width / float(record["W"]), height / float(record["H"]))
        elongation = np.maximum(width / np.maximum(height, 1e-12), height / np.maximum(width, 1e-12))
        centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / (2 * float(record["W"])),
                                   (boxes[:, 1] + boxes[:, 3]) / (2 * float(record["H"]))))
        if len(centers) >= 2:
            nearest = cKDTree(centers).query(centers, k=2)[0][:, 1]
            median_nearest = float(np.median(nearest))
        else:
            median_nearest = float("nan")
        return {"image_name": image_name, "coverage_group": "covered" if covered else "not_covered",
                "covered": covered, "image_width": float(record["W"]), "image_height": float(record["H"]),
                "vehicle_boxes": int(len(boxes)), "median_max_side_px": float(np.median(absolute)),
                "median_normalized_side": float(np.median(normalized)),
                "fraction_below_24px": float(np.mean(absolute < 24)),
                "fraction_below_norm015": float(np.mean(normalized < .015)),
                "median_elongation": float(np.median(elongation)),
                "median_nearest_center_distance": median_nearest}
    return {"image_name": image_name, "coverage_group": "covered" if covered else "not_covered",
            "covered": covered, "image_width": float(record["W"]), "image_height": float(record["H"]),
            "vehicle_boxes": 0, "median_max_side_px": float("nan"), "median_normalized_side": float("nan"),
            "fraction_below_24px": float("nan"), "fraction_below_norm015": float("nan"),
            "median_elongation": float("nan"), "median_nearest_center_distance": float("nan")}


def main() -> None:
    gt = gt_map("aitod")
    raw_names = set(pd.read_parquet(PRED, columns=["img_name"]).img_name.astype(str))
    covered_names = set(gt) & raw_names
    if len(gt) != 2804 or len(covered_names) != 1869:
        raise RuntimeError(f"unexpected AI-TOD coverage: {len(covered_names)}/{len(gt)}")
    image_features = pd.DataFrame([
        image_feature_row(name, gt[name], name in covered_names) for name in sorted(gt)
    ])

    features = ["vehicle_boxes", "median_max_side_px", "median_normalized_side", "image_width", "image_height",
                "fraction_below_24px", "fraction_below_norm015", "median_elongation",
                "median_nearest_center_distance"]
    covered = image_features[image_features.covered]
    noncovered = image_features[~image_features.covered]
    image_rows = []
    for feature in features:
        a, b = describe(covered, feature), describe(noncovered, feature)
        image_rows.append({"level": "image", "feature": feature,
                           **{f"covered_{k}": v for k, v in a.items()},
                           **{f"not_covered_{k}": v for k, v in b.items()},
                           "smd_covered_minus_not": standardized_mean_difference(covered[feature], noncovered[feature]),
                           "wasserstein": float(wasserstein_distance(finite(covered[feature]), finite(noncovered[feature])))})

    boxes = pd.read_parquet(SCALE)
    boxes = boxes[boxes.source.eq("aitod")].copy()
    boxes["covered"] = boxes.file_name.astype(str).isin(covered_names)
    boxes["coverage_group"] = np.where(boxes.covered, "covered", "not_covered")
    boxes["elongation"] = np.maximum(boxes.w / np.maximum(boxes.h, 1e-12), boxes.h / np.maximum(boxes.w, 1e-12))
    box_rows = []
    for feature in ("max_side_px", "normalized_side", "elongation"):
        a = boxes.loc[boxes.covered, feature]
        b = boxes.loc[~boxes.covered, feature]
        da, db = describe(pd.DataFrame({feature: a}), feature), describe(pd.DataFrame({feature: b}), feature)
        box_rows.append({"level": "box", "feature": feature,
                         **{f"covered_{k}": v for k, v in da.items()},
                         **{f"not_covered_{k}": v for k, v in db.items()},
                         "smd_covered_minus_not": standardized_mean_difference(a, b),
                         "wasserstein": float(wasserstein_distance(finite(a), finite(b)))})

    curve_rows = []
    for group, subset in (("full", boxes), ("covered", boxes[boxes.covered]), ("not_covered", boxes[~boxes.covered])):
        for mode, thresholds, field in (("absolute", ABS, "max_side_px"), ("normalized", NORM, "normalized_side")):
            values = subset[field].to_numpy(float)
            for threshold in thresholds:
                curve_rows.append({"coverage_group": group, "mode": mode, "threshold": float(threshold),
                                   "vehicle_boxes": int(len(values)),
                                   "retained_fraction": float(np.mean(values >= threshold)),
                                   "retained_percent": float(100 * np.mean(values >= threshold))})

    OUT.mkdir(parents=True, exist_ok=True)
    image_features.to_parquet(OUT / "aitod_image_features.parquet", index=False)
    pd.DataFrame(image_rows).to_csv(OUT / "aitod_image_qualification.csv", index=False)
    pd.DataFrame(box_rows).to_csv(OUT / "aitod_box_qualification.csv", index=False)
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(OUT / "aitod_support_curves.csv", index=False)
    decision = {
        "source": "AI-TOD", "denominator_images": len(gt), "covered_images": len(covered_names),
        "coverage_fraction": len(covered_names) / len(gt),
        "qualification": {"support_diagnosis": "pass", "covered_subset_metric_replay": "conditional_only",
                          "full_benchmark_metric": "fail", "full_benchmark_rank": "fail"},
        "reason": "declared prediction coverage is incomplete and support-selective",
        "no_extrapolation": True,
        "covered_boxes": int(boxes.covered.sum()), "not_covered_boxes": int((~boxes.covered).sum()),
        "headline_full_retention": {
            "absolute_24": float(curves[(curves.coverage_group == "full") & (curves["mode"] == "absolute") & np.isclose(curves.threshold, 24)].iloc[0].retained_fraction),
            "normalized_015": float(curves[(curves.coverage_group == "full") & (curves["mode"] == "normalized") & np.isclose(curves.threshold, .015)].iloc[0].retained_fraction),
        },
        "headline_covered_retention": {
            "absolute_24": float(curves[(curves.coverage_group == "covered") & (curves["mode"] == "absolute") & np.isclose(curves.threshold, 24)].iloc[0].retained_fraction),
            "normalized_015": float(curves[(curves.coverage_group == "covered") & (curves["mode"] == "normalized") & np.isclose(curves.threshold, .015)].iloc[0].retained_fraction),
        },
    }
    (OUT / "coverage_qualification.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(pd.DataFrame(image_rows).to_string(index=False))
    print("\nBox-level\n", pd.DataFrame(box_rows).to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
