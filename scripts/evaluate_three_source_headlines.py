#!/usr/bin/env python3
"""Replay one registered baseline artifact per source at the two headline rules.

The evaluator is intentionally identical to the final VisDrone V4 contract:
confidence/IoU=.25, score-ordered valid-GT-first matching, no source-ignore
neutralization for I or O, and COCO-crowd intersection-over-prediction-area
neutralization for N.  AI-TOD results are explicitly coverage-conditioned.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_cached_vehicle_grid import gt_map  # noqa: E402
from label_reliability.matching import greedy_match_with_ignores  # noqa: E402


OUT = ROOT / "outputs" / "three_source_headline" / "three_source_policy_endpoints.csv"
CONFIDENCE = .25
IOU = .25
POLICIES = ("I", "O", "N")
SOURCES = {
    "VisDrone": {
        "candidate": "B-640",
        "path": ROOT / "outputs" / "resubmission_controlled_pool" / "visdrone" /
                "visdrone_control_baseline640" / "pred_rows.parquet",
        "classes": {3, 4, 5, 8},
        "source_key": "visdrone",
        "denominator_images": 548,
    },
    "UAVDT": {
        "candidate": "UAVDT-Base",
        "path": Path("/root/zjh_UAV_detection/experiments/uavdt/oracle_route/"
                     "cache_val_baseline640/pred_rows.parquet"),
        "classes": {0, 1, 2},
        "source_key": "uavdt",
        "denominator_images": 305,
    },
    "AI-TOD": {
        "candidate": "AI-TOD-Base",
        "path": Path("/root/zjh_UAV_detection/experiments/aitod/oracle_route/"
                     "cache_val_baseline640/pred_rows.parquet"),
        "classes": {5},
        "source_key": "aitod",
        "denominator_images": 2804,
    },
}
RULES = (("absolute", 24.0), ("normalized", .015))


def f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def main() -> None:
    rows: list[dict] = []
    for display_source, spec in SOURCES.items():
        gt = gt_map(spec["source_key"])
        raw = pd.read_parquet(spec["path"])
        raw_names = set(raw.img_name.astype(str))
        names = sorted(set(gt) & raw_names)
        expected_covered = 1869 if display_source == "AI-TOD" else spec["denominator_images"]
        if len(names) != expected_covered:
            raise RuntimeError(f"{display_source}: expected {expected_covered} covered images, got {len(names)}")
        pred = raw[raw.cls.isin(spec["classes"])]
        groups = {name: group for name, group in pred.groupby("img_name", sort=False)}

        aggregate = {(mode, threshold, policy): [0, 0, 0] for mode, threshold in RULES for policy in POLICIES}
        retained_counts = {(mode, threshold): 0 for mode, threshold in RULES}
        valid_count = 0
        for name in names:
            record = gt[name]
            valid = np.asarray([box[:4] for box in record["boxes"]], float).reshape((-1, 4))
            valid_count += len(valid)
            sides = np.asarray([box[4] for box in record["boxes"]], float)
            source_ignore = np.asarray([box[:4] for box in record["ignore"]], float).reshape((-1, 4))
            group = groups.get(name, pd.DataFrame())
            boxes_all = group[["x1", "y1", "x2", "y2"]].to_numpy(float) if len(group) else np.empty((0, 4))
            scores_all = group.score.to_numpy(float) if len(group) else np.empty(0)
            keep = scores_all >= CONFIDENCE
            boxes, scores = boxes_all[keep], scores_all[keep]

            include = greedy_match_with_ignores(
                boxes, valid, np.empty((0, 4)), IOU, scores=scores,
                ignore_overlap="crowd_iop",
            )
            normalized = np.maximum(
                (valid[:, 2] - valid[:, 0]) / float(record["W"]),
                (valid[:, 3] - valid[:, 1]) / float(record["H"]),
            ) if len(valid) else np.empty(0)
            for mode, threshold in RULES:
                support = sides if mode == "absolute" else normalized
                retained = valid[support >= threshold]
                retained_counts[(mode, threshold)] += len(retained)
                results = {
                    "I": include,
                    "O": greedy_match_with_ignores(
                        boxes, retained, np.empty((0, 4)), IOU, scores=scores,
                        ignore_overlap="crowd_iop",
                    ),
                    "N": greedy_match_with_ignores(
                        boxes, retained, source_ignore, IOU, scores=scores,
                        ignore_overlap="crowd_iop",
                    ),
                }
                for policy, result in results.items():
                    aggregate[(mode, threshold, policy)] = [
                        x + int(y) for x, y in zip(aggregate[(mode, threshold, policy)], result[:3])
                    ]

        for mode, threshold in RULES:
            endpoints = {}
            for policy in POLICIES:
                tp, fp, fn = aggregate[(mode, threshold, policy)]
                endpoints[policy] = f1(tp, fp, fn)
            rows.append({
                "source": display_source,
                "candidate": spec["candidate"],
                "covered_images": len(names),
                "denominator_images": spec["denominator_images"],
                "coverage_conditioned": display_source == "AI-TOD",
                "mode": mode,
                "threshold": threshold,
                "retained_gt_covered": retained_counts[(mode, threshold)],
                "valid_gt_covered": valid_count,
                "retained_fraction_covered": retained_counts[(mode, threshold)] / valid_count,
                "I": endpoints["I"],
                "O": endpoints["O"],
                "N": endpoints["N"],
                "F1_policy_band": max(endpoints.values()) - min(endpoints.values()),
            })

    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
