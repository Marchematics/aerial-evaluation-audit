#!/usr/bin/env python3
"""Evaluate the frozen controlled VisDrone pool under a policy-explicit F1 grid.

This revision evaluator implements valid-GT-first, COCO-crowd-compatible
ignore precedence. Source-ignore regions neutralize only unmatched detections,
using intersection over detection area (the COCO ``iscrowd`` denominator), so
they cannot erase a true positive that also overlaps an ignore region. The
script materializes a complete finite-grid atlas; it makes no claim outside
that grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "resubmission_visdrone_controlled_pool.yaml"
OUT_DEFAULT = ROOT / "outputs" / "resubmission_controlled_pool" / "visdrone_grid"
CONTROLLED_CACHE_ROOT = ROOT / "outputs" / "resubmission_controlled_pool" / "visdrone"

from evaluate_cached_vehicle_grid import gt_map
from label_reliability.matching import greedy_match_with_ignores

CONF = [.10, .15, .20, .25, .30, .35, .40, .50]
IOU = [.25, .50]
ABS = [16, 20, 24, 28, 32, 40, 48, 64]
NORM = [.005, .0075, .010, .015, .020, .030]
POL = ["include_all", "exclude_source_ignore_off", "exclude_source_ignore_on"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--candidate", action="append", default=None, help="candidate id; repeat to subset")
    p.add_argument("--confidence", type=float, action="append", default=None)
    p.add_argument("--iou", type=float, action="append", default=None)
    p.add_argument("--absolute-thresholds", type=str, default=None)
    p.add_argument("--normalized-thresholds", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def f1(tp: int, fp: int, fn: int) -> float:
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    wanted = set(args.candidate or [x["id"] for x in cfg["candidates"]])
    candidates = [x for x in cfg["candidates"] if x["id"] in wanted]
    if len(candidates) != len(wanted):
        missing = wanted - {x["id"] for x in candidates}
        raise ValueError(f"Unknown candidate(s): {sorted(missing)}")
    confs = args.confidence or CONF
    ious = args.iou or IOU
    abss = [float(x) for x in args.absolute_thresholds.split(",")] if args.absolute_thresholds else ABS
    norms = [float(x) for x in args.normalized_thresholds.split(",")] if args.normalized_thresholds else NORM
    args.out.mkdir(parents=True, exist_ok=True)
    partial = args.out / "metrics_long.partial.parquet"
    old = pd.read_parquet(partial) if args.resume and partial.exists() else pd.DataFrame()
    done = set()
    if len(old):
        done = set(zip(old.candidate, old.scale_mode, old.scale_threshold, old.small_object_policy, old.confidence, old.iou))

    gt = gt_map("visdrone")
    records = old.to_dict("records") if len(old) else []
    for item in candidates:
        candidate = item["id"]
        pred_path = CONTROLLED_CACHE_ROOT / candidate / "pred_rows.parquet"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        raw = pd.read_parquet(pred_path)
        raw_names = set(raw.img_name.astype(str))
        names = sorted(raw_names & set(gt))
        if len(names) != 548:
            raise RuntimeError(f"{candidate}: raw prediction coverage {len(names)}/548")
        pred = raw[raw.cls.isin(cfg["shared_vehicle_native_classes"])]
        groups = {n: g for n, g in pred.groupby("img_name", sort=False)}
        print(json.dumps({"candidate": candidate, "raw_coverage": len(names), "vehicle_predictions": int(len(pred))}), flush=True)
        agg: dict[tuple, list[int]] = {}
        for mode, thresholds in (("absolute", abss), ("normalized", norms)):
            for threshold in thresholds:
                for policy in POL:
                    for conf in confs:
                        for iou in ious:
                            key = (candidate, mode, float(threshold), policy, float(conf), float(iou))
                            if key not in done:
                                agg[key] = [0, 0, 0, 0]  # TP FP FN neutralized
        if not agg:
            continue
        for ii, name in enumerate(names):
            r = gt[name]
            valid = np.asarray([b[:4] for b in r["boxes"]], float).reshape((-1, 4))
            sides = np.asarray([b[4] for b in r["boxes"]], float)
            source_ignore = np.asarray([b[:4] for b in r["ignore"]], float).reshape((-1, 4))
            pg = groups.get(name, pd.DataFrame())
            box_all = pg[["x1", "y1", "x2", "y2"]].to_numpy(float) if len(pg) else np.empty((0, 4))
            score_all = pg.score.to_numpy(float) if len(pg) else np.empty(0)
            for conf in confs:
                keep_pred = score_all >= conf
                boxes = box_all[keep_pred]
                scores = score_all[keep_pred]
                for iou in ious:
                    # Include-all is independent of support coordinate.
                    include = greedy_match_with_ignores(boxes, valid, np.empty((0, 4)), iou, scores=scores)
                    for mode, thresholds in (("absolute", abss), ("normalized", norms)):
                        # The registered normalized coordinate is
                        # max(w/W, h/H), not max(w,h)/max(W,H).  Preserve both
                        # box axes so a wide image cannot downweight a tall
                        # object (and vice versa).
                        support = sides if mode == "absolute" else np.maximum(
                            (valid[:, 2] - valid[:, 0]) / float(r["W"]),
                            (valid[:, 3] - valid[:, 1]) / float(r["H"]),
                        )
                        for threshold in thresholds:
                            for policy in POL:
                                key = (candidate, mode, float(threshold), policy, float(conf), float(iou))
                                if key not in agg:
                                    continue
                                if policy == "include_all":
                                    result = include
                                else:
                                    retained = valid[support >= threshold]
                                    ignore = np.empty((0, 4)) if policy == "exclude_source_ignore_off" else source_ignore
                                    result = greedy_match_with_ignores(
                                        boxes, retained, ignore, iou, scores=scores,
                                        ignore_overlap="crowd_iop",
                                    )
                                z = agg[key]
                                for j, value in enumerate(result):
                                    z[j] += int(value)
            if (ii + 1) % 100 == 0:
                print(f"{candidate}: {ii+1}/{len(names)}", flush=True)
        for (cand, mode, threshold, policy, conf, iou), (tp, fp, fn, neutralized) in agg.items():
            records.append({
                "source": "visdrone", "candidate": cand, "scale_mode": mode, "scale_threshold": threshold,
                "small_object_policy": policy, "confidence": conf, "iou": iou,
                "matching": "greedy_valid_gt_first_coco_crowd", "tp": tp, "fp": fp, "fn": fn,
                "neutralized_predictions": neutralized, "f1": f1(tp, fp, fn), "images": len(names),
            })
        pd.DataFrame(records).to_parquet(partial, index=False)
        print(json.dumps({"candidate": candidate, "completed_records": len(agg)}), flush=True)
    frame = pd.DataFrame(records).sort_values(["candidate", "scale_mode", "scale_threshold", "small_object_policy", "confidence", "iou"])
    expected_per = len(confs) * len(ious) * (len(abss) + len(norms)) * len(POL)
    expected = expected_per * len(candidates)
    if not args.resume and len(frame) != expected:
        raise RuntimeError(f"Expected {expected} records, found {len(frame)}")
    frame.to_parquet(args.out / "metrics_long.parquet", index=False)
    frame.to_csv(args.out / "metrics_long.csv", index=False)
    summary = {
        "candidate_count": len(candidates), "records": int(len(frame)), "confidence": confs, "iou": ious,
        "absolute_thresholds": abss, "normalized_thresholds": norms,
        "policies": POL, "ignore_precedence": "valid_gt_first_then_coco_crowd_ignore",
        "ignore_overlap": "intersection_over_prediction_area_ge_0.5",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
