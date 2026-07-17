#!/usr/bin/env python3
"""Replay five VisDrone candidates and two alternative support definitions.

The primary rules are max side at 24 pixels and normalized max side at .015.
Minimum side and geometric-mean side are calibrated with valid GT only to
match the corresponding primary retained count as closely as ties permit.
All candidates traverse A→F→R→S at confidence and valid-match IoU .25. The
frozen Y11m--ASD pair also receives the same 2-by-2 metric point estimates
under every support definition.
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
from label_reliability.contract_path import (  # noqa: E402
    CONTRACTS,
    TRANSITIONS,
    contract_regions,
    evaluate_contract_counts,
    micro_f1,
    score_ordered_detection_outcomes,
)

CONFIG = ROOT / "configs" / "resubmission_visdrone_controlled_pool.yaml"
CACHE_ROOT = ROOT / "outputs" / "resubmission_controlled_pool" / "visdrone"
OUT = ROOT / "outputs" / "visdrone_path_support_robustness"
CONFIDENCES = np.asarray([.10, .15, .20, .25, .30, .35, .40, .50], dtype=float)
OPERATING_CONFIDENCE = .25
IOUS = (.25, .50)
MAX_DETS = 2000
Y11M = "visdrone_control_yolo11m_hf640"
ASD = "visdrone_control_asd1280"
SHORT = {
    "visdrone_control_baseline640": "B-640",
    "visdrone_control_baseline1280": "B-1280",
    Y11M: "Y11m",
    ASD: "ASD",
    "visdrone_control_yolo11n_p2_1280": "Y11n-P2",
}


def support_values(boxes: np.ndarray, width: float, height: float, coordinate: str, definition: str) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=float)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    if coordinate == "normalized":
        w, h = w / float(width), h / float(height)
    if definition == "max_side":
        return np.maximum(w, h)
    if definition == "minimum_side":
        return np.minimum(w, h)
    if definition == "geometric_mean_side":
        return np.sqrt(w * h)
    raise ValueError(definition)


def closest_retention_threshold(values: np.ndarray, target_count: int) -> tuple[float, int]:
    ordered = np.sort(np.asarray(values, dtype=float))
    candidates = np.unique(ordered)
    retained = len(ordered) - np.searchsorted(ordered, candidates, side="left")
    difference = np.abs(retained - int(target_count))
    chosen = int(np.flatnonzero(difference == difference.min())[-1])
    return float(candidates[chosen]), int(retained[chosen])


def pooled_ap(record: dict[str, np.ndarray]) -> float:
    denominator = float(record["gt"].sum())
    if denominator <= 0 or len(record["tp"]) == 0:
        return 0.0
    tp = np.cumsum(record["tp"])
    fp = np.cumsum(record["fp"])
    recall = tp / denominator
    precision = tp / np.maximum(tp + fp, 1e-12)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    index = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    values = np.zeros(101, dtype=float)
    valid = index < len(envelope)
    values[valid] = envelope[index[valid]]
    return float(values.mean())


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    candidates = [item["id"] for item in cfg["candidates"]]
    vehicle_classes = set(cfg["shared_vehicle_native_classes"])
    gt = gt_map("visdrone")
    names = sorted(gt)
    if len(names) != 548 or len(candidates) != 5:
        raise RuntimeError("expected five candidates and 548 VisDrone images")

    pooled: dict[tuple[str, str], list[np.ndarray]] = {}
    for coordinate in ("absolute", "normalized"):
        for definition in ("max_side", "minimum_side", "geometric_mean_side"):
            pooled[(coordinate, definition)] = []
    for name in names:
        record = gt[name]
        boxes = np.asarray([box[:4] for box in record["boxes"]], dtype=float).reshape((-1, 4))
        for key in pooled:
            pooled[key].append(support_values(boxes, record["W"], record["H"], *key))
    pooled_values = {key: np.concatenate(parts) for key, parts in pooled.items()}

    primary = {"absolute": 24.0, "normalized": .015}
    threshold_rows: list[dict] = []
    rules: list[dict] = []
    for coordinate, primary_threshold in primary.items():
        target = int((pooled_values[(coordinate, "max_side")] >= primary_threshold).sum())
        total = len(pooled_values[(coordinate, "max_side")])
        rules.append({
            "rule_id": f"{coordinate}_max_side",
            "coordinate": coordinate,
            "definition": "max_side",
            "threshold": primary_threshold,
            "primary": True,
        })
        threshold_rows.append({
            "rule_id": f"{coordinate}_max_side", "coordinate": coordinate,
            "definition": "max_side", "threshold": primary_threshold,
            "target_retained_gt": target, "actual_retained_gt": target,
            "total_valid_gt": total, "retained_fraction": target / total,
            "absolute_count_difference": 0,
        })
        for definition in ("minimum_side", "geometric_mean_side"):
            threshold, actual = closest_retention_threshold(pooled_values[(coordinate, definition)], target)
            rule_id = f"{coordinate}_{definition}"
            rules.append({
                "rule_id": rule_id, "coordinate": coordinate,
                "definition": definition, "threshold": threshold, "primary": False,
            })
            threshold_rows.append({
                "rule_id": rule_id, "coordinate": coordinate, "definition": definition,
                "threshold": threshold, "target_retained_gt": target,
                "actual_retained_gt": actual, "total_valid_gt": total,
                "retained_fraction": actual / total,
                "absolute_count_difference": abs(actual - target),
            })

    path_rows: list[dict] = []
    factorial_count_rows: list[dict] = []
    ap_parts: dict[tuple[str, str, float], dict[str, list | np.ndarray]] = {}
    for candidate in candidates:
        raw = pd.read_parquet(CACHE_ROOT / candidate / "pred_rows.parquet")
        if raw.img_name.astype(str).nunique() != 548:
            raise RuntimeError(f"{candidate}: incomplete prediction coverage")
        pred = raw[raw.cls.isin(vehicle_classes)].copy()
        groups = {str(name): group for name, group in pred.groupby("img_name", sort=False)}
        if candidate in (Y11M, ASD):
            for rule in rules:
                for iou in IOUS:
                    ap_parts[(candidate, rule["rule_id"], iou)] = {
                        "score": [], "tp": [], "fp": [], "img": [],
                        "gt": np.zeros(len(names), dtype=np.int32),
                    }
        for image_index, name in enumerate(names):
            record = gt[name]
            valid = np.asarray([box[:4] for box in record["boxes"]], dtype=float).reshape((-1, 4))
            source_ignore = np.asarray([box[:4] for box in record["ignore"]], dtype=float).reshape((-1, 4))
            group = groups.get(name, pd.DataFrame())
            boxes = group[["x1", "y1", "x2", "y2"]].to_numpy(float) if len(group) else np.empty((0, 4))
            scores = group.score.to_numpy(float) if len(group) else np.empty(0)
            selected = scores >= OPERATING_CONFIDENCE
            for rule in rules:
                support = support_values(valid, record["W"], record["H"], rule["coordinate"], rule["definition"])
                for contract in CONTRACTS:
                    regions = contract_regions(valid, source_ignore, support, rule["threshold"], contract)
                    tp, fp, fn, neutralized = evaluate_contract_counts(
                        boxes[selected], scores[selected], regions, .25,
                    )
                    path_rows.append({
                        "candidate": candidate, "short_candidate": SHORT[candidate],
                        "image_name": name, "sequence_id": name.split("_", 1)[0],
                        **rule, "contract": contract, "confidence": OPERATING_CONFIDENCE,
                        "iou": .25, "valid_gt": len(valid), "scored_gt": len(regions.valid),
                        "tp": tp, "fp": fp, "fn": fn, "neutralized": neutralized,
                    })
                if candidate not in (Y11M, ASD):
                    continue
                terminal = contract_regions(valid, source_ignore, support, rule["threshold"], "S")
                for iou in IOUS:
                    for confidence in CONFIDENCES:
                        chosen = scores >= confidence
                        tp, fp, fn, neutralized = evaluate_contract_counts(
                            boxes[chosen], scores[chosen], terminal, iou,
                        )
                        factorial_count_rows.append({
                            "candidate": candidate, "short_candidate": SHORT[candidate],
                            "image_name": name, "sequence_id": name.split("_", 1)[0],
                            **rule, "contract": "S", "iou": iou,
                            "confidence": confidence, "tp": tp, "fp": fp, "fn": fn,
                            "neutralized": neutralized,
                        })
                    score, tp_flag, fp_flag = score_ordered_detection_outcomes(
                        boxes, scores, terminal, iou, max_dets=MAX_DETS,
                    )
                    accumulator = ap_parts[(candidate, rule["rule_id"], iou)]
                    accumulator["gt"][image_index] = len(terminal.valid)
                    if len(score):
                        accumulator["score"].append(score)
                        accumulator["tp"].append(tp_flag)
                        accumulator["fp"].append(fp_flag)
                        accumulator["img"].append(np.full(len(score), image_index, dtype=np.int32))
        print(json.dumps({"candidate": candidate, "completed_images": len(names)}), flush=True)

    path_images = pd.DataFrame(path_rows)
    path_group = ["candidate", "short_candidate", "rule_id", "coordinate", "definition", "threshold", "primary", "contract"]
    path_endpoints = path_images.groupby(path_group, as_index=False).agg(
        images=("image_name", "nunique"), valid_gt=("valid_gt", "sum"),
        scored_gt=("scored_gt", "sum"), tp=("tp", "sum"), fp=("fp", "sum"),
        fn=("fn", "sum"), neutralized=("neutralized", "sum"),
    )
    path_endpoints["f1"] = micro_f1(path_endpoints.tp, path_endpoints.fp, path_endpoints.fn)
    transition_rows: list[dict] = []
    for keys, block in path_endpoints.groupby(["candidate", "short_candidate", "rule_id", "coordinate", "definition", "threshold", "primary"]):
        indexed = block.set_index("contract")
        for before, after, component in TRANSITIONS:
            left, right = indexed.loc[before], indexed.loc[after]
            transition_rows.append({
                "candidate": keys[0], "short_candidate": keys[1], "rule_id": keys[2],
                "coordinate": keys[3], "definition": keys[4], "threshold": keys[5],
                "primary": keys[6], "transition": f"{before}->{after}", "component": component,
                "delta_tp": int(right.tp - left.tp), "delta_fp": int(right.fp - left.fp),
                "delta_fn": int(right.fn - left.fn),
                "delta_neutralized": int(right.neutralized - left.neutralized),
                "delta_f1": float(right.f1 - left.f1),
            })
        if not np.isclose(indexed.loc["S", "f1"] - indexed.loc["A", "f1"],
                          sum(row["delta_f1"] for row in transition_rows[-3:]), atol=1e-12):
            raise AssertionError(f"path identity failed: {keys}")

    finalized_ap: dict[tuple[str, str, float], dict[str, np.ndarray]] = {}
    for key, block in ap_parts.items():
        score = np.concatenate(block["score"]) if block["score"] else np.empty(0)
        tp = np.concatenate(block["tp"]) if block["tp"] else np.empty(0, dtype=np.int8)
        fp = np.concatenate(block["fp"]) if block["fp"] else np.empty(0, dtype=np.int8)
        img = np.concatenate(block["img"]) if block["img"] else np.empty(0, dtype=np.int32)
        order = np.argsort(-score, kind="mergesort")
        finalized_ap[key] = {"score": score[order], "tp": tp[order], "fp": fp[order],
                             "img": img[order], "gt": block["gt"]}

    factorial_counts = pd.DataFrame(factorial_count_rows)
    point_rows: list[dict] = []
    group_cols = ["candidate", "short_candidate", "rule_id", "coordinate", "definition", "threshold", "primary", "iou"]
    for keys, block in factorial_counts.groupby(group_cols):
        counts = block.groupby("confidence", as_index=False)[["tp", "fp", "fn"]].sum().sort_values("confidence")
        counts["f1"] = micro_f1(counts.tp, counts.fp, counts.fn)
        fixed = counts[np.isclose(counts.confidence, OPERATING_CONFIDENCE)].iloc[0]
        best = counts.sort_values(["f1", "confidence"], ascending=[False, True]).iloc[0]
        ap = pooled_ap(finalized_ap[(keys[0], keys[2], keys[7])])
        point_rows.append({
            "candidate": keys[0], "short_candidate": keys[1], "rule_id": keys[2],
            "coordinate": keys[3], "definition": keys[4], "threshold": keys[5],
            "primary": keys[6], "iou": keys[7], "contract": "S",
            "f1_at_025": float(fixed.f1), "max_f1": float(best.f1),
            "max_f1_confidence": float(best.confidence), "ap": ap,
        })
    points = pd.DataFrame(point_rows)
    pair_rows: list[dict] = []
    for keys, block in points.groupby(["rule_id", "coordinate", "definition", "threshold", "primary", "iou"]):
        indexed = block.set_index("candidate")
        for metric in ("f1_at_025", "max_f1", "ap"):
            pair_rows.append({
                "rule_id": keys[0], "coordinate": keys[1], "definition": keys[2],
                "threshold": keys[3], "primary": keys[4], "iou": keys[5],
                "metric": metric, "candidate_a": Y11M, "candidate_b": ASD,
                "point_a": float(indexed.loc[Y11M, metric]),
                "point_b": float(indexed.loc[ASD, metric]),
                "point_difference": float(indexed.loc[Y11M, metric] - indexed.loc[ASD, metric]),
            })

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(threshold_rows).to_csv(OUT / "support_definition_thresholds.csv", index=False)
    path_images.to_parquet(OUT / "path_image_counts.parquet", index=False)
    path_endpoints.to_csv(OUT / "path_endpoints.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(OUT / "path_transition_ledger.csv", index=False)
    factorial_counts.to_parquet(OUT / "pair_factorial_image_counts.parquet", index=False)
    points.to_csv(OUT / "pair_factorial_points.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(OUT / "pair_factorial_differences.csv", index=False)
    summary = {
        "freeze_record": "configs/extension_analysis_freeze_20260717.yaml",
        "candidates": candidates,
        "common_images": len(names),
        "contracts": list(CONTRACTS),
        "operating_confidence": OPERATING_CONFIDENCE,
        "path_iou": .25,
        "factorial_ious": list(IOUS),
        "confidence_grid": CONFIDENCES.tolist(),
        "support_definitions": ["max_side", "minimum_side", "geometric_mean_side"],
        "alternative_threshold_calibration": "valid-GT retained-count matching to the corresponding max-side headline",
        "thresholds": threshold_rows,
    }
    (OUT / "README.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(threshold_rows).to_string(index=False))
    print(pd.DataFrame(pair_rows).to_string(index=False))


if __name__ == "__main__":
    main()
