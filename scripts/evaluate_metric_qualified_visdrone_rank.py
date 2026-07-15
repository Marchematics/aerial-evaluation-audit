#!/usr/bin/env python3
"""Cross-metric, sequence-qualified rank audit for the controlled VisDrone pool.

The endpoint contract is ``S`` (filtered target, removed-object ignore, and
source-ignore enabled).  Point estimates cover all five common-coverage
configurations.  The declared Y11m--ASD pair is then evaluated with paired
image and filename-sequence bootstrap for F1@.25, max-F1 over the prespecified
confidence grid, and AP50.  AP is recomputed from globally pooled detections in
each replicate rather than averaged over images or sequences.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_cached_vehicle_grid import gt_map  # noqa: E402
from label_reliability.contract_path import (  # noqa: E402
    contract_regions,
    evaluate_contract_counts,
    micro_f1,
    score_ordered_detection_outcomes,
    support_values,
)


CONFIG = ROOT / "configs" / "resubmission_visdrone_controlled_pool.yaml"
CACHE_ROOT = ROOT / "outputs" / "resubmission_controlled_pool" / "visdrone"
OUT = ROOT / "outputs" / "metric_qualified_rank"
RULES = (("absolute", 24.0), ("normalized", .015))
CONFIDENCES = np.asarray([.10, .15, .20, .25, .30, .35, .40, .50], dtype=float)
F1_IOU = .25
AP_IOU = .50
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--ap-batch", type=int, default=24)
    return parser.parse_args()


def sequence_id(image_name: str) -> str:
    prefix = image_name.split("_", 1)[0]
    if len(prefix) != 7 or not prefix.isdigit():
        raise ValueError(f"unexpected VisDrone image name: {image_name}")
    return prefix


def prepare_predictions(candidate: str, vehicle_classes: set[int]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    raw = pd.read_parquet(CACHE_ROOT / candidate / "pred_rows.parquet")
    if raw.img_name.astype(str).nunique() != 548:
        raise RuntimeError(f"{candidate}: incomplete raw coverage")
    pred = raw[raw.cls.isin(vehicle_classes)].copy()
    return pred, {str(name): group for name, group in pred.groupby("img_name", sort=False)}


def ap50_from_record(record: dict, image_weights: np.ndarray) -> float:
    denominator = float(np.dot(image_weights, record["gt"]))
    if denominator <= 0:
        return float("nan")
    weights = image_weights[record["img"]]
    tp = np.cumsum(weights * record["tp"])
    fp = np.cumsum(weights * record["fp"])
    if not len(tp):
        return 0.0
    recall = tp / denominator
    precision = tp / np.maximum(tp + fp, 1e-12)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    index = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    values = np.zeros(101, dtype=float)
    valid = index < len(envelope)
    values[valid] = envelope[index[valid]]
    return float(values.mean())


def ap50_bootstrap_batch(
    record: dict,
    unit_counts: np.ndarray,
    image_unit: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Recompute pooled AP50 for every paired resample in bounded memory."""

    result = np.empty(len(unit_counts), dtype=float)
    recall_grid = np.linspace(0, 1, 101)
    for start in range(0, len(unit_counts), batch_size):
        stop = min(start + batch_size, len(unit_counts))
        image_weights = unit_counts[start:stop, image_unit]
        denominator = image_weights @ record["gt"]
        detection_weights = image_weights[:, record["img"]]
        tp = np.cumsum(detection_weights * record["tp"][None, :], axis=1)
        fp = np.cumsum(detection_weights * record["fp"][None, :], axis=1)
        recall = np.divide(tp, denominator[:, None], out=np.zeros_like(tp, dtype=float), where=denominator[:, None] > 0)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
        envelope = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
        for row in range(stop - start):
            index = np.searchsorted(recall[row], recall_grid, side="left")
            values = np.zeros(101, dtype=float)
            valid = index < envelope.shape[1]
            values[valid] = envelope[row, index[valid]]
            result[start + row] = values.mean()
    return result


def build_candidate_records(
    candidate: str,
    groups: dict[str, pd.DataFrame],
    names: list[str],
    gt: dict,
) -> tuple[list[dict], dict[tuple[str, float], dict]]:
    count_rows: list[dict] = []
    ap_records: dict[tuple[str, float], dict] = {}
    for mode, threshold in RULES:
        all_scores: list[np.ndarray] = []
        all_tp: list[np.ndarray] = []
        all_fp: list[np.ndarray] = []
        all_image: list[np.ndarray] = []
        gt_per_image = np.zeros(len(names), dtype=np.int32)
        for image_index, image_name in enumerate(names):
            record = gt[image_name]
            valid = np.asarray([box[:4] for box in record["boxes"]], float).reshape((-1, 4))
            source_ignore = np.asarray([box[:4] for box in record["ignore"]], float).reshape((-1, 4))
            support = support_values(valid, record["W"], record["H"], mode)
            regions = contract_regions(valid, source_ignore, support, threshold, "S")
            gt_per_image[image_index] = len(regions.valid)
            group = groups.get(image_name, pd.DataFrame())
            pred = group[["x1", "y1", "x2", "y2"]].to_numpy(float) if len(group) else np.empty((0, 4))
            scores = group.score.to_numpy(float) if len(group) else np.empty(0)
            for confidence in CONFIDENCES:
                chosen = scores >= confidence
                tp, fp, fn, neutralized = evaluate_contract_counts(
                    pred[chosen], scores[chosen], regions, F1_IOU,
                )
                count_rows.append({
                    "candidate": candidate, "short_candidate": SHORT[candidate],
                    "image_name": image_name, "sequence_id": sequence_id(image_name),
                    "mode": mode, "threshold": threshold, "contract": "S",
                    "confidence": float(confidence), "iou": F1_IOU,
                    "tp": int(tp), "fp": int(fp), "fn": int(fn), "neutralized": int(neutralized),
                })
            scores_ap, tp_ap, fp_ap = score_ordered_detection_outcomes(
                pred, scores, regions, AP_IOU, max_dets=MAX_DETS,
            )
            if len(scores_ap):
                all_scores.append(scores_ap); all_tp.append(tp_ap); all_fp.append(fp_ap)
                all_image.append(np.full(len(scores_ap), image_index, dtype=np.int32))
        score = np.concatenate(all_scores) if all_scores else np.empty(0)
        tp = np.concatenate(all_tp) if all_tp else np.empty(0, dtype=np.int8)
        fp = np.concatenate(all_fp) if all_fp else np.empty(0, dtype=np.int8)
        image = np.concatenate(all_image) if all_image else np.empty(0, dtype=np.int32)
        order = np.argsort(-score, kind="mergesort")
        score, tp, fp, image = score[order], tp[order], fp[order], image[order]
        # Detections after the final possible TP can never increase the COCO
        # precision envelope, including under nonnegative bootstrap weights.
        positive = np.flatnonzero(tp)
        if len(positive):
            end = int(positive[-1]) + 1
            score, tp, fp, image = score[:end], tp[:end], fp[:end], image[:end]
        ap_records[(mode, threshold)] = {"score": score, "tp": tp, "fp": fp, "img": image, "gt": gt_per_image}
    return count_rows, ap_records


def coco_ap50_crosscheck(
    candidate: str,
    pred: pd.DataFrame,
    names: list[str],
    gt: dict,
    mode: str,
    threshold: float,
) -> float:
    images, annotations, detections = [], [], []
    for image_index, image_name in enumerate(names, start=1):
        record = gt[image_name]
        images.append({"id": image_index, "file_name": image_name, "width": int(record["W"]), "height": int(record["H"])})
        valid = np.asarray([box[:4] for box in record["boxes"]], float).reshape((-1, 4))
        source_ignore = np.asarray([box[:4] for box in record["ignore"]], float).reshape((-1, 4))
        support = support_values(valid, record["W"], record["H"], mode)
        regions = contract_regions(valid, source_ignore, support, threshold, "S")
        for box in regions.valid:
            width, height = float(box[2] - box[0]), float(box[3] - box[1])
            annotations.append({"id": len(annotations) + 1, "image_id": image_index, "category_id": 1,
                                "bbox": [float(box[0]), float(box[1]), width, height], "area": width * height,
                                "iscrowd": 0, "ignore": 0})
        for box in regions.ignore:
            width, height = float(box[2] - box[0]), float(box[3] - box[1])
            annotations.append({"id": len(annotations) + 1, "image_id": image_index, "category_id": 1,
                                "bbox": [float(box[0]), float(box[1]), width, height], "area": width * height,
                                "iscrowd": 1, "ignore": 1})
    image_ids = {name: i for i, name in enumerate(names, start=1)}
    for row in pred.itertuples():
        detections.append({"image_id": image_ids[str(row.img_name)], "category_id": 1,
                           "bbox": [float(row.x1), float(row.y1), float(row.x2 - row.x1), float(row.y2 - row.y1)],
                           "score": float(row.score)})
    payload = {"info": {}, "licenses": [], "images": images, "annotations": annotations,
               "categories": [{"id": 1, "name": "vehicle"}]}
    with tempfile.TemporaryDirectory() as tmp:
        gt_path, dt_path = Path(tmp) / "gt.json", Path(tmp) / "dt.json"
        gt_path.write_text(json.dumps(payload)); dt_path.write_text(json.dumps(detections))
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(str(gt_path)); coco_dt = coco_gt.loadRes(str(dt_path))
            evaluator = COCOeval(coco_gt, coco_dt, "bbox")
            evaluator.params.catIds = [1]
            evaluator.params.imgIds = list(range(1, len(names) + 1))
            evaluator.params.iouThrs = np.asarray([AP_IOU])
            evaluator.params.maxDets = [1, 10, MAX_DETS]
            evaluator.evaluate(); evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    values = precision[precision > -1]
    return float(values.mean()) if len(values) else float("nan")


def summarize_points(image_counts: pd.DataFrame, ap_records: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for (candidate, mode, threshold), block in image_counts.groupby(["candidate", "mode", "threshold"]):
        by_conf = (block.groupby("confidence", as_index=False)[["tp", "fp", "fn"]].sum().sort_values("confidence"))
        by_conf["f1"] = micro_f1(by_conf.tp.to_numpy(), by_conf.fp.to_numpy(), by_conf.fn.to_numpy())
        fixed = by_conf[np.isclose(by_conf.confidence, .25)].iloc[0]
        best = by_conf.sort_values(["f1", "confidence"], ascending=[False, True]).iloc[0]
        record = ap_records[candidate][(mode, threshold)]
        ap50 = ap50_from_record(record, np.ones(len(record["gt"]), dtype=np.int16))
        rows.append({"candidate": candidate, "short_candidate": SHORT[candidate], "mode": mode,
                     "threshold": float(threshold), "contract": "S", "common_images": 548,
                     "f1_at_025": float(fixed.f1), "max_f1": float(best.f1),
                     "max_f1_confidence": float(best.confidence), "ap50": ap50, "ap_max_dets": MAX_DETS})
    frame = pd.DataFrame(rows)
    for metric in ("f1_at_025", "max_f1", "ap50"):
        frame[f"rank_{metric}"] = (frame.groupby(["mode", "threshold"])[metric]
                                    .rank(method="min", ascending=False).astype(int))
    return frame.sort_values(["mode", "threshold", "rank_f1_at_025"])


def bootstrap_count_metrics(
    a: np.ndarray,
    b: np.ndarray,
    unit_counts: np.ndarray,
    image_unit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    image_weights = unit_counts[:, image_unit]
    shape = a.shape
    a_sum = (image_weights @ a.reshape(shape[0], -1)).reshape(len(unit_counts), shape[1], 3)
    b_sum = (image_weights @ b.reshape(shape[0], -1)).reshape(len(unit_counts), shape[1], 3)
    af = micro_f1(a_sum[:, :, 0], a_sum[:, :, 1], a_sum[:, :, 2])
    bf = micro_f1(b_sum[:, :, 0], b_sum[:, :, 1], b_sum[:, :, 2])
    return af, bf


def interval_row(metric: str, mode: str, threshold: float, unit: str, a: float, b: float, diff: np.ndarray,
                 replicates: int, units: int) -> dict:
    return {"metric": metric, "mode": mode, "threshold": float(threshold), "contract": "S",
            "candidate_a": Y11M, "candidate_b": ASD, "point_a": float(a), "point_b": float(b),
            "point_difference": float(a - b), "ci95_low": float(np.quantile(diff, .025)),
            "ci95_high": float(np.quantile(diff, .975)), "probability_a_gt_b": float(np.mean(diff > 0)),
            "resampling_unit": unit, "sampling_units": int(units), "common_images": 548,
            "replicates": int(replicates)}


def main() -> None:
    args = parse_args()
    if args.replicates < 100:
        raise ValueError("at least 100 replicates are required")
    cfg = yaml.safe_load(CONFIG.read_text())
    candidates = [item["id"] for item in cfg["candidates"]]
    classes = set(cfg["shared_vehicle_native_classes"])
    gt = gt_map("visdrone")
    names = sorted(gt)
    if len(names) != 548:
        raise RuntimeError("expected 548 VisDrone validation images")
    sequences = [sequence_id(name) for name in names]
    unique_sequences = sorted(set(sequences))
    if len(unique_sequences) != 76:
        raise RuntimeError(f"expected 76 filename-derived sequences, got {len(unique_sequences)}")
    sequence_index = {value: index for index, value in enumerate(unique_sequences)}
    image_sequence = np.asarray([sequence_index[value] for value in sequences], dtype=np.int16)

    all_count_rows: list[dict] = []
    ap_records: dict[str, dict] = {}
    raw_predictions: dict[str, pd.DataFrame] = {}
    for candidate in candidates:
        pred, groups = prepare_predictions(candidate, classes)
        raw_predictions[candidate] = pred
        rows, records = build_candidate_records(candidate, groups, names, gt)
        all_count_rows.extend(rows); ap_records[candidate] = records
        print(json.dumps({"candidate": candidate, "count_rows": len(rows),
                          "ap_detections": {f"{m}:{t}": len(r["tp"]) for (m, t), r in records.items()}}), flush=True)
    image_counts = pd.DataFrame(all_count_rows)
    points = summarize_points(image_counts, ap_records)

    crosscheck_rows = []
    for candidate in (Y11M, ASD):
        for mode, threshold in RULES:
            custom = float(points[(points.candidate == candidate) & (points["mode"] == mode) &
                                  np.isclose(points.threshold, threshold)].iloc[0].ap50)
            coco = coco_ap50_crosscheck(candidate, raw_predictions[candidate], names, gt, mode, threshold)
            difference = custom - coco
            crosscheck_rows.append({"candidate": candidate, "mode": mode, "threshold": threshold,
                                    "contract": "S", "custom_ap50": custom, "cocoeval_ap50": coco,
                                    "difference": difference, "absolute_difference": abs(difference),
                                    "tolerance": 5e-4, "pass": abs(difference) <= 5e-4,
                                    "max_dets": MAX_DETS, "iou": AP_IOU})
    crosscheck = pd.DataFrame(crosscheck_rows)
    if not crosscheck["pass"].all():
        raise AssertionError("custom AP50 and COCOeval differ beyond the prespecified tolerance")

    rng = np.random.default_rng(args.seed)
    bootstrap_rows: list[dict] = []
    for unit, image_unit in (("image", np.arange(len(names), dtype=np.int16)),
                             ("sequence", image_sequence)):
        unit_count = int(image_unit.max()) + 1
        sampled = rng.multinomial(unit_count, np.full(unit_count, 1 / unit_count), size=args.replicates)
        for mode, threshold in RULES:
            subset = image_counts[(image_counts["mode"] == mode) & np.isclose(image_counts.threshold, threshold)]
            arrays = {}
            for candidate in (Y11M, ASD):
                q = subset[subset.candidate == candidate].copy()
                q["image_index"] = pd.Categorical(q.image_name, categories=names, ordered=True).codes
                q["confidence_index"] = pd.Categorical(q.confidence, categories=CONFIDENCES, ordered=True).codes
                q = q.sort_values(["image_index", "confidence_index"])
                arrays[candidate] = q[["tp", "fp", "fn"]].to_numpy(np.int64).reshape(len(names), len(CONFIDENCES), 3)
            af, bf = bootstrap_count_metrics(arrays[Y11M], arrays[ASD], sampled, image_unit)
            fixed_index = int(np.where(np.isclose(CONFIDENCES, .25))[0][0])
            point_block = points[(points["mode"] == mode) & np.isclose(points.threshold, threshold)].set_index("candidate")
            bootstrap_rows.append(interval_row("F1@.25", mode, threshold, unit,
                                               point_block.loc[Y11M, "f1_at_025"], point_block.loc[ASD, "f1_at_025"],
                                               af[:, fixed_index] - bf[:, fixed_index], args.replicates, unit_count))
            bootstrap_rows.append(interval_row("max-F1", mode, threshold, unit,
                                               point_block.loc[Y11M, "max_f1"], point_block.loc[ASD, "max_f1"],
                                               af.max(axis=1) - bf.max(axis=1), args.replicates, unit_count))
            a_ap = ap50_bootstrap_batch(ap_records[Y11M][(mode, threshold)], sampled, image_unit, args.ap_batch)
            b_ap = ap50_bootstrap_batch(ap_records[ASD][(mode, threshold)], sampled, image_unit, args.ap_batch)
            bootstrap_rows.append(interval_row("AP50", mode, threshold, unit,
                                               point_block.loc[Y11M, "ap50"], point_block.loc[ASD, "ap50"],
                                               a_ap - b_ap, args.replicates, unit_count))
            print(json.dumps({"unit": unit, "mode": mode, "threshold": threshold,
                              "completed_metrics": ["F1@.25", "max-F1", "AP50"]}), flush=True)

    bootstrap = pd.DataFrame(bootstrap_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    image_counts.to_parquet(OUT / "rank_image_counts.parquet", index=False)
    points.to_csv(OUT / "metric_rank_points.csv", index=False)
    bootstrap.to_csv(OUT / "paired_bootstrap.csv", index=False)
    crosscheck.to_csv(OUT / "evaluator_crosscheck.csv", index=False)
    summary = {
        "contract": "S", "common_images": len(names), "sequence_clusters": len(unique_sequences),
        "sequence_definition": "leading seven-digit VisDrone filename token",
        "confidence_grid": CONFIDENCES.tolist(), "f1_iou": F1_IOU, "ap_iou": AP_IOU,
        "ap_max_dets": MAX_DETS, "replicates": args.replicates, "seed": args.seed,
        "top_pair": [Y11M, ASD], "metrics": ["F1@.25", "max-F1", "AP50"],
        "ap_bootstrap": "pooled detections recomputed for every paired resample",
        "crosscheck_max_abs_difference": float(crosscheck.absolute_difference.max()),
    }
    (OUT / "README.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nPoint metrics\n", points.to_string(index=False))
    print("\nPaired bootstrap\n", bootstrap.to_string(index=False))
    print("\nEvaluator crosscheck\n", crosscheck.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
