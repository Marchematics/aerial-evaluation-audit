#!/usr/bin/env python3
"""Separate localization and score-integration effects for the Y11m--ASD pair.

The control reuses the two registered VisDrone headline supports and terminal
contract S.  It crosses IoU in {.25, .50} with threshold-optimized micro-F1
and score-integrated AP, and retains fixed-confidence F1 at confidence .25 as
an operational endpoint.  Every pairwise interval resamples the 76 filename-
derived sequences and recomputes pooled counts or pooled-detection AP.
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
OUT = ROOT / "outputs" / "metric_factorial_control"
RULES = (("absolute", 24.0), ("normalized", .015))
IOUS = np.asarray([.25, .50], dtype=float)
CONFIDENCES = np.asarray([.10, .15, .20, .25, .30, .35, .40, .50], dtype=float)
OPERATING_CONFIDENCE = .25
MAX_DETS = 2000
Y11M = "visdrone_control_yolo11m_hf640"
ASD = "visdrone_control_asd1280"
SHORT = {Y11M: "Y11m", ASD: "ASD"}


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


def pooled_ap(record: dict[str, np.ndarray], image_weights: np.ndarray) -> float:
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


def pooled_ap_bootstrap(
    record: dict[str, np.ndarray],
    unit_counts: np.ndarray,
    image_unit: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    result = np.empty(len(unit_counts), dtype=float)
    recall_grid = np.linspace(0, 1, 101)
    for start in range(0, len(unit_counts), batch_size):
        stop = min(start + batch_size, len(unit_counts))
        image_weights = unit_counts[start:stop, image_unit]
        denominator = image_weights @ record["gt"]
        detection_weights = image_weights[:, record["img"]]
        tp = np.cumsum(detection_weights * record["tp"][None, :], axis=1)
        fp = np.cumsum(detection_weights * record["fp"][None, :], axis=1)
        recall = np.divide(
            tp, denominator[:, None], out=np.zeros_like(tp, dtype=float), where=denominator[:, None] > 0,
        )
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
) -> tuple[list[dict], dict[tuple[str, float, float], dict[str, np.ndarray]]]:
    count_rows: list[dict] = []
    ap_records: dict[tuple[str, float, float], dict[str, np.ndarray]] = {}
    for mode, threshold in RULES:
        accumulators = {
            float(iou): {"score": [], "tp": [], "fp": [], "img": [], "gt": np.zeros(len(names), dtype=np.int32)}
            for iou in IOUS
        }
        for image_index, image_name in enumerate(names):
            record = gt[image_name]
            valid = np.asarray([box[:4] for box in record["boxes"]], dtype=float).reshape((-1, 4))
            source_ignore = np.asarray([box[:4] for box in record["ignore"]], dtype=float).reshape((-1, 4))
            support = support_values(valid, record["W"], record["H"], mode)
            regions = contract_regions(valid, source_ignore, support, threshold, "S")
            group = groups.get(image_name, pd.DataFrame())
            pred = group[["x1", "y1", "x2", "y2"]].to_numpy(float) if len(group) else np.empty((0, 4))
            scores = group.score.to_numpy(float) if len(group) else np.empty(0)
            for iou in IOUS:
                iou = float(iou)
                accumulators[iou]["gt"][image_index] = len(regions.valid)
                for confidence in CONFIDENCES:
                    chosen = scores >= confidence
                    tp, fp, fn, neutralized = evaluate_contract_counts(pred[chosen], scores[chosen], regions, iou)
                    count_rows.append({
                        "candidate": candidate,
                        "short_candidate": SHORT[candidate],
                        "image_name": image_name,
                        "sequence_id": sequence_id(image_name),
                        "mode": mode,
                        "threshold": float(threshold),
                        "contract": "S",
                        "confidence": float(confidence),
                        "iou": iou,
                        "tp": int(tp),
                        "fp": int(fp),
                        "fn": int(fn),
                        "neutralized": int(neutralized),
                    })
                score, tp_flag, fp_flag = score_ordered_detection_outcomes(
                    pred, scores, regions, iou, max_dets=MAX_DETS,
                )
                if len(score):
                    accumulators[iou]["score"].append(score)
                    accumulators[iou]["tp"].append(tp_flag)
                    accumulators[iou]["fp"].append(fp_flag)
                    accumulators[iou]["img"].append(np.full(len(score), image_index, dtype=np.int32))
        for iou in IOUS:
            iou = float(iou)
            block = accumulators[iou]
            score = np.concatenate(block["score"]) if block["score"] else np.empty(0)
            tp = np.concatenate(block["tp"]) if block["tp"] else np.empty(0, dtype=np.int8)
            fp = np.concatenate(block["fp"]) if block["fp"] else np.empty(0, dtype=np.int8)
            image = np.concatenate(block["img"]) if block["img"] else np.empty(0, dtype=np.int32)
            order = np.argsort(-score, kind="mergesort")
            score, tp, fp, image = score[order], tp[order], fp[order], image[order]
            positive = np.flatnonzero(tp)
            if len(positive):
                end = int(positive[-1]) + 1
                score, tp, fp, image = score[:end], tp[:end], fp[:end], image[:end]
            ap_records[(mode, float(threshold), iou)] = {
                "score": score, "tp": tp, "fp": fp, "img": image, "gt": block["gt"],
            }
    return count_rows, ap_records


def summarize_points(image_counts: pd.DataFrame, ap_records: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["candidate", "mode", "threshold", "iou"]
    for (candidate, mode, threshold, iou), block in image_counts.groupby(keys):
        by_confidence = block.groupby("confidence", as_index=False)[["tp", "fp", "fn"]].sum().sort_values("confidence")
        by_confidence["f1"] = micro_f1(
            by_confidence.tp.to_numpy(), by_confidence.fp.to_numpy(), by_confidence.fn.to_numpy(),
        )
        fixed = by_confidence[np.isclose(by_confidence.confidence, OPERATING_CONFIDENCE)].iloc[0]
        best = by_confidence.sort_values(["f1", "confidence"], ascending=[False, True]).iloc[0]
        record = ap_records[candidate][(mode, float(threshold), float(iou))]
        ap = pooled_ap(record, np.ones(len(record["gt"]), dtype=np.int16))
        rows.append({
            "candidate": candidate,
            "short_candidate": SHORT[candidate],
            "mode": mode,
            "threshold": float(threshold),
            "contract": "S",
            "common_images": 548,
            "iou": float(iou),
            "operating_confidence": OPERATING_CONFIDENCE,
            "f1_at_025": float(fixed.f1),
            "max_f1": float(best.f1),
            "max_f1_confidence": float(best.confidence),
            "ap": ap,
            "ap_max_dets": MAX_DETS,
        })
    return pd.DataFrame(rows).sort_values(["mode", "threshold", "iou", "candidate"])


def bootstrap_count_metrics(
    values: np.ndarray,
    unit_counts: np.ndarray,
    image_unit: np.ndarray,
) -> np.ndarray:
    image_weights = unit_counts[:, image_unit]
    shape = values.shape
    totals = (image_weights @ values.reshape(shape[0], -1)).reshape(len(unit_counts), shape[1], 3)
    return micro_f1(totals[:, :, 0], totals[:, :, 1], totals[:, :, 2])


def pairwise_row(
    metric: str,
    mode: str,
    threshold: float,
    iou: float,
    point_a: float,
    point_b: float,
    differences: np.ndarray,
    replicates: int,
) -> dict:
    return {
        "metric": metric,
        "mode": mode,
        "threshold": float(threshold),
        "contract": "S",
        "iou": float(iou),
        "candidate_a": Y11M,
        "candidate_b": ASD,
        "point_a": float(point_a),
        "point_b": float(point_b),
        "point_difference": float(point_a - point_b),
        "ci95_low": float(np.quantile(differences, .025)),
        "ci95_high": float(np.quantile(differences, .975)),
        "probability_a_gt_b": float(np.mean(differences > 0)),
        "resampling_unit": "sequence",
        "sampling_units": 76,
        "common_images": 548,
        "replicates": int(replicates),
    }


def cocoeval_ap(
    pred: pd.DataFrame,
    names: list[str],
    gt: dict,
    mode: str,
    threshold: float,
    iou: float,
) -> float:
    images: list[dict] = []
    annotations: list[dict] = []
    detections: list[dict] = []
    for image_index, image_name in enumerate(names, start=1):
        record = gt[image_name]
        images.append({
            "id": image_index, "file_name": image_name,
            "width": int(record["W"]), "height": int(record["H"]),
        })
        valid = np.asarray([box[:4] for box in record["boxes"]], dtype=float).reshape((-1, 4))
        source_ignore = np.asarray([box[:4] for box in record["ignore"]], dtype=float).reshape((-1, 4))
        support = support_values(valid, record["W"], record["H"], mode)
        regions = contract_regions(valid, source_ignore, support, threshold, "S")
        for box in regions.valid:
            width, height = float(box[2] - box[0]), float(box[3] - box[1])
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_index, "category_id": 1,
                "bbox": [float(box[0]), float(box[1]), width, height], "area": width * height,
                "iscrowd": 0, "ignore": 0,
            })
        for box in regions.ignore:
            width, height = float(box[2] - box[0]), float(box[3] - box[1])
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_index, "category_id": 1,
                "bbox": [float(box[0]), float(box[1]), width, height], "area": width * height,
                "iscrowd": 1, "ignore": 1,
            })
    image_ids = {name: index for index, name in enumerate(names, start=1)}
    for row in pred.itertuples():
        detections.append({
            "image_id": image_ids[str(row.img_name)], "category_id": 1,
            "bbox": [float(row.x1), float(row.y1), float(row.x2 - row.x1), float(row.y2 - row.y1)],
            "score": float(row.score),
        })
    payload = {
        "info": {}, "licenses": [], "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": "vehicle"}],
    }
    with tempfile.TemporaryDirectory() as tmp:
        gt_path, dt_path = Path(tmp) / "gt.json", Path(tmp) / "dt.json"
        gt_path.write_text(json.dumps(payload)); dt_path.write_text(json.dumps(detections))
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(str(gt_path)); coco_dt = coco_gt.loadRes(str(dt_path))
            evaluator = COCOeval(coco_gt, coco_dt, "bbox")
            evaluator.params.catIds = [1]
            evaluator.params.imgIds = list(range(1, len(names) + 1))
            evaluator.params.iouThrs = np.asarray([iou])
            evaluator.params.maxDets = [1, 10, MAX_DETS]
            evaluator.evaluate(); evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    values = precision[precision > -1]
    return float(values.mean()) if len(values) else float("nan")


def main() -> None:
    args = parse_args()
    if args.replicates < 100:
        raise ValueError("at least 100 replicates are required")
    cfg = yaml.safe_load(CONFIG.read_text())
    vehicle_classes = set(cfg["shared_vehicle_native_classes"])
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

    raw_predictions: dict[str, pd.DataFrame] = {}
    all_count_rows: list[dict] = []
    ap_records: dict[str, dict] = {}
    for candidate in (Y11M, ASD):
        pred, groups = prepare_predictions(candidate, vehicle_classes)
        raw_predictions[candidate] = pred
        rows, records = build_candidate_records(candidate, groups, names, gt)
        all_count_rows.extend(rows)
        ap_records[candidate] = records
        print(json.dumps({
            "candidate": candidate,
            "count_rows": len(rows),
            "ap_detections": {f"{mode}:{threshold}:{iou}": len(record["tp"])
                              for (mode, threshold, iou), record in records.items()},
        }), flush=True)

    image_counts = pd.DataFrame(all_count_rows)
    points = summarize_points(image_counts, ap_records)

    rng = np.random.default_rng(args.seed)
    unit_count = len(unique_sequences)
    # Consume the registered image-bootstrap draw first so the sequence draws
    # exactly match the existing metric-qualified rank record.
    rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=args.replicates)
    sampled = rng.multinomial(unit_count, np.full(unit_count, 1 / unit_count), size=args.replicates)
    pairwise_rows: list[dict] = []
    fixed_index = int(np.where(np.isclose(CONFIDENCES, OPERATING_CONFIDENCE))[0][0])
    for mode, threshold in RULES:
        for iou in IOUS:
            iou = float(iou)
            arrays: dict[str, np.ndarray] = {}
            for candidate in (Y11M, ASD):
                block = image_counts[
                    image_counts.candidate.eq(candidate)
                    & image_counts["mode"].eq(mode)
                    & np.isclose(image_counts.threshold, threshold)
                    & np.isclose(image_counts.iou, iou)
                ].copy()
                block["image_index"] = pd.Categorical(block.image_name, categories=names, ordered=True).codes
                block["confidence_index"] = pd.Categorical(
                    block.confidence, categories=CONFIDENCES, ordered=True,
                ).codes
                block = block.sort_values(["image_index", "confidence_index"])
                arrays[candidate] = block[["tp", "fp", "fn"]].to_numpy(np.int64).reshape(
                    len(names), len(CONFIDENCES), 3,
                )
            a_f1 = bootstrap_count_metrics(arrays[Y11M], sampled, image_sequence)
            b_f1 = bootstrap_count_metrics(arrays[ASD], sampled, image_sequence)
            point_block = points[
                points["mode"].eq(mode)
                & np.isclose(points.threshold, threshold)
                & np.isclose(points.iou, iou)
            ].set_index("candidate")
            pairwise_rows.append(pairwise_row(
                "F1@conf=.25", mode, threshold, iou,
                point_block.loc[Y11M, "f1_at_025"], point_block.loc[ASD, "f1_at_025"],
                a_f1[:, fixed_index] - b_f1[:, fixed_index], args.replicates,
            ))
            pairwise_rows.append(pairwise_row(
                "max-F1", mode, threshold, iou,
                point_block.loc[Y11M, "max_f1"], point_block.loc[ASD, "max_f1"],
                a_f1.max(axis=1) - b_f1.max(axis=1), args.replicates,
            ))
            a_ap = pooled_ap_bootstrap(
                ap_records[Y11M][(mode, float(threshold), iou)], sampled, image_sequence, args.ap_batch,
            )
            b_ap = pooled_ap_bootstrap(
                ap_records[ASD][(mode, float(threshold), iou)], sampled, image_sequence, args.ap_batch,
            )
            pairwise_rows.append(pairwise_row(
                "AP", mode, threshold, iou,
                point_block.loc[Y11M, "ap"], point_block.loc[ASD, "ap"],
                a_ap - b_ap, args.replicates,
            ))
            print(json.dumps({"mode": mode, "threshold": threshold, "iou": iou,
                              "completed_metrics": ["F1@conf=.25", "max-F1", "AP"]}), flush=True)

    crosscheck_rows: list[dict] = []
    for candidate in (Y11M, ASD):
        for mode, threshold in RULES:
            for iou in (.50,):
                custom = float(points[
                    points.candidate.eq(candidate)
                    & points["mode"].eq(mode)
                    & np.isclose(points.threshold, threshold)
                    & np.isclose(points.iou, iou)
                ].iloc[0].ap)
                reference = cocoeval_ap(raw_predictions[candidate], names, gt, mode, threshold, iou)
                difference = custom - reference
                crosscheck_rows.append({
                    "candidate": candidate, "mode": mode, "threshold": float(threshold),
                    "contract": "S", "iou": iou, "custom_ap": custom, "cocoeval_ap": reference,
                    "difference": difference, "absolute_difference": abs(difference),
                    "contract_aligned": True,
                    "tolerance": 5e-4, "pass": bool(abs(difference) <= 5e-4),
                    "max_dets": MAX_DETS,
                })
    crosscheck = pd.DataFrame(crosscheck_rows)
    if not crosscheck["pass"].all():
        raise AssertionError("custom AP and COCOeval differ beyond the registered tolerance")

    pairwise = pd.DataFrame(pairwise_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    image_counts.to_parquet(OUT / "factorial_image_counts.parquet", index=False)
    points.to_csv(OUT / "factorial_points.csv", index=False)
    pairwise.to_csv(OUT / "factorial_pairwise_bootstrap.csv", index=False)
    crosscheck.to_csv(OUT / "factorial_ap_crosscheck.csv", index=False)
    summary = {
        "purpose": "separate localization threshold from score integration",
        "contract": "S",
        "candidates": [Y11M, ASD],
        "common_images": 548,
        "sequence_clusters": 76,
        "sequence_definition": "leading seven-digit VisDrone filename token",
        "rules": [{"mode": mode, "threshold": threshold} for mode, threshold in RULES],
        "ious": IOUS.tolist(),
        "confidence_grid": CONFIDENCES.tolist(),
        "operating_confidence": OPERATING_CONFIDENCE,
        "metrics": ["F1@conf=.25", "max-F1", "AP"],
        "ap_max_dets": MAX_DETS,
        "replicates": args.replicates,
        "seed": args.seed,
        "ap_bootstrap": "pooled detections recomputed for every paired sequence resample",
        "cocoeval_contract_aligned_iou": .50,
        "ap25_cocoeval_note": "diagnostic only: COCOeval couples crowd neutralization to evaluated IoU, whereas contract S fixes neutralization at IoP=.5",
        "crosscheck_max_abs_difference": float(crosscheck.absolute_difference.max()),
    }
    (OUT / "README.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nFactorial points\n", points.to_string(index=False))
    print("\nSequence-paired differences\n", pairwise.to_string(index=False))
    print("\nAP evaluator crosscheck\n", crosscheck.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
