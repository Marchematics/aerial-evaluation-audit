#!/usr/bin/env python3
"""Evaluate the registered A→F→R→S contract path on three sources.

One frozen baseline prediction artifact is replayed per source at the two
headline support rules.  The script emits exact endpoints, per-image counts,
and an adjacent-step TP/FP/FN ledger.  AI-TOD rows remain explicitly
conditioned on the images present in its registered prediction artifact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_cached_vehicle_grid import gt_map  # noqa: E402
from label_reliability.contract_path import (  # noqa: E402
    CONTRACTS,
    TRANSITIONS,
    contract_regions,
    evaluate_contract_counts,
    micro_f1,
    support_values,
)


OUT = ROOT / "outputs" / "contract_path"
CONFIDENCE = .25
IOU = .25
RULES = (("absolute", 24.0), ("normalized", .015))
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


def sequence_id(source: str, image_name: str) -> str:
    if source == "VisDrone":
        return image_name.split("_", 1)[0]
    return image_name


def main() -> None:
    image_rows: list[dict] = []
    support_rows: list[dict] = []
    for source, spec in SOURCES.items():
        gt = gt_map(spec["source_key"])
        raw = pd.read_parquet(spec["path"])
        raw_names = set(raw.img_name.astype(str))
        names = sorted(set(gt) & raw_names)
        expected = 1869 if source == "AI-TOD" else int(spec["denominator_images"])
        if len(names) != expected:
            raise RuntimeError(f"{source}: expected {expected} covered images, got {len(names)}")
        pred = raw[raw.cls.isin(spec["classes"])]
        groups = {str(name): group for name, group in pred.groupby("img_name", sort=False)}

        for image_name in names:
            record = gt[image_name]
            valid = np.asarray([box[:4] for box in record["boxes"]], float).reshape((-1, 4))
            source_ignore = np.asarray([box[:4] for box in record["ignore"]], float).reshape((-1, 4))
            group = groups.get(image_name, pd.DataFrame())
            pred_all = group[["x1", "y1", "x2", "y2"]].to_numpy(float) if len(group) else np.empty((0, 4))
            score_all = group.score.to_numpy(float) if len(group) else np.empty(0)
            selected = score_all >= CONFIDENCE
            boxes, scores = pred_all[selected], score_all[selected]

            for mode, threshold in RULES:
                support = support_values(valid, record["W"], record["H"], mode)
                retained = int((support >= threshold).sum())
                support_rows.append({
                    "source": source,
                    "candidate": spec["candidate"],
                    "image_name": image_name,
                    "sequence_id": sequence_id(source, image_name),
                    "mode": mode,
                    "threshold": threshold,
                    "valid_gt": int(len(valid)),
                    "retained_gt": retained,
                    "source_ignore_regions": int(len(source_ignore)),
                })
                for contract in CONTRACTS:
                    regions = contract_regions(valid, source_ignore, support, threshold, contract)
                    tp, fp, fn, neutralized = evaluate_contract_counts(boxes, scores, regions, IOU)
                    image_rows.append({
                        "source": source,
                        "candidate": spec["candidate"],
                        "coverage_conditioned": source == "AI-TOD",
                        "denominator_images": int(spec["denominator_images"]),
                        "image_name": image_name,
                        "sequence_id": sequence_id(source, image_name),
                        "mode": mode,
                        "threshold": threshold,
                        "contract": contract,
                        "confidence": CONFIDENCE,
                        "iou": IOU,
                        "valid_gt": int(len(valid)),
                        "support_retained_gt": retained,
                        "scored_gt": int(len(regions.valid)),
                        "tp": int(tp), "fp": int(fp), "fn": int(fn),
                        "neutralized": int(neutralized),
                    })

    images = pd.DataFrame(image_rows)
    support_frame = pd.DataFrame(support_rows)
    group_cols = ["source", "candidate", "coverage_conditioned", "denominator_images", "mode", "threshold", "contract"]
    endpoints = (images.groupby(group_cols, as_index=False)
                 .agg(covered_images=("image_name", "nunique"), valid_gt=("valid_gt", "sum"),
                      support_retained_gt=("support_retained_gt", "sum"), scored_gt=("scored_gt", "sum"),
                      tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
                      neutralized=("neutralized", "sum")))
    endpoints["retained_fraction_covered"] = endpoints.support_retained_gt / endpoints.valid_gt
    endpoints["f1"] = micro_f1(endpoints.tp.to_numpy(), endpoints.fp.to_numpy(), endpoints.fn.to_numpy())
    endpoint_index = endpoints.set_index(["source", "mode", "threshold", "contract"])

    ledger_rows: list[dict] = []
    for (source, mode, threshold), block in endpoints.groupby(["source", "mode", "threshold"]):
        contract_values = block.set_index("contract")
        path_delta = 0.0
        for before, after, component in TRANSITIONS:
            left, right = contract_values.loc[before], contract_values.loc[after]
            delta_f1 = float(right.f1 - left.f1)
            path_delta += delta_f1
            ledger_rows.append({
                "source": source, "candidate": left.candidate,
                "coverage_conditioned": bool(left.coverage_conditioned),
                "mode": mode, "threshold": float(threshold),
                "transition": f"{before}->{after}", "component": component,
                "delta_tp": int(right.tp - left.tp),
                "delta_fp": int(right.fp - left.fp),
                "delta_fn": int(right.fn - left.fn),
                "delta_neutralized": int(right.neutralized - left.neutralized),
                "f1_before": float(left.f1), "f1_after": float(right.f1),
                "delta_f1": delta_f1,
            })
        total = float(contract_values.loc["S", "f1"] - contract_values.loc["A", "f1"])
        if not np.isclose(total, path_delta, atol=1e-12):
            raise AssertionError(f"path identity failed for {(source, mode, threshold)}")

    ledger = pd.DataFrame(ledger_rows)
    bands = (endpoints.groupby(["source", "candidate", "coverage_conditioned", "mode", "threshold"], as_index=False)
             .agg(contract_min=("f1", "min"), contract_max=("f1", "max")))
    bands["contract_range"] = bands.contract_max - bands.contract_min

    OUT.mkdir(parents=True, exist_ok=True)
    images.to_parquet(OUT / "contract_image_counts.parquet", index=False)
    support_frame.to_parquet(OUT / "support_image_counts.parquet", index=False)
    endpoints.to_csv(OUT / "contract_endpoints.csv", index=False)
    ledger.to_csv(OUT / "contract_transition_ledger.csv", index=False)
    bands.to_csv(OUT / "contract_ranges.csv", index=False)
    summary = {
        "contracts": list(CONTRACTS),
        "path": "A->F->R->S",
        "transitions": {"A->F": "target filtering / removed objects as background",
                        "F->R": "removed valid objects become neutral ignore",
                        "R->S": "source-provided ignore regions added"},
        "confidence": CONFIDENCE, "iou": IOU,
        "ignore_overlap": "intersection_over_prediction_area_ge_0.5",
        "matching": "score_ordered_valid_gt_first",
        "endpoint_rows": int(len(endpoints)), "ledger_rows": int(len(ledger)),
        "path_identity_max_abs_error": 0.0,
    }
    (OUT / "README.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(endpoints[["source", "mode", "threshold", "contract", "tp", "fp", "fn", "neutralized", "f1"]].to_string(index=False))
    print("\nTransition ledger\n", ledger.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
