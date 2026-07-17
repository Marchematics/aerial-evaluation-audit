#!/usr/bin/env python3
"""Replay A→F→R→S across existing configurations in three aerial sources.

The frozen cohort contains five VisDrone artifacts, four UAVDT configurations,
and three resolution slices of the coverage-conditioned AI-TOD artifact.  No
training or result-dependent configuration selection occurs in this script.
"""
from __future__ import annotations

import hashlib
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

OUT = ROOT / "outputs" / "cross_configuration_contract_paths"
FREEZE = ROOT / "configs" / "cross_configuration_path_freeze_20260717.yaml"
CONFIDENCE = .25
IOU = .25
RULES = (("absolute", 24.0), ("normalized", .015))

VD_ROOT = ROOT / "outputs" / "resubmission_controlled_pool" / "visdrone"
UAV_MULTI = Path("/root/zjh_UAV_detection/experiments/uavdt/oracle_route/cache_val_baseline640/pred_rows.parquet")
UAV_TILE = Path("/root/zjh_UAV_detection/experiments/uavdt/oracle_route/router_runs/tiling_baseline/pred_rows.parquet")
AITOD_MULTI = Path("/root/zjh_UAV_detection/experiments/aitod/oracle_route/cache_val_baseline640/pred_rows.parquet")

CANDIDATES = [
    # VisDrone: five separately materialized prediction artifacts.
    dict(source="VisDrone", source_key="visdrone", candidate="B-640",
         path=VD_ROOT / "visdrone_control_baseline640" / "pred_rows.parquet",
         classes={3, 4, 5, 8}, resolution=640, expected_images=548,
         coverage_conditioned=False, artifact_group="vd_b640"),
    dict(source="VisDrone", source_key="visdrone", candidate="B-1280",
         path=VD_ROOT / "visdrone_control_baseline1280" / "pred_rows.parquet",
         classes={3, 4, 5, 8}, resolution=1280, expected_images=548,
         coverage_conditioned=False, artifact_group="vd_b1280"),
    dict(source="VisDrone", source_key="visdrone", candidate="Y11m",
         path=VD_ROOT / "visdrone_control_yolo11m_hf640" / "pred_rows.parquet",
         classes={3, 4, 5, 8}, resolution=640, expected_images=548,
         coverage_conditioned=False, artifact_group="vd_y11m"),
    dict(source="VisDrone", source_key="visdrone", candidate="ASD",
         path=VD_ROOT / "visdrone_control_asd1280" / "pred_rows.parquet",
         classes={3, 4, 5, 8}, resolution=1280, expected_images=548,
         coverage_conditioned=False, artifact_group="vd_asd"),
    dict(source="VisDrone", source_key="visdrone", candidate="Y11n-P2",
         path=VD_ROOT / "visdrone_control_yolo11n_p2_1280" / "pred_rows.parquet",
         classes={3, 4, 5, 8}, resolution=1280, expected_images=548,
         coverage_conditioned=False, artifact_group="vd_y11np2"),
    # UAVDT: three resolution slices of one trained baseline plus tiling.
    dict(source="UAVDT", source_key="uavdt", candidate="Base-640",
         path=UAV_MULTI, classes={0, 1, 2}, resolution=640, expected_images=305,
         coverage_conditioned=False, artifact_group="uavdt_baseline_multires"),
    dict(source="UAVDT", source_key="uavdt", candidate="Base-960",
         path=UAV_MULTI, classes={0, 1, 2}, resolution=960, expected_images=305,
         coverage_conditioned=False, artifact_group="uavdt_baseline_multires"),
    dict(source="UAVDT", source_key="uavdt", candidate="Base-1280",
         path=UAV_MULTI, classes={0, 1, 2}, resolution=1280, expected_images=305,
         coverage_conditioned=False, artifact_group="uavdt_baseline_multires"),
    dict(source="UAVDT", source_key="uavdt", candidate="Tile-960",
         path=UAV_TILE, classes={0, 1, 2}, resolution=960, expected_images=305,
         coverage_conditioned=False, artifact_group="uavdt_tiling"),
    # AI-TOD: three resolution slices on the same 1,869 covered images.
    dict(source="AI-TOD", source_key="aitod", candidate="Base-640",
         path=AITOD_MULTI, classes={5}, resolution=640, expected_images=1869,
         coverage_conditioned=True, artifact_group="aitod_baseline_multires"),
    dict(source="AI-TOD", source_key="aitod", candidate="Base-960",
         path=AITOD_MULTI, classes={5}, resolution=960, expected_images=1869,
         coverage_conditioned=True, artifact_group="aitod_baseline_multires"),
    dict(source="AI-TOD", source_key="aitod", candidate="Base-1280",
         path=AITOD_MULTI, classes={5}, resolution=1280, expected_images=1869,
         coverage_conditioned=True, artifact_group="aitod_baseline_multires"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sign_code(value: float, tolerance: float = 1e-12) -> str:
    if value > tolerance:
        return "+"
    if value < -tolerance:
        return "-"
    return "0"


def main() -> None:
    if not FREEZE.exists():
        raise FileNotFoundError(FREEZE)
    image_rows: list[dict] = []
    provenance_rows: list[dict] = []

    gt_cache: dict[str, dict] = {}
    raw_cache: dict[Path, pd.DataFrame] = {}
    for spec in CANDIDATES:
        source_key = spec["source_key"]
        if source_key not in gt_cache:
            gt_cache[source_key] = gt_map(source_key)
        gt = gt_cache[source_key]
        path = Path(spec["path"])
        if path not in raw_cache:
            raw_cache[path] = pd.read_parquet(path)
        raw = raw_cache[path]
        if "resolution" in raw.columns:
            raw = raw[np.isclose(raw.resolution.astype(float), float(spec["resolution"]))]
        raw_names = set(raw.img_name.astype(str))
        names = sorted(set(gt) & raw_names)
        if len(names) != int(spec["expected_images"]):
            raise RuntimeError(
                f"{spec['source']}/{spec['candidate']}: expected {spec['expected_images']} images, got {len(names)}"
            )
        predictions = raw[raw.cls.isin(spec["classes"])].copy()
        groups = {str(name): group for name, group in predictions.groupby("img_name", sort=False)}
        provenance_rows.append({
            "source": spec["source"], "candidate": spec["candidate"],
            "resolution": int(spec["resolution"]), "path": str(path),
            "artifact_group": spec["artifact_group"], "sha256": sha256(path),
            "covered_images": len(names), "raw_rows": len(raw),
            "vehicle_rows": len(predictions),
            "coverage_conditioned": bool(spec["coverage_conditioned"]),
        })

        for image_name in names:
            record = gt[image_name]
            valid = np.asarray([box[:4] for box in record["boxes"]], float).reshape((-1, 4))
            source_ignore = np.asarray([box[:4] for box in record["ignore"]], float).reshape((-1, 4))
            group = groups.get(image_name, pd.DataFrame())
            boxes = group[["x1", "y1", "x2", "y2"]].to_numpy(float) if len(group) else np.empty((0, 4))
            scores = group.score.to_numpy(float) if len(group) else np.empty(0)
            selected = scores >= CONFIDENCE
            boxes, scores = boxes[selected], scores[selected]
            for mode, threshold in RULES:
                support = support_values(valid, record["W"], record["H"], mode)
                for contract in CONTRACTS:
                    regions = contract_regions(valid, source_ignore, support, threshold, contract)
                    tp, fp, fn, neutralized = evaluate_contract_counts(boxes, scores, regions, IOU)
                    image_rows.append({
                        "source": spec["source"], "candidate": spec["candidate"],
                        "resolution": int(spec["resolution"]),
                        "artifact_group": spec["artifact_group"],
                        "coverage_conditioned": bool(spec["coverage_conditioned"]),
                        "image_name": image_name, "mode": mode, "threshold": threshold,
                        "contract": contract, "confidence": CONFIDENCE, "iou": IOU,
                        "valid_gt": len(valid), "scored_gt": len(regions.valid),
                        "tp": tp, "fp": fp, "fn": fn, "neutralized": neutralized,
                    })
        print(json.dumps({"source": spec["source"], "candidate": spec["candidate"],
                          "images": len(names), "vehicle_rows": len(predictions)}), flush=True)

    images = pd.DataFrame(image_rows)
    group = ["source", "candidate", "resolution", "artifact_group", "coverage_conditioned",
             "mode", "threshold", "contract"]
    endpoints = images.groupby(group, as_index=False).agg(
        images=("image_name", "nunique"), valid_gt=("valid_gt", "sum"),
        scored_gt=("scored_gt", "sum"), tp=("tp", "sum"), fp=("fp", "sum"),
        fn=("fn", "sum"), neutralized=("neutralized", "sum"),
    )
    endpoints["f1"] = micro_f1(endpoints.tp, endpoints.fp, endpoints.fn)

    ledger_rows: list[dict] = []
    identity_error = 0.0
    keys = ["source", "candidate", "resolution", "artifact_group", "coverage_conditioned", "mode", "threshold"]
    for key, block in endpoints.groupby(keys):
        indexed = block.set_index("contract")
        deltas = []
        for before, after, component in TRANSITIONS:
            left, right = indexed.loc[before], indexed.loc[after]
            delta = float(right.f1 - left.f1)
            deltas.append(delta)
            ledger_rows.append({
                **dict(zip(keys, key)), "transition": f"{before}->{after}", "component": component,
                "delta_tp": int(right.tp - left.tp), "delta_fp": int(right.fp - left.fp),
                "delta_fn": int(right.fn - left.fn),
                "delta_neutralized": int(right.neutralized - left.neutralized),
                "f1_before": float(left.f1), "f1_after": float(right.f1), "delta_f1": delta,
            })
        error = abs(float(indexed.loc["S", "f1"] - indexed.loc["A", "f1"] - sum(deltas)))
        identity_error = max(identity_error, error)
        if error > 1e-12:
            raise AssertionError(f"path identity failed for {key}: {error}")
    ledger = pd.DataFrame(ledger_rows)

    range_rows = []
    for key, block in endpoints.groupby(keys):
        values = block.set_index("contract").f1
        deltas = {
            "A->F": float(values.loc["F"] - values.loc["A"]),
            "F->R": float(values.loc["R"] - values.loc["F"]),
            "R->S": float(values.loc["S"] - values.loc["R"]),
        }
        range_rows.append({
            **dict(zip(keys, key)), "contract_min": float(values.min()),
            "contract_max": float(values.max()), "contract_range": float(values.max() - values.min()),
            "path_signature": "".join(sign_code(deltas[t]) for t in ("A->F", "F->R", "R->S")),
        })
    ranges = pd.DataFrame(range_rows)

    summary = ledger.groupby(["source", "mode", "threshold", "transition"], as_index=False).agg(
        configurations=("candidate", "size"), delta_f1_min=("delta_f1", "min"),
        delta_f1_max=("delta_f1", "max"), positive=("delta_f1", lambda x: int((x > 1e-12).sum())),
        negative=("delta_f1", lambda x: int((x < -1e-12).sum())),
        zero=("delta_f1", lambda x: int((np.abs(x) <= 1e-12).sum())),
    )
    range_summary = ranges.groupby(["source", "mode", "threshold"], as_index=False).agg(
        configurations=("candidate", "size"), contract_range_min=("contract_range", "min"),
        contract_range_max=("contract_range", "max"), distinct_path_signatures=("path_signature", "nunique"),
    )
    summary = summary.merge(range_summary, on=["source", "mode", "threshold", "configurations"], how="left")

    OUT.mkdir(parents=True, exist_ok=True)
    images.to_parquet(OUT / "image_counts.parquet", index=False)
    endpoints.to_csv(OUT / "contract_endpoints.csv", index=False)
    ledger.to_csv(OUT / "transition_ledger.csv", index=False)
    ranges.to_csv(OUT / "candidate_path_ranges.csv", index=False)
    summary.to_csv(OUT / "cross_configuration_summary.csv", index=False)
    pd.DataFrame(provenance_rows).to_csv(OUT / "artifact_provenance.csv", index=False)
    metadata = {
        "freeze_record": str(FREEZE.relative_to(ROOT)),
        "configurations": int(len(CANDIDATES)),
        "source_configuration_counts": {k: int(v) for k, v in ranges.groupby("source").candidate.nunique().items()},
        "rules": [{"mode": m, "threshold": t} for m, t in RULES],
        "contracts": list(CONTRACTS), "confidence": CONFIDENCE, "iou": IOU,
        "path_identity_max_abs_error": identity_error,
        "ai_tod_scope": "coverage-conditioned on 1,869/2,804 images",
        "independence_note": "Resolution slices share a raw cache and checkpoint; artifact_group identifies shared provenance.",
    }
    (OUT / "README.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
