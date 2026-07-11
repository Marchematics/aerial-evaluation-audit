"""Materialize the compact, predeclared AP sensitivity surface for the paper.

This is deliberately *not* an AP x confidence lattice: AP integrates over the
score ordering.  It evaluates every frozen candidate only on its own declared
source, at the two predeclared headline scale rules (24 px and .015 normalized
side) and three small-object policies.  DOTA/xView remain structural-only
because no frozen prediction cache exists for them.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import ast
from pathlib import Path

import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from label_reliability.inventory import load_ontology, load_uavdt_coco, load_visdrone_split
from label_reliability.aitod_provenance_bridge import load_aitod
from evaluate_cached_vehicle_grid import CANDIDATES

POLICIES = ("include_all", "exclude_without_ignore_protection", "exclude_with_ignore_protection")
HEADLINES = (("absolute", 24.0), ("normalized", 0.015))


def source_records(source: str):
    ontology = load_ontology(ROOT / "label_reliability/config/ontology.json")
    if source == "visdrone":
        return load_visdrone_split(Path("/root/zjh_UAV_detection/experiments/visdrone/data/VisDrone/VisDrone2019-DET-val"), "val", ontology)
    if source == "uavdt":
        return load_uavdt_coco(Path("/root/zjh_UAV_detection/external/zoomdet_yolo_official/data/UAVDT_local/annotations/test.json"), ontology)
    if source == "aitod":
        return load_aitod(Path("/root/zjh_UAV_detection/datasets/aitod/annotations/aitod_val.json"), Path("/root/zjh_UAV_detection/datasets/aitod/images/val"))
    raise ValueError(source)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-12)


def evaluate(candidate: str, mode: str, threshold: float, policy: str):
    source, pred_path, classes = CANDIDATES[candidate]
    records = source_records(source)
    gt = {r["file_name"]: r for r in records}
    raw = pd.read_parquet(pred_path)
    covered_names = set(raw.img_name.unique()).intersection(gt)
    pred = raw[raw.cls.isin(classes)]
    groups = {name: group for name, group in pred.groupby("img_name", sort=False)}
    names = sorted(covered_names)
    images, annotations, result_rows = [], [], []
    image_id = {name: i for i, name in enumerate(names, start=1)}
    for name in names:
        record = gt[name]
        iid = image_id[name]
        images.append({"id": iid, "file_name": name, "width": int(record["width"]), "height": int(record["height"])})
        for box in record["boxes"]:
            if box.is_source_ignore or box.shared_coarse != "vehicle":
                continue
            size = box.max_side if mode == "absolute" else box.max_side / max(record["width"], record["height"])
            if policy != "include_all" and size < threshold:
                continue
            annotations.append({"id": len(annotations) + 1, "image_id": iid, "category_id": 1,
                                "bbox": [box.x1, box.y1, box.width, box.height], "area": box.width * box.height, "iscrowd": 0})
        ignores = [box for box in record["boxes"] if box.is_source_ignore]
        for row in groups.get(name, pd.DataFrame()).itertuples():
            box = (row.x1, row.y1, row.x2, row.y2)
            if policy == "exclude_with_ignore_protection" and any(iou(box, (x.x1, x.y1, x.x2, x.y2)) >= 0.5 for x in ignores):
                continue
            result_rows.append({"image_id": iid, "category_id": 1,
                                "bbox": [float(row.x1), float(row.y1), float(row.x2 - row.x1), float(row.y2 - row.y1)],
                                "score": float(row.score)})
    # COCO's maxDets convention bounds the same object across all configurations.
    result_rows.sort(key=lambda x: -x["score"])
    payload = {"info": {}, "licenses": [], "images": images, "annotations": annotations,
               "categories": [{"id": 1, "name": "vehicle"}]}
    with tempfile.TemporaryDirectory() as tmp:
        gt_path, dt_path = Path(tmp) / "gt.json", Path(tmp) / "dt.json"
        gt_path.write_text(json.dumps(payload)); dt_path.write_text(json.dumps(result_rows))
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(str(gt_path)); coco_dt = coco_gt.loadRes(str(dt_path))
            evaluator = COCOeval(coco_gt, coco_dt, "bbox")
            evaluator.params.catIds = [1]; evaluator.params.imgIds = list(image_id.values())
            evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        stats = evaluator.stats
    return {"candidate": candidate, "source": source, "mode": mode, "threshold": threshold, "small_object_policy": policy,
            "images_with_predictions": len(names), "vehicle_annotations": len(annotations), "predictions": len(result_rows),
            "AP50_95": float(stats[0]), "AP50": float(stats[1]), "AP75": float(stats[2]),
            "AP_small": float(stats[3]), "AP_medium": float(stats[4]), "AP_large": float(stats[5])}


def main():
    out = ROOT / "outputs" / "ap_headline"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    # Preserve work if an evaluation is interrupted: each evaluation is a
    # deterministic independent record and is validated by its key below.
    partial = out / "ap_headline_surface.partial.csv"
    if partial.exists():
        rows = pd.read_csv(partial).to_dict("records")
    elif (out / "run.log").exists():
        for line in (out / "run.log").read_text(errors="replace").splitlines():
            if line.startswith("{") and line.endswith("}"):
                try:
                    rows.append(ast.literal_eval(line))
                except (ValueError, SyntaxError):
                    pass
        if rows:
            pd.DataFrame(rows).drop_duplicates(["candidate", "mode", "threshold", "small_object_policy"]).to_csv(partial, index=False)
            rows = pd.read_csv(partial).to_dict("records")
    done = {(r["candidate"], r["mode"], float(r["threshold"]), r["small_object_policy"]) for r in rows}
    for candidate in CANDIDATES:
        for mode, threshold in HEADLINES:
            for policy in POLICIES:
                if (candidate, mode, threshold, policy) in done:
                    continue
                row = evaluate(candidate, mode, threshold, policy)
                rows.append(row)
                pd.DataFrame(rows).to_csv(partial, index=False)
                print(row, flush=True)
    frame = pd.DataFrame(rows)
    expected = len(CANDIDATES) * len(HEADLINES) * len(POLICIES)
    if len(frame) != expected:
        raise RuntimeError(f"incomplete headline surface: {len(frame)}/{expected}")
    frame.to_parquet(out / "ap_headline_surface.parquet", index=False)
    frame.to_csv(out / "ap_headline_surface.csv", index=False)
    bands = (frame.groupby(["source", "candidate", "mode", "threshold"], as_index=False)
             .agg(AP50_min=("AP50", "min"), AP50_max=("AP50", "max"), AP50_95_min=("AP50_95", "min"), AP50_95_max=("AP50_95", "max")))
    bands["AP50_policy_band"] = bands.AP50_max - bands.AP50_min
    bands["AP50_95_policy_band"] = bands.AP50_95_max - bands.AP50_95_min
    bands.to_csv(out / "ap_policy_band.csv", index=False)
    winners = (frame.sort_values(["source", "mode", "threshold", "small_object_policy", "AP50"], ascending=[True, True, True, True, False])
               .groupby(["source", "mode", "threshold", "small_object_policy"], as_index=False).first())
    winners.to_csv(out / "ap_policy_winners.csv", index=False)
    (out / "README.md").write_text("# Headline AP surface\n\n42 rows = 7 frozen candidates x 2 declared scale rules x 3 small-object policies. Candidates are evaluated only on their declared source. DOTA-v2/xView are structural-only because no frozen raw-prediction cache is available. AP integrates score ranking; confidence is not crossed with AP.\n")
    print(json.dumps({"rows": len(frame), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
