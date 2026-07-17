#!/usr/bin/env python3
"""Stratify VisDrone contract paths and report operational consequences."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs" / "scene_consequence_freeze_20260717.yaml"
INPUT = ROOT / "outputs" / "visdrone_path_support_robustness" / "path_image_counts.parquet"
OUT = ROOT / "outputs" / "visdrone_scene_consequences"
ANN = Path("/root/zjh_UAV_detection/experiments/visdrone/data/VisDrone/VisDrone2019-DET-val/annotations")
IMAGES = Path("/root/zjh_UAV_detection/experiments/visdrone/data/VisDrone/VisDrone2019-DET-val/images")
VEHICLE_CLASSES = {4, 5, 6, 9}
TRANSITIONS = (("A", "F", "delta_target"), ("F", "R", "delta_removed"), ("R", "S", "delta_source"))


def micro_f1(tp: float, fp: float, fn: float) -> float:
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def parse_scene_features() -> pd.DataFrame:
    from PIL import Image

    rows = []
    for path in sorted(ANN.glob("*.txt")):
        image_name = f"{path.stem}.jpg"
        with Image.open(IMAGES / image_name) as image:
            width, height = image.size
        boxes = []
        occlusion = []
        for raw in path.read_text().splitlines():
            parts = [item.strip() for item in raw.split(",")]
            if len(parts) < 8:
                continue
            x, y, w, h = map(float, parts[:4])
            score, class_id = int(float(parts[4])), int(float(parts[5]))
            if score == 0 or class_id not in VEHICLE_CLASSES or w <= 0 or h <= 0:
                continue
            boxes.append((x, y, w, h))
            occlusion.append(int(float(parts[7])) > 0)
        density = len(boxes)
        if boxes:
            sides = np.asarray([max(w, h) for _, _, w, h in boxes], dtype=float)
            centers = np.asarray([((x + w / 2) / width, (y + h / 2) / height)
                                  for x, y, w, h in boxes], dtype=float)
            median_side = float(np.median(sides))
            occluded_fraction = float(np.mean(occlusion))
        else:
            centers = np.empty((0, 2))
            median_side = np.nan
            occluded_fraction = np.nan
        if len(centers) >= 2:
            delta = centers[:, None, :] - centers[None, :, :]
            distance = np.sqrt((delta ** 2).sum(axis=2))
            np.fill_diagonal(distance, np.inf)
            nearest_neighbor = float(np.median(distance.min(axis=1)))
        else:
            nearest_neighbor = np.nan
        rows.append({
            "image_name": image_name, "sequence_id": image_name.split("_", 1)[0],
            "width": width, "height": height, "density": density,
            "median_target_side_px": median_side,
            "median_normalized_nearest_neighbor": nearest_neighbor,
            "occluded_fraction": occluded_fraction,
        })
    return pd.DataFrame(rows)


def tertile_membership(features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    definitions = {
        "density": ("density", ["sparse", "moderate", "dense"]),
        "target_size": ("median_target_side_px", ["small", "medium", "large"]),
        "nearest_neighbor": ("median_normalized_nearest_neighbor", ["crowded", "intermediate", "dispersed"]),
        "occlusion": ("occluded_fraction", ["low", "middle", "high"]),
    }
    rows = []
    cutpoint_record = {}
    for stratifier, (column, labels) in definitions.items():
        eligible = features[features[column].notna()].copy()
        q1, q2 = eligible[column].quantile([1 / 3, 2 / 3]).to_numpy(float)
        if np.isclose(q1, q2):
            order = eligible[column].rank(method="first")
            bins = pd.qcut(order, 3, labels=labels)
            method = "rank_tertiles_due_to_tied_quantiles"
        else:
            bins = pd.cut(eligible[column], [-np.inf, q1, q2, np.inf], labels=labels,
                          include_lowest=True, ordered=True)
            method = "value_tertiles"
        eligible["stratum"] = bins.astype(str)
        eligible["stratum_order"] = eligible.stratum.map({label: index for index, label in enumerate(labels)})
        for row in eligible.itertuples():
            rows.append({
                "image_name": row.image_name, "sequence_id": row.sequence_id,
                "stratifier": stratifier, "feature": column,
                "feature_value": float(getattr(row, column)), "stratum": row.stratum,
                "stratum_order": int(row.stratum_order),
            })
        counts = eligible.stratum.value_counts().reindex(labels, fill_value=0)
        cutpoint_record[stratifier] = {
            "feature": column, "method": method, "lower_cutpoint": float(q1),
            "upper_cutpoint": float(q2), "eligible_images": int(len(eligible)),
            "stratum_counts": {label: int(counts[label]) for label in labels},
        }
    return pd.DataFrame(rows), cutpoint_record


def add_operational_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["f1"] = [micro_f1(tp, fp, fn) for tp, fp, fn in zip(frame.tp, frame.fp, frame.fn)]
    frame["false_positives_per_image"] = frame.fp / frame.images
    frame["false_negatives_per_100_scored_vehicles"] = 100 * frame.fn / frame.scored_gt.replace(0, np.nan)
    frame["confirmed_vehicles_per_100_scored_alerts"] = 100 * frame.tp / (frame.tp + frame.fp).replace(0, np.nan)
    return frame


def main() -> None:
    if not FREEZE.exists():
        raise FileNotFoundError(FREEZE)
    features = parse_scene_features()
    if len(features) != 548:
        raise RuntimeError(f"expected 548 VisDrone images, got {len(features)}")
    membership, cutpoints = tertile_membership(features)

    counts = pd.read_parquet(INPUT)
    counts = counts[counts.primary & counts.definition.eq("max_side")].copy()
    if counts.image_name.nunique() != 548 or counts.candidate.nunique() != 5:
        raise RuntimeError("frozen path input does not contain five configurations on 548 images")
    stratified = counts.merge(membership, on=["image_name", "sequence_id"], how="inner", validate="many_to_many")
    group = ["candidate", "short_candidate", "rule_id", "coordinate", "threshold",
             "stratifier", "stratum", "stratum_order", "contract"]
    endpoints = stratified.groupby(group, as_index=False).agg(
        images=("image_name", "nunique"), valid_gt=("valid_gt", "sum"),
        scored_gt=("scored_gt", "sum"), tp=("tp", "sum"), fp=("fp", "sum"),
        fn=("fn", "sum"), neutralized=("neutralized", "sum"),
    )
    endpoints = add_operational_metrics(endpoints)

    path_rows = []
    path_keys = ["candidate", "short_candidate", "rule_id", "coordinate", "threshold",
                 "stratifier", "stratum", "stratum_order"]
    for key, block in endpoints.groupby(path_keys):
        indexed = block.set_index("contract")
        row = dict(zip(path_keys, key))
        for before, after, name in TRANSITIONS:
            row[name] = float(indexed.loc[after, "f1"] - indexed.loc[before, "f1"])
        row["contract_range"] = float(indexed.f1.max() - indexed.f1.min())
        row["images"] = int(indexed.images.iloc[0])
        row["scored_gt_terminal"] = int(indexed.loc["S", "scored_gt"])
        path_rows.append(row)
    paths = pd.DataFrame(path_rows)

    summary = paths.groupby(
        ["rule_id", "coordinate", "threshold", "stratifier", "stratum", "stratum_order"], as_index=False
    ).agg(
        configurations=("candidate", "size"), images=("images", "first"),
        delta_target_median=("delta_target", "median"), delta_target_min=("delta_target", "min"),
        delta_target_max=("delta_target", "max"), delta_removed_median=("delta_removed", "median"),
        delta_removed_min=("delta_removed", "min"), delta_removed_max=("delta_removed", "max"),
        contract_range_median=("contract_range", "median"), contract_range_min=("contract_range", "min"),
        contract_range_max=("contract_range", "max"),
    )

    terminal = endpoints[endpoints.contract.eq("S")].copy()
    pair = terminal[terminal.short_candidate.isin(["Y11m", "ASD"])].pivot_table(
        index=["rule_id", "coordinate", "threshold", "stratifier", "stratum", "stratum_order", "images"],
        columns="short_candidate", values="f1",
    ).reset_index()
    pair["y11m_minus_asd_f1"] = pair.Y11m - pair.ASD

    overall_group = ["candidate", "short_candidate", "rule_id", "coordinate", "threshold", "contract"]
    overall = counts.groupby(overall_group, as_index=False).agg(
        images=("image_name", "nunique"), valid_gt=("valid_gt", "sum"),
        scored_gt=("scored_gt", "sum"), tp=("tp", "sum"), fp=("fp", "sum"),
        fn=("fn", "sum"), neutralized=("neutralized", "sum"),
    )
    overall = add_operational_metrics(overall[overall.contract.eq("S")])
    operational_by_stratum = terminal[[
        "candidate", "short_candidate", "rule_id", "coordinate", "threshold", "stratifier",
        "stratum", "stratum_order", "images", "scored_gt", "tp", "fp", "fn",
        "false_positives_per_image", "false_negatives_per_100_scored_vehicles",
        "confirmed_vehicles_per_100_scored_alerts", "f1",
    ]].copy()

    OUT.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT / "scene_features.csv", index=False)
    membership.to_csv(OUT / "stratum_membership.csv", index=False)
    (OUT / "stratum_cutpoints.json").write_text(json.dumps(cutpoints, indent=2) + "\n")
    endpoints.to_csv(OUT / "stratified_contract_endpoints.csv", index=False)
    paths.to_csv(OUT / "stratified_candidate_paths.csv", index=False)
    summary.to_csv(OUT / "stratified_path_summary.csv", index=False)
    pair.to_csv(OUT / "stratified_pair_differences.csv", index=False)
    overall.to_csv(OUT / "operational_metrics_overall.csv", index=False)
    operational_by_stratum.to_csv(OUT / "operational_metrics_by_stratum.csv", index=False)
    metadata = {
        "freeze_record": str(FREEZE.relative_to(ROOT)), "images": 548,
        "configurations": 5, "stratifiers": list(cutpoints),
        "path_rules": 2, "path_stratum_rows": int(len(paths)),
        "pair_stratum_rows": int(len(pair)), "operational_definition": {
            "false_positives_per_image": "FP divided by images at terminal S and confidence .25",
            "false_negatives_per_100_scored_vehicles": "100*FN divided by scored GT at terminal S",
            "confirmed_vehicles_per_100_scored_alerts": "100*TP divided by TP+FP after contract neutralization",
        },
    }
    (OUT / "README.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(cutpoints, indent=2))
    print("\nOverall operational metrics\n", overall.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
