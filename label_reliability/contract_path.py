"""Registered evaluation-contract path for support-conditioned detection.

The four contracts alter one evaluator choice at a time:

``A`` all mapped valid GT, no ignore protection;
``F`` support-filtered valid GT, removed objects remain background;
``R`` the same filtered target, with removed valid GT acting as neutral ignore;
``S`` the ``R`` contract plus source-provided ignore regions.

All matching is score ordered and valid-GT first.  Ignore regions are offered
only to detections that did not match a retained valid target.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from label_reliability.matching import greedy_match_with_ignores, iou_matrix


CONTRACTS = ("A", "F", "R", "S")
TRANSITIONS = (
    ("A", "F", "target"),
    ("F", "R", "removed"),
    ("R", "S", "source"),
)


@dataclass(frozen=True)
class ContractRegions:
    """Valid and neutral regions implied by one declared contract."""

    valid: np.ndarray
    ignore: np.ndarray
    retained_count: int
    removed_count: int


def _boxes(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape((-1, 4))


def support_values(valid: np.ndarray, width: float, height: float, mode: str) -> np.ndarray:
    """Return absolute or axis-normalized max-side support for valid boxes."""

    valid = _boxes(valid)
    if mode == "absolute":
        return np.maximum(valid[:, 2] - valid[:, 0], valid[:, 3] - valid[:, 1])
    if mode == "normalized":
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        return np.maximum(
            (valid[:, 2] - valid[:, 0]) / float(width),
            (valid[:, 3] - valid[:, 1]) / float(height),
        )
    raise ValueError("mode must be 'absolute' or 'normalized'")


def contract_regions(
    valid: np.ndarray,
    source_ignore: np.ndarray,
    support: np.ndarray,
    threshold: float,
    contract: str,
) -> ContractRegions:
    """Materialize the target and ignore regions for ``A/F/R/S``."""

    valid = _boxes(valid)
    source_ignore = _boxes(source_ignore)
    support = np.asarray(support, dtype=float)
    if contract not in CONTRACTS:
        raise ValueError(f"unknown contract {contract!r}; expected one of {CONTRACTS}")
    if len(support) != len(valid):
        raise ValueError("support and valid boxes must have equal length")

    keep = support >= float(threshold)
    retained = valid if contract == "A" else valid[keep]
    removed = valid[~keep]
    if contract in {"A", "F"}:
        ignore = np.empty((0, 4), dtype=float)
    elif contract == "R":
        ignore = removed
    else:  # S
        ignore = np.concatenate((removed, source_ignore), axis=0)
    return ContractRegions(
        valid=_boxes(retained),
        ignore=_boxes(ignore),
        retained_count=int(len(retained)),
        removed_count=int(len(removed) if contract != "A" else 0),
    )


def evaluate_contract_counts(
    pred: np.ndarray,
    scores: np.ndarray,
    regions: ContractRegions,
    iou_threshold: float,
    ignore_threshold: float = .5,
) -> tuple[int, int, int, int]:
    """Evaluate one contract and return ``TP, FP, FN, neutralized``."""

    return greedy_match_with_ignores(
        _boxes(pred),
        regions.valid,
        regions.ignore,
        float(iou_threshold),
        float(ignore_threshold),
        scores=np.asarray(scores, dtype=float),
        ignore_overlap="crowd_iop",
    )


def score_ordered_detection_outcomes(
    pred: np.ndarray,
    scores: np.ndarray,
    regions: ContractRegions,
    iou_threshold: float,
    max_dets: int = 2000,
    ignore_threshold: float = .5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return score-ordered AP outcomes after valid-first ignore handling.

    Neutral detections are omitted.  The returned arrays are ``score``,
    ``TP flag``, and ``FP flag`` and reproduce the count evaluator when
    summed.  ``max_dets`` is applied per image before matching.
    """

    pred = _boxes(pred)
    scores = np.asarray(scores, dtype=float)
    if len(pred) != len(scores):
        raise ValueError("scores and predictions must have equal length")
    if max_dets <= 0:
        raise ValueError("max_dets must be positive")

    order = np.argsort(-scores, kind="mergesort")[: int(max_dets)]
    pred, scores = pred[order], scores[order]
    valid_iou = iou_matrix(pred, regions.valid)
    ignore = regions.ignore
    if len(pred) and len(ignore):
        x1 = np.maximum(pred[:, None, 0], ignore[None, :, 0])
        y1 = np.maximum(pred[:, None, 1], ignore[None, :, 1])
        x2 = np.minimum(pred[:, None, 2], ignore[None, :, 2])
        y2 = np.minimum(pred[:, None, 3], ignore[None, :, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        pred_area = np.maximum(0, pred[:, 2] - pred[:, 0]) * np.maximum(0, pred[:, 3] - pred[:, 1])
        ignore_iop = inter / (pred_area[:, None] + 1e-12)
    else:
        ignore_iop = np.zeros((len(pred), len(ignore)), dtype=float)

    unused = np.ones(len(regions.valid), dtype=bool)
    kept_scores: list[float] = []
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    for i in range(len(pred)):
        hit = False
        if unused.any():
            candidate_iou = np.where(unused, valid_iou[i], -1.0)
            j = int(np.argmax(candidate_iou))
            if candidate_iou[j] >= float(iou_threshold):
                unused[j] = False
                hit = True
        if hit:
            kept_scores.append(float(scores[i])); tp_flags.append(1); fp_flags.append(0)
        elif len(ignore) and float(ignore_iop[i].max()) >= float(ignore_threshold):
            continue
        else:
            kept_scores.append(float(scores[i])); tp_flags.append(0); fp_flags.append(1)

    return (
        np.asarray(kept_scores, dtype=float),
        np.asarray(tp_flags, dtype=np.int8),
        np.asarray(fp_flags, dtype=np.int8),
    )


def micro_f1(tp: np.ndarray | int, fp: np.ndarray | int, fn: np.ndarray | int) -> np.ndarray | float:
    """Compute global micro-F1 for scalar or array counts."""

    tp = np.asarray(tp)
    fp = np.asarray(fp)
    fn = np.asarray(fn)
    denominator = 2 * tp + fp + fn
    value = np.divide(2 * tp, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
    return float(value) if value.ndim == 0 else value
