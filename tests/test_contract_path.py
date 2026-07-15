import numpy as np

from label_reliability.contract_path import (
    CONTRACTS,
    TRANSITIONS,
    contract_regions,
    evaluate_contract_counts,
    score_ordered_detection_outcomes,
)


def test_registered_path_changes_one_ignore_source_at_a_time():
    valid = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], float)
    source_ignore = np.array([[40, 0, 50, 10]], float)
    support = np.array([20, 5], float)
    regions = {c: contract_regions(valid, source_ignore, support, 10, c) for c in CONTRACTS}
    assert len(regions["A"].valid) == 2 and len(regions["A"].ignore) == 0
    assert len(regions["F"].valid) == 1 and len(regions["F"].ignore) == 0
    assert len(regions["R"].valid) == 1 and np.array_equal(regions["R"].ignore, valid[1:])
    assert len(regions["S"].valid) == 1 and len(regions["S"].ignore) == 2
    assert [name for _, _, name in TRANSITIONS] == ["target", "removed", "source"]


def test_removed_and_source_steps_neutralize_only_unmatched_predictions():
    valid = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], float)
    source_ignore = np.array([[40, 0, 50, 10]], float)
    support = np.array([20, 5], float)
    pred = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], float)
    scores = np.array([.9, .8, .7])
    counts = {
        c: evaluate_contract_counts(pred, scores, contract_regions(valid, source_ignore, support, 10, c), .5)
        for c in CONTRACTS
    }
    assert counts["A"] == (2, 1, 0, 0)
    assert counts["F"] == (1, 2, 0, 0)
    assert counts["R"] == (1, 1, 0, 1)
    assert counts["S"] == (1, 0, 0, 2)


def test_detection_outcomes_reproduce_count_endpoint_at_both_registered_ious():
    valid = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], float)
    source_ignore = np.array([[40, 0, 50, 10]], float)
    support = np.array([20, 5], float)
    pred = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10], [60, 0, 70, 10]], float)
    scores = np.array([.9, .8, .7, .6])
    regions = contract_regions(valid, source_ignore, support, 10, "S")
    for iou in (.25, .50):
        endpoint = evaluate_contract_counts(pred, scores, regions, iou)
        _, tp, fp = score_ordered_detection_outcomes(pred, scores, regions, iou)
        assert (int(tp.sum()), int(fp.sum()), len(regions.valid) - int(tp.sum())) == endpoint[:3]
