"""Deterministic IoU matching primitives used by the rule audit."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred=np.asarray(pred,float).reshape((-1,4)); gt=np.asarray(gt,float).reshape((-1,4))
    if not len(pred) or not len(gt): return np.zeros((len(pred),len(gt)))
    x1=np.maximum(pred[:,None,0],gt[None,:,0]); y1=np.maximum(pred[:,None,1],gt[None,:,1]); x2=np.minimum(pred[:,None,2],gt[None,:,2]); y2=np.minimum(pred[:,None,3],gt[None,:,3])
    inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1); ap=np.maximum(0,pred[:,2]-pred[:,0])*np.maximum(0,pred[:,3]-pred[:,1]); ag=np.maximum(0,gt[:,2]-gt[:,0])*np.maximum(0,gt[:,3]-gt[:,1])
    return inter/(ap[:,None]+ag[None,:]-inter+1e-12)

def greedy_match(pred: np.ndarray, gt: np.ndarray, threshold: float, scores: np.ndarray|None=None) -> tuple[int,int,int]:
    mat=iou_matrix(pred,gt); scores=np.zeros(len(pred)) if scores is None else np.asarray(scores)
    unused=np.ones(len(gt),dtype=bool); tp=0
    for i in sorted(range(len(pred)),key=lambda x:(-float(scores[x]),x)):
        if not unused.any(): break
        candidate_iou=np.where(unused,mat[i],-1.0)
        j=int(np.argmax(candidate_iou))
        if candidate_iou[j]>=threshold: unused[j]=False; tp+=1
    return tp,len(pred)-tp,len(gt)-tp


def greedy_match_with_ignores(
    pred: np.ndarray,
    gt: np.ndarray,
    ignore_regions: np.ndarray,
    threshold: float,
    ignore_threshold: float = .5,
    scores: np.ndarray | None = None,
    ignore_overlap: str = "iou",
) -> tuple[int, int, int, int]:
    """Greedy match with valid-GT-first ignore precedence.

    A detection is first offered to the highest-IoU unmatched *retained valid*
    ground-truth box. Only an otherwise unmatched detection is neutralized when
    it overlaps a declared ignore region. This makes an ignore region protect
    against a false positive without suppressing a valid true positive that
    happens to overlap it.

    ``ignore_overlap='iou'`` uses ordinary symmetric box IoU. The
    ``'crowd_iop'`` option uses intersection over prediction area, matching the
    overlap denominator used by COCO for an ``iscrowd=1`` ground-truth region.
    In both modes, retained valid GT takes precedence over ignore regions. The
    return is ``(TP, FP, FN, neutralized)``.
    """
    pred = np.asarray(pred, float).reshape((-1, 4))
    gt = np.asarray(gt, float).reshape((-1, 4))
    ignore_regions = np.asarray(ignore_regions, float).reshape((-1, 4))
    scores = np.zeros(len(pred), dtype=float) if scores is None else np.asarray(scores, float)
    if len(scores) != len(pred):
        raise ValueError("scores and predictions must have equal length")
    valid_iou = iou_matrix(pred, gt)
    if ignore_overlap == "iou":
        ignore_iou = iou_matrix(pred, ignore_regions)
    elif ignore_overlap == "crowd_iop":
        if not len(pred) or not len(ignore_regions):
            ignore_iou = np.zeros((len(pred), len(ignore_regions)))
        else:
            x1 = np.maximum(pred[:, None, 0], ignore_regions[None, :, 0])
            y1 = np.maximum(pred[:, None, 1], ignore_regions[None, :, 1])
            x2 = np.minimum(pred[:, None, 2], ignore_regions[None, :, 2])
            y2 = np.minimum(pred[:, None, 3], ignore_regions[None, :, 3])
            inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            pred_area = np.maximum(0, pred[:, 2] - pred[:, 0]) * np.maximum(0, pred[:, 3] - pred[:, 1])
            ignore_iou = inter / (pred_area[:, None] + 1e-12)
    else:
        raise ValueError("ignore_overlap must be 'iou' or 'crowd_iop'")
    unused = np.ones(len(gt), dtype=bool)
    tp = fp = neutralized = 0
    for i in sorted(range(len(pred)), key=lambda x: (-float(scores[x]), x)):
        if unused.any():
            candidate_iou = np.where(unused, valid_iou[i], -1.0)
            j = int(np.argmax(candidate_iou))
            if candidate_iou[j] >= threshold:
                unused[j] = False
                tp += 1
                continue
        if len(ignore_regions) and float(ignore_iou[i].max()) >= ignore_threshold:
            neutralized += 1
        else:
            fp += 1
    return tp, fp, int(unused.sum()), neutralized

def hungarian_match(pred: np.ndarray, gt: np.ndarray, threshold: float, scores: np.ndarray|None=None) -> tuple[int,int,int]:
    mat=iou_matrix(pred,gt)
    if not len(pred) or not len(gt): return 0,len(pred),len(gt)
    # Threshold-first lexicographic assignment: maximize valid-match
    # cardinality, then total IoU.  Dummy nodes allow an unmatched object at
    # zero cost and prevent sub-threshold edges from crowding out valid ones.
    npred,ngt=mat.shape; n=npred+ngt; big=float(n+1)
    cost=np.zeros((n,n),dtype=float); valid=mat>=threshold
    cost[:npred,:ngt][valid]=-(big+mat[valid])
    rr,cc=linear_sum_assignment(cost)
    tp=int(sum(i<npred and j<ngt and valid[i,j] for i,j in zip(rr,cc)))
    return tp,len(pred)-tp,len(gt)-tp
