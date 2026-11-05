"""Panoptic Quality exactly as in the organizers' self-evaluation notebook (per expert annotation set; IoU > 0.5 is a hit)."""
import numpy as np
from pycocotools import mask as mu


def score_image(gt_sets, pred):
    """gt_sets: {expert_image_id: [rle]} for one image; pred: [rle]. Returns (list of TP IoUs, FP count, FN count)."""
    tp, fp, fn = [], 0, 0
    for rles in gt_sets.values():
        ng, npd = len(rles), len(pred)
        if ng == 0:
            fp += npd; continue
        if npd == 0:
            fn += ng; continue
        iou = np.array(mu.iou(pred, rles, [0] * ng)).reshape(npd, ng).T
        hit = iou > 0.5
        tp += iou[hit].tolist(); fp += int((hit.sum(0) == 0).sum()); fn += int((hit.sum(1) == 0).sum())
    return tp, fp, fn


def pq_from(results):
    """Aggregate (tp, fp, fn) tuples of several images into PQ / SQ / RQ."""
    tp = [x for r in results for x in r[0]]; fp = sum(r[1] for r in results); fn = sum(r[2] for r in results)
    den = len(tp) + 0.5 * fp + 0.5 * fn
    return dict(pq=sum(tp) / den if den else 0.0, sq=float(np.mean(tp)) if tp else 0.0,
                rq=len(tp) / den if den else 0.0, tp=len(tp), fp=fp, fn=fn)
