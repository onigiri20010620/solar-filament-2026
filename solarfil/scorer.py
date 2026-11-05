"""Instance scorer: 20 per-instance features -> expected PQ contribution (hit probability x IoU given a hit)."""
import pickle

import cv2
import numpy as np
from pycocotools import mask as mu

from .candidates import orientation
from .config import FULL, SCORER

FEATS = ["log_area", "elong", "length", "width", "fill", "p_mean", "p_max", "p_std", "p_q10", "f07", "f09",
         "m_std", "m_min", "ring_p", "contrast", "core_contrast", "ring_tex", "inside_int", "r", "nn_dist"]


def disk_radius(img_full):
    """Solar disk radius (px at 2048) from the radial intensity profile (the bright limb line)."""
    img = cv2.resize(img_full, (1024, 1024), interpolation=cv2.INTER_AREA)
    yy, xx = np.indices(img.shape); rr = np.hypot(yy - 512, xx - 512).astype(int)
    prof = np.bincount(rr.ravel(), img.ravel()) / np.maximum(np.bincount(rr.ravel()), 1)
    return 2 * (358 + int(np.argmax(prof[358:512])))


def features(pred, probs, img, R):
    """pred: [rle] at 2048 px; probs: list of 1024 px probability maps (one per model family; here one);
    img: 2048 px uint8; R: disk radius. Returns float32 [N, len(FEATS)]."""
    ps = [cv2.resize(p.astype(np.float32), (FULL, FULL), interpolation=cv2.INTER_LINEAR) for p in probs]
    pe = sum(ps) / len(ps)
    img = img.astype(np.float32)
    masks = [mu.decode(r).astype(bool) for r in pred]
    anym = np.zeros((FULL, FULL), bool)
    for m in masks:
        anym |= m
    cents = [(ys.mean(), xs.mean()) for ys, xs in (np.nonzero(m) for m in masks)]
    k7, k21 = np.ones((15, 15), np.uint8), np.ones((43, 43), np.uint8)
    rows = []
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = max(ys.min() - 30, 0), min(ys.max() + 31, FULL), max(xs.min() - 30, 0), min(xs.max() + 31, FULL)
        mc = m[y0:y1, x0:x1].astype(np.uint8)
        ring = (cv2.dilate(mc, k7) > 0) & (mc == 0)
        ring_far = (cv2.dilate(mc, k21) > 0) & (mc == 0) & ~anym[y0:y1, x0:x1]
        pv = pe[y0:y1, x0:x1][mc > 0]
        iv = img[y0:y1, x0:x1]
        inside = iv[mc > 0]
        bg = iv[ring_far] if ring_far.sum() > 10 else iv[ring]
        bgm = float(bg.mean()) if len(bg) else float(inside.mean())
        core = inside[pv >= np.quantile(pv, 0.7)] if len(pv) > 5 else inside
        _, el = orientation(ys, xs)
        cov = np.cov(np.vstack([xs, ys]).astype(float)) if len(ys) > 2 else np.eye(2)
        lam = np.sort(np.linalg.eigvalsh(cov))
        length = 4 * np.sqrt(max(lam[1], 1e-6)); area = float(m.sum())
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        mm = [float(p[y0:y1, x0:x1][mc > 0].mean()) for p in ps]
        cy, cx = cents[i]
        nn = min([np.hypot(cy - c[0], cx - c[1]) for j, c in enumerate(cents) if j != i], default=FULL) / R
        rows.append([np.log(area), el, length / R, area / max(length, 1), area / (bw * bh),
                     float(pv.mean()), float(pv.max()), float(pv.std()), float(np.quantile(pv, 0.1)),
                     float((pv > 0.7).mean()), float((pv > 0.9).mean()),
                     float(np.std(mm)), float(np.min(mm)), float(pe[y0:y1, x0:x1][ring].mean()) if ring.any() else 0.0,
                     (bgm - float(inside.mean())) / max(bgm, 1), (bgm - float(core.mean())) / max(bgm, 1),
                     float(bg.std()) / max(bgm, 1) if len(bg) else 0.0, float(inside.mean()) / 255,
                     float(np.hypot(cy - FULL / 2, cx - FULL / 2) / R), nn])
    return np.array(rows, np.float32).reshape(-1, len(FEATS))


class Scorer:
    """Hurdle model: h = P(hit) regressor, m = E[IoU | hit] regressor (both HistGradientBoostingRegressor)."""

    def __init__(self, path):
        d = pickle.load(open(path, "rb"))
        self.h, self.m, self.rule = d["h"], d["m"], d.get("rule", SCORER)
        assert d.get("feats", FEATS) == FEATS

    def score(self, X):
        if not len(X):
            return np.zeros(0)
        return np.clip(self.h.predict(X), 0, 1) * np.clip(self.m.predict(X), 0, 1)

    def keep(self, X):
        """Boolean mask of candidates to keep: score >= clf and p_mean >= p_mean and r <= r_max."""
        if not len(X):
            return np.zeros(0, bool)
        s = self.score(X)
        return (s >= self.rule["clf"]) & (X[:, FEATS.index("p_mean")] >= self.rule["p_mean"]) & (X[:, FEATS.index("r")] <= self.rule["r_max"])
