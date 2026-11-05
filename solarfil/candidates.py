"""Candidate extraction: probability map -> one mask per filament instance.

hysteresis : connected components of p > lo are one instance if they contain a core pixel p > hi;
             output pixels are p > out.
join2      : fragments whose end points face each other (distance <= gap2, orientation within theta)
             are merged and the gap is filled with a line of width fill2.
Lengths/areas in `cfg` are given in 1024 px units; `scale=2` produces 2048 px masks.
"""
import cv2
import numpy as np
from pycocotools import mask as mu


def _ellipse(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def orientation(ys, xs):
    """Principal-axis angle (deg, 0-180) and elongation (ratio of axis standard deviations) of a pixel set."""
    if len(ys) < 3:
        return 0.0, 1.0
    c = np.cov(np.vstack([xs, ys]).astype(np.float64))
    w, v = np.linalg.eigh(c)
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1])) % 180
    return ang, float(np.sqrt(max(w[1], 1e-9) / max(w[0], 1e-9)))


def _join(lab, n, gap, angle, fill):
    """Merge labelled fragments closer than `gap` px whose principal axes agree within `angle` deg."""
    H, W = lab.shape
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    info = {}
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) == 0:
            continue
        ang, elong = orientation(ys, xs)
        info[i] = dict(ys=ys, xs=xs, ang=ang, elong=elong, box=(ys.min(), ys.max(), xs.min(), xs.max()))
    lines = np.zeros((H, W), np.uint8)
    ids = sorted(info)
    for a_i, i in enumerate(ids):
        bi = info[i]["box"]
        y0, y1 = max(bi[0] - gap, 0), min(bi[1] + gap + 1, H)
        x0, x1 = max(bi[2] - gap, 0), min(bi[3] + gap + 1, W)
        cand = [j for j in ids[a_i + 1:]
                if not (info[j]["box"][0] > y1 or info[j]["box"][1] < y0 or info[j]["box"][2] > x1 or info[j]["box"][3] < x0)]
        if not cand:
            continue
        crop = (lab[y0:y1, x0:x1] != i).astype(np.uint8)
        dt = cv2.distanceTransform(crop, cv2.DIST_L2, 3)
        for j in cand:
            sel = (info[j]["ys"] >= y0) & (info[j]["ys"] < y1) & (info[j]["xs"] >= x0) & (info[j]["xs"] < x1)
            if not sel.any():
                continue
            jy, jx = info[j]["ys"][sel] - y0, info[j]["xs"][sel] - x0
            d = dt[jy, jx]
            k = int(np.argmin(d))
            if d[k] > gap:
                continue
            if angle < 90:
                da = abs(info[i]["ang"] - info[j]["ang"]) % 180
                da = min(da, 180 - da)
                if info[i]["elong"] > 2 and info[j]["elong"] > 2 and da > angle:
                    continue
            parent[find(j)] = find(i)
            if fill:
                py, px = jy[k] + y0, jx[k] + x0
                ii = int(np.argmin((info[i]["ys"] - py) ** 2 + (info[i]["xs"] - px) ** 2))
                cv2.line(lines, (int(px), int(py)), (int(info[i]["xs"][ii]), int(info[i]["ys"][ii])), 1, int(fill))
    new = np.zeros_like(lab)
    for i in ids:
        new[lab == i] = find(i)
    if fill:
        add = (lines > 0) & (new == 0)
        cur = new.copy()
        for _ in range(gap + 2):
            if not (add & (cur == 0)).any():
                break
            d = cv2.dilate(cur.astype(np.float32), np.ones((3, 3), np.uint8)).astype(cur.dtype)
            m = add & (cur == 0)
            cur[m] = d[m]
        new = cur
    return new, lines


def _ends(ys, xs, R):
    """Two end points of an elongated blob and the outward direction at each (unit vectors)."""
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    w, v = np.linalg.eigh(np.cov((pts - c).T))
    axis = v[:, 1]
    proj = (pts - c) @ axis
    out = []
    for idx, sgn in ((int(np.argmin(proj)), -1), (int(np.argmax(proj)), 1)):
        e = pts[idx]
        near = pts[((pts - e) ** 2).sum(1) <= R * R]
        d = e - near.mean(0)
        nd = np.linalg.norm(d)
        d = d / nd if nd >= 1 else sgn * axis
        out.append((e, d))
    return out


def _join_ends(lab, gap2, theta, R, fill2, min_elong):
    """Merge fragments whose end points face each other (join2)."""
    ids = [int(i) for i in np.unique(lab) if i > 0]
    if len(ids) < 2:
        return lab
    info = {}
    for i in ids:
        ys, xs = np.nonzero(lab == i)
        if len(ys) < 5:
            continue
        _, elong = orientation(ys, xs)
        if elong < min_elong:
            continue
        info[i] = _ends(ys, xs, R)
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cos_t = np.cos(np.radians(theta))
    keys = sorted(info)
    lines = []
    for ai, i in enumerate(keys):
        for j in keys[ai + 1:]:
            best = None
            for ea, da in info[i]:
                for eb, db in info[j]:
                    v = eb - ea
                    dist = np.linalg.norm(v)
                    if dist > gap2:
                        continue
                    if dist >= 2:
                        v = v / dist
                        if da @ v < cos_t or db @ (-v) < cos_t:
                            continue
                    if best is None or dist < best[0]:
                        best = (dist, ea, eb)
            if best is not None:
                parent[find(j)] = find(i)
                lines.append((best[1], best[2], i))
    new = np.zeros_like(lab)
    for i in ids:
        new[lab == i] = find(i)
    if fill2:
        for ea, eb, i in lines:
            ln = np.zeros(lab.shape, np.uint8)
            cv2.line(ln, (int(ea[0]), int(ea[1])), (int(eb[0]), int(eb[1])), 1, max(int(fill2), 1))
            new[(ln > 0) & (new == 0)] = find(i)
    return new


def instances(prob_small, cfg, scale=2):
    """prob_small: probability map (1024 px, float in [0,1] or uint8). Returns a list of COCO RLE masks at 1024*scale px."""
    p = prob_small.astype(np.float32) / (255.0 if prob_small.dtype == np.uint8 else 1.0)
    size = 1024 * scale
    if p.shape[0] != size:
        p = cv2.resize(p, (size, size), interpolation=cv2.INTER_LINEAR)
    close_r = cfg.get("close_r", 0) * scale
    low = (p > cfg["lo"]).astype(np.uint8)
    if close_r:
        low = cv2.morphologyEx(low, cv2.MORPH_CLOSE, _ellipse(close_r))
    n, lab = cv2.connectedComponents(low, connectivity=8)
    seed_ids = np.unique(lab[(p > cfg["hi"]) & (low > 0)])
    keep = np.zeros(n, bool); keep[seed_ids] = True; keep[0] = False
    out_mask = (p > cfg["out"]).astype(np.uint8)
    if close_r:
        out_mask = cv2.morphologyEx(out_mask, cv2.MORPH_CLOSE, _ellipse(close_r))
    lab = np.where(keep[lab] & (out_mask > 0), lab, 0).astype(np.int32)
    if cfg.get("gap", 0) > 0:
        lab, _ = _join(lab, int(lab.max()), cfg["gap"] * scale, cfg.get("angle", 90), cfg.get("fill", 0) * scale)
    if cfg.get("gap2", 0) > 0:
        lab = _join_ends(lab, cfg["gap2"] * scale, cfg.get("theta", 45), cfg.get("R", 15) * scale,
                         cfg.get("fill2", 0) * scale, cfg.get("min_elong", 2.0))
    n = int(lab.max()) + 1
    areas = np.bincount(lab.ravel(), minlength=n)
    min_a = cfg.get("min_area", 0) * scale * scale
    return [mu.encode(np.asfortranarray((lab == i).astype(np.uint8))) for i in range(1, n) if areas[i] >= min_a and areas[i] > 0]
