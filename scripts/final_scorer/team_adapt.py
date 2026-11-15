"""Per-candidate refiner thresholds and refiner-based scorer features for v18 (out of fold, saved refiner outputs).

Variants of the paste threshold (per candidate):
  g0.45        the current global threshold
  e0.5/e1/e2   the threshold in 0.30..0.60 maximising the expected IoU of the refiner probability q, sharpened as
               q^a / (q^a + (1-q)^a), a = 0.5 / 1 / 2
  s40-45 ...   by candidate area (2048 px units; below / above 1000 px): (0.40, 0.45), (0.45, 0.50), (0.40, 0.50), (0.50, 0.45)
Feature sets: base = v18's 20; ref = + 8 from the refiner mean output; parts = + 4 from the per-model outputs
(team_ref_parts.py), only if <work>/parts exists.
Output <work>/<out>/<stem>.pkl in team_refine_sweep.py's feats layout, keys (variant, featset); read by team_select_any.py.
Usage: python team_adapt.py --team <dir> --root <data> --oof <team_oof> --work <team_sweep> --outdir feats_adapt
"""
import argparse, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from pycocotools import mask as mu

FULL = 2048
GRID = np.arange(0.30, 0.601, 0.05)
SIZE = {"s40-45": (0.40, 0.45), "s45-50": (0.45, 0.50), "s40-50": (0.40, 0.50), "s50-45": (0.50, 0.45)}


def paste_thr(cands, W, refs, thr, keep_small=0.3):
    """solarfil.refiner.paste (comp='seed') with one threshold per candidate; also returns the candidate index and
    whether the output is the refined mask (False = candidate kept unchanged)."""
    best = np.zeros((FULL, FULL), np.float32); owner = np.full((FULL, FULL), -1, np.int32)
    for i, ((y0, x0, side), ref) in enumerate(zip(W, refs)):
        p = cv2.resize(ref.astype(np.float32) / 255, (side, side), interpolation=cv2.INTER_LINEAR)
        ya, xa, yb, xb = max(y0, 0), max(x0, 0), min(y0 + side, FULL), min(x0 + side, FULL)
        pp = p[ya - y0:yb - y0, xa - x0:xb - x0]
        sub_b, sub_o = best[ya:yb, xa:xb], owner[ya:yb, xa:xb]
        win = (pp >= thr[i]) & (pp > sub_b)
        sub_b[win] = pp[win]; sub_o[win] = i
    out, refd = [], []
    for i, r in enumerate(cands):
        m = (owner == i)
        if m.sum() < keep_small * mu.area(r):
            out.append(r); refd.append(False); continue
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if n > 2:
            seed = cv2.dilate(mu.decode(r), np.ones((3, 3), np.uint8))
            hit = np.unique(lab[(seed > 0) & (lab > 0)])
            m = np.isin(lab, hit) if len(hit) else lab == 1 + int(st[1:, cv2.CC_STAT_AREA].argmax())
        out.append(mu.encode(np.asfortranarray(m.astype(np.uint8)))); refd.append(True)
    taken = np.zeros((FULL, FULL), np.uint8); tmp = {}
    for i in sorted(range(len(out)), key=lambda i: -int(mu.area(out[i]))):
        m = mu.decode(out[i]) & (1 - taken); taken |= m
        tmp[i] = mu.encode(np.asfortranarray(m)) if m.any() else None
    idx = [i for i in range(len(out)) if tmp[i] is not None]
    return [tmp[i] for i in idx], idx, [refd[i] for i in idx]


def eiou_thr(q, a):
    qa = q ** a / (q ** a + (1 - q) ** a + 1e-9); tot = qa.sum(); best = (-1, 0.45)
    for t in GRID:
        S = q >= t; n = S.sum()
        if n == 0:
            continue
        ins = qa[S].sum(); e = ins / (n + tot - ins)
        if e > best[0]:
            best = (e, float(t))
    return best[1]


def one(args):
    team, root, s, oof, work, outdir, gsets = args
    sys.path.insert(0, team)
    from solarfil.data import read_image
    from solarfil.postprocess import shrink_all
    from solarfil.scorer import disk_radius, features
    Q = pickle.load(open(os.path.join(work, "refs", s + ".pkl"), "rb")); T = pickle.load(open(os.path.join(oof, s + ".pkl"), "rb"))
    pf = os.path.join(work, "parts", s + ".pkl"); P = pickle.load(open(pf, "rb")) if os.path.exists(pf) else None
    img = read_image(os.path.join(root, "train", "train_images", s + ".jpeg")); Rd = disk_radius(img)
    p = T["prob"].astype(np.float32) / 255; cands, W, R = Q["cands"], Q["W"], Q["R"]; n = len(cands)
    q = R.astype(np.float32) / 255; area = np.array([float(mu.area(r)) for r in cands])
    var = {"g0.45": np.full(n, 0.45)}
    for a_ in (0.5, 1.0, 2.0):
        var["e%g" % a_] = np.array([eiou_thr(q[i], a_) for i in range(n)])
    for nm, (lo, hi) in SIZE.items():
        var[nm] = np.where(area < 1000, lo, hi)
    res = {}
    for nm, thr in var.items():
        pred, idx, refd = paste_thr(cands, W, R, thr) if n else ([], [], [])
        X = features(pred, [p], img, Rd) if pred else np.zeros((0, 20), np.float32)
        E, G = [], []
        for r, i, rf in zip(pred, idx, refd):
            qi = q[i]; S = qi >= thr[i]; ns = max(S.sum(), 1); ins = qi[S].sum()
            ao = float(mu.area(r)); inter = float(mu.area(mu.merge([r, cands[i]], intersect=True)))
            E.append([ins / ns, float(np.quantile(qi[S], 0.1)) if S.any() else 0, ((qi > 0.25) & (qi < 0.75)).sum() / ns,
                      ins / (ns + qi.sum() - ins), qi.sum() / ns, ao / area[i], inter / (ao + area[i] - inter), float(rf)])
            if P is not None:
                a1, a2 = P["Rpp"][i] >= thr[i] * 255, P["Rcx"][i] >= thr[i] * 255; u = (a1 | a2).sum()
                d = np.abs(P["Rpp"][i].astype(np.float32) - P["Rcx"][i]) / 255
                G.append([(a1 & a2).sum() / max(u, 1), d[a1 | a2].mean() if u else 0, P["Rsd"][i][S].mean() / 510 if S.any() else 0,
                          a1.sum() / max(a2.sum(), 1)])
        E = np.array(E, np.float32).reshape(-1, 8); G = np.array(G, np.float32).reshape(-1, 4)
        sh = [(shrink_all([r], 0.5) or [r])[0] for r in pred]
        ious = {e: (np.asarray(mu.iou(sh, g, [0] * len(g))).reshape(len(sh), len(g)) if sh and g else np.zeros((len(sh), len(g)))) for e, g in gsets.items()}
        res[(nm, "base")] = (X, ious); res[(nm, "ref")] = (np.hstack([X, E]), ious)
        if P is not None:
            res[(nm, "parts")] = (np.hstack([X, E, G]), ious)
    pickle.dump({"fold": Q["fold"], "res": res, "thr": {k: v.tolist() for k, v in var.items()}}, open(os.path.join(work, outdir, s + ".pkl"), "wb"))
    return s


def main():
    ap = argparse.ArgumentParser()
    for k in ("team", "root", "oof", "work", "outdir"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args(); sys.path.insert(0, a.team)
    from solarfil.data import find_data, load_annotations
    gt = load_annotations(find_data(a.root)["ann"]); os.makedirs(os.path.join(a.work, a.outdir), exist_ok=True)
    stems = sorted(f[:-4] for f in os.listdir(os.path.join(a.work, "refs")) if f.endswith(".pkl"))
    jobs = [(a.team, a.root, s, a.oof, a.work, a.outdir, gt[s]) for s in stems if not os.path.exists(os.path.join(a.work, a.outdir, s + ".pkl"))]
    with ProcessPoolExecutor(10) as ex:
        n = sum(1 for _ in ex.map(one, jobs, chunksize=2))
    print("feats", n)


if __name__ == "__main__":
    main()
