"""Why the refiner helps (final configuration, out of fold): draft (Stage-1 candidate) vs re-drawn mask.

prep: per image the drafts' 20 features and IoU tables (0.5 px-eroded, as the re-drawn masks) and mask areas
      -> <work>/why/<stem>.pkl and <work>/feats_draft/<stem>.pkl (team_select_any.py layout, key ('draft', 'base')).
stat: pairs (candidate, expert) with draft IoU >= 0.2 to its best ground truth: IoU / area ratio before and after;
      error headroom of the final prediction (cross-fitted scorer, rule 0.18 / 0.6) -> json + npy.
"""
import argparse, json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pycocotools import mask as mu


def one(args):
    team, root, s, oof, work, gsets = args
    sys.path.insert(0, team); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "final_scorer"))
    from solarfil.data import read_image
    from solarfil.postprocess import shrink_all
    from solarfil.scorer import disk_radius, features
    from team_adapt import paste_thr
    Q = pickle.load(open(os.path.join(work, "refs", s + ".pkl"), "rb")); T = pickle.load(open(os.path.join(oof, s + ".pkl"), "rb"))
    img = read_image(os.path.join(root, "train", "train_images", s + ".jpeg")); Rd = disk_radius(img)
    p = T["prob"].astype(np.float32) / 255; cands, W, R = Q["cands"], Q["W"], Q["R"]; n = len(cands)
    pred, idx, refd = paste_thr(cands, W, R, np.full(n, 0.45)) if n else ([], [], [])
    er = lambda L: [(shrink_all([r], 0.5) or [r])[0] for r in L]
    io = lambda sh: {e: (np.asarray(mu.iou(sh, g, [0] * len(g))).reshape(len(sh), len(g)) if sh and g else np.zeros((len(sh), len(g)))) for e, g in gsets.items()}
    sd, sr = er(cands), er(pred)
    Xd = features(cands, [p], img, Rd) if n else np.zeros((0, 20), np.float32)
    ar = lambda L: np.array([float(mu.area(r)) for r in L])
    out = {"fold": Q["fold"], "idx": idx, "refd": refd, "ious_d": io(sd), "ious_d0": io(list(cands)), "area_d0": ar(cands), "ious_r": io(sr), "area_d": ar(sd), "area_r": ar(sr),
           "area_gt": {e: ar(g) for e, g in gsets.items()}}
    pickle.dump(out, open(os.path.join(work, "why", s + ".pkl"), "wb"))
    pickle.dump({"fold": Q["fold"], "res": {("draft", "base"): (Xd, out["ious_d"]), ("draft0", "base"): (Xd, out["ious_d0"])}}, open(os.path.join(work, "feats_draft", s + ".pkl"), "wb"))
    return s


def prep(a):
    from solarfil.data import find_data, load_annotations
    gt = load_annotations(find_data(a.root)["ann"])
    for d in ("why", "feats_draft"):
        os.makedirs(os.path.join(a.work, d), exist_ok=True)
    stems = sorted(f[:-4] for f in os.listdir(os.path.join(a.work, "refs")) if f.endswith(".pkl"))
    with ProcessPoolExecutor(12) as ex:
        print("prep", sum(1 for _ in ex.map(one, [(a.team, a.root, s, a.oof, a.work, gt[s]) for s in stems], chunksize=2)))


def stat(a):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from solarfil.scorer import FEATS
    ip, ir = FEATS.index("p_mean"), FEATS.index("r")
    mk = lambda: HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400, min_samples_leaf=30, l2_regularization=1.0, random_state=0)
    stems = sorted(f[:-4] for f in os.listdir(os.path.join(a.work, "why")) if f.endswith(".pkl"))
    Wy = {s: pickle.load(open(os.path.join(a.work, "why", s + ".pkl"), "rb")) for s in stems}
    Fa = {s: pickle.load(open(os.path.join(a.work, "feats_adapt", s + ".pkl"), "rb"))["res"][("g0.45", "parts")] for s in stems}
    Xs = [Fa[s][0] for s in stems]; Is = [Fa[s][1] for s in stems]
    X = np.concatenate(Xs); F = np.concatenate([np.full(len(x), Wy[s]["fold"]) for s, x in zip(stems, Xs)])
    H, M, Wt = [], [], []
    for x, I in zip(Xs, Is):
        n = len(x); ne = max(len(I), 1); h = np.zeros(n); m = np.zeros(n)
        for J in I.values():
            if J.size:
                b = J.max(1); h += b > .5; m += np.where(b > .5, b, 0)
        H.append(h / ne); M.append(m / ne); Wt.append(np.full(n, ne, float))
    H, M, Wt = map(np.concatenate, (H, M, Wt)); sc = np.zeros(len(X))
    for k in range(5):
        tr, te = F != k, F == k; pos = tr & (H > 0)
        sc[te] = np.clip(mk().fit(X[tr], H[tr], sample_weight=Wt[tr]).predict(X[te]), 0, 1) * \
            np.clip(mk().fit(X[pos], M[pos] / H[pos], sample_weight=(Wt * H)[pos]).predict(X[te]), 0, 1)
    keep_all = (sc >= 0.18) & (X[:, ip] >= 0.6) & (X[:, ir] <= 0.95)
    # --- headroom of the final prediction
    c = dict(siou=0.0, tp=0, fp_shape=0, fp_bg=0, fn_shape=0, fn_disc=0, fn_miss=0); o = 0; pairs = []
    for s, x in zip(stems, Xs):
        w = Wy[s]; n = len(x); keep = keep_all[o:o + n]; o += n
        assert n == len(w["idx"]), s
        for e, J in w["ious_r"].items():
            assert np.allclose(J, Fa[s][1][e])
            ng = J.shape[1]; K = J[keep]
            if ng == 0:
                c["fp_bg"] += len(K); continue
            hit = K > .5; c["siou"] += K[hit].sum(); c["tp"] += hit.sum()
            if len(K):
                b = K.max(1); c["fp_shape"] += ((b <= .5) & (b >= .1)).sum(); c["fp_bg"] += (b < .1).sum()
            bg = K.max(0) if len(K) else np.zeros(ng); ba = J.max(0) if len(J) else np.zeros(ng); Jd = w["ious_d"][e]
            bd = Jd.max(0) if len(Jd) else np.zeros(ng); un = bg <= .5
            c["fn_shape"] += (un & (bg >= .1)).sum(); c["fn_disc"] += (un & (bg < .1) & (np.maximum(ba, bd) >= .1)).sum()
            c["fn_miss"] += (un & (bg < .1) & (np.maximum(ba, bd) < .1)).sum()
            for j, i in enumerate(w["idx"]):          # pairs: draft i -> re-drawn j, same ground truth (the draft's best)
                g = int(Jd[i].argmax())
                if Jd[i, g] >= .2:
                    pairs.append([Jd[i, g], J[j, g], w["area_d"][i] / w["area_gt"][e][g], w["area_r"][j] / w["area_gt"][e][g],
                                  w["refd"][j], keep[j], w["area_gt"][e][g]])
    c = {k: float(v) for k, v in c.items()}; fp = c["fp_shape"] + c["fp_bg"]; fn = c["fn_shape"] + c["fn_disc"] + c["fn_miss"]
    pq = lambda si, tp, fp, fn: si / (tp + .5 * fp + .5 * fn); sq = c["siou"] / c["tp"]; base = pq(c["siou"], c["tp"], fp, fn)
    head = {"PQ": base, "SQ": sq, "RQ": base / sq, "counts": c,
            "fix shape (fn_shape -> TP at SQ, fp_shape removed)": pq(c["siou"] + sq * c["fn_shape"], c["tp"] + c["fn_shape"], c["fp_bg"], fn - c["fn_shape"]) - base,
            "fix discarded (fn_disc -> TP at SQ)": pq(c["siou"] + sq * c["fn_disc"], c["tp"] + c["fn_disc"], fp, fn - c["fn_disc"]) - base,
            "fix missed (fn_miss -> TP at SQ)": pq(c["siou"] + sq * c["fn_miss"], c["tp"] + c["fn_miss"], fp, fn - c["fn_miss"]) - base,
            "fix background (fp_bg removed)": pq(c["siou"], c["tp"], c["fp_shape"], fn) - base,
            "perfect outlines of the TPs (SQ = 1)": pq(c["tp"], c["tp"], fp, fn) - base}
    P = np.array(pairs, float); d, r = P[:, 0], P[:, 1]

    def summ(m):
        d_, r_ = d[m], r[m]
        return {"n": int(m.sum()), "mean IoU draft": d_.mean(), "mean IoU re-drawn": r_.mean(), "fraction improved": (r_ > d_).mean(),
                "up across 0.5": int(((d_ <= .5) & (r_ > .5)).sum()), "down across 0.5": int(((d_ > .5) & (r_ <= .5)).sum()),
                "hit rate draft": (d_ > .5).mean(), "hit rate re-drawn": (r_ > .5).mean(),
                "median area ratio draft": float(np.median(P[m, 2])), "median area ratio re-drawn": float(np.median(P[m, 3])),
                "IQR log2 area ratio draft": float(np.subtract(*np.percentile(np.log2(P[m, 2]), [75, 25]))),
                "IQR log2 area ratio re-drawn": float(np.subtract(*np.percentile(np.log2(np.maximum(P[m, 3], 1e-3)), [75, 25])))}
    ag = P[:, 6]
    out = {"headroom": head, "pairs all": summ(np.ones(len(P), bool)), "pairs kept by the scorer": summ(P[:, 5] > 0),
           "pairs by GT area": {f"{lo}-{hi}": summ((ag >= lo) & (ag < hi)) for lo, hi in ((0, 300), (300, 1000), (1000, 3000), (3000, 1e9))}}
    np.save(a.out[:-5] + "_pairs.npy", P)
    json.dump(out, open(a.out, "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["prep", "stat"])
    for k in ("team", "root", "oof", "work", "out"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args(); sys.path.insert(0, a.team)
    {"prep": prep, "stat": stat}[a.cmd](a)
