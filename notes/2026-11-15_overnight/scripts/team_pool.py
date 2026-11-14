"""One candidate pool from both pipelines, one cross-fitted scorer, greedy non-overlapping selection (out of fold).

team_complement.py: v18 and ours each hit ~450 annotated filaments the other misses (oracle +0.023). Here:
  pool per train image = v18's refined candidates (team_oof.py, before its scorer) + our final filaments
  (tree selection tau 0.24 + fine.py re-draw t0.5), all described by v18's 20 features (on v18's probability map,
  nearest neighbour within the same source) + src, best IoU with the other source's set, our probability inside,
  our expected score u_hat (ours) / u_hat of the best-overlapping ours (v18's).
  hurdle scorer (HistGB h and m, weights = experts) cross-fitted on the team's folds;
  per image: candidates by score, kept if score >= clf (and v18's p_mean / r rule if 'rule'), skipped if >= 20 % of
  it is already taken; 0.5 px erosion; PQ as solarfil.metrics. Baseline = v18's candidates through the same code.
Usage: python team_pool.py --team <dir> --root <data> --oof <team_oof dir> --fine <dir> --runs <our ens prefix> --tree <dir> --out <json>
"""
import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from pycocotools import mask as mu

EXTRA = ["src", "iou_other", "our_p", "our_u"]


def work(args):
    team, root, s, T, meta_img, preds_img, prob8, u = args
    sys.path.insert(0, team); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import data as D
    import fine as FN
    from solarfil.data import read_image
    from solarfil.scorer import disk_radius, features
    from tree_select import build_tree, dp_select
    img = read_image(os.path.join(root, "train", "train_images", s + ".jpeg")); R = disk_radius(img)
    pv = T["prob"].astype(np.float32) / 255
    ours = [mu.encode(np.asfortranarray(m.astype(np.uint8))) for m in FN.paste(meta_img, preds_img, 0.5, False)] if meta_img else []
    # our u_hat for our filaments, in the same order as fine.py's seeds (dp_select order)
    if ours:
        disk = D.disk_mask(img)
        nodes, parent = build_tree(cv2.resize(prob8.astype(np.float32) / 255, (2048, 2048), interpolation=cv2.INTER_LINEAR), disk)
        u_ours = u[dp_select(parent, u - 0.24)] if len(u) == len(nodes) else np.zeros(len(ours))
        if len(u_ours) != len(ours):
            u_ours = np.zeros(len(ours))
    else:
        u_ours = np.zeros(0)
    v = T["pred"]
    ourp = cv2.resize(prob8.astype(np.float32) / 255, (2048, 2048), interpolation=cv2.INTER_LINEAR)
    Fv = T["feats"] if len(v) else np.zeros((0, 20), np.float32)
    Fo = features(ours, [pv], img, R) if ours else np.zeros((0, 20), np.float32)
    I = np.asarray(mu.iou(v, ours, [0] * len(ours))).reshape(len(v), len(ours)) if v and ours else np.zeros((len(v), len(ours)))
    ev = np.c_[np.zeros(len(v)), I.max(1) if I.shape[1] else np.zeros(len(v)),
               [ourp[mu.decode(r) > 0].mean() for r in v], [u_ours[I[i].argmax()] if I.shape[1] and I[i].max() > 0.5 else 0.0 for i in range(len(v))]]
    eo = np.c_[np.ones(len(ours)), I.max(0) if I.shape[0] else np.zeros(len(ours)),
               [ourp[mu.decode(r) > 0].mean() for r in ours], u_ours]
    X = np.r_[np.c_[Fv, ev.reshape(len(v), 4)], np.c_[Fo, eo.reshape(len(ours), 4)]].astype(np.float32)
    return s, list(v) + ours, X


def main():
    ap = argparse.ArgumentParser()
    for k in ("team", "root", "oof", "fine", "runs", "tree", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.team)
    from sklearn.ensemble import HistGradientBoostingRegressor
    from solarfil.data import find_data, load_annotations
    from solarfil.metrics import pq_from, score_image
    from solarfil.postprocess import shrink_all
    from solarfil.scorer import FEATS
    gt = load_annotations(find_data(a.root)["ann"])
    stems = sorted(f[:-4] for f in os.listdir(a.oof) if f.endswith(".pkl"))
    meta = pickle.load(open(os.path.join(a.fine, "meta.pkl"), "rb"))["meta"]; oof = np.load(os.path.join(a.fine, "oofplain.npy"))
    per = defaultdict(list)
    for i, m in enumerate(meta):
        per[m["name"][:-5]].append(i)
    probs = {}
    for f in range(5):
        z = np.load(f"{a.runs}{f}/val_probs.npz"); probs.update({n[:-5]: p for n, p in zip(z["names"], z["probs"])})
    rows = pickle.load(open(os.path.join(a.tree, "tree.pkl"), "rb"))["rows"]; uh = np.load(os.path.join(a.tree, "u_hat.npy"))
    u_of = defaultdict(list)
    for r, u in zip(rows, uh):
        u_of[r["name"][:-5]].append(u)
    jobs = []
    for s in stems:
        T = pickle.load(open(os.path.join(a.oof, s + ".pkl"), "rb"))
        jobs.append((a.team, a.root, s, T, [meta[i] for i in per.get(s, [])], [oof[i] for i in per.get(s, [])], probs[s], np.array(u_of[s])))
    fold = {s: j[3]["fold"] for s, j in zip(stems, jobs)}
    P, X = {}, {}
    with ProcessPoolExecutor(10) as ex:
        for s, rl, x in ex.map(work, jobs, chunksize=2):
            P[s], X[s] = rl, x
    del jobs
    # targets
    Xa, H, M, Wt, F, own = [], [], [], [], [], []
    for s in stems:
        pred = P[s]; ne = max(len(gt[s]), 1); h = np.zeros(len(pred)); m = np.zeros(len(pred))
        for g in gt[s].values():
            if g and pred:
                iou = np.array(mu.iou(pred, g, [0] * len(g))).reshape(len(pred), len(g)).max(1)
                h += iou > 0.5; m += np.where(iou > 0.5, iou, 0)
        Xa.append(X[s]); H.append(h / ne); M.append(m / ne); Wt.append(np.full(len(pred), ne, float)); F += [fold[s]] * len(pred); own += [s] * len(pred)
    Xa, H, M, Wt = map(np.concatenate, (Xa, H, M, Wt)); F = np.array(F); own = np.array(own)
    src = Xa[:, 20]

    def make():
        return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400, min_samples_leaf=30, l2_regularization=1.0, random_state=0)

    def crossfit(cols, rowsel):
        sc = np.full(len(Xa), -1.0); Xc = Xa[:, cols]
        for k in range(5):
            tr, te = (F != k) & rowsel, (F == k) & rowsel; pos = tr & (H > 0)
            mh = make().fit(Xc[tr], H[tr], sample_weight=Wt[tr]); mm = make().fit(Xc[pos], M[pos] / H[pos], sample_weight=(Wt * H)[pos])
            sc[te] = np.clip(mh.predict(Xc[te]), 0, 1) * np.clip(mm.predict(Xc[te]), 0, 1)
        return sc

    idx_of = defaultdict(list)
    for i, s in enumerate(own):
        idx_of[s].append(i)
    ip, ir = FEATS.index("p_mean"), FEATS.index("r")

    def run(sc, clf, rule, rowsel):
        per_img = {}
        for s in stems:
            ids = [i for i in idx_of.get(s, []) if rowsel[i] and sc[i] >= clf and (not rule or (Xa[i, ip] >= 0.7 and Xa[i, ir] <= 0.95))]
            ids.sort(key=lambda i: -sc[i])
            taken = np.zeros((2048, 2048), np.uint8); keep = []
            base = idx_of[s][0] if idx_of.get(s) else 0
            for i in ids:
                m = mu.decode(P[s][i - base])
                if (m & taken).sum() >= 0.2 * m.sum():
                    continue
                mm = m & (1 - taken); taken |= mm; keep.append(mu.encode(np.asfortranarray(mm)))
            per_img[s] = score_image(gt[s], shrink_all(keep, 0.5))
        return per_img

    out = {"candidates": {"v18": int((src == 0).sum()), "ours": int((src == 1).sum())}}
    allc = list(range(24)); v18c = list(range(20))
    s_base = crossfit(v18c, src == 0)
    s_pool = crossfit(allc, np.ones(len(Xa), bool))
    s_v18x = crossfit(allc, src == 0)          # v18's candidates only, but with the extra (agreement) features
    R = {}
    for name, sc, sel in (("A v18 candidates, v18 features", s_base, src == 0), ("B v18 candidates + agreement features", s_v18x, src == 0),
                          ("C pooled candidates, all features", s_pool, np.ones(len(Xa), bool))):
        for clf in (0.16, 0.2, 0.24, 0.28):
            for rule in (True, False):
                r = run(sc, clf, rule, sel); key = f"{name}, clf {clf}, rule {rule}"
                R[key] = r; out[key] = round(pq_from(list(r.values()))["pq"], 4); print(key, out[key], flush=True)
    base = R["A v18 candidates, v18 features, clf 0.2, rule True"]
    for nm in ("B", "C"):
        best = max((k for k in R if k.startswith(nm)), key=lambda k: out[k])
        L = stems; rng = np.random.default_rng(0); d = []
        for _ in range(5000):
            b = rng.integers(0, len(L), len(L))
            d.append(pq_from([R[best][L[i]] for i in b])["pq"] - pq_from([base[L[i]] for i in b])["pq"])
        d = np.array(d)
        out[f"{nm} best vs A (clf 0.2, rule)"] = {"setting": best, "mean": round(float(d.mean()), 4),
                                                  "95% interval": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                                                  "P(better)": round(float((d > 0).mean()), 3)}
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
