"""Does our pipeline add information to the team's v18? Cross-fitted on the team's own folds, scored with their metric.

Input: team_oof.py output (per image: v18 refined candidates + their 20 scorer features). For every candidate, our
features from OUR out-of-fold results (each made by models that never saw the image):
  o_pmean / o_pmax : our 1024 px ensemble probability (R34 + B3, 8 views) inside the candidate
  o_uhat / o_iou   : expected score (u_hat) of the tree node of ours that overlaps the candidate best, and that IoU
  o_sel            : best IoU with a filament our final rule selected (tau 0.24)
Scorer = the team's hurdle model (HistGradientBoosting h and m, weights = number of experts), refit per fold on the
other four folds; rule h*m >= clf and p_mean >= 0.7 and r <= 0.95; 0.5 px erosion; PQ exactly as solarfil.metrics.
Compared: their 20 features vs 20 + ours, at clf 0.2 and at the clf chosen on the training folds; paired bootstrap
over images (10,000) for the difference.
Usage: python team_combine.py --team <dir> --root <data> --oof <team_oof dir> --runs <our ens prefix> --tree <dir> --out <json>
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

OURS = ["o_pmean", "o_pmax", "o_uhat", "o_iou", "o_sel"]
G = {}


def our_feats(args):
    root, stem, pred, prob8, u = args
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import data as D
    from tree_select import build_tree, dp_select
    if not pred:
        return stem, np.zeros((0, len(OURS)), np.float32)
    img = D.read_image(os.path.join(root, "train", "train_images", stem + ".jpeg")); disk = D.disk_mask(img)
    prob = cv2.resize(prob8.astype(np.float32) / 255, (2048, 2048), interpolation=cv2.INTER_LINEAR)
    nodes, parent = build_tree(prob, disk)
    nr = [mu.encode(np.asfortranarray(m.astype(np.uint8))) for _, m in nodes]
    sel = [nr[i] for i in dp_select(parent, u - 0.24)] if len(u) == len(nodes) else []
    I = np.asarray(mu.iou(pred, nr, [0] * len(nr))).reshape(len(pred), len(nr)) if nr else np.zeros((len(pred), 0))
    S = np.asarray(mu.iou(pred, sel, [0] * len(sel))).reshape(len(pred), len(sel)) if sel else np.zeros((len(pred), 0))
    out = []
    for i, r in enumerate(pred):
        m = mu.decode(r).astype(bool); pv = prob[m]
        j = int(I[i].argmax()) if I.shape[1] else -1
        out.append([pv.mean(), pv.max(), float(u[j]) if j >= 0 and len(u) == len(nodes) else 0.0,
                    float(I[i, j]) if j >= 0 else 0.0, float(S[i].max()) if S.shape[1] else 0.0])
    return stem, np.array(out, np.float32)


def main():
    ap = argparse.ArgumentParser()
    for k in ("team", "root", "oof", "runs", "tree", "out"):
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
    T = {s: pickle.load(open(os.path.join(a.oof, s + ".pkl"), "rb")) for s in stems}
    # our out-of-fold maps and u_hat
    probs = {}
    for f in range(5):
        z = np.load(f"{a.runs}{f}/val_probs.npz"); probs.update({n[:-5]: p for n, p in zip(z["names"], z["probs"])})
    rows = pickle.load(open(os.path.join(a.tree, "tree.pkl"), "rb"))["rows"]; uh = np.load(os.path.join(a.tree, "u_hat.npy"))
    u_of = defaultdict(list)
    for r, u in zip(rows, uh):
        u_of[r["name"][:-5]].append(u)
    jobs = [(a.root, s, T[s]["pred"], probs[s], np.array(u_of[s])) for s in stems]
    OF = {}
    with ProcessPoolExecutor(10) as ex:
        for s, x in ex.map(our_feats, jobs, chunksize=2):
            OF[s] = x
    # targets (team definition)
    X0, X1, H, M, W, fold, owner = [], [], [], [], [], [], []
    for s in stems:
        pred = T[s]["pred"]; ne = max(len(gt[s]), 1); h = np.zeros(len(pred)); m = np.zeros(len(pred))
        for g in gt[s].values():
            if g and pred:
                iou = np.array(mu.iou(pred, g, [0] * len(g))).reshape(len(pred), len(g)).max(1)
                h += iou > 0.5; m += np.where(iou > 0.5, iou, 0)
        X0.append(T[s]["feats"]); X1.append(np.c_[T[s]["feats"], OF[s]]); H.append(h / ne); M.append(m / ne)
        W.append(np.full(len(pred), ne, float)); fold += [T[s]["fold"]] * len(pred); owner += [s] * len(pred)
    X0, X1, H, M, W = map(np.concatenate, (X0, X1, H, M, W)); fold = np.array(fold); owner = np.array(owner)
    ip, ir = FEATS.index("p_mean"), FEATS.index("r")

    def make():
        return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400, min_samples_leaf=30, l2_regularization=1.0, random_state=0)

    def oof_score(X):
        sc = np.zeros(len(X))
        for k in range(5):
            tr, te = fold != k, fold == k; pos = tr & (H > 0)
            mh = make().fit(X[tr], H[tr], sample_weight=W[tr]); mm = make().fit(X[pos], M[pos] / H[pos], sample_weight=(W * H)[pos])
            sc[te] = np.clip(mh.predict(X[te]), 0, 1) * np.clip(mm.predict(X[te]), 0, 1)
        return sc

    by_img = defaultdict(list)
    for i, s in enumerate(owner):
        by_img[s].append(i)
    shrunk = {s: None for s in stems}

    def per_image(sc, clf):
        res = {}
        for s in stems:
            ids = by_img.get(s, [])
            keep = [i for i in ids if sc[i] >= clf and X0[i, ip] >= 0.7 and X0[i, ir] <= 0.95]
            pred = [T[s]["pred"][ids.index(i)] for i in keep]
            res[s] = score_image(gt[s], shrink_all(pred, 0.5))
        return res

    out = {"images": len(stems), "candidates": int(len(X0))}
    S0, S1 = oof_score(X0), oof_score(X1)
    sp = pickle.load(open(os.path.join(a.team, "weights", "scorer.pkl"), "rb"))     # fit on all 707 images: in-sample
    Sin = np.clip(sp["h"].predict(X0), 0, 1) * np.clip(sp["m"].predict(X0), 0, 1)
    out["v18 as submitted (scorer.pkl, in-sample - optimistic)"] = round(pq_from(list(per_image(Sin, 0.2).values()))["pq"], 4)
    grid = np.round(np.arange(0.12, 0.32, 0.02), 2)
    R = {}
    for name, sc in (("v18 features (cross-fitted)", S0), ("v18 + ours (cross-fitted)", S1)):
        curve = {}
        for c in grid:
            r = per_image(sc, c); curve[float(c)] = (pq_from(list(r.values()))["pq"], r)
        out[name] = {"PQ at clf 0.20": round(curve[0.2][0], 4), "curve": {k: round(v[0], 4) for k, v in curve.items()}}
        R[name] = curve
    # paired bootstrap at clf 0.20
    r0, r1 = R["v18 features (cross-fitted)"][0.2][1], R["v18 + ours (cross-fitted)"][0.2][1]
    rng = np.random.default_rng(0); diffs = []
    L = list(stems)
    for _ in range(10000):
        b = rng.integers(0, len(L), len(L))
        diffs.append(pq_from([r1[L[i]] for i in b])["pq"] - pq_from([r0[L[i]] for i in b])["pq"])
    d = np.array(diffs)
    out["difference at clf 0.20 (ours added - v18)"] = {"mean": round(float(d.mean()), 4), "95% interval": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)], "P(better)": round(float((d > 0).mean()), 3)}
    # importance check: correlation of score with true h
    out["corr(score, h): v18 / v18+ours"] = [round(float(np.corrcoef(S0, H)[0, 1]), 4), round(float(np.corrcoef(S1, H)[0, 1]), 4)]
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
