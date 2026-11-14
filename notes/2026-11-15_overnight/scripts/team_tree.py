"""v18's refiner on a component TREE of v18's own probability map, then the exact tree selection (out of fold).

v18 decides once how a structure is split (one threshold cut + joining) and then refines / scores those pieces.
Here every node of the nested threshold tree (whole structure, its pieces at higher thresholds, ...) is re-drawn by
v18's refiners and scored by a hurdle model (expected per-annotator hit share x IoU), and the antichain with the largest
total expected gain is chosen - the split itself is decided by the expected score over the annotators.

stage "nodes" (GPU): per train image, v18 probability (team_oof.py) -> tree (tree_select.build_tree, 15 levels; a node
  covering >= 97 % of its parent's area is merged into it) -> refiners of the image's fold (2 archs x 4 rotations;
  the 'other candidates' channel = the other nodes of the same level) -> refined mask per node (threshold 0.5, parts
  touching the node, the node itself if < 30 % remains) -> v18's 20 features + tree features; targets h, m.
stage "select": hurdle model cross-fitted on the team's folds, DP over the tree with score - tau (nodes failing v18's
  p_mean >= 0.7 / r <= 0.95 rule get -inf), overlaps resolved by expected score, 0.5 px erosion, PQ (solarfil.metrics).
  Baseline = v18's own candidates with the same cross-fitted scorer (clf rule as shipped).
Usage: python team_tree.py nodes --team <dir> --root <data> --oof <team_oof dir> --out <dir> [--shard 0/2]
       python team_tree.py select --team <dir> --root <data> --oof <team_oof dir> --out <dir>
"""
import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import cv2
import numpy as np
from pycocotools import mask as mu

TREE_FEATS = ["level", "area_vs_parent", "n_children", "depth", "is_root"]


def predict_rot4(models, X, device="cuda", batch=48):
    """Same as solarfil.refiner.predict_refined (mean over models x 4 rotations) but in batches of 48 (GPU memory)."""
    import torch
    out = np.zeros((len(X), X.shape[2], X.shape[3]), np.uint8)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.from_numpy(X[i:i + batch].astype(np.float32) / 255).to(device); x[:, 0] = (x[:, 0] - 0.45) / 0.25; ps = 0
            for m in models:
                for k in range(4):
                    with torch.autocast("cuda", dtype=torch.float16):
                        q = torch.sigmoid(m(torch.rot90(x, k, (2, 3))).float())
                    ps = ps + torch.rot90(q, -k, (2, 3))
            out[i:i + len(x)] = (ps[:, 0] / (4 * len(models)) * 255).round().clamp(0, 255).byte().cpu().numpy()
    return out


def predict_rot1(models, X, device="cuda", batch=48):
    """Screening mode: like solarfil.refiner.predict_refined but without the 4-rotation averaging (4x less GPU)."""
    import torch
    out = np.zeros((len(X), X.shape[2], X.shape[3]), np.uint8)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.from_numpy(X[i:i + batch].astype(np.float32) / 255).to(device); x[:, 0] = (x[:, 0] - 0.45) / 0.25; ps = 0
            for m in models:
                with torch.autocast("cuda", dtype=torch.float16):
                    ps = ps + torch.sigmoid(m(x).float())
            out[i:i + len(x)] = (ps[:, 0] / len(models) * 255).round().clamp(0, 255).byte().cpu().numpy()
    return out


def nodes_stage(a):
    sys.path.insert(0, a.team); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from solarfil import config as C
    from solarfil.data import read_image
    from solarfil.refiner import build_refiner, crop, predict_refined, window
    from solarfil.scorer import disk_radius, features
    import data as D
    from tree_select import build_tree
    k_, n_ = map(int, a.shard.split("/"))
    stems = sorted(f[:-4] for f in os.listdir(a.oof) if f.endswith(".pkl"))
    todo = [s for s in stems if not os.path.exists(os.path.join(a.out, s + ".pkl"))]
    todo = (todo[::-1] if a.reverse else todo)[k_::n_]      # --reverse: extra workers start from the end
    os.makedirs(a.out, exist_ok=True)
    W = os.path.join(a.team, "weights")
    folds_needed = range(5) if a.fold < 0 else [a.fold]           # --fold k: this worker only does images of fold k
    ref = {k: [build_refiner(r["arch"], r["encoder"], f"{W}/{r['name']}/fold{k}.pt", "cuda") for r in C.REFINERS] for k in folds_needed}
    img_dir = os.path.join(a.root, "train", "train_images")
    for i, s in enumerate(todo, 1):
        if os.path.exists(os.path.join(a.out, s + ".pkl")):  # another worker got here first
            continue
        T = pickle.load(open(os.path.join(a.oof, s + ".pkl"), "rb")); k = T["fold"]
        if k not in ref:
            continue
        img = read_image(os.path.join(img_dir, s + ".jpeg")); disk = D.disk_mask(img)
        p = T["prob"].astype(np.float32) / 255
        p2 = cv2.resize(p, (2048, 2048), interpolation=cv2.INTER_LINEAR)
        nodes, parent = build_tree(p2, disk)
        # merge chains: a node that keeps >= 97 % of its parent's area replaces nothing new
        area = np.array([m.sum() for _, m in nodes], float); keep = np.ones(len(nodes), bool); newpar = parent.copy()
        for j in range(len(nodes)):
            q = parent[j]
            if q >= 0 and area[j] >= a.merge * area[q]:
                keep[j] = False
        remap = {}
        for j in range(len(nodes)):                       # parent of a kept node = nearest kept ancestor
            q = parent[j]
            while q >= 0 and not keep[q]:
                q = parent[q]
            newpar[j] = q
        idx = np.nonzero(keep)[0]; pos = {j: t for t, j in enumerate(idx)}
        nodes = [nodes[j] for j in idx]; parent = np.array([pos[newpar[j]] if newpar[j] >= 0 else -1 for j in idx], int)
        if not nodes:
            pickle.dump({"fold": k, "rles": [], "parent": parent, "X": np.zeros((0, 25), np.float32)}, open(os.path.join(a.out, s + ".pkl"), "wb")); continue
        rles = [mu.encode(np.asfortranarray(m.astype(np.uint8))) for _, m in nodes]
        lvl_union = defaultdict(lambda: np.zeros((2048, 2048), np.uint8))
        for (li, m) in nodes:
            lvl_union[li] |= m.astype(np.uint8)
        p8 = (p2 * 255).clip(0, 255).astype(np.uint8)
        X, Wn = [], []
        for (li, m), r in zip(nodes, rles):
            y0, x0, side = window(r); mm = m.astype(np.uint8)
            other = ((lvl_union[li] & (1 - mm)) * 255).astype(np.uint8)
            X.append(np.stack([crop(img, y0, x0, side), crop(p8, y0, x0, side),
                               crop(mm * 255, y0, x0, side), crop(other, y0, x0, side)]))
            Wn.append((y0, x0, side))
        refs = (predict_rot1 if a.rot1 else predict_rot4)(ref[k], np.array(X, np.uint8), "cuda")
        out = []
        for (li, m), r, (y0, x0, side), rf in zip(nodes, rles, Wn, refs):
            pr = cv2.resize(rf.astype(np.float32) / 255, (side, side), interpolation=cv2.INTER_LINEAR) >= C.REFINE["thr"]
            full = np.zeros((2048, 2048), bool)
            ya, xa, yb, xb = max(y0, 0), max(x0, 0), min(y0 + side, 2048), min(x0 + side, 2048)
            full[ya:yb, xa:xb] = pr[ya - y0:yb - y0, xa - x0:xb - x0]
            n, lab = cv2.connectedComponents(full.astype(np.uint8), connectivity=8)
            seed = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            hit = np.unique(lab[seed & (lab > 0)])
            full = np.isin(lab, hit) if len(hit) else full
            if full.sum() < C.REFINE["keep_small"] * m.sum():
                full = m
            out.append(mu.encode(np.asfortranarray(full.astype(np.uint8))))
        F = features(out, [p], img, disk_radius(img))
        ch = np.bincount(parent[parent >= 0], minlength=len(nodes)); depth = np.zeros(len(nodes))
        for j in range(len(nodes)):
            if parent[j] >= 0:
                depth[j] = depth[parent[j]] + 1
        tf = np.c_[[li for li, _ in nodes], [area[idx[j]] / area[idx[parent[j]]] if parent[j] >= 0 else 1.0 for j in range(len(nodes))],
                   ch, depth, parent < 0].astype(np.float32)
        pickle.dump({"fold": k, "rles": out, "parent": parent, "X": np.c_[F, tf].astype(np.float32)}, open(os.path.join(a.out, s + ".pkl"), "wb"))
        if i % 20 == 0:
            print(i, len(todo), s, len(nodes), "nodes", flush=True)
    print("done", len(todo))


def select_stage(a):
    sys.path.insert(0, a.team); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sklearn.ensemble import HistGradientBoostingRegressor
    from solarfil.data import find_data, load_annotations
    from solarfil.metrics import pq_from, score_image
    from solarfil.postprocess import shrink_all
    from solarfil.scorer import FEATS
    from tree_select import dp_select
    gt = load_annotations(find_data(a.root)["ann"])
    stems = sorted(f[:-4] for f in os.listdir(a.out) if f.endswith(".pkl"))
    N = {s: pickle.load(open(os.path.join(a.out, s + ".pkl"), "rb")) for s in stems}
    B = {s: pickle.load(open(os.path.join(a.oof, s + ".pkl"), "rb")) for s in stems}
    ip, ir = FEATS.index("p_mean"), FEATS.index("r")

    def targets(pred, gs):
        ne = max(len(gs), 1); h = np.zeros(len(pred)); m = np.zeros(len(pred))
        for g in gs.values():
            if g and pred:
                iou = np.array(mu.iou(pred, g, [0] * len(g))).reshape(len(pred), len(g)).max(1)
                h += iou > 0.5; m += np.where(iou > 0.5, iou, 0)
        return h / ne, m / ne, np.full(len(pred), ne, float)

    def make():
        return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400, min_samples_leaf=30, l2_regularization=1.0, random_state=0)

    def crossfit(Xs, key):
        X = np.concatenate([Xs[s] for s in stems]); fold = np.concatenate([np.full(len(Xs[s]), N[s]["fold"]) for s in stems])
        tg = [targets(key(s), gt[s]) for s in stems]
        H = np.concatenate([t[0] for t in tg]); M = np.concatenate([t[1] for t in tg]); Wt = np.concatenate([t[2] for t in tg])
        sc = np.zeros(len(X))
        for k in range(5):
            tr, te = fold != k, fold == k; pos = tr & (H > 0)
            mh = make().fit(X[tr], H[tr], sample_weight=Wt[tr]); mm = make().fit(X[pos], M[pos] / H[pos], sample_weight=(Wt * H)[pos])
            sc[te] = np.clip(mh.predict(X[te]), 0, 1) * np.clip(mm.predict(X[te]), 0, 1)
        out, o = {}, 0
        for s in stems:
            n = len(Xs[s]); out[s] = sc[o:o + n]; o += n
        return out, float(np.corrcoef(sc, H)[0, 1])

    res = {}
    # baseline: v18 candidates, cross-fitted scorer, shipped rule
    sb, cb = crossfit({s: B[s]["feats"] for s in stems}, lambda s: B[s]["pred"])
    for clf in (0.16, 0.2, 0.24):
        per = {}
        for s in stems:
            X = B[s]["feats"]; keep = [j for j in range(len(X)) if sb[s][j] >= clf and X[j, ip] >= 0.7 and X[j, ir] <= 0.95]
            per[s] = score_image(gt[s], shrink_all([B[s]["pred"][j] for j in keep], 0.5))
        res[f"v18 candidates, clf {clf}"] = round(pq_from(list(per.values()))["pq"], 4)
        if clf == 0.2:
            base = per
    # tree: all refined nodes, DP
    st, ct = crossfit({s: N[s]["X"] for s in stems}, lambda s: N[s]["rles"])
    res["corr(score, h): v18 candidates / tree nodes"] = [round(cb, 4), round(ct, 4)]
    best = None
    for tau in (0.14, 0.17, 0.2, 0.23, 0.26):
        for rule in (True, False):
            per = {}
            for s in stems:
                X, par, rl = N[s]["X"], N[s]["parent"], N[s]["rles"]
                if not len(rl):
                    per[s] = score_image(gt[s], []); continue
                g = st[s] - tau
                if rule:
                    g = np.where((X[:, ip] >= 0.7) & (X[:, ir] <= 0.95), g, -1.0)
                ch = sorted(dp_select(par, g), key=lambda j: -st[s][j])
                taken = np.zeros((2048, 2048), np.uint8); keep = []
                for j in ch:
                    m = mu.decode(rl[j]); mm = m & (1 - taken)
                    if mm.sum() >= 0.5 * m.sum():
                        taken |= mm; keep.append(mu.encode(np.asfortranarray(mm)))
                per[s] = score_image(gt[s], shrink_all(keep, 0.5))
            v = round(pq_from(list(per.values()))["pq"], 4); key = f"tree + v18 refiner + DP, tau {tau}, v18 rule {rule}"
            res[key] = v; print(key, v, flush=True)
            if best is None or v > best[0]:
                best = (v, key, per)
    # paired bootstrap of the best tree setting vs the baseline (setting chosen on all images: slightly optimistic)
    L = stems; rng = np.random.default_rng(0); d = []
    for _ in range(5000):
        b = rng.integers(0, len(L), len(L))
        d.append(pq_from([best[2][L[i]] for i in b])["pq"] - pq_from([base[L[i]] for i in b])["pq"])
    d = np.array(d)
    res["best tree setting"] = best[1]
    res["difference best tree - v18 (clf 0.2)"] = {"mean": round(float(d.mean()), 4), "95% interval": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)], "P(better)": round(float((d > 0).mean()), 3)}
    json.dump(res, open(os.path.join(a.out, "select.json"), "w"), indent=1); print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["nodes", "select"])
    for k in ("team", "root", "oof", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--fold", type=int, default=-1); ap.add_argument("--merge", type=float, default=0.97)
    ap.add_argument("--rot1", action="store_true", help="screening: refiners without the 4-rotation averaging")
    a = ap.parse_args()
    nodes_stage(a) if a.cmd == "nodes" else select_stage(a)


if __name__ == "__main__":
    main()
