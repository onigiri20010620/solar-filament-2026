"""Fast version of team_pool.py: the heavy per-candidate work is done ONCE and cached, every setting is table look-up.

cache (10 processes, once): per train image the pooled candidates (v18 refined candidates + our final filaments),
  their 24 features (as team_pool.py), the 0.5 px-eroded masks' IoU with every annotator's filaments, and the
  pairwise overlap share among candidates (|i & j| / |i|).  -> <cache>/<stem>.pkl
select (seconds per setting): cross-fitted hurdle scorers; per image candidates by score, a candidate is skipped if the
  candidates already taken cover >= 20 % of it (sum of pairwise shares - trimming of small overlaps is ignored, which
  the exact team_pool.py run checks); PQ from the cached IoU tables (IoU > 0.5 per annotator, as solarfil.metrics).
  Settings: A v18 only / B v18 + agreement features / C pooled, learned scorer / D pooled, fixed voting rule
  (both systems agree (IoU > 0.5) -> keep if the pooled score >= c_agree; one system only -> keep if >= c_single).
Usage: python team_pool_fast.py cache --team <dir> --root <data> --oof <dir> --fine <dir> --runs <prefix> --tree <dir> --cache <dir>
       python team_pool_fast.py select --team <dir> --root <data> --cache <dir> --out <json>
"""
import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pycocotools import mask as mu


def cache_one(args):
    team, root, s, T, meta_img, preds_img, prob8, u, gsets, out = args
    sys.path.insert(0, team); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from team_pool import work
    from solarfil.postprocess import shrink_all
    _, rl, X = work((team, root, s, T, meta_img, preds_img, prob8, u))
    sh = [shrink_all([r], 0.5) for r in rl]
    sh = [x[0] if x else r for x, r in zip(sh, rl)]
    ious = {}
    for e, g in gsets.items():
        ious[e] = np.asarray(mu.iou(sh, g, [0] * len(g))).reshape(len(sh), len(g)) if sh and g else np.zeros((len(sh), len(g)))
    area = np.array([mu.area(r) for r in rl], float)
    inter = np.zeros((len(rl), len(rl)))
    if len(rl) > 1:
        I = np.asarray(mu.iou(rl, rl, [0] * len(rl))).reshape(len(rl), len(rl))    # IoU -> intersection
        inter = I * (area[:, None] + area[None, :]) / (1 + I)
        np.fill_diagonal(inter, 0)
    ov = inter / np.maximum(area[:, None], 1)
    pickle.dump({"X": X, "ious": ious, "ov": ov, "n_gt": {e: len(g) for e, g in gsets.items()}, "fold": T["fold"]},
                open(os.path.join(out, s + ".pkl"), "wb"))
    return s


def cache(a):
    sys.path.insert(0, a.team)
    from solarfil.data import find_data, load_annotations
    gt = load_annotations(find_data(a.root)["ann"])
    os.makedirs(a.cache, exist_ok=True)
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

    def jobs():
        for s in stems:
            if os.path.exists(os.path.join(a.cache, s + ".pkl")):
                continue
            T = pickle.load(open(os.path.join(a.oof, s + ".pkl"), "rb"))
            yield (a.team, a.root, s, T, [meta[i] for i in per.get(s, [])], [oof[i] for i in per.get(s, [])], probs[s],
                   np.array(u_of[s]), gt[s], a.cache)
    n = 0
    with ProcessPoolExecutor(12) as ex:
        for _ in ex.map(cache_one, jobs(), chunksize=2):
            n += 1
            if n % 100 == 0:
                print(n, flush=True)
    print("cached", n)


def select(a):
    sys.path.insert(0, a.team)
    from sklearn.ensemble import HistGradientBoostingRegressor
    from solarfil.scorer import FEATS
    stems = sorted(f[:-4] for f in os.listdir(a.cache) if f.endswith(".pkl"))
    C = {s: pickle.load(open(os.path.join(a.cache, s + ".pkl"), "rb")) for s in stems}
    X = np.concatenate([C[s]["X"] for s in stems]); F = np.concatenate([np.full(len(C[s]["X"]), C[s]["fold"]) for s in stems])
    H, M, Wt = [], [], []
    for s in stems:
        n = len(C[s]["X"]); ne = max(len(C[s]["ious"]), 1); h = np.zeros(n); m = np.zeros(n)
        for I in C[s]["ious"].values():
            if I.size:
                b = I.max(1); h += b > 0.5; m += np.where(b > 0.5, b, 0)
        H.append(h / ne); M.append(m / ne); Wt.append(np.full(n, ne, float))
    H, M, Wt = map(np.concatenate, (H, M, Wt))
    src = X[:, 20]; agree = X[:, 21] > 0.5
    off = np.cumsum([0] + [len(C[s]["X"]) for s in stems])
    ip, ir = FEATS.index("p_mean"), FEATS.index("r")

    def make():
        return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400, min_samples_leaf=30, l2_regularization=1.0, random_state=0)

    def crossfit(cols, rowsel):
        sc = np.full(len(X), -1.0); Xc = X[:, cols]
        for k in range(5):
            tr, te = (F != k) & rowsel, (F == k) & rowsel; pos = tr & (H > 0)
            mh = make().fit(Xc[tr], H[tr], sample_weight=Wt[tr]); mm = make().fit(Xc[pos], M[pos] / H[pos], sample_weight=(Wt * H)[pos])
            sc[te] = np.clip(mh.predict(Xc[te]), 0, 1) * np.clip(mm.predict(Xc[te]), 0, 1)
        return sc

    def run(keepfn, sc):
        """keepfn(global idx array) -> bool array; returns per-image (sum IoU, TP, FP, FN)."""
        res = {}
        for t, s in enumerate(stems):
            g = np.arange(off[t], off[t + 1]); c = C[s]
            ok = keepfn(g); cand = np.argsort(-sc[g]); chosen = []
            for i in cand:
                if not ok[i]:
                    continue
                if chosen and c["ov"][i, chosen].sum() >= 0.2:
                    continue
                chosen.append(i)
            tot = np.zeros(4)
            for e, I in c["ious"].items():
                ng = I.shape[1]
                if ng == 0:
                    tot[2] += len(chosen); continue
                J = I[chosen] if chosen else np.zeros((0, ng)); hit = J > 0.5
                tot += [J[hit].sum(), hit.sum(), (~hit.any(1)).sum() if len(chosen) else 0, (~hit.any(0)).sum()]
            res[s] = tot
        return res

    pq = lambda r: (lambda v: v[0] / (v[1] + .5 * v[2] + .5 * v[3]))(sum(r.values()))
    out = {"candidates": {"v18": int((src == 0).sum()), "ours": int((src == 1).sum())}}
    allc, v18c = list(range(24)), list(range(20))
    S = {"A": (crossfit(v18c, src == 0), src == 0), "B": (crossfit(allc, src == 0), src == 0), "C": (crossfit(allc, np.ones(len(X), bool)), np.ones(len(X), bool))}
    rule = (X[:, ip] >= 0.7) & (X[:, ir] <= 0.95)
    R = {}
    for nm, (sc, sel) in S.items():
        for clf in (0.12, 0.16, 0.2, 0.24, 0.28, 0.32):
            for rl in (True, False):
                key = f"{nm} clf {clf} rule {rl}"
                R[key] = run(lambda g, clf=clf, rl=rl, sel=sel, sc=sc: sel[g] & (sc[g] >= clf) & (rule[g] if rl else True), sc)
                out[key] = round(float(pq(R[key])), 4)
    sc = S["C"][0]
    for ca in (0.1, 0.15, 0.2):
        for cs in (0.25, 0.3, 0.35, 0.4):
            key = f"D vote: agree >= {ca}, single >= {cs}"
            R[key] = run(lambda g, ca=ca, cs=cs: np.where(agree[g], sc[g] >= ca, sc[g] >= cs), sc)
            out[key] = round(float(pq(R[key])), 4)
    base = R["A clf 0.2 rule True"]
    for nm in ("B", "C", "D"):
        best = max((k for k in R if k.startswith(nm)), key=lambda k: out[k])
        rng = np.random.default_rng(0); d = []
        for _ in range(5000):
            b = rng.integers(0, len(stems), len(stems))
            d.append(pq({i: R[best][stems[j]] for i, j in enumerate(b)}) - pq({i: base[stems[j]] for i, j in enumerate(b)}))
        d = np.array(d)
        out[f"{nm} best vs A (clf 0.2, rule)"] = {"setting": best, "PQ": out[best], "mean diff": round(float(d.mean()), 4),
                                                  "95% interval": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                                                  "P(better)": round(float((d > 0).mean()), 3)}
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if " vs " in k or k.startswith("cand") or k == "A clf 0.2 rule True"}, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["cache", "select"])
    for k in ("team", "root", "oof", "fine", "runs", "tree", "cache", "out"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args()
    cache(a) if a.cmd == "cache" else select(a)


if __name__ == "__main__":
    main()
