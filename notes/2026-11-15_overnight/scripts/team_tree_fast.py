"""Fast evaluation of team_tree.py nodes (v18 refiner on every node of the threshold tree of v18's map).

cache (12 processes, once): per image the 0.5 px-eroded refined node masks' IoU with every annotator's filaments and
  the pairwise overlap share among refined nodes (|i & j| / |i|).
select: hurdle scorer (HistGB h, m; weights = experts) cross-fitted on the team's folds over the 25 node features;
  per image the antichain of the tree with the largest sum of (score - tau) (tree_select.dp_select; nodes failing
  v18's p_mean >= 0.7 / r <= 0.95 rule excluded when 'rule'), then by score, a node is dropped if already-taken nodes
  cover >= 20 % of it; PQ from the cached IoU tables. Baseline = v18 (team_pool_fast cache, setting A clf 0.2 rule),
  compared with a paired bootstrap over images.
Usage: python team_tree_fast.py cache --tree_out <team_tree dir> --root <data> --team <dir> --cache <dir>
       python team_tree_fast.py select --cache <dir> --pool_cache <team_pool_cache> --team <dir> --out <json>
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pycocotools import mask as mu


def cache_one(args):
    team, s, path, gsets, out = args
    sys.path.insert(0, team)
    from solarfil.postprocess import shrink_all
    N = pickle.load(open(path, "rb")); rl = N["rles"]
    sh = [(shrink_all([r], 0.5) or [r])[0] for r in rl]
    ious = {e: (np.asarray(mu.iou(sh, g, [0] * len(g))).reshape(len(sh), len(g)) if sh and g else np.zeros((len(sh), len(g)))) for e, g in gsets.items()}
    area = np.array([mu.area(r) for r in rl], float); ov = np.zeros((len(rl), len(rl)))
    if len(rl) > 1:
        I = np.asarray(mu.iou(rl, rl, [0] * len(rl))).reshape(len(rl), len(rl))
        inter = I * (area[:, None] + area[None, :]) / (1 + I); np.fill_diagonal(inter, 0)
        ov = inter / np.maximum(area[:, None], 1)
    pickle.dump({"X": N["X"], "parent": N["parent"], "fold": N["fold"], "ious": ious, "ov": ov}, open(os.path.join(out, s + ".pkl"), "wb"))
    return s


def cache(a):
    sys.path.insert(0, a.team)
    from solarfil.data import find_data, load_annotations
    gt = load_annotations(find_data(a.root)["ann"]); os.makedirs(a.cache, exist_ok=True)
    stems = sorted(f[:-4] for f in os.listdir(a.tree_out) if f.endswith(".pkl"))
    jobs = [(a.team, s, os.path.join(a.tree_out, s + ".pkl"), gt[s], a.cache) for s in stems if not os.path.exists(os.path.join(a.cache, s + ".pkl"))]
    with ProcessPoolExecutor(12) as ex:
        n = sum(1 for _ in ex.map(cache_one, jobs, chunksize=2))
    print("cached", n)


def evaluate(C, stems, sc, choose):
    res = {}
    for s in stems:
        c = C[s]; chosen_raw = choose(s, c, sc[s]); chosen = []
        for i in sorted(chosen_raw, key=lambda i: -sc[s][i]):
            if chosen and c["ov"][i, chosen].sum() >= 0.2:
                continue
            chosen.append(i)
        tot = np.zeros(4)
        for I in c["ious"].values():
            ng = I.shape[1]
            if ng == 0:
                tot[2] += len(chosen); continue
            J = I[chosen] if chosen else np.zeros((0, ng)); hit = J > 0.5
            tot += [J[hit].sum(), hit.sum(), (~hit.any(1)).sum() if len(chosen) else 0, (~hit.any(0)).sum()]
        res[s] = tot
    return res


def select(a):
    sys.path.insert(0, a.team); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sklearn.ensemble import HistGradientBoostingRegressor
    from solarfil.scorer import FEATS
    from tree_select import dp_select
    stems = sorted(f[:-4] for f in os.listdir(a.cache) if f.endswith(".pkl"))
    C = {s: pickle.load(open(os.path.join(a.cache, s + ".pkl"), "rb")) for s in stems}
    P = {s: pickle.load(open(os.path.join(a.pool_cache, s + ".pkl"), "rb")) for s in stems}
    ip, ir = FEATS.index("p_mean"), FEATS.index("r")

    def make():
        return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400, min_samples_leaf=30, l2_regularization=1.0, random_state=0)

    def crossfit(D, rowsel=None):
        X = np.concatenate([D[s]["X"] for s in stems]); F = np.concatenate([np.full(len(D[s]["X"]), D[s]["fold"]) for s in stems])
        H, M, Wt = [], [], []
        for s in stems:
            n = len(D[s]["X"]); ne = max(len(D[s]["ious"]), 1); h = np.zeros(n); m = np.zeros(n)
            for I in D[s]["ious"].values():
                if I.size:
                    b = I.max(1); h += b > 0.5; m += np.where(b > 0.5, b, 0)
            H.append(h / ne); M.append(m / ne); Wt.append(np.full(n, ne, float))
        H, M, Wt = map(np.concatenate, (H, M, Wt))
        sel = np.ones(len(X), bool) if rowsel is None else np.concatenate([rowsel(D[s]) for s in stems])
        sc = np.full(len(X), -1.0)
        for k in range(5):
            tr, te = (F != k) & sel, (F == k) & sel; pos = tr & (H > 0)
            mh = make().fit(X[tr], H[tr], sample_weight=Wt[tr]); mm = make().fit(X[pos], M[pos] / H[pos], sample_weight=(Wt * H)[pos])
            sc[te] = np.clip(mh.predict(X[te]), 0, 1) * np.clip(mm.predict(X[te]), 0, 1)
        out, o = {}, 0
        for s in stems:
            n = len(D[s]["X"]); out[s] = sc[o:o + n]; o += n
        return out

    pq = lambda r: (lambda v: v[0] / (v[1] + .5 * v[2] + .5 * v[3]))(sum(r.values()))
    # baseline: v18 candidates (pool cache rows with src == 0), v18's 20 features, clf 0.2 + rule
    Pv = {s: {"X": P[s]["X"][P[s]["X"][:, 20] == 0][:, :20], "ious": {e: I[P[s]["X"][:, 20] == 0] for e, I in P[s]["ious"].items()},
              "ov": P[s]["ov"][np.ix_(P[s]["X"][:, 20] == 0, P[s]["X"][:, 20] == 0)], "fold": P[s]["fold"]} for s in stems}
    sb = crossfit(Pv)
    base = evaluate(Pv, stems, sb, lambda s, c, v: [i for i in range(len(v)) if v[i] >= 0.2 and c["X"][i, ip] >= 0.7 and c["X"][i, ir] <= 0.95])
    out = {"v18 (cross-fitted scorer, clf 0.2, rule)": round(float(pq(base)), 4)}
    st = crossfit(C)
    R = {}
    for tau in (0.12, 0.16, 0.2, 0.24, 0.28):
        for rule in (True, False):
            def choose(s, c, v, tau=tau, rule=rule):
                g = v - tau
                if rule:
                    g = np.where((c["X"][:, ip] >= 0.7) & (c["X"][:, ir] <= 0.95), g, -1.0)
                return dp_select(c["parent"], g) if len(v) else []
            key = f"tree + v18 refiner + DP, tau {tau}, rule {rule}"
            R[key] = evaluate(C, stems, st, choose); out[key] = round(float(pq(R[key])), 4); print(key, out[key], flush=True)
    best = max(R, key=lambda k: out[k])
    rng = np.random.default_rng(0); d = []
    for _ in range(5000):
        b = rng.integers(0, len(stems), len(stems))
        d.append(pq({i: R[best][stems[j]] for i, j in enumerate(b)}) - pq({i: base[stems[j]] for i, j in enumerate(b)}))
    d = np.array(d)
    out["best vs v18"] = {"setting": best, "PQ": out[best], "mean diff": round(float(d.mean()), 4),
                          "95% interval": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)], "P(better)": round(float((d > 0).mean()), 3)}
    folds = sorted({C[s]["fold"] for s in stems})
    out["per fold (v18, best)"] = [[round(float(pq({s: base[s] for s in stems if C[s]["fold"] == k})), 4), round(float(pq({s: R[best][s] for s in stems if C[s]["fold"] == k})), 4)] for k in folds]
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["cache", "select"])
    for k in ("tree_out", "root", "team", "cache", "pool_cache", "out"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args()
    cache(a) if a.cmd == "cache" else select(a)


if __name__ == "__main__":
    main()
