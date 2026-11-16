"""Bootstrap (over images) 95% intervals of PQ / SQ / RQ: ours on 707 images, ours and expert-vs-expert on the multi-expert images.
Out of fold, final configuration, from the cached IoU tables.
"""
import argparse, json, os, pickle, sys

import numpy as np
from pycocotools import mask as mu


def main(a):
    from solarfil.data import find_data, load_annotations
    gt = load_annotations(find_data(a.root)["ann"]); C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb"))
    O, X, multi = [], [], []
    for s, keep in C["keep"].items():
        w = pickle.load(open(os.path.join(a.work, "why", s + ".pkl"), "rb")); es = list(gt[s]); o = np.zeros(4); x = np.zeros(4)
        for e in es:
            J = w["ious_r"][e][keep]; ng = len(gt[s][e]); hit = J > .5; t = hit.sum(); o += [J[hit].sum(), t, len(J) - t, ng - t]
            for q in es:
                if q != e:
                    G, B = gt[s][e], gt[s][q]
                    Y = np.asarray(mu.iou(B, G, [0] * len(G))).reshape(len(B), len(G)) if B and G else np.zeros((len(B), len(G)))
                    h = Y > .5; t2 = h.sum(); x += [Y[h].sum(), t2, len(B) - t2, len(G) - t2]
        O.append(o); X.append(x); multi.append(len(es) > 1)
    O, X, multi = np.array(O), np.array(X), np.array(multi); rng = np.random.default_rng(0)

    def stats(v):
        return np.array([v[0] / (v[1] + .5 * v[2] + .5 * v[3]), v[0] / v[1], v[1] / (v[1] + .5 * v[2] + .5 * v[3])])

    def ci(M, D=None):
        n = len(M); bs = []
        for _ in range(5000):
            i = rng.integers(0, n, n); bs.append(stats(M[i].sum(0)) - (stats(D[i].sum(0)) if D is not None else 0))
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0); pt = stats(M.sum(0)) - (stats(D.sum(0)) if D is not None else 0)
        return {k: [float(pt[j]), float(lo[j]), float(hi[j])] for j, k in enumerate(("PQ", "SQ", "RQ"))}

    out = {"ours, 707 images": ci(O), "ours, multi-expert images": ci(O[multi]), "expert vs expert, multi-expert images": ci(X[multi]),
           "ours minus experts (paired), multi-expert images": ci(O[multi], X[multi]), "n multi": int(multi.sum())}
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("team", "root", "oof", "work", "eda", "out"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args(); sys.path.insert(0, a.team); main(a)
