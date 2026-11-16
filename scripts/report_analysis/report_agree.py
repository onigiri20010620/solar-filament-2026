"""Our hit rate on an expert's filament as a function of how many OTHER experts drew the same filament (IoU > 0.5).
Multi-expert training images only, out of fold, final configuration. Also PQ of ours and of expert-vs-expert on these images.
"""
import argparse, json, os, pickle, sys

import numpy as np
from pycocotools import mask as mu


def main(a):
    from solarfil.data import find_data, load_annotations
    gt = load_annotations(find_data(a.root)["ann"]); C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb"))
    rows = []; ours = np.zeros(4); xp = np.zeros(4); n_img = 0          # siou, tp, fp, fn
    for s, keep in C["keep"].items():
        es = list(gt[s])
        if len(es) < 2:
            continue
        n_img += 1; w = pickle.load(open(os.path.join(a.work, "why", s + ".pkl"), "rb"))
        for e in es:
            G = gt[s][e]; J = w["ious_r"][e][keep]; ng = len(G)
            hit = J > .5; t = hit.sum(); ours += [J[hit].sum(), t, len(J) - t, ng - t]
            best = J.max(0) if len(J) and ng else np.zeros(ng)
            agree = np.zeros(ng, int); partial = np.zeros(ng, int)
            for o in es:
                if o == e:
                    continue
                B = gt[s][o]
                X = np.asarray(mu.iou(B, G, [0] * ng)).reshape(len(B), ng) if B and ng else np.zeros((len(B), ng))
                bo = X.max(0) if len(B) and ng else np.zeros(ng)
                agree += bo > .5; partial += (bo > 0) & (bo <= .5)
                h2 = X > .5; t2 = h2.sum(); xp += [X[h2].sum(), t2, len(B) - t2, ng - t2]
            for g in range(ng):
                rows.append((len(es) - 1, agree[g], partial[g], best[g], float(mu.area(G[g]))))
    R = np.array(rows, float); pq = lambda v: v[0] / (v[1] + .5 * v[2] + .5 * v[3])
    out = {"multi-expert images": n_img, "expert filaments": len(R), "PQ ours": pq(ours), "PQ expert vs expert": pq(xp)}
    fr = R[:, 1] / R[:, 0]
    for nm, m in (("no other expert drew it (IoU > 0.5)", fr == 0), ("some but not all others", (fr > 0) & (fr < 1)), ("all other experts drew it", fr == 1)):
        out[nm] = {"n": int(m.sum()), "share": float(m.mean()), "our hit rate": float((R[m, 3] > .5).mean()), "our mean IoU when overlapping": float(R[m & (R[:, 3] > 0), 3].mean()),
                   "median area": float(np.median(R[m, 4]))}
    m0 = fr == 0
    out["of 'no other expert': others overlap partly (0 < IoU <= 0.5)"] = float((R[m0, 2] > 0).mean())
    out["of 'no other expert': nobody else drew anything there"] = float((R[m0, 2] == 0).mean())
    miss = R[:, 3] <= .5
    out["our misses: share that no other expert matched either"] = float((fr[miss] == 0).mean())
    out["our misses: share that all others matched"] = float((fr[miss] == 1).mean())
    json.dump(out, open(a.out, "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("team", "root", "oof", "work", "eda", "out"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args(); sys.path.insert(0, a.team); main(a)
