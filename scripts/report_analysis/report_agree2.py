"""Expert agreement, part 2 (multi-expert training images, out of fold, final configuration).
A  hit rate of the best draft / best re-drawn candidate / final kept prediction by agreement class of the expert's filament
B  3-expert images: on filaments of expert e that expert o1 confirmed (IoU > 0.5), hit rate and IoU of the third expert o2 vs ours
C  kept and rejected candidates by the fraction of experts that match them
D  our false positives against one expert: share that match (IoU > 0.5) a filament of another expert of the same image
"""
import argparse, json, os, pickle, sys

import numpy as np
from pycocotools import mask as mu


def main(a):
    from solarfil.data import find_data, load_annotations
    gt = load_annotations(find_data(a.root)["ann"]); C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb"))
    A, B, Ck, Cr, D = [], [], [], [], []; nexp = {}
    for s, keep in C["keep"].items():
        es = list(gt[s]); nexp[len(es)] = nexp.get(len(es), 0) + 1
        if len(es) < 2:
            continue
        w = pickle.load(open(os.path.join(a.work, "why", s + ".pkl"), "rb"))
        X = {}
        for e in es:
            for o in es:
                if o != e:
                    G, Bm = gt[s][e], gt[s][o]
                    X[e, o] = np.asarray(mu.iou(Bm, G, [0] * len(G))).reshape(len(Bm), len(G)) if Bm and G else np.zeros((len(Bm), len(G)))
        mx = lambda J, n: J.max(0) if J.shape[0] and n else np.zeros(n)
        for e in es:
            ng = len(gt[s][e]); J = w["ious_r"][e]; J0 = w["ious_d0"][e]; K = J[keep]
            bo = np.array([mx(X[e, o], ng) for o in es if o != e]); fr = (bo > .5).mean(0)
            bd, br, bk = mx(J0, ng), mx(J, ng), mx(K, ng)
            for g in range(ng):
                A.append((fr[g], bd[g], br[g], bk[g]))
            if len(es) >= 3:
                for o1 in es:
                    for o2 in es:
                        if len({e, o1, o2}) == 3:
                            c1 = mx(X[e, o1], ng) > .5; b2 = mx(X[e, o2], ng)
                            for g in np.nonzero(c1)[0]:
                                B.append((b2[g], bk[g], bd[g]))
        # candidates: fraction of experts matching
        n = len(keep); h = np.zeros(n)
        for e in es:
            J = w["ious_r"][e]
            if J.shape[1]:
                h += J.max(1) > .5
        h /= len(es); Ck += h[keep].tolist(); Cr += h[~keep].tolist()
        for e in es:
            J = w["ious_r"][e][keep]; fp = (J.max(1) <= .5) if J.shape[1] else np.ones(len(J), bool)
            oth = np.zeros(len(J), bool)
            for o in es:
                if o != e:
                    Jo = w["ious_r"][o][keep]
                    if Jo.shape[1]:
                        oth |= Jo.max(1) > .5
            D += oth[fp].tolist()
    A = np.array(A); B = np.array(B); Ck = np.array(Ck); Cr = np.array(Cr); D = np.array(D); out = {"images by number of experts": nexp}
    for nm, m in (("none", A[:, 0] == 0), ("some", (A[:, 0] > 0) & (A[:, 0] < 1)), ("all", A[:, 0] == 1), ("any filament", np.ones(len(A), bool))):
        out[f"A {nm}"] = {"n": int(m.sum()), "hit best draft": float((A[m, 1] > .5).mean()), "hit best re-drawn": float((A[m, 2] > .5).mean()), "hit final": float((A[m, 3] > .5).mean()),
                          "mean IoU best draft (overlapping)": float(A[m & (A[:, 1] > 0), 1].mean()), "mean IoU best re-drawn (overlapping)": float(A[m & (A[:, 2] > 0), 2].mean())}
    out["B n (filaments confirmed by a second expert, 3-expert images)"] = len(B)
    out["B third expert: hit rate"] = float((B[:, 0] > .5).mean()); out["B ours: hit rate"] = float((B[:, 1] > .5).mean()); out["B best draft: hit rate"] = float((B[:, 2] > .5).mean())
    both = (B[:, 0] > .5) & (B[:, 1] > .5)
    out["B both hit: mean IoU third expert / ours"] = [float(B[both, 0].mean()), float(B[both, 1].mean())]
    out["B mean IoU (all, zeros included) third expert / ours"] = [float(B[:, 0].mean()), float(B[:, 1].mean())]
    for nm, v in (("kept", Ck), ("rejected", Cr)):
        out[f"C {nm}"] = {"n": len(v), "no expert matches": float((v == 0).mean()), "some": float(((v > 0) & (v < 1)).mean()), "all experts match": float((v == 1).mean())}
    out["D false positives against one expert"] = len(D); out["D share matching another expert of the same image"] = float(D.mean())
    json.dump(out, open(a.out, "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("team", "root", "oof", "work", "eda", "out"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args(); sys.path.insert(0, a.team); main(a)
