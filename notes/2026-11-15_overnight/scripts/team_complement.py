"""How much information does our pipeline hold that v18 lacks? Upper bounds of three kinds of mixing (out of fold).

Per annotator filament (707 train images), against v18's final output (refined candidates kept by its scorer, 0.5 px
erosion; the shipped scorer.pkl - in-sample, so v18 is if anything favoured) and ours (tree selection tau 0.24 +
fine.py re-draw t 0.5):
  (3) detection: hit (IoU > 0.5) by both / v18 only / ours only / neither  -> PQ if ours-only hits were added to v18
  (2) shape: on filaments both hit, IoU of v18, ours, the better of the two (oracle), the intersection and the union
      of the two masks -> PQ if v18's hit were replaced by the oracle / by the union / by the intersection
  (1) selection: v18 false positives that ours did not output, and v18 hits ours did not output (agreement)
Usage: python team_complement.py --team <dir> --root <data> --oof <team_oof dir> --fine <dir> --out <json>
"""
import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pycocotools import mask as mu

G = {}


def one(args):
    stem, gt_sets, v18, ours = args
    vr = v18; orl = ours
    res = {"cnt": Counter(), "iou": [], "pq": defaultdict(lambda: np.zeros(4))}
    def add(key, preds, g):
        if not g:
            return
        if not preds:
            res["pq"][key][3] += len(g); return
        I = np.asarray(mu.iou(preds, g, [0] * len(g))).reshape(len(preds), len(g)); hit = I > 0.5
        res["pq"][key] += [I[hit].sum(), hit.sum(), (~hit.any(1)).sum(), (~hit.any(0)).sum()]
    for g in gt_sets.values():
        if not g:
            continue
        add("v18", vr, g); add("ours", orl, g)
        Iv = np.asarray(mu.iou(vr, g, [0] * len(g))).reshape(len(vr), len(g)) if vr else np.zeros((0, len(g)))
        Io = np.asarray(mu.iou(orl, g, [0] * len(g))).reshape(len(orl), len(g)) if orl else np.zeros((0, len(g)))
        # oracle variants built per annotator: v18 output, with additions / replacements
        add_only, rep_best, rep_union, rep_inter = list(vr), list(vr), list(vr), list(vr)
        for j in range(len(g)):
            hv = Iv[:, j].max() > 0.5 if len(vr) else False
            ho = Io[:, j].max() > 0.5 if len(orl) else False
            res["cnt"]["both" if hv and ho else "v18 only" if hv else "ours only" if ho else "neither"] += 1
            if ho and not hv:
                add_only.append(orl[int(Io[:, j].argmax())])
            if hv and ho:
                iv, io = int(Iv[:, j].argmax()), int(Io[:, j].argmax())
                mv, mo = mu.decode(vr[iv]).astype(bool), mu.decode(orl[io]).astype(bool); gm = mu.decode(g[j]).astype(bool)
                f = lambda m: float((m & gm).sum() / max((m | gm).sum(), 1))
                a, b, u, n = f(mv), f(mo), f(mv | mo), f(mv & mo)
                res["iou"].append((a, b, u, n))
                enc = lambda m: mu.encode(np.asfortranarray(m.astype(np.uint8)))
                rep_best[iv] = enc(mo) if b > a else vr[iv]; rep_union[iv] = enc(mv | mo); rep_inter[iv] = enc(mv & mo)
        add("(3) v18 + ours-only hits (oracle)", add_only, g); add("(2) v18, better shape of the two (oracle)", rep_best, g)
        add("(2) v18, union of the two shapes", rep_union, g); add("(2) v18, intersection of the two shapes", rep_inter, g)
    res["pq"] = dict(res["pq"])                     # a defaultdict with a lambda cannot travel back from a worker
    return res


def main():
    ap = argparse.ArgumentParser()
    for k in ("team", "root", "oof", "fine", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.team); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from solarfil.data import find_data, load_annotations
    from solarfil.postprocess import shrink_all
    import fine as FN
    gt = load_annotations(find_data(a.root)["ann"])
    sp = pickle.load(open(os.path.join(a.team, "weights", "scorer.pkl"), "rb"))
    from solarfil.scorer import FEATS
    ip, ir = FEATS.index("p_mean"), FEATS.index("r")
    meta = pickle.load(open(os.path.join(a.fine, "meta.pkl"), "rb"))["meta"]; oof = np.load(os.path.join(a.fine, "oofplain.npy"))
    per = defaultdict(list)
    for i, m in enumerate(meta):
        per[m["name"][:-5]].append(i)
    jobs = []
    for f in sorted(os.listdir(a.oof)):
        s = f[:-4]; T = pickle.load(open(os.path.join(a.oof, f), "rb"))
        X = T["feats"]
        if len(X):
            sc = np.clip(sp["h"].predict(X), 0, 1) * np.clip(sp["m"].predict(X), 0, 1)
            keep = (sc >= 0.2) & (X[:, ip] >= 0.7) & (X[:, ir] <= 0.95)
            v18 = shrink_all([r for r, k in zip(T["pred"], keep) if k], 0.5)
        else:
            v18 = []
        ids = per.get(s, [])
        ours = [mu.encode(np.asfortranarray(m.astype(np.uint8))) for m in FN.paste([meta[i] for i in ids], [oof[i] for i in ids], 0.5, False)] if ids else []
        jobs.append((s, gt[s], v18, ours))
    cnt = Counter(); ious = []; pq = defaultdict(lambda: np.zeros(4))
    with ProcessPoolExecutor(10) as ex:
        for r in ex.map(one, jobs, chunksize=2):
            cnt.update(r["cnt"]); ious += r["iou"]
            for k, v in r["pq"].items():
                pq[k] += v
    I = np.array(ious)
    out = {"PQ": {k: round(float(v[0] / (v[1] + .5 * v[2] + .5 * v[3])), 4) for k, v in pq.items()},
           "annotated filaments: hit by": dict(cnt),
           "on filaments both hit, mean IoU": {"v18": round(float(I[:, 0].mean()), 4), "ours": round(float(I[:, 1].mean()), 4),
                                               "better of the two": round(float(I[:, :2].max(1).mean()), 4),
                                               "union": round(float(I[:, 2].mean()), 4), "intersection": round(float(I[:, 3].mean()), 4),
                                               "corr(v18 IoU, ours IoU)": round(float(np.corrcoef(I[:, 0], I[:, 1])[0, 1]), 3)}}
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
