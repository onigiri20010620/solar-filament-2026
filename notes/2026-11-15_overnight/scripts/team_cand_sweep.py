"""Re-tune v18's candidate extraction (hysteresis lo / hi / out) for the refiner at threshold 0.45, out of fold.

v18 tuned the candidate settings before the refiner existed; with the refiner re-drawing every candidate the best
cut may differ. For each setting: candidates from the saved out-of-fold probability (team_oof.py) -> refiners of the
image's fold (2 archs x 4 rotations) -> paste at 0.45 -> v18's 20 features + IoU of the 0.5 px-eroded masks.
  crops  (CPU, 12 processes): candidates + refiner crops per image and setting
  refine (GPU, one process):  all crops of a fold through that fold's refiners
  feats  (CPU, 12 processes): paste, features, IoU
  select: per setting the cross-fitted hurdle scorer; rule (clf, p_mean) grid; NESTED choice of (setting, rule) on 4
          folds scored on the 5th, vs the current best (0.45 / clf 0.22 / p_mean 0.6 on the shipped candidates).
Usage: python team_cand_sweep.py crops|refine|feats|select --team <dir> --root <data> --oof <dir> --work <dir> [--out <json>]
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pycocotools import mask as mu

SETTINGS = [(0.15, 0.8, 0.6), (0.15, 0.8, 0.5), (0.15, 0.8, 0.55), (0.15, 0.8, 0.65),
            (0.15, 0.7, 0.6), (0.15, 0.7, 0.5), (0.1, 0.8, 0.6), (0.1, 0.7, 0.5)]
tag = lambda st: "lo%.2f_hi%.2f_out%.2f" % st


def crops_one(args):
    team, root, s, oof, work = args
    sys.path.insert(0, team)
    from solarfil import config as C
    from solarfil.candidates import instances
    from solarfil.data import read_image
    from solarfil.refiner import make_crops
    T = pickle.load(open(os.path.join(oof, s + ".pkl"), "rb")); p = T["prob"].astype(np.float32) / 255
    img = read_image(os.path.join(root, "train", "train_images", s + ".jpeg"))
    for st in SETTINGS:
        f = os.path.join(work, tag(st), "crops", s + ".pkl")
        if os.path.exists(f):
            continue
        cfg = dict(C.CANDIDATES); cfg["lo"], cfg["hi"], cfg["out"] = st
        cands = instances(p, cfg, scale=2)
        X, W = make_crops(img, p, cands) if cands else (np.zeros((0, 4, 256, 256), np.uint8), [])
        pickle.dump({"fold": T["fold"], "cands": cands, "W": W, "X": X}, open(f, "wb"))
    return s


def crops(a):
    for st in SETTINGS:
        os.makedirs(os.path.join(a.work, tag(st), "crops"), exist_ok=True)
    stems = sorted(f[:-4] for f in os.listdir(a.oof) if f.endswith(".pkl"))
    with ProcessPoolExecutor(12) as ex:
        n = sum(1 for _ in ex.map(crops_one, [(a.team, a.root, s, a.oof, a.work) for s in stems], chunksize=2))
    print("crops", n)


def refine(a):
    sys.path.insert(0, a.team); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from solarfil import config as C
    from solarfil.refiner import build_refiner
    from team_tree import predict_rot4
    W = os.path.join(a.team, "weights")
    for k in (range(5) if a.fold < 0 else [a.fold]):      # --fold k: one GPU process per fold, run in parallel
        ref = [build_refiner(r["arch"], r["encoder"], f"{W}/{r['name']}/fold{k}.pt", "cuda") for r in C.REFINERS]
        for st in SETTINGS:
            d = os.path.join(a.work, tag(st), "crops"); o = os.path.join(a.work, tag(st), "refs"); os.makedirs(o, exist_ok=True)
            for f in sorted(os.listdir(d)):
                if os.path.exists(os.path.join(o, f)):
                    continue
                Q = pickle.load(open(os.path.join(d, f), "rb"))
                if Q["fold"] != k:
                    continue
                R = predict_rot4(ref, Q["X"]) if len(Q["X"]) else np.zeros((0, 256, 256), np.uint8)
                pickle.dump({"fold": k, "cands": Q["cands"], "W": Q["W"], "R": R}, open(os.path.join(o, f), "wb"))
        del ref
        print("fold", k, "refined", flush=True)


def feats_one(args):
    team, root, s, oof, work, gsets = args
    sys.path.insert(0, team)
    from solarfil.data import read_image
    from solarfil.postprocess import shrink_all
    from solarfil.refiner import paste
    from solarfil.scorer import disk_radius, features
    T = pickle.load(open(os.path.join(oof, s + ".pkl"), "rb")); p = T["prob"].astype(np.float32) / 255
    img = read_image(os.path.join(root, "train", "train_images", s + ".jpeg")); R = disk_radius(img)
    for st in SETTINGS:
        f = os.path.join(work, tag(st), "feats", s + ".pkl")
        if os.path.exists(f):
            continue
        Q = pickle.load(open(os.path.join(work, tag(st), "refs", s + ".pkl"), "rb"))
        pred = paste(Q["cands"], Q["W"], Q["R"], 0.45, 0.3, "seed") if Q["cands"] else []
        X = features(pred, [p], img, R) if pred else np.zeros((0, 20), np.float32)
        sh = [(shrink_all([r], 0.5) or [r])[0] for r in pred]
        ious = {e: (np.asarray(mu.iou(sh, g, [0] * len(g))).reshape(len(sh), len(g)) if sh and g else np.zeros((len(sh), len(g)))) for e, g in gsets.items()}
        pickle.dump({"fold": Q["fold"], "X": X, "ious": ious}, open(f, "wb"))
    return s


def feats(a):
    sys.path.insert(0, a.team)
    from solarfil.data import find_data, load_annotations
    gt = load_annotations(find_data(a.root)["ann"])
    for st in SETTINGS:
        os.makedirs(os.path.join(a.work, tag(st), "feats"), exist_ok=True)
    stems = sorted(f[:-4] for f in os.listdir(a.oof) if f.endswith(".pkl"))
    with ProcessPoolExecutor(12) as ex:
        n = sum(1 for _ in ex.map(feats_one, [(a.team, a.root, s, a.oof, a.work, gt[s]) for s in stems], chunksize=2))
    print("feats", n)


def select(a):
    sys.path.insert(0, a.team)
    from sklearn.ensemble import HistGradientBoostingRegressor
    from solarfil.scorer import FEATS
    ip, ir = FEATS.index("p_mean"), FEATS.index("r")
    mk = lambda: HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400, min_samples_leaf=30, l2_regularization=1.0, random_state=0)
    RULES = [(c, pm) for c in (0.18, 0.2, 0.22, 0.24, 0.26) for pm in (0.5, 0.6, 0.7)]
    per = {}                                           # (setting, rule) -> {stem: [sumIoU, TP, FP, FN]}
    stems = None
    for st in SETTINGS:
        d = os.path.join(a.work, tag(st), "feats")
        ss = sorted(f[:-4] for f in os.listdir(d)); stems = ss if stems is None else stems
        D = {s: pickle.load(open(os.path.join(d, s + ".pkl"), "rb")) for s in ss}
        Xs = [D[s]["X"] for s in ss]; Is = [D[s]["ious"] for s in ss]
        X = np.concatenate(Xs); F = np.concatenate([np.full(len(x), D[s]["fold"]) for s, x in zip(ss, Xs)])
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
        for clf, pm in RULES:
            out, o = {}, 0
            for s, x, I in zip(ss, Xs, Is):
                n = len(x); keep = np.nonzero((sc[o:o + n] >= clf) & (x[:, ip] >= pm) & (x[:, ir] <= 0.95))[0]; o += n; t = np.zeros(4)
                for J in I.values():
                    ng = J.shape[1]
                    if ng == 0:
                        t[2] += len(keep); continue
                    K = J[keep]; hit = K > .5; t += [K[hit].sum(), hit.sum(), (~hit.any(1)).sum() if len(keep) else 0, (~hit.any(0)).sum()]
                out[s] = t
            per[(tag(st), clf, pm)] = out
        print(tag(st), "best rule PQ", max(round(float((lambda v: v[0] / (v[1] + .5 * v[2] + .5 * v[3]))(sum(per[(tag(st), c, p)].values()))), 4) for c, p in RULES), flush=True)
    fold = {s: pickle.load(open(os.path.join(a.work, tag(SETTINGS[0]), "feats", s + ".pkl"), "rb"))["fold"] for s in stems}
    pq = lambda r, ss: (lambda v: v[0] / (v[1] + .5 * v[2] + .5 * v[3]))(sum(r[s] for s in ss))
    cur = (tag(SETTINGS[0]), 0.22, 0.6)
    res = {"%s clf %.2f pmean %.1f" % k: round(float(pq(v, stems)), 4) for k, v in per.items()}
    tb, tn, chosen = [], [], []
    for f in range(5):
        tr = [s for s in stems if fold[s] != f]; te = [s for s in stems if fold[s] == f]
        kb = max(per, key=lambda k: pq(per[k], tr)); chosen.append(kb)
        tb += [per[cur][s] for s in te]; tn += [per[kb][s] for s in te]
    f2 = lambda L: (lambda v: v[0] / (v[1] + .5 * v[2] + .5 * v[3]))(sum(L))
    best = max(per, key=lambda k: pq(per[k], stems))
    out = {"current (shipped candidates, 0.45, clf 0.22, p_mean 0.6)": res["%s clf %.2f pmean %.1f" % cur],
           "best on all 707 (optimistic)": ["%s clf %.2f pmean %.1f" % best, res["%s clf %.2f pmean %.1f" % best]],
           "nested held-out: current / tuned": [round(f2(tb), 4), round(f2(tn), 4)],
           "chosen per outer fold": ["%s clf %.2f pmean %.1f" % k for k in chosen],
           "best per candidate setting": {tag(st): max(res[k] for k in res if k.startswith(tag(st))) for st in SETTINGS}}
    json.dump({**out, "all": res}, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["crops", "refine", "feats", "select"])
    for k in ("team", "root", "oof", "work", "out"):
        ap.add_argument(f"--{k}")
    ap.add_argument("--fold", type=int, default=-1)
    a = ap.parse_args()
    {"crops": crops, "refine": refine, "feats": feats, "select": select}[a.cmd](a)


if __name__ == "__main__":
    main()
