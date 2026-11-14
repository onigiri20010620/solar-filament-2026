"""Sweep v18's refiner paste settings (threshold, keep_small) out of fold; only config values change.

refs (GPU): per train image v18's candidates (team_oof.py) re-drawn by the refiners of the image's fold (2 archs x
  4 rotations, as predict_refined) - the raw 256 px probabilities are saved with the windows.
feats (CPU): for each setting (thr x keep_small) solarfil.refiner.paste -> v18's 20 features + IoU of the 0.5 px-eroded
  masks with every annotator's filaments.
select: per setting the hurdle scorer cross-fitted on the team's folds; keep h*m >= clf and p_mean >= 0.7 and
  r <= 0.95 (clf 0.16 / 0.2 / 0.24); PQ; paired bootstrap of the best setting vs the shipped one (thr 0.5, keep 0.3, clf 0.2).
Usage: python team_refine_sweep.py refs  --team <dir> --root <data> --oof <team_oof dir> --work <dir> [--shard 0/3]
       python team_refine_sweep.py feats --team <dir> --root <data> --oof <team_oof dir> --work <dir>
       python team_refine_sweep.py select --team <dir> --work <dir> --out <json>
"""
import argparse
import itertools
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pycocotools import mask as mu

THR = [0.4, 0.45, 0.5, 0.55, 0.6]
KEEP = [0.2, 0.3, 0.4]


def refs(a):
    sys.path.insert(0, a.team); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from solarfil import config as C
    from solarfil.data import read_image
    from solarfil.refiner import build_refiner, make_crops
    from team_tree import predict_rot4
    k_, n_ = map(int, a.shard.split("/"))
    out = os.path.join(a.work, "refs"); os.makedirs(out, exist_ok=True)
    stems = sorted(f[:-4] for f in os.listdir(a.oof) if f.endswith(".pkl"))
    todo = [s for s in stems if not os.path.exists(os.path.join(out, s + ".pkl"))][k_::n_]
    W = os.path.join(a.team, "weights")
    ref = {k: [build_refiner(r["arch"], r["encoder"], f"{W}/{r['name']}/fold{k}.pt", "cuda") for r in C.REFINERS] for k in range(5)}
    for s in todo:
        T = pickle.load(open(os.path.join(a.oof, s + ".pkl"), "rb")); k = T["fold"]
        img = read_image(os.path.join(a.root, "train", "train_images", s + ".jpeg"))
        p = T["prob"].astype(np.float32) / 255; cands = T["cands"]
        if cands:
            X, Wn = make_crops(img, p, cands); R = predict_rot4(ref[k], X)
        else:
            Wn, R = [], np.zeros((0, 256, 256), np.uint8)
        pickle.dump({"fold": k, "cands": cands, "W": Wn, "R": R}, open(os.path.join(out, s + ".pkl"), "wb"))
    print("done", len(todo))


def feats_one(args):
    team, root, s, oof, work, gsets = args
    sys.path.insert(0, team)
    from solarfil.data import read_image
    from solarfil.postprocess import shrink_all
    from solarfil.refiner import paste
    from solarfil.scorer import disk_radius, features
    Q = pickle.load(open(os.path.join(work, "refs", s + ".pkl"), "rb")); T = pickle.load(open(os.path.join(oof, s + ".pkl"), "rb"))
    img = read_image(os.path.join(root, "train", "train_images", s + ".jpeg")); R = disk_radius(img)
    p = T["prob"].astype(np.float32) / 255
    res = {}
    for thr, ks in itertools.product(THR, KEEP):
        pred = paste(Q["cands"], Q["W"], Q["R"], thr, ks, "seed") if Q["cands"] else []
        X = features(pred, [p], img, R) if pred else np.zeros((0, 20), np.float32)
        sh = [(shrink_all([r], 0.5) or [r])[0] for r in pred]
        ious = {e: (np.asarray(mu.iou(sh, g, [0] * len(g))).reshape(len(sh), len(g)) if sh and g else np.zeros((len(sh), len(g)))) for e, g in gsets.items()}
        res[(thr, ks)] = (X, ious)
    pickle.dump({"fold": Q["fold"], "res": res}, open(os.path.join(work, "feats", s + ".pkl"), "wb"))
    return s


def feats(a):
    sys.path.insert(0, a.team)
    from solarfil.data import find_data, load_annotations
    gt = load_annotations(find_data(a.root)["ann"]); os.makedirs(os.path.join(a.work, "feats"), exist_ok=True)
    stems = sorted(f[:-4] for f in os.listdir(os.path.join(a.work, "refs")) if f.endswith(".pkl"))
    jobs = [(a.team, a.root, s, a.oof, a.work, gt[s]) for s in stems if not os.path.exists(os.path.join(a.work, "feats", s + ".pkl"))]
    with ProcessPoolExecutor(12) as ex:
        n = sum(1 for _ in ex.map(feats_one, jobs, chunksize=2))
    print("feats", n)


def select(a):
    sys.path.insert(0, a.team)
    from sklearn.ensemble import HistGradientBoostingRegressor
    from solarfil.scorer import FEATS
    ip, ir = FEATS.index("p_mean"), FEATS.index("r")
    d = os.path.join(a.work, "feats"); stems = sorted(f[:-4] for f in os.listdir(d) if f.endswith(".pkl"))
    Dd = {s: pickle.load(open(os.path.join(d, s + ".pkl"), "rb")) for s in stems}

    def make():
        return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400, min_samples_leaf=30, l2_regularization=1.0, random_state=0)

    def run(key, clfs):
        Xs = [Dd[s]["res"][key][0] for s in stems]; Is = [Dd[s]["res"][key][1] for s in stems]
        X = np.concatenate(Xs); F = np.concatenate([np.full(len(x), Dd[s]["fold"]) for s, x in zip(stems, Xs)])
        H, M, Wt = [], [], []
        for x, I in zip(Xs, Is):
            n = len(x); ne = max(len(I), 1); h = np.zeros(n); m = np.zeros(n)
            for J in I.values():
                if J.size:
                    b = J.max(1); h += b > 0.5; m += np.where(b > 0.5, b, 0)
            H.append(h / ne); M.append(m / ne); Wt.append(np.full(n, ne, float))
        H, M, Wt = map(np.concatenate, (H, M, Wt)); sc = np.zeros(len(X))
        for k in range(5):
            tr, te = F != k, F == k; pos = tr & (H > 0)
            mh = make().fit(X[tr], H[tr], sample_weight=Wt[tr]); mm = make().fit(X[pos], M[pos] / H[pos], sample_weight=(Wt * H)[pos])
            sc[te] = np.clip(mh.predict(X[te]), 0, 1) * np.clip(mm.predict(X[te]), 0, 1)
        out = {}
        for clf in clfs:
            per, o = {}, 0
            for s, x, I in zip(stems, Xs, Is):
                n = len(x); keep = np.nonzero((sc[o:o + n] >= clf) & (x[:, ip] >= 0.7) & (x[:, ir] <= 0.95))[0]; o += n
                tot = np.zeros(4)
                for J in I.values():
                    ng = J.shape[1]
                    if ng == 0:
                        tot[2] += len(keep); continue
                    K = J[keep]; hit = K > 0.5
                    tot += [K[hit].sum(), hit.sum(), (~hit.any(1)).sum() if len(keep) else 0, (~hit.any(0)).sum()]
                per[s] = tot
            out[clf] = per
        return out

    pq = lambda r: (lambda v: v[0] / (v[1] + .5 * v[2] + .5 * v[3]))(sum(r.values()))
    R, res = {}, {}
    for thr, ks in itertools.product(THR, KEEP):
        o = run((thr, ks), (0.16, 0.2, 0.24))
        for clf, per in o.items():
            k = f"thr {thr} keep {ks} clf {clf}"; R[k] = per; res[k] = round(float(pq(per)), 4)
        print(thr, ks, [res[f"thr {thr} keep {ks} clf {c}"] for c in (0.16, 0.2, 0.24)], flush=True)
    base_k = "thr 0.5 keep 0.3 clf 0.2"; best = max(res, key=res.get)
    rng = np.random.default_rng(0); dd = []
    for _ in range(5000):
        b = rng.integers(0, len(stems), len(stems))
        dd.append(pq({i: R[best][stems[j]] for i, j in enumerate(b)}) - pq({i: R[base_k][stems[j]] for i, j in enumerate(b)}))
    dd = np.array(dd)
    out = {"shipped setting (thr 0.5, keep 0.3, clf 0.2)": res[base_k], "best": best, "best PQ": res[best],
           "best - shipped": {"mean": round(float(dd.mean()), 4), "95% interval": [round(float(np.percentile(dd, 2.5)), 4), round(float(np.percentile(dd, 97.5)), 4)],
                              "P(better)": round(float((dd > 0).mean()), 3)},
           "note": "best chosen on the same 707 images (slightly optimistic)", "all": res}
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "all"}, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["refs", "feats", "select"])
    for k in ("team", "root", "oof", "work", "out"):
        ap.add_argument(f"--{k}")
    ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
    {"refs": refs, "feats": feats, "select": select}[a.cmd](a)


if __name__ == "__main__":
    main()
