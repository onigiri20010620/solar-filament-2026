"""Fit the instance scorer (hurdle model) on out-of-fold candidates of the training images and save weights/scorer.pkl.

Inputs
  --data      MAGFiLO_1.0_Kaggle_2026 root (annotations + train_images)
  --oof_probs five 1024 px probability archives (npz, one per detector fold; each holds its own validation images)
  --refined   pickle {stem: [rle]} of refined out-of-fold candidates (output of train_refiner.py apply)
Targets per candidate (from the experts of that image): h = fraction of experts for whom the candidate is a hit (IoU > 0.5),
m = mean IoU over hits. Rows are weighted by the number of experts of the image (the metric counts an image once per expert).

    python scripts/fit_scorer.py --data <root> --oof_probs runs/detector/fold*/val_prob.npz --refined runs/refiner/refined_0.5ppcx.pkl --out weights/scorer.pkl
"""
import argparse
import glob
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solarfil.config import SCORER
from solarfil.data import find_data, load_annotations, read_image
from solarfil.scorer import FEATS, disk_radius, features

G = {}


def init(prob_files, img_dir, refined):
    zs = [np.load(f) for f in prob_files]
    G["where"] = {s: z for z in zs for s in z.files}
    G["img_dir"] = img_dir
    G["refined"] = pickle.load(open(refined, "rb"))


def work(stem):
    p = G["where"][stem][stem].astype(np.float32)
    img = read_image(os.path.join(G["img_dir"], stem + ".jpeg"))
    pred = G["refined"].get(stem, [])
    X = features(pred, [p], img, disk_radius(img)) if pred else np.zeros((0, len(FEATS)), np.float32)
    return stem, pred, X


def targets(pred, gt_sets):
    h = np.zeros(len(pred)); m = np.zeros(len(pred)); ne = max(len(gt_sets), 1)
    for gts in gt_sets.values():
        if not gts or not pred:
            continue
        iou = np.array(mu.iou(pred, gts, [0] * len(gts))).reshape(len(pred), len(gts)).max(1)
        h += iou > 0.5; m += np.where(iou > 0.5, iou, 0)
    return h / ne, m / ne


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--oof_probs", nargs="+", required=True)
    ap.add_argument("--refined", required=True)
    ap.add_argument("--out", default="weights/scorer.pkl")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingRegressor
    files = [f for pat in a.oof_probs for f in sorted(glob.glob(pat))]
    paths = find_data(a.data)
    gt = load_annotations(paths["ann"])
    stems = sorted(s for f in files for s in np.load(f).files)
    with Pool(a.workers, initializer=init, initargs=(files, paths["train_img"], a.refined)) as pool:
        out = pool.map(work, stems, chunksize=4)
    X = np.concatenate([x for _, _, x in out])
    H, M, W = [], [], []
    for s, pred, x in out:
        h, m = targets(pred, gt[s]); H.append(h); M.append(m); W.append(np.full(len(x), max(len(gt[s]), 1), float))
    H, M, W = np.concatenate(H), np.concatenate(M), np.concatenate(W)
    print(f"{len(stems)} images, {len(X)} candidates, mean hit fraction {H.mean():.3f}")

    def make():
        return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400, min_samples_leaf=30,
                                             l2_regularization=1.0, random_state=0)
    mh = make().fit(X, H, sample_weight=W)
    pos = H > 0
    mm = make().fit(X[pos], M[pos] / H[pos], sample_weight=(W * H)[pos])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pickle.dump(dict(h=mh, m=mm, feats=FEATS, rule=dict(SCORER)), open(a.out, "wb"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
