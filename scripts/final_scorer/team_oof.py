"""Out-of-fold run of the team's v18 pipeline (D:/kaggle/team_v18, solarfil package) on the 707 train images.

Each image is predicted only by the models of ITS fold of the team's image-level split (solarfil.data.folds, seed 0):
detector fold k (8-way TTA) -> candidates (join2) -> refiners refiner_pp + refiner_cx of fold k (4 rotations) -> paste.
The scorer is NOT applied (weights/scorer.pkl was fit on all out-of-fold candidates); its 20 features are saved so a
cross-fitted scorer can be refit (team_combine.py). Saved per image: 1024 px probability (uint8), refined rles, features.
Usage: python team_oof.py --team D:/kaggle/team_v18 --root <data> --out <dir> [--shard 0/3]
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
    sys.path.insert(0, a.team)
    from solarfil import config as C
    from solarfil.candidates import instances
    from solarfil.data import folds, list_stems, read_image
    from solarfil.detector import build_detector, predict_prob, to_small
    from solarfil.refiner import build_refiner, make_crops, paste, predict_refined
    from solarfil.scorer import disk_radius, features
    k_, n_ = map(int, a.shard.split("/"))
    img_dir = os.path.join(a.root, "train", "train_images")
    stems = list_stems(img_dir); fo = folds(stems)
    todo = [s for s in stems if not os.path.exists(os.path.join(a.out, s + ".pkl"))][k_::n_]
    os.makedirs(a.out, exist_ok=True)
    W = os.path.join(a.team, "weights")
    det = {k: build_detector(f"{W}/detector/fold{k}.pt", "cuda") for k in range(5)}
    ref = {k: [build_refiner(r["arch"], r["encoder"], f"{W}/{r['name']}/fold{k}.pt", "cuda") for r in C.REFINERS] for k in range(5)}
    t0 = time.time()
    for i, s in enumerate(todo, 1):
        k = fo[s]; img = read_image(os.path.join(img_dir, s + ".jpeg"))
        p = to_small(predict_prob(det[k], img, "cuda", True)).astype(np.float32)
        cands = instances(p, C.CANDIDATES, scale=2)
        if cands:
            X, Wn = make_crops(img, p, cands)
            pred = paste(cands, Wn, predict_refined(ref[k], X, "cuda"), **C.REFINE)
        else:
            pred = []
        F = features(pred, [p], img, disk_radius(img)) if pred else np.zeros((0, 20), np.float32)
        pickle.dump({"fold": k, "prob": np.round(p * 255).astype(np.uint8), "cands": cands, "pred": pred, "feats": F},
                    open(os.path.join(a.out, s + ".pkl"), "wb"))
        if i % 20 == 0:
            print(f"{i}/{len(todo)} {(time.time() - t0) / i:.2f} s/image", flush=True)
    print("done", len(todo))


if __name__ == "__main__":
    main()
