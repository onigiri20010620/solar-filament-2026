"""End-to-end inference: test images -> submission CSV, using only the stored weights.

    python scripts/predict.py --images <dir with .jpeg> --weights weights --out submission.csv [--limit N] [--cpu]

Per image: detector (5 folds x 8-way TTA) -> candidates -> refiners (2 archs x 5 folds x 4 rotations)
           -> scorer -> 0.5 px erosion -> RLE rows.  Optionally saves intermediate results with --save_debug.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solarfil import config as C
from solarfil.candidates import instances
from solarfil.data import list_stems, read_image, write_submission
from solarfil.detector import build_detector, ensemble_prob
from solarfil.postprocess import overlap_count, shrink_all
from solarfil.refiner import build_refiner, make_crops, paste, predict_refined
from solarfil.scorer import disk_radius, features


def load_models(weights, device, scorer=True):
    det = [build_detector(f"{weights}/detector/fold{k}.pt", device) for k in range(5)]
    ref = [build_refiner(r["arch"], r["encoder"], f"{weights}/{r['name']}/fold{k}.pt", device) for r in C.REFINERS for k in range(5)]
    if scorer:
        from solarfil.scorer import Scorer   # needs scikit-learn
        scorer = Scorer(f"{weights}/scorer.pkl")
    return det, ref, (scorer or None)


def predict_image(img, det, ref, scorer, device, timing=None):
    t = time.time()
    p = ensemble_prob(det, img, device)                                  # 1024 px probability, mean of folds
    t1 = time.time()
    cands = instances(p, C.CANDIDATES, scale=2)                           # 2048 px candidate masks
    if cands:
        X, W = make_crops(img, p, cands)
        refs = predict_refined(ref, X, device)
        pred = paste(cands, W, refs, **C.REFINE)
    else:
        pred = []
    t2 = time.time()
    if pred and scorer is not None:
        F = features(pred, [p], img, disk_radius(img))
        keep = scorer.keep(F)
        pred = [r for r, k in zip(pred, keep) if k]
    final = shrink_all(pred, C.SHRINK_PX)
    t3 = time.time()
    if timing is not None:
        timing.append((t1 - t, t2 - t1, t3 - t2))
    return final, dict(prob=p, cands=cands, refined=pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder with 2048x2048 H-alpha JPEGs")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (for a quick check)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--save_debug", default=None, help="pickle with probability maps / candidates / refined masks per image")
    ap.add_argument("--no_scorer", action="store_true", help="skip the instance scorer (debugging only; output is not a valid submission)")
    a = ap.parse_args()
    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    t0 = time.time()
    det, ref, scorer = load_models(a.weights, device, scorer=not a.no_scorer)
    print(f"[{time.time() - t0:6.1f} s] loaded {len(det)} detector folds, {len(ref)} refiners, scorer; device={device}", flush=True)
    stems = list_stems(a.images)
    if a.limit:
        stems = stems[:a.limit]
    sub, dbg, timing = {}, {}, []
    for i, s in enumerate(stems, 1):
        img = read_image(os.path.join(a.images, s + ".jpeg"))
        final, d = predict_image(img, det, ref, scorer, device, timing)
        assert overlap_count(final) == 0, s
        sub[s] = final
        if a.save_debug:
            dbg[s] = d
        if i % 10 == 0 or i == len(stems):
            td, tr, ts = np.mean(timing, 0)
            print(f"[{time.time() - t0:6.1f} s] {i}/{len(stems)} images, {sum(len(v) for v in sub.values())} filaments; "
                  f"per image: detector {td:.2f} s, candidates+refiner {tr:.2f} s, scorer+postproc {ts:.2f} s", flush=True)
    write_submission(a.out, sub)
    if a.save_debug:
        pickle.dump(dbg, open(a.save_debug, "wb"))
    print(f"wrote {a.out}: {sum(len(v) for v in sub.values())} filaments in {len(sub)} images", flush=True)


if __name__ == "__main__":
    main()
