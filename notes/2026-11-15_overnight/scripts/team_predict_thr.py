"""v18 with a different refiner threshold, for a leaderboard check (the team repository is not modified).

1. scorer: refit exactly as scripts/fit_scorer.py (HistGB h and m on ALL out-of-fold candidates of the 707 train
   images, weights = number of experts) on the candidates pasted at --thr (team_refine_sweep.py feats cache).
2. test: scripts/predict.py's predict_image with solarfil.config.REFINE["thr"] = --thr and that scorer.
Usage: python team_predict_thr.py --team <dir> --root <data> --work <team_sweep dir> --thr 0.45 --out <csv>
"""
import argparse
import os
import pickle
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    for k in ("team", "root", "work", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--thr", type=float, required=True); ap.add_argument("--keep", type=float, default=0.3)
    ap.add_argument("--clf", type=float, default=None); ap.add_argument("--pmean", type=float, default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.team); sys.path.insert(0, os.path.join(a.team, "scripts"))
    import torch
    from sklearn.ensemble import HistGradientBoostingRegressor
    from solarfil import config as C
    from solarfil.data import list_stems, read_image, write_submission
    from solarfil.postprocess import overlap_count
    from solarfil.scorer import FEATS
    if a.clf is not None:
        C.SCORER["clf"] = a.clf
    if a.pmean is not None:
        C.SCORER["p_mean"] = a.pmean
    print("scorer rule", C.SCORER)
    # 1. scorer on all out-of-fold candidates at this threshold
    d = os.path.join(a.work, "feats"); stems = sorted(f[:-4] for f in os.listdir(d))
    X, H, M, W = [], [], [], []
    for s in stems:
        x, I = pickle.load(open(os.path.join(d, s + ".pkl"), "rb"))["res"][(a.thr, a.keep)]
        n = len(x); ne = max(len(I), 1); h = np.zeros(n); m = np.zeros(n)
        for J in I.values():
            if J.size:
                b = J.max(1); h += b > 0.5; m += np.where(b > 0.5, b, 0)
        X.append(x); H.append(h / ne); M.append(m / ne); W.append(np.full(n, ne, float))
    X, H, M, W = map(np.concatenate, (X, H, M, W))
    make = lambda: HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400, min_samples_leaf=30, l2_regularization=1.0, random_state=0)
    pos = H > 0
    sc_path = os.path.splitext(a.out)[0] + "_scorer.pkl"
    pickle.dump(dict(h=make().fit(X, H, sample_weight=W), m=make().fit(X[pos], M[pos] / H[pos], sample_weight=(W * H)[pos]),
                     feats=FEATS, rule=dict(C.SCORER)), open(sc_path, "wb"))
    print("scorer", sc_path, len(X), "candidates")
    # 2. test with the changed threshold
    C.REFINE["thr"] = a.thr; C.REFINE["keep_small"] = a.keep
    import predict as P
    det, ref, _ = P.load_models(os.path.join(a.team, "weights"), "cuda", scorer=False)
    from solarfil.scorer import Scorer
    scorer = Scorer(sc_path)
    img_dir = os.path.join(a.root, "test", "test_images"); sub = {}
    for i, s in enumerate(list_stems(img_dir), 1):
        final, _ = P.predict_image(read_image(os.path.join(img_dir, s + ".jpeg")), det, ref, scorer, "cuda")
        assert overlap_count(final) == 0, s
        sub[s] = final
        if i % 30 == 0:
            print(i, sum(len(v) for v in sub.values()), flush=True)
    write_submission(a.out, sub)
    print("wrote", a.out, sum(len(v) for v in sub.values()), "filaments in", len(sub), "images")


if __name__ == "__main__":
    main()
