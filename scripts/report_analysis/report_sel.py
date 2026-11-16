"""How much of the final PQ comes from selection (instance scorer)? Out of fold, final configuration, cached tables.
Rules compared on the re-drawn masks: keep all; p_mean >= 0.6 only; cross-fitted scorer with threshold t (sweep);
the threshold predicted by the metric (keep iff h*m > PQ/2).
"""
import argparse, json, os, pickle, sys

import numpy as np


def main(a):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from solarfil.scorer import FEATS
    ip, ir = FEATS.index("p_mean"), FEATS.index("r")
    mk = lambda: HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400, min_samples_leaf=30, l2_regularization=1.0, random_state=0)
    stems = sorted(f[:-4] for f in os.listdir(os.path.join(a.work, "why")) if f.endswith(".pkl"))
    Wy = {s: pickle.load(open(os.path.join(a.work, "why", s + ".pkl"), "rb")) for s in stems}
    Fa = {s: pickle.load(open(os.path.join(a.work, "feats_adapt", s + ".pkl"), "rb"))["res"][("g0.45", "parts")] for s in stems}
    Xs = [Fa[s][0] for s in stems]; Is = [Fa[s][1] for s in stems]
    X = np.concatenate(Xs); F = np.concatenate([np.full(len(x), Wy[s]["fold"]) for s, x in zip(stems, Xs)])
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

    def pq(keep_all):
        si = tp = fp = fn = 0.0; o = 0
        for x, I in zip(Xs, Is):
            n = len(x); kp = keep_all[o:o + n]; o += n
            for J in I.values():
                K = J[kp]; hit = K > .5; t = hit.sum(); si += K[hit].sum(); tp += t; fp += len(K) - t; fn += J.shape[1] - t
        d = tp + .5 * fp + .5 * fn
        return dict(PQ=si / d, SQ=si / max(tp, 1), RQ=tp / d, TP=int(tp), FP=int(fp), FN=int(fn), kept=int(keep_all.sum()))

    base = (X[:, ip] >= 0.6) & (X[:, ir] <= 0.95); out = {}
    out["keep all"] = pq(np.ones(len(X), bool)); out["p_mean >= 0.6, r <= 0.95 only"] = pq(base)
    for t in (0.05, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30):
        out[f"scorer >= {t:.2f} + base"] = pq((sc >= t) & base)
    out["scorer >= 0.18 alone"] = pq(sc >= 0.18)
    # oracle: the true expected contribution h*m (in-sample target) at the same threshold form
    out["oracle target >= 0.18"] = pq(M >= 0.18)
    # calibration of the score against the target
    q = np.quantile(sc, np.linspace(0, 1, 11)); cal = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (sc >= lo) & (sc <= hi); cal.append((float(sc[m].mean()), float(M[m].mean())))
    out["calibration (mean score, mean target) per decile"] = cal
    out["corr(score, target)"] = float(np.corrcoef(sc, M)[0, 1])
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("team", "root", "oof", "work", "out"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args(); sys.path.insert(0, a.team); main(a)
