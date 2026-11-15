"""Compare refiner settings from several feats folders (layout of team_refine_sweep.py feats) with one protocol.

Per setting: the hurdle scorer cross-fitted on the team's folds; keep h*m >= clf, p_mean >= pm, r <= 0.95.
Reported: best rule per setting on all 707 images (optimistic), the NESTED estimate (setting + rule chosen on 4 folds,
scored on the 5th) against --current, and a paired bootstrap of the overall best vs --current.
Usage: python team_select_any.py --team <dir> --dirs base=<dir> fine=<dir> ... --current "base|(0.45, 0.3)|0.22|0.6" --out <json>
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True); ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--current", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="substrings: keep only settings whose name contains one of them")
    a = ap.parse_args()
    sys.path.insert(0, a.team)
    from sklearn.ensemble import HistGradientBoostingRegressor
    from solarfil.scorer import FEATS
    ip, ir = FEATS.index("p_mean"), FEATS.index("r")
    mk = lambda: HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400, min_samples_leaf=30, l2_regularization=1.0, random_state=0)
    RULES = [(c, pm) for c in (0.18, 0.2, 0.22, 0.24, 0.26) for pm in (0.5, 0.6, 0.7)]
    per, stems, fold = {}, None, None
    for spec in a.dirs:
        tag, d = spec.split("=", 1)
        ss = sorted(f[:-4] for f in os.listdir(d) if f.endswith(".pkl"))
        D = {s: pickle.load(open(os.path.join(d, s + ".pkl"), "rb")) for s in ss}
        if stems is None:
            stems = ss; fold = {s: D[s]["fold"] for s in ss}
        assert ss == stems, (tag, len(ss))
        keys = sorted(set.intersection(*[set(D[s]["res"]) for s in ss]), key=str)
        for key in keys:
            name = f"{tag}|{key}"
            if a.only and not any(o in name for o in a.only) and name != a.current.rsplit("|", 2)[0]:
                continue
            Xs = [D[s]["res"][key][0] for s in ss]; Is = [D[s]["res"][key][1] for s in ss]
            X = np.concatenate(Xs); F = np.concatenate([np.full(len(x), fold[s]) for s, x in zip(ss, Xs)])
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
                per[(name, clf, pm)] = out
            print(name, "done", flush=True)
    pq = lambda r, ss: (lambda v: v[0] / (v[1] + .5 * v[2] + .5 * v[3]))(sum(r[s] for s in ss))
    cn, cc, cp = a.current.rsplit("|", 2); cur = (cn, float(cc), float(cp))
    names = sorted({k[0] for k in per})
    best_per = {n: max((round(float(pq(per[k], stems)), 4), f"clf {k[1]} pm {k[2]}") for k in per if k[0] == n) for n in names}
    tb, tn, chosen = [], [], []
    for f in range(5):
        tr = [s for s in stems if fold[s] != f]; te = [s for s in stems if fold[s] == f]
        kb = max(per, key=lambda k: pq(per[k], tr)); chosen.append("%s clf %.2f pm %.1f" % kb)
        tb += [per[cur][s] for s in te]; tn += [per[kb][s] for s in te]
    f2 = lambda L: (lambda v: v[0] / (v[1] + .5 * v[2] + .5 * v[3]))(sum(L))
    best = max(per, key=lambda k: pq(per[k], stems))
    rng = np.random.default_rng(0); d = []
    for _ in range(5000):
        b = rng.integers(0, len(stems), len(stems)); bs = [stems[j] for j in b]
        d.append(f2([per[best][s] for s in bs]) - f2([per[cur][s] for s in bs]))
    d = np.array(d)
    out = {"current": [a.current, round(float(pq(per[cur], stems)), 4)], "best per setting (all 707, optimistic)": best_per,
           "nested held-out: current / tuned": [round(f2(tb), 4), round(f2(tn), 4)], "chosen per outer fold": chosen,
           "overall best vs current": {"setting": "%s clf %.2f pm %.1f" % best, "PQ": round(float(pq(per[best], stems)), 4),
                                       "mean diff": round(float(d.mean()), 4), "95% interval": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                                       "P(better)": round(float((d > 0).mean()), 3),
                                       "per fold (current, best)": [[round(float(pq(per[cur], [s for s in stems if fold[s] == f])), 4), round(float(pq(per[best], [s for s in stems if fold[s] == f])), 4)] for f in range(5)]}}
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
