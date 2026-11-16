"""Figures of the final report (draft -> fair copy story). All from out-of-fold outputs of the final configuration.

cache : cross-fitted scorer -> kept instances; table of (candidate, expert) pairs with identifiers -> <eda>/why_cache.pkl
sheet : contact sheets of pairs crossing IoU 0.5 upwards / downwards (to choose examples)
Usage: python report_figs.py <cmd> --team .. --root .. --oof .. --work .. --eda .. --fig ..
"""
import argparse, os, pickle, sys

import numpy as np
from pycocotools import mask as mu

GREEN, ORANGE, CYAN = "#3ddc84", "#ff9f1c", "#39c5ff"


def load_gt(a):
    from solarfil.data import find_data, load_annotations
    return load_annotations(find_data(a.root)["ann"])


def cache(a):
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
    keep_all = (sc >= 0.18) & (X[:, ip] >= 0.6) & (X[:, ir] <= 0.95)
    keep, pairs, o = {}, [], 0
    for s, x in zip(stems, Xs):
        w = Wy[s]; n = len(x); keep[s] = keep_all[o:o + n]; o += n
        for e, J in w["ious_r"].items():
            J0 = w["ious_d0"][e]
            if J.shape[1] == 0:
                continue
            for j, i in enumerate(w["idx"]):
                g = int(J0[i].argmax())
                if J0[i, g] >= .2:
                    pairs.append(dict(stem=s, e=e, i=i, j=j, g=g, d=float(J0[i, g]), r=float(J[j, g]), ag=float(w["area_gt"][e][g]),
                                      rd=float(w["area_d0"][i] / w["area_gt"][e][g]), rr=float(w["area_r"][j] / w["area_gt"][e][g]),
                                      keep=bool(keep[s][j]), ne=len(w["ious_r"])))
    pickle.dump({"keep": keep, "pairs": pairs}, open(os.path.join(a.eda, "why_cache.pkl"), "wb"))
    print("pairs", len(pairs), "kept instances", int(keep_all.sum()))


def masks_of(a, s):
    """image, prob, candidates, re-drawn masks (+ index of the source candidate) of one training image."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "final_scorer"))
    from solarfil.data import read_image
    from team_adapt import paste_thr
    Q = pickle.load(open(os.path.join(a.work, "refs", s + ".pkl"), "rb")); T = pickle.load(open(os.path.join(a.oof, s + ".pkl"), "rb"))
    img = read_image(os.path.join(a.root, "train", "train_images", s + ".jpeg"))
    pred, idx, refd = paste_thr(Q["cands"], Q["W"], Q["R"], np.full(len(Q["cands"]), 0.45))
    return img, T["prob"], Q, pred, idx


def window(ms, pad=0.35, lo=90):
    m = np.zeros((2048, 2048), bool)
    for r in ms:
        m |= mu.decode(r).astype(bool)
    ys, xs = np.nonzero(m); cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    h = max(max(ys.max() - ys.min(), xs.max() - xs.min()) * (1 + pad) / 2, lo / 2)
    y0, x0 = int(np.clip(cy - h, 0, 2048 - 2 * h)), int(np.clip(cx - h, 0, 2048 - 2 * h)); return y0, x0, int(2 * h)


def show(ax, img, win, outlines=(), fills=(), lw=1.4):
    y0, x0, sd = win; c = img[y0:y0 + sd, x0:x0 + sd]
    ax.imshow(c, cmap="gray", vmin=np.percentile(c, 1), vmax=np.percentile(c, 99.5), interpolation="bilinear")
    for r, col, al in fills:
        m = mu.decode(r)[y0:y0 + sd, x0:x0 + sd].astype(float)
        if m.any():
            ax.contourf(m, levels=[0.5, 1.5], colors=[col], alpha=al)
    for r, col, ls in outlines:
        m = mu.decode(r)[y0:y0 + sd, x0:x0 + sd].astype(float)
        if m.any():
            ax.contour(m, levels=[0.5], colors=[col], linewidths=lw, linestyles=ls)
    ax.set_xticks([]); ax.set_yticks([])


def sheet(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb")); gt = load_gt(a); P = C["pairs"]
    for name, sel in (("up", [p for p in P if p["keep"] and p["d"] <= .5 < p["r"] and 1200 <= p["ag"] <= 12000]),
                      ("down", [p for p in P if p["d"] > .5 >= p["r"] and 800 <= p["ag"] <= 12000])):
        sel.sort(key=lambda p: -abs(p["r"] - p["d"])); seen, pick = set(), []
        for p in sel:
            if (p["stem"], p["i"]) not in seen:
                seen.add((p["stem"], p["i"])); pick.append(p)
        pick = pick[:48]; print(name, len(sel), "->", len(pick))
        fig, axs = plt.subplots(6, 8, figsize=(24, 18)); cacheM = {}
        for k, (p, ax) in enumerate(zip(pick, axs.ravel())):
            if p["stem"] not in cacheM:
                cacheM[p["stem"]] = masks_of(a, p["stem"])
            img, _, Q, pred, idx = cacheM[p["stem"]]; g = gt[p["stem"]][p["e"]][p["g"]]
            show(ax, img, window([g, Q["cands"][p["i"]], pred[p["j"]]]), [(g, GREEN, "solid"), (Q["cands"][p["i"]], ORANGE, "solid"), (pred[p["j"]], CYAN, "solid")])
            ax.set_title(f"#{k} {p['stem'][:8]} c{p['i']} {p['d']:.2f}->{p['r']:.2f} A{p['ag']:.0f} n{p['ne']}", fontsize=9)
        for ax in axs.ravel()[len(pick):]:
            ax.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(a.eda, f"why_sheet_{name}.png"), dpi=60); plt.close(fig)
        pickle.dump(pick, open(os.path.join(a.eda, f"why_pick_{name}.pkl"), "wb"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd")
    for k in ("team", "root", "oof", "work", "eda", "fig"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args(); sys.path.insert(0, a.team)
    globals()[a.cmd](a)
