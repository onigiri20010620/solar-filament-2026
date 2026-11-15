"""v18 + refiner threshold 0.45 + 12 refiner-agreement features in the scorer -> test CSV (team repository not modified).

Scorer: HistGB h and m on ALL out-of-fold candidates (team_adapt.py feats, key ('g0.45', 'parts'), 32 features).
Test: as scripts/predict.py (detector 5 folds x 8 TTA -> candidates -> refiners 2 archs x 5 folds x 4 rotations -> paste
at 0.45). The 12 extra features are computed PER FOLD (that fold's UNet++ and ConvNeXt outputs, as out of fold on
train) and averaged over the 5 folds, so that their distribution matches training.
Usage: python team_predict_parts.py --team <dir> --root <data> --work <team_sweep> --out <csv> --clf 0.18 --pmean 0.6
"""
import argparse, os, pickle, sys
import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from team_adapt import paste_thr

T0 = 0.45
EXTRA = ["rq_mean", "rq_q10", "rq_band", "rq_eiou", "rq_soft", "area_ratio", "iou_cand", "refined", "arch_iou", "arch_absdiff", "tta_std", "arch_area_ratio"]


def extra_fold(pred, idx, refd, cands, Rpp, Rcx, Rsd):
    out = []
    for r, i, rf in zip(pred, idx, refd):
        qi = (Rpp[i].astype(np.float32) + Rcx[i]) / 510; S = qi >= T0; ns = max(S.sum(), 1); ins = qi[S].sum()
        ao = float(mu.area(r)); ac = float(mu.area(cands[i])); inter = float(mu.area(mu.merge([r, cands[i]], intersect=True)))
        a1, a2 = Rpp[i] >= T0 * 255, Rcx[i] >= T0 * 255; u = (a1 | a2).sum(); d = np.abs(Rpp[i].astype(np.float32) - Rcx[i]) / 255
        out.append([ins / ns, float(np.quantile(qi[S], 0.1)) if S.any() else 0, ((qi > 0.25) & (qi < 0.75)).sum() / ns,
                    ins / (ns + qi.sum() - ins), qi.sum() / ns, ao / ac, inter / (ao + ac - inter), float(rf),
                    (a1 & a2).sum() / max(u, 1), d[a1 | a2].mean() if u else 0, Rsd[i][S].mean() / 510 if S.any() else 0, a1.sum() / max(a2.sum(), 1)])
    return np.array(out, np.float32).reshape(-1, 12)


def main():
    ap = argparse.ArgumentParser()
    for k in ("team", "root", "work", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--clf", type=float, default=0.18); ap.add_argument("--pmean", type=float, default=0.6)
    a = ap.parse_args()
    sys.path.insert(0, a.team); sys.path.insert(0, os.path.join(a.team, "scripts"))
    import torch
    from sklearn.ensemble import HistGradientBoostingRegressor
    from solarfil import config as C
    from solarfil.candidates import instances
    from solarfil.data import list_stems, read_image, write_submission
    from solarfil.detector import ensemble_prob
    from solarfil.postprocess import overlap_count, shrink_all
    from solarfil.refiner import make_crops
    from solarfil.scorer import FEATS, disk_radius, features
    d = os.path.join(a.work, "feats_adapt"); stems = sorted(f[:-4] for f in os.listdir(d))
    X, H, M, W = [], [], [], []
    for s in stems:
        x, I = pickle.load(open(os.path.join(d, s + ".pkl"), "rb"))["res"][("g0.45", "parts")]
        n = len(x); ne = max(len(I), 1); h = np.zeros(n); m = np.zeros(n)
        for J in I.values():
            if J.size:
                b = J.max(1); h += b > 0.5; m += np.where(b > 0.5, b, 0)
        X.append(x); H.append(h / ne); M.append(m / ne); W.append(np.full(n, ne, float))
    X, H, M, W = map(np.concatenate, (X, H, M, W))
    make = lambda: HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400, min_samples_leaf=30, l2_regularization=1.0, random_state=0)
    pos = H > 0; mh = make().fit(X, H, sample_weight=W); mm = make().fit(X[pos], M[pos] / H[pos], sample_weight=(W * H)[pos])
    rule = dict(clf=a.clf, p_mean=a.pmean, r_max=0.95); sc_path = os.path.splitext(a.out)[0] + "_scorer.pkl"
    pickle.dump(dict(h=mh, m=mm, feats=FEATS + EXTRA, rule=rule), open(sc_path, "wb"))
    print("scorer", sc_path, X.shape, rule, flush=True)
    import predict as P
    det, ref, _ = P.load_models(os.path.join(a.team, "weights"), "cuda", scorer=False)      # ref: arch-major, 5 folds each
    na = len(C.REFINERS); assert len(ref) == 5 * na == 10
    img_dir = os.path.join(a.root, "test", "test_images"); sub = {}; XT = []
    ip, ir = FEATS.index("p_mean"), FEATS.index("r")
    for it, s in enumerate(list_stems(img_dir), 1):
        img = read_image(os.path.join(img_dir, s + ".jpeg")); p = ensemble_prob(det, img, "cuda")
        cands = instances(p, C.CANDIDATES, scale=2); pred = []
        if cands:
            Xc, Wn = make_crops(img, p, cands); n = len(Xc)
            Rm = np.zeros((10, n, 256, 256), np.float32); Rs = np.zeros((5, n, 256, 256), np.float32)
            with torch.no_grad():
                for i in range(0, n, 32):
                    x = torch.from_numpy(Xc[i:i + 32].astype(np.float32) / 255).cuda(); x[:, 0] = (x[:, 0] - 0.45) / 0.25; qs = []
                    for m in ref:
                        q4 = []
                        for k in range(4):
                            with torch.autocast("cuda", dtype=torch.float16):
                                q = torch.sigmoid(m(torch.rot90(x, k, (2, 3))).float())
                            q4.append(torch.rot90(q, -k, (2, 3))[:, 0])
                        qs.append(torch.stack(q4))
                    qs = torch.stack(qs)                                    # [10 models, 4 rot, b, 256, 256]
                    Rm[:, i:i + len(x)] = qs.mean(1).cpu().numpy()
                    for f in range(5):
                        Rs[f, i:i + len(x)] = torch.cat([qs[f], qs[5 + f]]).std(0).cpu().numpy()
            R = np.round(Rm.mean(0) * 255).clip(0, 255).astype(np.uint8)
            pred, idx, refd = paste_thr(cands, Wn, R, np.full(n, T0))
        if pred:
            F = features(pred, [p], img, disk_radius(img)); u8 = lambda v: np.round(v * 255).clip(0, 255).astype(np.uint8)
            E = np.mean([extra_fold(pred, idx, refd, cands, u8(Rm[f]), u8(Rm[5 + f]), np.round(Rs[f] * 510).clip(0, 255).astype(np.uint8)) for f in range(5)], 0)
            Xt = np.hstack([F, E]); XT.append(Xt)
            sc = np.clip(mh.predict(Xt), 0, 1) * np.clip(mm.predict(Xt), 0, 1)
            keep = (sc >= a.clf) & (Xt[:, ip] >= a.pmean) & (Xt[:, ir] <= 0.95)
            pred = [r for r, k in zip(pred, keep) if k]
        final = shrink_all(pred, C.SHRINK_PX); assert overlap_count(final) == 0, s
        sub[s] = final
        if it % 30 == 0:
            print(it, sum(len(v) for v in sub.values()), flush=True)
    write_submission(a.out, sub)
    print("wrote", a.out, sum(len(v) for v in sub.values()), "filaments in", len(sub), "images")
    XT = np.concatenate(XT); names = FEATS + EXTRA
    print("feature means train / test (extra features and p_mean):")
    for j in [ip] + list(range(20, 32)):
        print("  %-16s %.4f  %.4f" % (names[j], X[:, j].mean(), XT[:, j].mean()))


if __name__ == "__main__":
    main()
