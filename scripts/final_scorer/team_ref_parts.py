"""Per-model refiner outputs for v18's out-of-fold candidates (same crops as team_refine_sweep.py refs).

Per train image: Rpp / Rcx = each refiner's mean over 4 rotations (uint8, prob x 255), Rsd = std over the 8
(model, rotation) predictions (uint8, std x 510). The mean of Rpp and Rcx equals team_refine_sweep.py's R.
Usage: python team_ref_parts.py --team <dir> --root <data> --oof <team_oof> --work <team_sweep> [--shard 0/2]
"""
import argparse, os, pickle, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    for k in ("team", "root", "oof", "work"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
    sys.path.insert(0, a.team)
    import torch
    from solarfil import config as C
    from solarfil.data import read_image
    from solarfil.refiner import build_refiner, make_crops
    k_, n_ = map(int, a.shard.split("/"))
    out = os.path.join(a.work, "parts"); os.makedirs(out, exist_ok=True)
    stems = sorted(f[:-4] for f in os.listdir(a.oof) if f.endswith(".pkl"))
    todo = [s for s in stems if not os.path.exists(os.path.join(out, s + ".pkl"))][k_::n_]
    W = os.path.join(a.team, "weights")
    ref = {k: [build_refiner(r["arch"], r["encoder"], f"{W}/{r['name']}/fold{k}.pt", "cuda") for r in C.REFINERS] for k in range(5)}
    for s in todo:
        T = pickle.load(open(os.path.join(a.oof, s + ".pkl"), "rb")); k = T["fold"]
        img = read_image(os.path.join(a.root, "train", "train_images", s + ".jpeg"))
        p = T["prob"].astype(np.float32) / 255; cands = T["cands"]
        P = [np.zeros((0, 256, 256), np.uint8)] * 3
        if cands:
            X, _ = make_crops(img, p, cands); P = [[], [], []]
            with torch.no_grad():
                for i in range(0, len(X), 48):
                    x = torch.from_numpy(X[i:i + 48].astype(np.float32) / 255).cuda(); x[:, 0] = (x[:, 0] - 0.45) / 0.25; qs = []
                    for m in ref[k]:
                        for r in range(4):
                            with torch.autocast("cuda", dtype=torch.float16):
                                q = torch.sigmoid(m(torch.rot90(x, r, (2, 3))).float())
                            qs.append(torch.rot90(q, -r, (2, 3))[:, 0])
                    qs = torch.stack(qs)                                   # [8, n, 256, 256]
                    for j, v in enumerate((qs[:4].mean(0) * 255, qs[4:].mean(0) * 255, qs.std(0) * 510)):
                        P[j].append(v.round().clamp(0, 255).byte().cpu().numpy())
            P = [np.concatenate(v) for v in P]
        pickle.dump({"fold": k, "Rpp": P[0], "Rcx": P[1], "Rsd": P[2]}, open(os.path.join(out, s + ".pkl"), "wb"))
    print("done", len(todo))


if __name__ == "__main__":
    main()
