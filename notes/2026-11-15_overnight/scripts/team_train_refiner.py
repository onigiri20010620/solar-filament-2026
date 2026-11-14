"""
Decoupled detector / contour refiner: a second-stage net (refiner) that redraws the contour of each candidate

Detector = candidates built from the probability maps of the current best single model (2048px + Focal+Dice, models_cv/r2048f), using the tune_v5 recipe.
Refiner = crops a square around one candidate (candidate size x 1.3 + 48 px, resized to 256px) and,
  from 4 input channels [image, probability, this candidate's mask, other candidates' mask], a U-Net draws the ground-truth contour of that one filament.
  Target = the ground-truth filament of each expert that overlaps the candidate most (IoU >= 0.2); each expert gives a separate sample.
  Candidates the detector got wrong (overlapping no ground truth) are not used for training (the refiner learns shape only).
Training-image candidates come from out-of-fold probabilities; the refiner is trained on the same 5 folds and only redraws the candidates of its held-out fold (evaluated on 707 images).
Test uses the average of the 5 fold refiners.

Usage:
  python refiner.py cands              # build candidates and crops -> analysis/refine/
  python refiner.py train --fold k     # train the refiner and redraw the fold-k and test candidates
  python refiner.py apply --thr 0.5    # map the redrawn contours back to 2048px -> analysis/refine/refined_<thr>.pkl
Then run cv_classifier.py --refined analysis/refine/refined_<thr>.pkl for threshold filtering, classifier, scoring and the submission CSV.
"""
import argparse, json, os, pickle, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.environ.get("TEAM_REPO", "D:/kaggle/team_v18"))   # copy of the team's scripts/train_refiner.py; crop size via REF_S

import cv2
import numpy as np
from pycocotools import mask as mu

FULL, S = 2048, int(os.environ.get("REF_S", "256"))
ENC = os.environ.get("REF_ENC", "runs/detector")   # folder with fold<k>/val_prob.npz and fold<k>/test_prob.npz of the detector
OUT = os.environ.get("REF_OUT", "runs/refiner")    # working folder for crops, refiner weights and refined masks
G = {}


def window(rle):
    """Square window (y0, x0, side) around a candidate, in 2048px coordinates; may extend beyond the image"""
    x, y, w, h = mu.toBbox(rle)
    side = int(max(w, h) * 1.3 + 48); side = max(side, 96)
    cy, cx = y + h / 2, x + w / 2
    return int(round(cy - side / 2)), int(round(cx - side / 2)), side


def crop(a, y0, x0, side, interp):
    """Crop the window from a 2048px array (zero outside) and resize to S x S"""
    out = np.zeros((side, side), a.dtype)
    ya, xa, yb, xb = max(y0, 0), max(x0, 0), min(y0 + side, FULL), min(x0 + side, FULL)
    if yb > ya and xb > xa:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = a[ya:yb, xa:xb]
    return cv2.resize(out, (S, S), interpolation=interp)


def cinit(data):
    from solarfil.config import CANDIDATES
    from solarfil.data import find_data, load_annotations
    paths = find_data(data)
    G["gt"] = load_annotations(paths["ann"])
    G["cfg"] = dict(CANDIDATES)
    G["img_dir"], G["test_dir"] = paths["train_img"], paths["test_img"]
    G["val"] = [np.load(f"{ENC}/fold{k}/val_prob.npz") for k in range(5)]
    G["where"] = {s: k for k in range(5) for s in G["val"][k].files}
    G["test"] = [np.load(f"{ENC}/fold{k}/test_prob.npz") for k in range(5)] if os.path.exists(f"{ENC}/fold0/test_prob.npz") else []


def cwork(task):
    """Candidates, crops and ground truth for one image"""
    from solarfil.candidates import instances
    s, split = task
    if split == "train":
        p = G["val"][G["where"][s]][s].astype(np.float32)
        img = cv2.imread(f"{G['img_dir']}/{s}.jpeg", cv2.IMREAD_GRAYSCALE)
    else:
        p = sum(z[s].astype(np.float32) for z in G["test"]) / 5
        img = cv2.imread(f"{G['test_dir']}/{s}.jpeg", cv2.IMREAD_GRAYSCALE)
    pred = instances(p, G["cfg"], 2)
    p2 = (cv2.resize(p, (FULL, FULL), interpolation=cv2.INTER_LINEAR) * 255).clip(0, 255).astype(np.uint8)
    ms = [mu.decode(r) for r in pred]
    allm = np.zeros((FULL, FULL), np.uint8)
    for m in ms:
        allm |= m
    X, W, T, owner = [], [], [], []
    gt = G["gt"].get(s, {}) if split == "train" else {}
    for i, (r, m) in enumerate(zip(pred, ms)):
        y0, x0, side = window(r)
        other = (allm & (1 - m)) * 255
        x = np.stack([crop(img, y0, x0, side, cv2.INTER_AREA), crop(p2, y0, x0, side, cv2.INTER_AREA),
                      crop(m * 255, y0, x0, side, cv2.INTER_AREA), crop(other.astype(np.uint8), y0, x0, side, cv2.INTER_AREA)])
        X.append(x); W.append((y0, x0, side))
        for k, gts in gt.items():   # per expert: the single most-overlapping ground-truth filament
            if not gts:
                continue
            iou = np.array(mu.iou([r], gts, [0] * len(gts))).reshape(-1)
            j = int(iou.argmax())
            if iou[j] >= 0.2:
                T.append(crop(mu.decode(gts[j]) * 255, y0, x0, side, cv2.INTER_AREA)); owner.append(i)
    return s, pred, (np.array(X, np.uint8).reshape(-1, 4, S, S), W, np.array(T, np.uint8).reshape(-1, S, S), owner)


def cmd_cands(a):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    cinit(a.data)
    stems = sorted(G["where"])
    tests = sorted(os.path.splitext(f)[0] for f in os.listdir(G["test_dir"])) if G["test"] else []
    with Pool(16, initializer=cinit, initargs=(a.data,)) as pool:
        tr = pool.map(cwork, [(s, "train") for s in stems], chunksize=4)
        te = pool.map(cwork, [(s, "test") for s in tests], chunksize=4) if tests else []
    for name, res in [("train", tr), ("test", te)] if tests else [("train", tr)]:
        cands = {s: pred for s, pred, _ in res}
        X = np.concatenate([d[0] for _, _, d in res]); W = [w for _, _, d in res for w in d[1]]
        stem_of = np.array([s for s, pred, _ in res for _ in pred])
        off = np.cumsum([0] + [len(d[0]) for _, _, d in res])
        T = np.concatenate([d[2] for _, _, d in res]) if name == "train" else np.zeros((0, S, S), np.uint8)
        own = np.concatenate([np.array(d[3], int) + off[i] for i, (_, _, d) in enumerate(res)]) if name == "train" else np.zeros(0, int)
        np.save(f"{OUT}/{name}_X.npy", X); np.save(f"{OUT}/{name}_T.npy", T); np.save(f"{OUT}/{name}_owner.npy", own)
        pickle.dump(dict(cands=cands, windows=W, stem_of=stem_of, fold_of=G["where"]), open(f"{OUT}/{name}_meta.pkl", "wb"))
        print(f"[{(time.time() - t0) / 60:.1f} min] {name}: {len(X)} candidates, {len(T)} (candidate, expert) targets", flush=True)


def cmd_train(a):
    import torch, torch.nn.functional as F
    import segmentation_models_pytorch as smp
    torch.manual_seed(a.fold)
    dev = "cuda"
    X = np.load(f"{OUT}/train_X.npy", mmap_mode="r"); T = np.load(f"{OUT}/train_T.npy", mmap_mode="r")
    own = np.load(f"{OUT}/train_owner.npy"); meta = pickle.load(open(f"{OUT}/train_meta.pkl", "rb"))
    if a.consensus:   # one sample per candidate: mean of the ground truths of the experts who drew that filament (soft ground truth)
        u, inv = np.unique(own, return_inverse=True)
        acc = np.zeros((len(u), S, S), np.float32); cnt = np.bincount(inv).astype(np.float32)
        for i in range(len(own)):
            acc[inv[i]] += T[i]
        T = (acc / cnt[:, None, None]).round().astype(np.uint8); own = u
    fold = np.array([meta["fold_of"][s] for s in meta["stem_of"]])
    tr = np.nonzero(fold[own] != a.fold)[0]; va = np.nonzero(fold[own] == a.fold)[0]
    model = getattr(smp, a.arch)(a.encoder, encoder_weights="imagenet", in_channels=4, classes=1, **ATTN(a)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = a.epochs * (len(tr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.1)
    focal = smp.losses.FocalLoss(mode="binary", gamma=2.0)
    scaler = torch.amp.GradScaler()
    t0 = time.time(); log = lambda *x: print(f"[{(time.time() - t0) / 60:6.1f} min]", *x, flush=True)
    log(f"fold {a.fold}: train {len(tr)} / val {len(va)} (candidate, expert) pairs")

    def batch(idx, aug):
        x = np.stack([X[own[i]] for i in idx]).astype(np.float32) / 255
        y = np.stack([T[i] for i in idx]).astype(np.float32)[:, None] / 255
        if aug:
            for b in range(len(idx)):
                k = np.random.randint(4); f = np.random.rand() < 0.5
                x[b] = np.rot90(x[b], k, (1, 2)); y[b] = np.rot90(y[b], k, (1, 2))
                if f:
                    x[b] = x[b][:, :, ::-1]; y[b] = y[b][:, :, ::-1]
        x[:, 0] = (x[:, 0] - 0.45) / 0.25
        return torch.from_numpy(np.ascontiguousarray(x)).to(dev), torch.from_numpy(np.ascontiguousarray(y)).to(dev)

    def loss_fn(logit, y):
        p = torch.sigmoid(logit.float())
        inter = (p * y).sum((2, 3)); den = (p + y).sum((2, 3))
        return a.focal_w * focal(logit.float(), y) + 1 - ((2 * inter + 1) / (den + 1)).mean()

    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(tr); tot = 0
        for i in range(0, len(perm) - a.batch + 1, a.batch):
            x, y = batch(perm[i:i + a.batch], True)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(x)
            l = loss_fn(logit, y)
            opt.zero_grad(); scaler.scale(l).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += l.item()
        if ep % 5 == 4 or ep == a.epochs - 1:
            model.eval(); inter = den = 0; inter0 = den0 = 0
            with torch.no_grad():
                for i in range(0, len(va), 64):
                    x, y = batch(va[i:i + 64], False)
                    with torch.autocast("cuda", dtype=torch.float16):
                        p = (torch.sigmoid(model(x).float()) > 0.5).float()
                    c = (x[:, 2:3] > 0.5).float(); yb = (y > 0.5).float()
                    inter += (p * yb).sum().item(); den += (p + yb).sum().item()
                    inter0 += (c * yb).sum().item(); den0 += (c + yb).sum().item()
            log(f"epoch {ep + 1} loss {tot / max(len(perm) // a.batch, 1):.4f} val_dice {2 * inter / den:.4f} (candidate as is {2 * inter0 / den0:.4f})")

    D = f"{OUT}/fold{a.fold}{a.tag}"
    os.makedirs(D, exist_ok=True)
    torch.save(model.state_dict(), f"{D}/refiner.pt")

    def predict(Xa, idx):
        out = np.zeros((len(idx), S, S), np.uint8)
        model.eval()
        with torch.no_grad():
            for i in range(0, len(idx), 64):
                x = torch.from_numpy(np.stack([Xa[j] for j in idx[i:i + 64]]).astype(np.float32) / 255).to(dev)
                x[:, 0] = (x[:, 0] - 0.45) / 0.25
                ps = 0
                for k in range(4):   # average over 4 rotations
                    with torch.autocast("cuda", dtype=torch.float16):
                        ps = ps + torch.rot90(torch.sigmoid(model(torch.rot90(x, k, (2, 3))).float()), -k, (2, 3))
                out[i:i + len(x)] = (ps[:, 0] / 4 * 255).round().clamp(0, 255).byte().cpu().numpy()
        return out
    vi = np.nonzero(fold == a.fold)[0]
    np.save(f"{D}/val_ref.npy", predict(X, vi)); np.save(f"{D}/val_idx.npy", vi)
    if os.path.exists(f"{OUT}/test_X.npy"):
        Xt = np.load(f"{OUT}/test_X.npy", mmap_mode="r")
        np.save(f"{D}/test_ref.npy", predict(Xt, np.arange(len(Xt))))
    log("saved")


def ATTN(a):
    """--attn scse: add scSE attention to the decoder (+0.005 in filseg v4)"""
    return {"decoder_attention_type": "scse"} if a.attn == "scse" else {}


def cmd_predict(a):
    """Redraw the candidates in OUT with a trained refiner (fold<k><tag>/refiner.pt under --src) without retraining"""
    import torch
    import segmentation_models_pytorch as smp
    dev = "cuda"
    X = np.load(f"{OUT}/train_X.npy", mmap_mode="r"); Xt = np.load(f"{OUT}/test_X.npy", mmap_mode="r")
    meta = pickle.load(open(f"{OUT}/train_meta.pkl", "rb"))
    fold = np.array([meta["fold_of"][s] for s in meta["stem_of"]])
    for k in range(5):
        model = getattr(smp, a.arch)(a.encoder, encoder_weights=None, in_channels=4, classes=1, **ATTN(a)).to(dev)
        model.load_state_dict(torch.load(f"{a.src}/fold{k}{a.tag}/refiner.pt", map_location=dev)); model.eval()

        def predict(Xa, idx):
            out = np.zeros((len(idx), S, S), np.uint8)
            with torch.no_grad():
                for i in range(0, len(idx), 64):
                    x = torch.from_numpy(np.stack([Xa[j] for j in idx[i:i + 64]]).astype(np.float32) / 255).to(dev)
                    x[:, 0] = (x[:, 0] - 0.45) / 0.25
                    ps = 0; nv = 0
                    for fl in ([False, True] if a.tta8 else [False]):   # --tta8: 4 rotations x horizontal flip (D4 group, as in filseg v4)
                        xf = x.flip(3) if fl else x
                        for r in range(4):
                            with torch.autocast("cuda", dtype=torch.float16):
                                q = torch.rot90(torch.sigmoid(model(torch.rot90(xf, r, (2, 3))).float()), -r, (2, 3))
                            ps = ps + (q.flip(3) if fl else q); nv += 1
                    out[i:i + len(x)] = (ps[:, 0] / nv * 255).round().clamp(0, 255).byte().cpu().numpy()
            return out
        D = f"{OUT}/fold{k}{a.tag}{a.dst}"; os.makedirs(D, exist_ok=True)
        vi = np.nonzero(fold == k)[0]
        np.save(f"{D}/val_ref.npy", predict(X, vi)); np.save(f"{D}/val_idx.npy", vi)
        np.save(f"{D}/test_ref.npy", predict(Xt, np.arange(len(Xt))))
        print(f"fold{k}{a.tag}{a.dst}: {len(vi)} val, {len(Xt)} test", flush=True)


def paste(cands, W, refs, thr, keep_small, comp="largest"):
    """Map the redrawn probabilities back to 2048px; overlaps go to the higher probability. Candidates that vanished or shrank drastically are kept as originals"""
    best = np.zeros((FULL, FULL), np.float32); owner = np.full((FULL, FULL), -1, np.int32)
    for i, ((y0, x0, side), ref) in enumerate(zip(W, refs)):
        p = cv2.resize(ref.astype(np.float32) / 255, (side, side), interpolation=cv2.INTER_LINEAR)
        ya, xa, yb, xb = max(y0, 0), max(x0, 0), min(y0 + side, FULL), min(x0 + side, FULL)
        pp = p[ya - y0:yb - y0, xa - x0:xb - x0]
        sub_b, sub_o = best[ya:yb, xa:xb], owner[ya:yb, xa:xb]
        win = (pp >= thr) & (pp > sub_b)
        sub_b[win] = pp[win]; sub_o[win] = i
    out = []
    for i, r in enumerate(cands):
        m = (owner == i)
        if m.sum() < keep_small * mu.area(r):
            out.append(r)   # vanished or shrank a lot when redrawn -> original candidate (but without overlapping others)
            continue
        # Keep only the largest connected component (avoid fragments)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if n > 2 and comp == "largest":
            m = lab == 1 + int(st[1:, cv2.CC_STAT_AREA].argmax())
        elif n > 2 and comp == "seed":   # keep every connected component touching the original candidate
            seed = cv2.dilate(mu.decode(r), np.ones((3, 3), np.uint8))
            hit = np.unique(lab[(seed > 0) & (lab > 0)])
            m = np.isin(lab, hit) if len(hit) else lab == 1 + int(st[1:, cv2.CC_STAT_AREA].argmax())
        out.append(mu.encode(np.asfortranarray(m.astype(np.uint8))))
    # Remove the parts of kept-original candidates that overlap redrawn candidates
    taken = np.zeros((FULL, FULL), np.uint8); res = []
    order = sorted(range(len(out)), key=lambda i: -mu.area(out[i]))
    tmp = {}
    for i in order:
        m = mu.decode(out[i]) & (1 - taken)
        taken |= m
        tmp[i] = mu.encode(np.asfortranarray(m)) if m.any() else None
    return [tmp[i] for i in range(len(out)) if tmp[i] is not None]


def awork(task):
    s, cands, W, refs, thr, ks, comp = task
    return s, paste(cands, W, refs, thr, ks, comp)


def cmd_apply(a):
    tags = ["" if t == "base" else t for t in a.tags]
    name_out = f"{OUT}/refined_{a.thr}{a.out_tag}.pkl"
    for name in ["train", "test"]:
        meta = pickle.load(open(f"{OUT}/{name}_meta.pkl", "rb"))
        n = len(meta["windows"])
        if name == "train":
            ref = np.zeros((n, S, S), np.float32)
            for t in tags:
                for k in range(5):
                    ref[np.load(f"{OUT}/fold{k}{t}/val_idx.npy")] += np.load(f"{OUT}/fold{k}{t}/val_ref.npy") / len(tags)
        else:
            ref = sum(np.load(f"{OUT}/fold{k}{t}/test_ref.npy").astype(np.float32) for k in range(5) for t in tags) / (5 * len(tags))
        ref = ref.round().astype(np.uint8)
        idx = {}
        for i, s in enumerate(meta["stem_of"]):
            idx.setdefault(s, []).append(i)
        tasks = [(s, meta["cands"][s], [meta["windows"][i] for i in idx.get(s, [])], ref[idx.get(s, [])], a.thr, a.keep_small, a.comp)
                 for s in meta["cands"]]
        with Pool(16) as pool:
            res = dict(pool.map(awork, tasks, chunksize=4))
        old = {}
        if os.path.exists(name_out):
            old = pickle.load(open(name_out, "rb"))
        old.update(res)
        pickle.dump(old, open(name_out, "wb"))
        print(name, len(res), "images ->", name_out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cands", "train", "apply", "predict"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--focal_w", type=float, default=10.0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--keep_small", type=float, default=0.3, help="use the original candidate if the redrawn area falls below this fraction of it")
    ap.add_argument("--tag", default="", help="train: save to fold<k><tag> (to build several refiners side by side)")
    ap.add_argument("--tags", nargs="+", default=["base"], help="apply: refiners to average (base = the first resnet34)")
    ap.add_argument("--out_tag", default="", help="apply: refined_<thr><out_tag>.pkl")
    ap.add_argument("--attn", default="", help="scse: add scSE attention to the decoder")
    ap.add_argument("--arch", default="Unet", help="train: smp model name (Unet, UnetPlusPlus, etc.)")
    ap.add_argument("--src", default="runs/refiner", help="predict: location of the trained refiners")
    ap.add_argument("--consensus", action="store_true", help="train: use as target the mean of the ground truths of the experts who drew that filament")
    ap.add_argument("--tta8", action="store_true", help="predict: average over 8 variants (4 rotations plus horizontal flip)")
    ap.add_argument("--dst", default="", help="predict: save to fold<k><tag><dst> (keeps the original predictions under a different name)")
    ap.add_argument("--comp", default="largest", choices=["largest", "seed"], help="apply: components to keep (largest / all touching the candidate)")
    a = ap.parse_args()
    {"cands": cmd_cands, "train": cmd_train, "apply": cmd_apply, "predict": cmd_predict}[a.cmd](a)
