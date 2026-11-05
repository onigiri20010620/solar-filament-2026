"""
Train a U-Net at the full 2048px resolution (detector, stage 1).

Differences from train_unet_local.py (which downsamples to 1024px):
  - Images are not downsampled; 768px tiles are cropped from the 2048px image for training.
  - Tile centers: with probability 0.7, pick one filament at random, then one point inside it.
    Sampling per pixel would favor large filaments; sampling per filament trains small ones at the same frequency.
  - Inference feeds the whole 2048px image (the network is fully convolutional, so any size works).
  - After training, the probability maps of the validation images are saved (the post-processing grid search is done by tune_postproc.py).

Usage:
  python train_unet_2048.py --data data --smoke          # sanity check
  python train_unet_2048.py --data data                  # full run
Outputs (--out, default output_2048): best.pt, report.json, val_prob_1024.npz, val_prob_2048.npz
"""
import argparse, glob, json, os, random, time
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="output_2048")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--crop", type=int, default=768)
ap.add_argument("--encoder", default="resnet34")
ap.add_argument("--arch", default="Unet", help="smp model name (Unet, UnetPlusPlus, etc.)")
ap.add_argument("--spe", type=float, default=2.0, help="tiles per epoch = spe x number of training samples")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--val_every", type=int, default=4)
ap.add_argument("--pos_frac", type=float, default=0.7, help="fraction of tiles centered on a filament")
ap.add_argument("--no-amp", action="store_true")
ap.add_argument("--loss", default="bcedice", choices=["bcedice", "lovasz", "focaldice"],
                help="focaldice: Focal loss + Dice (emphasizes hard pixels; weight set by --focal_w) / lovasz: BCE + Lovasz (at 1024px it reduced false positives and gave a model with different behavior, useful for ensembling)")
ap.add_argument("--focal_w", type=float, default=10.0, help="Focal loss weight for focaldice (10 was best at 1024px)")
ap.add_argument("--stack", nargs="*", default=None,
                help="train a second-stage (stacking) model at 2048px: input = original image + first-stage probability maps (models_cv/<name>, 1024px upsampled to 2048px). "
                     "Training/validation images use out-of-fold maps; test uses the 5-fold average. Example: --stack r34,effb3,crop768,lovasz r2048c,r2048e,r2048l")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--ema", type=float, default=0, help="exponential moving average of weights (e.g. 0.999); validation, saving and prediction use the averaged weights")
ap.add_argument("--seeing", action="store_true", help="seeing augmentation: randomly adds blur, gamma change and noise")
ap.add_argument("--pred_fp32", action="store_true", help="compute probability maps in full precision (EfficientNetV2 produces NaN in half precision)")
ap.add_argument("--tta8", action="store_true", help="average predictions over 8 orientations (flip x four 90-degree rotations)")
ap.add_argument("--fold", type=int, default=0, help="which of the 5 folds is used for validation (fold 0 = the original validation images, same split as train_unet_local.py)")
ap.add_argument("--last", action="store_true", help="use the last-epoch weights (keeps out-of-fold probabilities unbiased)")
ap.add_argument("--predict_only", action="store_true", help="skip training; only produce probability maps with best.pt in --out")
ap.add_argument("--test", action="store_true", help="also save test-image probability maps (1024px, float16) to test_prob.npz")
args = ap.parse_args()

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mu
import segmentation_models_pytorch as smp

t0 = time.time()
def log(*a):
    print(f"[{(time.time() - t0) / 60:6.1f} min]", *a, flush=True)

FULL, CROP, SEED = 2048, args.crop, 0
EPOCHS = 2 if args.smoke else args.epochs
OUT = args.out; os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = dev.type == "cuda" and not args.no_amp
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ann_path = glob.glob(os.path.join(args.data, "**", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"), recursive=True)[0]
ROOT = os.path.dirname(os.path.dirname(ann_path))
TRAIN_IMG = f"{ROOT}/train/train_images"

d = json.load(open(ann_path))
anns = defaultdict(list)
for a in d["annotations"]:
    anns[a["image_id"]].append(a)
gt_full = defaultdict(dict)
for im in d["images"]:
    stem = os.path.splitext(im["file_name"])[0]
    gt_full[stem][im["id"]] = [mu.merge(mu.frPyObjects(a["segmentation"], FULL, FULL)) for a in anns[im["id"]]]
stems = sorted(gt_full)
random.Random(SEED).shuffle(stems)           # same split as train_unet_local.py
n_val = len(stems) // 5
lo, hi = args.fold * n_val, (args.fold + 1) * n_val if args.fold < 4 else len(stems)
val_stems, tr_stems = stems[lo:hi], stems[:lo] + stems[hi:]
if args.smoke:
    val_stems, tr_stems = val_stems[:10], tr_stems[:60]

imgs = {s: cv2.imread(f"{TRAIN_IMG}/{s}.jpeg", cv2.IMREAD_GRAYSCALE) for s in tr_stems + val_stems}

_stack_cache = {}


def stack_probs(split, stem):
    """Average the first-stage probability maps per group (uint8, 1024px). split = val (out-of-fold) / test (5-fold average)"""
    chans = []
    for grp in args.stack:
        acc = None
        for e in grp.split(","):
            key = (split, e)
            if key not in _stack_cache:
                zs = [np.load(f"models_cv/{e}/fold{k}/{split}_prob.npz") for k in range(5)]
                _stack_cache[key] = (zs, {st: i for i, z in enumerate(zs) for st in z.files})
            zs, where = _stack_cache[key]
            p = zs[where[stem]][stem].astype(np.float32) if split == "val" else sum(z[stem].astype(np.float32) for z in zs) / 5
            acc = p if acc is None else acc + p
        chans.append(np.round(acc / len(grp.split(",")) * 255).astype(np.uint8))
    return chans + [chans[-1]] * (2 - len(chans))


probs = {s: stack_probs("val", s) for s in tr_stems + val_stems} if args.stack else {}
log(f"loaded {len(imgs)} images")

# Training sample = one per (image, expert): union mask plus pixel coordinates of each filament (up to 300 points)
samples = []
for s in tr_stems:
    for rles in gt_full[s].values():
        m = np.zeros((FULL, FULL), np.uint8)
        pts = []
        for r in rles:
            mi = mu.decode(r); m |= mi
            ys, xs = np.nonzero(mi)
            if len(ys):
                k = np.random.choice(len(ys), min(300, len(ys)), replace=False)
                pts.append(np.stack([ys[k], xs[k]], 1))
        samples.append((s, m, pts))
val_union = {s: (sum(mu.decode(r).astype(np.uint8) for r in next(iter(gt_full[s].values()))) > 0)
             if next(iter(gt_full[s].values())) else np.zeros((FULL, FULL), bool) for s in val_stems}
log(f"train images {len(tr_stems)} (samples {len(samples)}), val images {len(val_stems)}")


def norm(x, ch=None):
    """x: grayscale image. ch: the two probability maps when stacking (same size as x, 0-255)"""
    x = (x.astype(np.float32) / 255.0 - 0.45) / 0.225
    if ch is None:
        return np.repeat(x[None], 3, 0)
    return np.stack([x] + [(c.astype(np.float32) / 255.0 - 0.45) / 0.225 for c in ch])


class TrainDS(torch.utils.data.Dataset):
    def __len__(self):
        return int(len(samples) * args.spe)

    def __getitem__(self, i):
        s, m, pts = samples[random.randrange(len(samples))]
        img = imgs[s]
        h = CROP // 2
        if pts and random.random() < args.pos_frac:
            p = random.choice(pts); cy, cx = p[random.randrange(len(p))]
            cy += random.randint(-CROP // 3, CROP // 3); cx += random.randint(-CROP // 3, CROP // 3)
        else:
            cy, cx = random.randint(h, FULL - h), random.randint(h, FULL - h)
        cy, cx = int(np.clip(cy, h, FULL - h)) // 2 * 2, int(np.clip(cx, h, FULL - h)) // 2 * 2
        im, mk = img[cy - h:cy + h, cx - h:cx + h], m[cy - h:cy + h, cx - h:cx + h]
        ch = None
        if args.stack:   # crop the same region from the 1024px probability maps and upsample to the tile size
            y0, x0, hh = cy // 2, cx // 2, h // 2
            ch = [cv2.resize(c[y0 - hh:y0 + hh, x0 - hh:x0 + hh], (CROP, CROP), interpolation=cv2.INTER_LINEAR) for c in probs[s]]
        fl, fu, k = random.random() < 0.5, random.random() < 0.5, random.randint(0, 3)

        def aug(a):
            if fl: a = a[:, ::-1]
            if fu: a = a[::-1]
            return np.rot90(a, k)
        im, mk = aug(im), aug(mk)
        if ch is not None: ch = [aug(c) for c in ch]
        im = np.clip(im.astype(np.float32) * random.uniform(0.85, 1.15) + random.uniform(-15, 15), 0, 255)
        if args.seeing:   # mimic atmospheric seeing, brightness quirks and noise
            if random.random() < 0.5:
                im = cv2.GaussianBlur(im, (0, 0), random.uniform(0.3, 1.5))
            if random.random() < 0.5:
                im = 255.0 * (im / 255.0) ** random.uniform(0.8, 1.25)
            if random.random() < 0.3:
                im = np.clip(im + np.random.normal(0, random.uniform(1, 4), im.shape).astype(np.float32), 0, 255)
        return torch.from_numpy(norm(im, ch)), torch.from_numpy(np.ascontiguousarray(mk)[None].astype(np.float32))


try:
    model = getattr(smp, args.arch)(args.encoder, encoder_weights="imagenet", in_channels=3, classes=1)
    log(f"{args.arch} encoder: {args.encoder} imagenet")
except Exception as e:
    log("pretrained weights unavailable -> random init", e)
    model = getattr(smp, args.arch)(args.encoder, encoder_weights=None, in_channels=3, classes=1)
model = model.to(dev)


_focal = smp.losses.FocalLoss(mode="binary", gamma=2.0)
_lovasz = smp.losses.LovaszLoss(mode="binary", from_logits=True) if args.loss == "lovasz" else None


def loss_fn(logit, y):
    if args.loss == "focaldice":
        p = torch.sigmoid(logit.float())
        inter = (p * y).sum((2, 3)); den = (p + y).sum((2, 3))
        return args.focal_w * _focal(logit.float(), y) + 1 - ((2 * inter + 1) / (den + 1)).mean()
    bce = F.binary_cross_entropy_with_logits(logit, y)
    if _lovasz is not None:
        return bce + _lovasz(logit.float(), (y > 0.5).float())
    p = torch.sigmoid(logit)
    inter = (p * y).sum((2, 3)); den = (p + y).sum((2, 3))
    return bce + 1 - ((2 * inter + 1) / (den + 1)).mean()


@torch.no_grad()
def predict_prob(img_full, ch=None):
    """2048px image -> 2048px probability map (average of flip TTA). ch: first-stage probability maps when stacking (1024px, uint8)"""
    net = ema.module if args.ema else model
    net.eval()
    if ch is not None:
        ch = [cv2.resize(c, (FULL, FULL), interpolation=cv2.INTER_LINEAR) for c in ch]
    x = torch.from_numpy(norm(img_full, ch))[None].to(dev)
    if args.tta8:
        p = 0
        for f in (False, True):
            xf = x.flip(-1) if f else x
            for k in range(4):
                with torch.autocast(dev.type, enabled=USE_AMP and not args.pred_fp32):
                    q = torch.sigmoid(net(torch.rot90(xf, k, (-2, -1)))).float()
                q = torch.rot90(q, -k, (-2, -1))
                p = p + (q.flip(-1) if f else q)
        return (p[0, 0] / 8).cpu().numpy()
    with torch.autocast(dev.type, enabled=USE_AMP and not args.pred_fp32):
        p = torch.sigmoid(net(x)) + torch.sigmoid(net(x.flip(-1))).flip(-1)
    return (p[0, 0] / 2).float().cpu().numpy()


dl = torch.utils.data.DataLoader(TrainDS(), batch_size=args.batch, shuffle=False, num_workers=0, drop_last=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=EPOCHS * len(dl), pct_start=0.1)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
ema = None
if args.ema:
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)

best_dice, history = -1, []
for ep in range(1, 0 if args.predict_only else EPOCHS + 1):
    model.train(); tl = 0
    for x, y in dl:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(dev.type, enabled=USE_AMP):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if ema is not None: ema.update_parameters(model)
        tl += loss.item()
    if ep % args.val_every == 0 or ep == EPOCHS:
        inter = den = 0
        for s in val_stems:
            p = predict_prob(imgs[s], probs.get(s)) > 0.5; g = val_union[s]
            inter += (p & g).sum(); den += p.sum() + g.sum()
        vd = 2 * inter / max(den, 1)
        history.append(dict(epoch=ep, loss=tl / len(dl), val_dice=float(vd)))
        if vd > best_dice:
            best_dice = vd; torch.save((ema.module if ema is not None else model).state_dict(), f"{OUT}/best.pt")
        log(f"epoch {ep} loss {tl / len(dl):.4f} val_dice {vd:.4f} (best {best_dice:.4f})"
            + (f" gpu_mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB" if dev.type == "cuda" else ""))
    else:
        log(f"epoch {ep} loss {tl / len(dl):.4f}")

# Save the probability maps of the validation images (uint8 = probability x 255)
if args.last and not args.predict_only:
    torch.save((ema.module if ema is not None else model).state_dict(), f"{OUT}/best.pt")
else:
    model.load_state_dict(torch.load(f"{OUT}/best.pt", map_location=dev))
    if ema is not None: ema.module.load_state_dict(model.state_dict())
p2048, p1024 = {}, {}
for s in val_stems:
    p = predict_prob(imgs[s], probs.get(s))
    p2048[s] = (p * 255).round().astype(np.uint8)
    p1024[s] = (cv2.resize(p, (1024, 1024), interpolation=cv2.INTER_AREA) * 255).round().astype(np.uint8)
np.savez_compressed(f"{OUT}/val_prob_2048.npz", **p2048)
np.savez_compressed(f"{OUT}/val_prob_1024.npz", **p1024)
# format used by cv_classifier.py etc. (1024px, float16)
np.savez_compressed(f"{OUT}/val_prob.npz", **{k: (v / 255).astype(np.float16) for k, v in p1024.items()})
if args.test:
    TEST_IMG = f"{ROOT}/test/test_images"
    tp = {}
    for fn in sorted(os.listdir(TEST_IMG)):
        st = os.path.splitext(fn)[0]
        p = predict_prob(cv2.imread(f"{TEST_IMG}/{fn}", cv2.IMREAD_GRAYSCALE), stack_probs("test", st) if args.stack else None)
        tp[os.path.splitext(fn)[0]] = cv2.resize(p, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float16)
    np.savez_compressed(f"{OUT}/test_prob.npz", **tp)
json.dump(dict(args=vars(args), epochs=EPOCHS, best_val_dice=float(best_dice), history=history,
               n_train=len(tr_stems), n_val=len(val_stems)), open(f"{OUT}/report.json", "w"), indent=1)
log("done")
