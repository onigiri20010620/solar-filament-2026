"""Stage 2 refiner: re-draw each candidate from an instance-conditioned crop.

Input (4 channels, 256 x 256): image, probability map, the candidate mask, the union of all other candidates.
Output: refined probability for that single filament, pasted back at 2048 px.
"""
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from pycocotools import mask as mu

from .config import FULL, CROP


def window(rle):
    """Square window (y0, x0, side) around a candidate: side = 1.3 * extent + 48 px (min 96), may exceed the image."""
    x, y, w, h = mu.toBbox(rle)
    side = int(max(w, h) * 1.3 + 48); side = max(side, 96)
    cy, cx = y + h / 2, x + w / 2
    return int(round(cy - side / 2)), int(round(cx - side / 2)), side


def crop(a, y0, x0, side):
    """Cut the window out of a 2048 px array (zero outside the image) and resize to CROP x CROP (area interpolation)."""
    out = np.zeros((side, side), a.dtype)
    ya, xa, yb, xb = max(y0, 0), max(x0, 0), min(y0 + side, FULL), min(x0 + side, FULL)
    if yb > ya and xb > xa:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = a[ya:yb, xa:xb]
    return cv2.resize(out, (CROP, CROP), interpolation=cv2.INTER_AREA)


def make_crops(img, prob_small, cands):
    """img: 2048 px uint8; prob_small: 1024 px probability (float); cands: [rle].
    Returns (X uint8 [N,4,CROP,CROP], windows [(y0,x0,side)])."""
    p2 = (cv2.resize(prob_small.astype(np.float32), (FULL, FULL), interpolation=cv2.INTER_LINEAR) * 255).clip(0, 255).astype(np.uint8)
    ms = [mu.decode(r) for r in cands]
    allm = np.zeros((FULL, FULL), np.uint8)
    for m in ms:
        allm |= m
    X, W = [], []
    for r, m in zip(cands, ms):
        y0, x0, side = window(r)
        other = ((allm & (1 - m)) * 255).astype(np.uint8)
        X.append(np.stack([crop(img, y0, x0, side), crop(p2, y0, x0, side), crop(m * 255, y0, x0, side), crop(other, y0, x0, side)]))
        W.append((y0, x0, side))
    return np.array(X, np.uint8).reshape(-1, 4, CROP, CROP), W


def build_refiner(arch, encoder, weights=None, device="cuda"):
    model = getattr(smp, arch)(encoder, encoder_weights=None, in_channels=4, classes=1)
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict_refined(models, X, device="cuda", batch=64):
    """Average of all refiner models with 4-rotation TTA. X: uint8 [N,4,CROP,CROP] -> uint8 [N,CROP,CROP] (prob x 255)."""
    out = np.zeros((len(X), CROP, CROP), np.uint8)
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i:i + batch].astype(np.float32) / 255).to(device)
        x[:, 0] = (x[:, 0] - 0.45) / 0.25
        ps = 0
        for model in models:
            for k in range(4):
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=device.startswith("cuda")):
                    q = torch.sigmoid(model(torch.rot90(x, k, (2, 3))).float())
                ps = ps + torch.rot90(q, -k, (2, 3))
        out[i:i + len(x)] = (ps[:, 0] / (4 * len(models)) * 255).round().clamp(0, 255).byte().cpu().numpy()
    return out


def paste(cands, W, refs, thr=0.5, keep_small=0.3, comp="seed"):
    """Paste refined crops back at 2048 px. Overlaps go to the higher probability; a candidate whose refined area is
    below keep_small x original is kept unchanged; comp='seed' keeps every component touching the original candidate."""
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
            out.append(r)
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if n > 2 and comp == "largest":
            m = lab == 1 + int(st[1:, cv2.CC_STAT_AREA].argmax())
        elif n > 2 and comp == "seed":
            seed = cv2.dilate(mu.decode(r), np.ones((3, 3), np.uint8))
            hit = np.unique(lab[(seed > 0) & (lab > 0)])
            m = np.isin(lab, hit) if len(hit) else lab == 1 + int(st[1:, cv2.CC_STAT_AREA].argmax())
        out.append(mu.encode(np.asfortranarray(m.astype(np.uint8))))
    # candidates kept unchanged must not overlap the refined ones: larger masks take precedence
    taken = np.zeros((FULL, FULL), np.uint8); tmp = {}
    for i in sorted(range(len(out)), key=lambda i: -int(mu.area(out[i]))):
        m = mu.decode(out[i]) & (1 - taken)
        taken |= m
        tmp[i] = mu.encode(np.asfortranarray(m)) if m.any() else None
    return [tmp[i] for i in range(len(out)) if tmp[i] is not None]
