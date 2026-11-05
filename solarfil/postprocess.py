"""Final mask post-processing: sub-pixel erosion and overlap check."""
import cv2
import numpy as np
from pycocotools import mask as mu

from .config import FULL

K3 = np.ones((3, 3), np.uint8)


def shrink_mask(m, px):
    """Erode a binary mask by `px` pixels. px=0.5: erode once at 2x resolution and downsample (>= 50 % coverage)."""
    if px == 0:
        return m
    if px == 1:
        return cv2.erode(m, K3)
    if px == 0.5:
        big = cv2.resize(m, (2 * FULL, 2 * FULL), interpolation=cv2.INTER_NEAREST)
        big = cv2.erode(big, K3)
        return (cv2.resize(big.astype(np.float32), (FULL, FULL), interpolation=cv2.INTER_AREA) >= 0.5).astype(np.uint8)
    raise ValueError(px)


def shrink_all(rles, px):
    out = []
    for r in rles:
        m = shrink_mask(mu.decode(r), px)
        if m.any():
            out.append(mu.encode(np.asfortranarray(m)))
    return out


def overlap_count(rles):
    """Number of pixels covered by more than one mask (must be 0 for a valid submission)."""
    acc = np.zeros((FULL, FULL), np.uint8)
    for r in rles:
        acc += mu.decode(r)
    return int((acc > 1).sum())
