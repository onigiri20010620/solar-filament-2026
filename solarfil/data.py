"""Data access: MAGFiLO annotations, image folders, the 5-fold split, and submission CSV I/O."""
import csv
import glob
import os
import random

import cv2
import numpy as np
from pycocotools import mask as mu

from .config import FULL

SEED = 0


def find_data(root):
    """Locate train/test folders and the annotation file under `root` (the extracted Kaggle archive)."""
    ann = glob.glob(os.path.join(root, "**", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"), recursive=True)
    train_img = glob.glob(os.path.join(root, "**", "train_images"), recursive=True)
    test_img = glob.glob(os.path.join(root, "**", "test_images"), recursive=True)
    return dict(ann=ann[0] if ann else None, train_img=train_img[0] if train_img else None, test_img=test_img[0] if test_img else None)


def load_annotations(ann_path):
    """Returns {image_stem: {expert_image_id: [rle, ...]}} with one entry per (image, expert)."""
    import json
    d = json.load(open(ann_path))
    anns = {}
    for a in d["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    gt = {}
    for im in d["images"]:
        stem = os.path.splitext(im["file_name"])[0]
        gt.setdefault(stem, {})[im["id"]] = [mu.merge(mu.frPyObjects(a["segmentation"], FULL, FULL)) for a in anns.get(im["id"], [])]
    return gt


def folds(stems, n_folds=5, seed=SEED):
    """Image-level 5-fold split used for every model: shuffle sorted stems with a fixed seed, slice into 5 parts.
    Returns {stem: fold}."""
    s = sorted(stems)
    random.Random(seed).shuffle(s)
    n = len(s) // n_folds
    out = {}
    for k in range(n_folds):
        lo, hi = k * n, (k + 1) * n if k < n_folds - 1 else len(s)
        for st in s[lo:hi]:
            out[st] = k
    return out


def read_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def list_stems(img_dir):
    return sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith((".jpeg", ".jpg", ".png")))


def rle_to_str(rle):
    return rle["counts"].decode("ascii")


def str_to_rle(s):
    return {"size": [FULL, FULL], "counts": s.encode("ascii")}


def write_submission(path, per_image):
    """per_image: {stem: [rle, ...]} -> CSV with columns filament_id, segmentation_rle."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filament_id", "segmentation_rle"])
        for stem in sorted(per_image):
            for k, r in enumerate(per_image[stem], 1):
                w.writerow([f"{stem}_{k}", rle_to_str(r)])


def read_submission(path):
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stem = row["filament_id"].rsplit("_", 1)[0]
            out.setdefault(stem, []).append(str_to_rle(row["segmentation_rle"]))
    return out
