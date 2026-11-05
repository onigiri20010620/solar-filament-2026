"""Compare two submission CSVs image by image (used to check that the release pipeline reproduces the Kaggle submission).

    python scripts/compare_submissions.py new.csv reference.csv
"""
import argparse
import os
import sys

import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solarfil.data import read_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new"); ap.add_argument("ref")
    a = ap.parse_args()
    A, B = read_submission(a.new), read_submission(a.ref)
    stems = sorted(A)
    n_a = n_b = n_same = n_match = 0
    for s in stems:
        ra, rb = A[s], B.get(s, [])
        n_a += len(ra); n_b += len(rb)
        if ra and rb:
            iou = np.array(mu.iou(ra, rb, [0] * len(rb))).reshape(len(ra), len(rb))
            n_same += int((iou.max(1) > 0.999).sum()) if iou.size else 0
            n_match += int((iou.max(1) > 0.9).sum()) if iou.size else 0
    print(f"{len(stems)} images compared: new {n_a} filaments, reference {n_b}; identical masks {n_same}, IoU>0.9 {n_match}")


if __name__ == "__main__":
    main()
