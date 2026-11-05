"""Score a submission-format CSV against the training annotations, exactly like the organizers' self-evaluation
(every expert annotation set of an image is scored separately against the same prediction set).

    python scripts/evaluate.py --csv predictions.csv --data <MAGFiLO_1.0_Kaggle_2026 root> [--per_expert]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solarfil.data import find_data, load_annotations, read_submission
from solarfil.metrics import pq_from, score_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--per_expert", action="store_true", help="also print PQ per expert id (first 6 characters of the annotation id)")
    a = ap.parse_args()
    gt = load_annotations(find_data(a.data)["ann"])
    pred = read_submission(a.csv)
    stems = sorted(s for s in gt if s in pred or True)   # images without a prediction row count as all-FN
    res = [score_image(gt[s], pred.get(s, [])) for s in stems]
    r = pq_from(res)
    print(f"{len(stems)} images, {sum(len(v) for v in gt.values())} annotation sets: "
          f"PQ {r['pq']:.4f}  SQ {r['sq']:.4f}  RQ {r['rq']:.4f}  TP {r['tp']} FP {r['fp']} FN {r['fn']}")
    if a.per_expert:
        by = {}
        for s in stems:
            for eid, rles in gt[s].items():
                by.setdefault(eid.split("-")[0], []).append(score_image({eid: rles}, pred.get(s, [])))
        for e in sorted(by, key=lambda e: -len(by[e])):
            q = pq_from(by[e])
            print(f"  expert {e}: {len(by[e]):4d} images  PQ {q['pq']:.3f}  SQ {q['sq']:.3f}  RQ {q['rq']:.3f}")


if __name__ == "__main__":
    main()
