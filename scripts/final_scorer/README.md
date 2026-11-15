# How the final scorer (`weights/scorer.pkl`) and settings were obtained (2026-11-15)

Final configuration = v18 with three changes, all evaluated out of fold on the 707 training images with the
scorer cross-fitted on the same 5 folds (paired bootstrap over images, nested selection of the rule):

| change | OOF PQ | public LB |
|---|---|---|
| v18 (refiner threshold 0.5; rule h*m >= 0.2, p_mean >= 0.7) | 0.4500 | 0.39 |
| refiner threshold 0.45 | 0.4551 (+0.0051 [+0.0022, +0.0080]) | 0.40 |
| rule h*m >= 0.22, p_mean >= 0.6 | 0.4566 (nested) | 0.40 |
| + 12 refiner-agreement features in the scorer, rule h*m >= 0.18, p_mean >= 0.6 (final) | 0.458 (+0.0018 [0.0000, +0.0036], up in all 5 folds) | 0.40 |

(0.4500 for v18 is with a cross-fitted scorer; the shipped v18 `scorer.pkl`, fitted on all 707 images, gives 0.4574 in-sample.)

Steps (paths are arguments; `team_tree.py` is only used for `predict_rot4`):

1. `team_oof.py` - v18 detector / candidates out of fold for every training image.
2. `team_refine_sweep.py refs` - the two refiners of the image's fold on its candidates (mean output).
3. `team_ref_parts.py` - the same, kept per architecture, plus the std over (architecture, rotation).
4. `team_adapt.py` - paste at 0.45, the 20 + 12 features and the IoU tables (key `('g0.45', 'parts')`).
5. `team_select_any.py` - cross-fitted scorer, rule grid, nested estimate, bootstrap.
6. `team_predict_parts.py` - fits the scorer on all out-of-fold candidates (-> `weights/scorer.pkl`) and writes the
   test CSV. `scripts/predict.py` in this repository reproduces that CSV (180 images: the same 1,132 filaments,
   1,121 pixel-identical, all with IoU > 0.9; the rest differ by GPU float16 rounding).

Tried on top and not adopted (no gain): per-candidate thresholds, erosion 0 / 1 px, 8-fold refiner TTA, zoomed / wider
refiner windows, a third / fourth refiner architecture (U-Net R34, U-Net ResNeXt50), other candidate thresholds.
