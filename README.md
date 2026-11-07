# Solar Filament Segmentation Challenge 2026 — detect-then-refine pipeline

Code, weights and demo notebook for our submission to the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(Kaggle user `husataro`). Everything needed to reproduce the submitted CSV from the test images is in this repository;
no re-training is required. Only the competition training data was used (no external, temporally adjacent or leaked data).

## Method in one paragraph

A full-resolution (2048 px) U-Net with a ResNet-34 encoder, trained with a Focal + Dice loss on 1024 px crops, predicts a
filament probability map (5 folds, 8-way dihedral test-time augmentation). Instance candidates are cut out of the map by
hysteresis thresholding (0.15 / 0.8, mask at 0.6) and end-point gap bridging. Each candidate is then re-drawn by two
instance-conditioned refiners (UNet++ ResNet-34 and ConvNeXt-T U-Net; input = image, probability map, the candidate,
the other candidates; 256 px crop, 4-rotation TTA, averaged) and pasted back. A gradient-boosted hurdle scorer
(P(hit) × E[IoU | hit] from 20 per-instance features, trained on out-of-fold candidates and weighted by the number of
experts per image) keeps candidates with expected PQ contribution ≥ 0.2 and mean probability ≥ 0.7. Masks are eroded
by 0.5 px and written as non-overlapping COCO RLE. Out-of-fold PQ on the 707 training images against every expert:
**0.461** (public leaderboard 0.39–0.40). Details are in the technical report (`report/`).

## Repository layout

```
solarfil/            the pipeline as a small package
  config.py          all fixed thresholds of the submitted configuration
  data.py            annotations, 5-fold split, submission CSV I/O
  detector.py        Stage 1 U-Net, TTA inference
  candidates.py      hysteresis thresholding + gap bridging
  refiner.py         Stage 2 crops, refiner inference, paste-back
  scorer.py          instance features and the hurdle scorer
  postprocess.py     0.5 px erosion, overlap check
  metrics.py         Panoptic Quality (identical to the organizers' self-evaluation code)
scripts/
  predict.py         test images -> submission.csv (inference only)
  evaluate.py        score a submission CSV against the training annotations (per expert, PQ/SQ/RQ)
  train_detector.py  Stage 1 training (one fold)
  train_refiner.py   Stage 2 training (one fold, one architecture)
  fit_scorer.py      instance scorer training on out-of-fold candidates
notebooks/demo.ipynb end-to-end demo: load weights, predict the test set, write and validate the CSV
weights/             detector/fold{0-4}.pt, refiner_pp/fold{0-4}.pt, refiner_cx/fold{0-4}.pt, scorer.pkl
report/              technical report (LaTeX sources and figures)
requirements.txt
```

## Reproducing the submission

The model weights (1.6 GB) are stored with Git LFS, so install [git-lfs](https://git-lfs.com) before cloning
(or run `git lfs pull` after a plain clone). Files in `weights/` must be real binaries, not 130-byte pointer files.

```bash
git lfs install
git clone https://github.com/<owner>/<repo>.git && cd <repo>
pip install -r requirements.txt            # Python 3.12, CUDA GPU with >= 6 GB recommended (CPU works, ~50x slower)
python scripts/predict.py --images <path>/test/test_images --weights weights --out submission.csv
```

`submission.csv` has the two columns required by the competition (`filament_id`, `segmentation_rle`) and no overlapping
masks. Inference takes about 5.5 s per image on an RTX 3070 Ti (8 GB): detector with 5 folds × 8 TTA 2.9 s,
candidates + two refiners 1.8 s, scorer + post-processing 0.7 s; the 180 test images take 16.5 min. Running this
command on the test set reproduced our submitted CSV up to floating-point noise (1088 vs 1089 instances, every
instance matching with IoU > 0.9, 80 % bit-identical). The notebook `notebooks/demo.ipynb` runs the same steps
with intermediate visualisations.

To score a prediction on the training images (out-of-fold predictions are needed for a fair number):

```bash
python scripts/evaluate.py --csv submission_train.csv --data <path to MAGFiLO_1.0_Kaggle_2026>
```

## Training from scratch (optional)

1. `scripts/train_detector.py --fold k` for k = 0..4 (≈95 min per fold on an RTX 3070 Ti) writes `best.pt` and the
   out-of-fold / test probability maps.
2. `scripts/train_refiner.py cands`, then `train --fold k --arch UnetPlusPlus` and
   `train --fold k --encoder tu-convnext_tiny` for k = 0..4 (10–20 min per fold).
3. `scripts/fit_scorer.py` fits the hurdle scorer on the out-of-fold candidates and writes `weights/scorer.pkl`.

All random seeds are fixed; the image-level 5-fold split is `solarfil.data.folds`.

## License

Code: MIT. Model weights: released under the same license; the ImageNet-pretrained encoders come from
`segmentation_models_pytorch` / `timm` (Apache-2.0 / MIT-licensed weights).
