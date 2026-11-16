# Analyses and figures of the technical report

These scripts produced the numbers and figures of `report/report.pdf`. They are provided for transparency:
they read the out-of-fold caches written by steps 1-4 of `scripts/final_scorer/` (about 707 pickles per step,
not shipped with the repository), so they do not run from a fresh clone without re-creating those caches.

All scripts take the same path arguments:

```
--team  repository root (for `solarfil/`)
--root  MAGFiLO_1.0_Kaggle_2026 data directory
--oof   output of final_scorer/team_oof.py            (Stage-1 probability maps and candidates, out of fold)
--work  output of final_scorer/team_refine_sweep.py, team_ref_parts.py, team_adapt.py
--eda   directory for the intermediate tables written here
--fig   directory for the figures (report/figures)
--out   json written by the analysis scripts
```

Order:

1. `team_why.py prep`, then `team_why.py stat` - per image: IoU tables and areas of the Stage-1 candidates and of
   the re-drawn masks against every expert (`<work>/why/`); pairs crossing IoU 0.5 (398 up, 120 down), area ratios
   (0.93 -> 1.00) and the error counts (1,047 filaments without a candidate).
2. `report_figs.py cache` - cross-fitted scorer and final rule -> kept instances per image (`<eda>/why_cache.pkl`).
3. Analyses (each writes one json):
   - `report_agree.py` - PQ of ours and of expert vs. expert on the 296 multi-expert images; hit rate by how many
     other experts drew the same filament (89 / 68 / 35 %); share of our misses that no other expert matched (70 %).
   - `report_agree2.py` - third expert vs. ours on filaments confirmed by a second expert (73 % vs. 84 %); false
     positives that match another expert (57 %); kept and rejected candidates by expert agreement.
   - `report_sel.py` - contribution of selection: keep all 0.446, fixed rule 0.454, scorer 0.458, oracle 0.513.
   - `report_ci.py` - bootstrap intervals over images of PQ / SQ / RQ (Table 2).
4. Figures (`report_figs4.py <command>`; `report_figs2.py` and `report_figs3.py` hold shared helpers and earlier
   versions of the figures):

   | figure | command |
   |---|---|
   | Fig. 1 `concept` | `agree --ex 20141216222634Lh,840,560,560` (image, window y0, x0, side) |
   | Fig. 2 `pipeline` | `pipe2` |
   | Fig. 3 `fulldisk` | `disk2 --ex 20141216222634Lh,840,560,560` |
   | Fig. 4 `eval_dist` | `eval2` |
   | Fig. 5 `agree` | `evid2` |

   `report_figs4.py sheet` writes the contact sheet from which the example of Fig. 1 was chosen by hand.
