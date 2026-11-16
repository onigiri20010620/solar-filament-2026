"""Report figures, third version: few words; colour = who drew it (green expert, orange detector, blue re-drawn).
Usage: python report_figs3.py <concept|why|gallery|pipeline|evald|all> --team .. --root .. --oof .. --work .. --eda .. --fig ..
"""
import argparse, json, os, pickle, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pycocotools import mask as mu

from report_figs import CYAN, GREEN, ORANGE, load_gt, masks_of, show, window
from report_figs2 import DOWN, TEASER, UP, pair_info, save

BLUE, RED, OKG, GREY = "#0072B2", "#D55E00", "#009E73", "#999999"
CYAN, ORANGE, GREEN = "#56B4E9", "#E69F00", "#009E73"
DORANGE = "#E69F00"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "font.family": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42})


def verdict(ax, iou):
    hit = iou > 0.5
    ax.text(0.5, 0.045, f"IoU {iou:.2f} " + ("> 0.5: hit" if hit else "< 0.5: miss"), transform=ax.transAxes, color="w", fontsize=8.5,
            ha="center", va="bottom", weight="bold", bbox=dict(boxstyle="round,pad=0.3", fc=(OKG if hit else RED), ec="none"))


def arrows(fig, axs, y):
    for k in range(len(axs) - 1):
        x = (axs[k].get_position().x1 + axs[k + 1].get_position().x0) / 2
        fig.text(x, y, "▶", ha="center", va="center", fontsize=10, color="#444")


def concept(a):
    gt = load_gt(a); s, i = TEASER; img, prob, Q, pred, idx, cand, red, ex = pair_info(a, gt, s, i)
    e = ex[0]; win = window([x["g"] for x in ex] + [cand, red], pad=0.45)
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.55))
    show(axs[0], img, win, [(e["g"], GREEN, "solid")], lw=2.0); axs[0].set_title("Expert", color=OKG, weight="bold", fontsize=10)
    show(axs[1], img, win, [(e["g"], GREEN, "solid"), (cand, ORANGE, "solid")], [(cand, ORANGE, 0.28)], lw=2.0)
    axs[1].set_title("Detector: too small", color=DORANGE, weight="bold", fontsize=10); verdict(axs[1], e["d"])
    show(axs[2], img, win, [(e["g"], GREEN, "solid"), (red, CYAN, "solid")], [(red, CYAN, 0.25)], lw=2.0)
    axs[2].set_title("Re-drawn", color=BLUE, weight="bold", fontsize=10); verdict(axs[2], e["r"])
    fig.subplots_adjust(wspace=0.09, left=0, right=1, top=0.9, bottom=0); arrows(fig, axs, 0.45)
    save(fig, a, "concept"); print("teaser", e["d"], e["r"])


def pipeline(a):
    s, i = TEASER; img, prob, Q, pred, idx = masks_of(a, s)
    C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb")); keep = C["keep"][s]
    ys, xs = np.nonzero(mu.decode(Q["cands"][i])); cy, cx = int(ys.mean()), int(xs.mean()); sd = 640
    y0, x0 = int(np.clip(cy - sd // 2, 0, 2048 - sd)), int(np.clip(cx - sd // 2, 0, 2048 - sd)); win = (y0, x0, sd)
    fig, ax = plt.subplots(1, 4, figsize=(7.1, 2.15))
    show(ax[0], img, win); ax[0].set_title("Image", weight="bold")
    ax[1].imshow(np.kron(prob, np.ones((2, 2), np.uint8))[y0:y0 + sd, x0:x0 + sd], cmap="magma", vmin=0, vmax=255); ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].set_title("1  Detect", weight="bold")
    show(ax[2], img, win, [(r, ORANGE, "solid") for r in Q["cands"]], lw=0.9); ax[2].set_title("2  Cut candidates", weight="bold", color=DORANGE)
    show(ax[3], img, win, [(r, CYAN, "solid") for r, k in zip(pred, keep) if k], lw=0.9); ax[3].set_title("3  Re-draw, select", weight="bold", color=BLUE)
    for x, t in zip(ax, ("H-alpha, 2048 px", "U-Net: filament probability", "threshold, split", "UNet++ / ConvNeXt + scorer")):
        x.set_xlabel(t, fontsize=7.5, labelpad=3)
    fig.subplots_adjust(wspace=0.1, left=0, right=1, top=0.9, bottom=0.08); arrows(fig, ax, 0.5)
    save(fig, a, "pipeline")


def why(a):
    C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb")); P = C["pairs"]; H = json.load(open(os.path.join(a.eda, "team_why.json")))["headroom"]
    d = np.array([p["d"] for p in P]); r = np.array([p["r"] for p in P]); rd = np.array([p["rd"] for p in P]); rr = np.array([p["rr"] for p in P])
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.25), gridspec_kw=dict(width_ratios=[1.0, 0.92, 1.0]))
    # (a)
    ax = axs[0]
    rows = [("wrong outline", H["fix shape (fn_shape -> TP at SQ, fp_shape removed)"], CYAN),
            ("not found", H["fix missed (fn_miss -> TP at SQ)"], "#9aa5b1"),
            ("rejected", H["fix discarded (fn_disc -> TP at SQ)"], "#9aa5b1"),
            ("false alarm", H["fix background (fp_bg removed)"], "#9aa5b1")]
    y = np.arange(len(rows))[::-1]; ax.barh(y, [v for _, v, _ in rows], color=[c_ for _, _, c_ in rows], height=0.6)
    for yy, (_, v, _) in zip(y, rows):
        ax.text(v + 0.004, yy, f"+{v:.3f}", va="center", fontsize=8.5, weight=("bold" if yy == y[0] else "normal"))
    ax.set_yticks(y); ax.set_yticklabels([t for t, _, _ in rows], fontsize=8.5); ax.get_yticklabels()[0].set_weight("bold")
    ax.set_xlim(0, 0.15); ax.set_xticks([0, 0.05, 0.1]); ax.set_xlabel("PQ lost")
    ax.set_title("(a) Main loss: drawing", weight="bold", loc="left"); ax.spines[["top", "right"]].set_visible(False)
    # (b)
    ax = axs[1]; lo, hi = 0.2, 0.95
    ax.add_patch(Rectangle((lo, 0.5), 0.5 - lo, hi - 0.5, fc=CYAN, alpha=0.18, ec="none")); ax.add_patch(Rectangle((0.5, lo), hi - 0.5, 0.5 - lo, fc=RED, alpha=0.12, ec="none"))
    ax.scatter(d, r, s=1.2, color="#555", alpha=0.25, lw=0, rasterized=True)
    ax.plot([lo, hi], [lo, hi], color="k", lw=0.6, ls=":"); ax.axhline(0.5, color="k", lw=0.6); ax.axvline(0.5, color="k", lw=0.6)
    up = int(((d <= .5) & (r > .5)).sum()); dn = int(((d > .5) & (r <= .5)).sum())
    ax.text(0.215, 0.935, f"miss → hit\n{up}", color=BLUE, fontsize=9, va="top", weight="bold"); ax.text(0.935, 0.215, f"hit → miss\n{dn}", color=RED, fontsize=9, ha="right", weight="bold")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal"); ax.set_xticks([0.25, 0.5, 0.75]); ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_xlabel("IoU, detector"); ax.set_ylabel("IoU, re-drawn", labelpad=1)
    ax.set_title("(b) Re-drawing helps", weight="bold", loc="left")
    # (c)
    ax = axs[2]; b = np.linspace(-1.2, 1.2, 41); md, mr = np.median(rd), np.median(rr)
    ax.hist(np.log2(rd), bins=b, color=ORANGE, alpha=0.6); ax.hist(np.log2(np.maximum(rr, 1e-3)), bins=b, histtype="step", color=BLUE, lw=1.4)
    top = ax.get_ylim()[1]
    ax.axvline(np.log2(md), color=DORANGE, lw=1.6); ax.axvline(np.log2(mr), color=BLUE, lw=1.6)
    ax.text(np.log2(md) - 0.06, top * 1.03, f"detector\n{md:.2f}", color=DORANGE, ha="right", fontsize=8.5, weight="bold")
    ax.text(np.log2(mr) + 0.06, top * 1.03, f"re-drawn\n{mr:.2f}", color=BLUE, ha="left", fontsize=8.5, weight="bold")
    ax.set_ylim(0, top * 1.38); ax.set_xticks([-1, 0, 1]); ax.set_xticklabels(["0.5", "1", "2"]); ax.set_yticks([])
    ax.set_xlabel("mask area / expert's area")
    ax.set_title("(c) Size bias removed", weight="bold", loc="left"); ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(w_pad=1.6); save(fig, a, "why")


def gallery(a):
    gt = load_gt(a); ex = UP + DOWN; n = len(ex); C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb"))
    fig, axs = plt.subplots(2, n, figsize=(7.1, 1.85))
    for k, (s, i) in enumerate(ex):
        img, prob, Q, pred, idx, cand, red, E = pair_info(a, gt, s, i)
        p = max((p for p in C["pairs"] if p["stem"] == s and p["i"] == i), key=lambda p: abs(p["r"] - p["d"]))
        g = gt[s][p["e"]][p["g"]]; win = window([g, cand, red], pad=0.3)
        for ax, m, col, v in ((axs[0, k], cand, ORANGE, p["d"]), (axs[1, k], red, CYAN, p["r"])):
            show(ax, img, win, [(g, GREEN, "solid"), (m, col, "solid")], lw=1.0)
            ax.text(0.04, 0.96, f"{v:.2f}", transform=ax.transAxes, color="w", fontsize=6.5, va="top", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc=(OKG if v > .5 else RED), ec="none", alpha=0.95))
    axs[0, 0].set_ylabel("detector", fontsize=8.5, color=DORANGE, weight="bold"); axs[1, 0].set_ylabel("re-drawn", fontsize=8.5, color=BLUE, weight="bold")
    fig.subplots_adjust(wspace=0.03, hspace=0.03, left=0.025, right=1, top=0.87, bottom=0)
    x0, x1 = axs[0, 0].get_position().x0, axs[0, len(UP) - 1].get_position().x1; x2, x3 = axs[0, len(UP)].get_position().x0, axs[0, n - 1].get_position().x1
    for xa, xb, col, t in ((x0, x1, OKG, "miss → hit"), (x2, x3, RED, "hit → miss")):
        fig.add_artist(Line2D([xa + 0.004, xb - 0.004], [0.895, 0.895], color=col, lw=2)); fig.text((xa + xb) / 2, 0.915, t, ha="center", va="bottom", fontsize=8.5, color=col, weight="bold")
    fig.legend([Line2D([], [], color=GREEN, lw=1.8)], ["expert"], loc="upper left", bbox_to_anchor=(0.02, 1.03), fontsize=8, frameon=False, handlelength=1.4)
    fig.text(0.17, 0.955, "number: IoU", ha="left", va="center", fontsize=8, color="#333")
    save(fig, a, "gallery")


def evald(a):
    C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb")); gt = load_gt(a)
    mi, gcnt, pcnt, xi, xg = [], [], [], [], []
    for s, keep in C["keep"].items():
        w = pickle.load(open(os.path.join(a.work, "why", s + ".pkl"), "rb"))
        for e, J in w["ious_r"].items():
            K = J[keep]; mi += K[K > 0].tolist(); gcnt += (K > 0).sum(0).tolist(); pcnt += (K > 0).sum(1).tolist()
        es = list(gt[s])
        for ea in es:
            for eb in es:
                if ea != eb:
                    A, B = gt[s][ea], gt[s][eb]
                    J = np.asarray(mu.iou(B, A, [0] * len(A))).reshape(len(B), len(A)) if A and B else np.zeros((len(B), len(A)))
                    xi += J[J > 0].tolist(); xg += (J > 0).sum(0).tolist()
    mi, xi = np.array(mi), np.array(xi); dice = lambda v: 2 * v / (1 + v); GREY = "#6b7178"
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 1.95), gridspec_kw=dict(width_ratios=[1, 1, 1.15])); b = np.linspace(0, 1, 41)
    for ax, f, nm, tag in ((axs[0], lambda v: v, "IoU", "a"), (axs[1], dice, "Dice", "b")):
        ax.hist(f(xi), bins=b, weights=np.full(len(xi), 1 / len(xi)), color="#b8bcc2"); ax.hist(f(mi), bins=b, weights=np.full(len(mi), 1 / len(mi)), color=BLUE, alpha=0.6)
        ax.axvline(f(xi).mean(), color=GREY, ls="--", lw=0.9); ax.axvline(f(mi).mean(), color=BLUE, ls="--", lw=0.9)
        ax.text(0.02, 0.95, f"ours\nmean {f(mi).mean():.2f}", transform=ax.transAxes, color=BLUE, va="top", fontsize=8, weight="bold")
        ax.text(0.02, 0.62, f"2nd expert\nmean {f(xi).mean():.2f}", transform=ax.transAxes, color=GREY, va="top", fontsize=8, weight="bold")
        ax.set_xlabel(f"{nm} with an expert"); ax.set_yticks([]); ax.set_xticks([0, 0.5, 1])
        ax.set_title(f"({tag}) {nm}: ours > 2nd expert", weight="bold", loc="left"); ax.spines[["top", "right", "left"]].set_visible(False)
    frac = lambda v: np.array([(np.minimum(np.array(v), 3) == k).mean() for k in range(4)]) * 100
    ax = axs[2]; x = np.arange(3); wd = 0.26
    f3 = lambda v: np.array([frac(v)[0], frac(v)[1], frac(v)[2] + frac(v)[3]])
    for k, (v, col, lb) in enumerate(((gcnt, BLUE, "expert → ours"), (pcnt, CYAN, "ours → expert"), (xg, "#b8bcc2", "expert → 2nd expert"))):
        ax.bar(x + (k - 1) * wd, f3(v), wd, color=col, label=lb)
        for xx, vv in zip(x + (k - 1) * wd, f3(v)):
            ax.text(xx, vv + 1.5, f"{vv:.0f}" if vv >= 10 else f"{vv:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(["0", "1", "2+"]); ax.set_ylabel("% of filaments"); ax.set_ylim(0, 135); ax.set_yticks([0, 50, 100])
    ax.legend(fontsize=7, frameon=False, loc="upper right", borderaxespad=0.0, handlelength=1.0, labelspacing=0.2); ax.set_xlabel("overlapping partners")
    ax.set_title("(c) Splits and merges are rare", weight="bold", loc="left"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.0); save(fig, a, "eval_dist")
    print("f3 GT ours", f3(gcnt), "pred ours", f3(pcnt), "experts", f3(xg))


def why(a):          # simple version (replaces the one above): counts and medians only
    C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb")); P = C["pairs"]; c = json.load(open(os.path.join(a.eda, "team_why.json")))["headroom"]["counts"]
    d = np.array([p["d"] for p in P]); r = np.array([p["r"] for p in P]); rd = np.array([p["rd"] for p in P]); rr = np.array([p["rr"] for p in P])
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.3), gridspec_kw=dict(width_ratios=[1.35, 0.8, 1.15])); GREYB = "#9aa5b1"
    # (a) what happens to the experts' filaments
    ax = axs[0]; v = [c["tp"], c["fn_shape"], c["fn_disc"], c["fn_miss"]]; tot = sum(v)
    lab = ["hit", "outline\noff", "rejected", "not\nfound"]; col = [OKG, RED, GREYB, GREYB]
    ax.bar(range(4), v, color=col, width=0.68)
    for k, x in enumerate(v):
        ax.text(k, x + tot * 0.012, f"{x:,.0f}", ha="center", fontsize=8.5, weight=("bold" if k == 1 else "normal"), color=(RED if k == 1 else "k"))
    ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=8.5); ax.get_xticklabels()[1].set_color(RED); ax.get_xticklabels()[1].set_weight("bold")
    yb = v[0] * 1.2; ax.plot([-0.34, -0.34, 2.34, 2.34], [yb * 0.97, yb, yb, yb * 0.97], color="k", lw=0.8)
    ax.text(1.0, yb * 1.02, f"found by the detector: {100 * sum(v[:3]) / tot:.0f}%", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, v[0] * 1.42); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(f"(a) The experts' {tot:,.0f} filaments", weight="bold", loc="left")
    # (b) crossings of the hit threshold
    ax = axs[1]; up = int(((d <= .5) & (r > .5)).sum()); dn = int(((d > .5) & (r <= .5)).sum())
    ax.bar([0, 1], [up, dn], color=[BLUE, RED], width=0.68)
    for k, x in enumerate((up, dn)):
        ax.text(k, x + up * 0.02, f"{x}", ha="center", fontsize=10, weight="bold", color=(BLUE, RED)[k])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["miss\n→ hit", "hit\n→ miss"], fontsize=8.5); ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, up * 1.25); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("(b) After re-drawing", weight="bold", loc="left")
    # (c) size relative to the expert
    ax = axs[2]
    for yy, x, colr, nm in ((1, rd, DORANGE, "detector"), (0, rr, BLUE, "re-drawn")):
        q1, m, q3 = np.percentile(x, [25, 50, 75])
        ax.plot([q1, q3], [yy, yy], color=colr, lw=5, alpha=0.35, solid_capstyle="round"); ax.plot([m], [yy], "o", color=colr, ms=8)
        ax.text(q3 * 1.06, yy, f"{nm} {m:.2f}", color=colr, ha="left", va="center", fontsize=9, weight="bold")
    ax.axvline(1, color=OKG, lw=1.4); ax.text(1.04, 1.85, "expert = 1", color=OKG, ha="left", va="bottom", fontsize=9, weight="bold")
    ax.set_xscale("log"); ax.set_xlim(0.55, 1.75); ax.set_xticks([0.6, 0.8, 1, 1.25, 1.6]); ax.set_xticklabels(["0.6", "0.8", "1", "1.25", "1.6"]); ax.minorticks_off()
    ax.set_ylim(-0.55, 2.25); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("← smaller      mask area / expert's      larger →", fontsize=7.5)
    ax.set_title("(c) Mask size", weight="bold", loc="left")
    fig.tight_layout(w_pad=1.8); save(fig, a, "why")
    print("counts", v, "found %", 100 * sum(v[:3]) / tot, "IQR det", np.percentile(rd, [25, 75]), "IQR red", np.percentile(rr, [25, 75]))


def evald(a):        # simple version (replaces the one above)
    C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb")); gt = load_gt(a)
    mi, gcnt, pcnt, xi, xg = [], [], [], [], []
    for s, keep in C["keep"].items():
        w = pickle.load(open(os.path.join(a.work, "why", s + ".pkl"), "rb"))
        for e, J in w["ious_r"].items():
            K = J[keep]; mi += K[K > 0].tolist(); gcnt += (K > 0).sum(0).tolist(); pcnt += (K > 0).sum(1).tolist()
        es = list(gt[s])
        for ea in es:
            for eb in es:
                if ea != eb:
                    A, B = gt[s][ea], gt[s][eb]
                    J = np.asarray(mu.iou(B, A, [0] * len(A))).reshape(len(B), len(A)) if A and B else np.zeros((len(B), len(A)))
                    xi += J[J > 0].tolist(); xg += (J > 0).sum(0).tolist()
    mi, xi = np.array(mi), np.array(xi); dice = lambda v: 2 * v / (1 + v); GREY = "#6b7178"
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 1.8), gridspec_kw=dict(width_ratios=[1, 1, 1.45])); b = np.linspace(0, 1, 41)
    for ax, f, nm, tag in ((axs[0], lambda v: v, "IoU", "a"), (axs[1], dice, "Dice", "b")):
        ax.hist(f(mi), bins=b, weights=np.full(len(mi), 1 / len(mi)), color=BLUE, alpha=0.75, label=f"ours (mean {f(mi).mean():.2f})")
        ax.hist(f(xi), bins=b, weights=np.full(len(xi), 1 / len(xi)), histtype="step", color="#444444", lw=1.1, label=f"two experts ({f(xi).mean():.2f})")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.38); ax.legend(fontsize=7.5, frameon=False, loc="upper left", borderaxespad=0.0, handlelength=1.2, labelspacing=0.25)
        ax.set_xlabel(f"{nm} with the expert's filament"); ax.set_yticks([]); ax.set_xticks([0, 0.5, 1])
        ax.set_title(f"({tag}) {nm}", weight="bold", loc="left"); ax.spines[["top", "right", "left"]].set_visible(False)
    frac = lambda v: np.array([(np.array(v) == 0).mean(), (np.array(v) == 1).mean(), (np.array(v) >= 2).mean()]) * 100
    ax = axs[2]; rows = (("expert's\nfilaments", gcnt, 2), ("our\nfilaments", pcnt, 1), ("two\nexperts", xg, 0)); cols = ["#CCCCCC", BLUE, RED]
    for nm, v, yy in rows:
        f = frac(v); left = 0
        for k in range(3):
            ax.barh(yy, f[k], left=left, color=cols[k], height=0.62, alpha=(0.55 if yy == 0 else 1))
            if f[k] > 8:
                ax.text(left + f[k] / 2, yy, f"{f[k]:.0f}%", ha="center", va="center", fontsize=8, color=("w" if k == 1 else "#333"), weight="bold")
            left += f[k]
        ax.text(101, yy, f"{f[2]:.1f}%", va="center", fontsize=8, color=RED, weight="bold")
    ax.set_yticks([r_[2] for r_ in rows]); ax.set_yticklabels([r_[0] for r_ in rows], fontsize=8); ax.set_xlim(0, 112); ax.set_xticks([]); ax.set_ylim(-0.5, 3.15)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False); ax.tick_params(length=0)
    for x, t, colr in ((0, "none", "#6b7178"), (42, "exactly one", BLUE), (112, "split / merged", RED)):
        ax.text(x, 2.5, t, ha=("left" if x == 0 else "center" if x == 42 else "right"), va="bottom", fontsize=8, color=colr, weight="bold")
    ax.set_title("(c) Overlapping partners", weight="bold", loc="left")
    fig.tight_layout(w_pad=1.2); save(fig, a, "eval_dist")
    print("ours GT", frac(gcnt), "ours pred", frac(pcnt), "experts", frac(xg))


def all(a):
    for f in (concept, pipeline, why, gallery, evald):
        f(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd")
    for k in ("team", "root", "oof", "work", "eda", "fig"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args(); sys.path.insert(0, a.team)
    globals()[a.cmd](a)
