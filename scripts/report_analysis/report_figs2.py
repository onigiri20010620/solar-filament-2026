"""Report figures (see report_figs.py for the cache): concept, why, gallery, pipeline, eval.
Usage: python report_figs2.py <concept|why|gallery|pipeline|evald|all> --team .. --root .. --oof .. --work .. --eda .. --fig ..
"""
import argparse, json, os, pickle, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from pycocotools import mask as mu

from report_figs import CYAN, GREEN, ORANGE, load_gt, masks_of, show, window

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "font.family": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42})
UP = [("20160609133654Bh", 1), ("20180925000030Lh", 1), ("20110119082634Lh", 3), ("20130110104734Ch", 0), ("20141106195854Bh", 10),
      ("20110615133454Bh", 0), ("20210201020330Lh", 2), ("20170310173450Bh", 2)]
DOWN = [("20131016185334Ch", 6), ("20161205133714Th", 3)]
TEASER = ("20141214201834Ch", 5)


def save(fig, a, name):
    from matplotlib.text import Text
    from matplotlib.colors import to_rgb
    for t in fig.findobj(Text):
        t.set_fontweight("normal")
        if sum(to_rgb(t.get_color())) < 2.7:
            t.set_color("#222222")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.fig, f"{name}.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig); print("saved", name)


def pair_info(a, gt, s, i):
    """masks and IoUs of candidate i of image s against its best ground truth of every expert."""
    w = pickle.load(open(os.path.join(a.work, "why", s + ".pkl"), "rb")); img, prob, Q, pred, idx = masks_of(a, s); j = idx.index(i)
    ex = []
    for e in w["ious_d0"]:
        J0 = w["ious_d0"][e]
        if J0.shape[1] == 0:
            continue
        g = int(J0[i].argmax())
        if J0[i, g] >= 0.2:
            ex.append(dict(e=e, g=gt[s][e][g], d=float(J0[i, g]), r=float(w["ious_r"][e][j, g])))
    return img, prob, Q, pred, idx, Q["cands"][i], pred[j], ex


def mark(v):
    return ("hit" if v > 0.5 else "miss")


def concept(a):
    gt = load_gt(a); s, i = TEASER; img, prob, Q, pred, idx, cand, red, ex = pair_info(a, gt, s, i)
    win = window([e["g"] for e in ex] + [cand, red], pad=0.5)
    fig, axs = plt.subplots(1, 4, figsize=(7.1, 1.95)); cols = [GREEN, "#c6f04d", "#18b89b"]
    show(axs[0], img, win); axs[0].set_title("(a) H-$\\alpha$ image")
    show(axs[1], img, win, [(e["g"], c, "solid") for e, c in zip(ex, cols)], lw=1.1); axs[1].set_title(f"(b) outlines of {len(ex)} experts")
    for ax, m, col, key, ttl in ((axs[2], cand, ORANGE, "d", "(c) draft: Stage-1 mask"), (axs[3], red, CYAN, "r", "(d) fair copy: re-drawn mask")):
        show(ax, img, win, [(m, col, "solid")], [(e["g"], GREEN, 0.16) for e in ex[:1]], lw=1.5); ax.set_title(ttl)
        txt = "IoU vs. experts\n" + "\n".join(f"{e[key]:.2f}  {mark(e[key])}" for e in ex)
        ax.text(0.03, 0.03, txt, transform=ax.transAxes, color="w", fontsize=7, va="bottom", ha="left", family="DejaVu Sans Mono",
                bbox=dict(boxstyle="round,pad=0.25", fc="k", ec="none", alpha=0.6))
    fig.subplots_adjust(wspace=0.04, left=0, right=1, top=0.9, bottom=0); save(fig, a, "concept")
    print("teaser IoUs", [(round(e["d"], 2), round(e["r"], 2)) for e in ex])


def why(a):
    C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb")); P = C["pairs"]; H = json.load(open(os.path.join(a.eda, "team_why.json")))["headroom"]
    d = np.array([p["d"] for p in P]); r = np.array([p["r"] for p in P]); rd = np.array([p["rd"] for p in P]); rr = np.array([p["rr"] for p in P])
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.2), gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.0]))
    ax = axs[0]; c = H["counts"]
    rows = [("wrong outline\n(overlap, IoU 0.1-0.5)", H["fix shape (fn_shape -> TP at SQ, fp_shape removed)"], f"{c['fn_shape']:.0f} FN + {c['fp_shape']:.0f} FP"),
            ("not found\n(no candidate)", H["fix missed (fn_miss -> TP at SQ)"], f"{c['fn_miss']:.0f} FN"),
            ("found, rejected\nby the scorer", H["fix discarded (fn_disc -> TP at SQ)"], f"{c['fn_disc']:.0f} FN"),
            ("spurious\n(no overlap)", H["fix background (fp_bg removed)"], f"{c['fp_bg']:.0f} FP")]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [v for _, v, _ in rows], color=[CYAN, "#9aa5b1", "#9aa5b1", "#9aa5b1"], height=0.62)
    for yy, (_, v, n) in zip(y, rows):
        ax.text(v + 0.003, yy, f"+{v:.3f}\n{n}", va="center", fontsize=6.5)
    ax.set_yticks(y); ax.set_yticklabels([t for t, _, _ in rows], fontsize=7); ax.set_xlim(0, 0.19)
    ax.set_xlabel("PQ gained if this error class were fixed"); ax.set_title("(a) where the remaining PQ is")
    ax.spines[["top", "right"]].set_visible(False)
    ax = axs[1]
    ax.hist2d(d, r, bins=np.linspace(0.2, 1, 41), cmap="Greys", norm=matplotlib.colors.LogNorm(vmin=1))
    ax.plot([0.2, 1], [0.2, 1], color="k", lw=0.6, ls=":"); ax.axhline(0.5, color="crimson", lw=0.7); ax.axvline(0.5, color="crimson", lw=0.7)
    up = int(((d <= .5) & (r > .5)).sum()); dn = int(((d > .5) & (r <= .5)).sum())
    ax.text(0.23, 0.92, f"{up} gained", color=CYAN, fontsize=8, va="top", weight="bold",
            bbox=dict(fc="w", ec="none", alpha=0.8, pad=1)); ax.text(0.93, 0.23, f"{dn} lost", color="crimson", fontsize=8, ha="right", weight="bold",
                                                                    bbox=dict(fc="w", ec="none", alpha=0.8, pad=1))
    ax.set_xlim(0.2, 0.95); ax.set_ylim(0.2, 0.95); ax.set_aspect("equal")
    ax.set_xticks([0.25, 0.5, 0.75]); ax.set_yticks([0.25, 0.5, 0.75]); ax.set_xlabel("IoU of the draft"); ax.set_ylabel("IoU re-drawn", labelpad=1); ax.set_title("(b) each candidate, before / after")
    ax = axs[2]; b = np.linspace(-1.5, 1.5, 49)
    ax.hist(np.log2(rd), bins=b, color=ORANGE, alpha=0.55, label=f"draft, median {np.median(rd):.2f}")
    ax.hist(np.log2(np.maximum(rr, 1e-3)), bins=b, histtype="step", color="#0b84c6", lw=1.3, label=f"re-drawn, median {np.median(rr):.2f}")
    ax.axvline(0, color="k", lw=0.6, ls=":"); ax.set_xticks([-1, 0, 1]); ax.set_xticklabels(["0.5", "1", "2"]); ax.set_yticks([])
    ax.set_xlabel("mask area / expert's area"); ax.set_title("(c) the draft is too thin"); ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25); ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(w_pad=1.6); save(fig, a, "why")
    print("n pairs", len(P), "up", up, "down", dn, "mean IoU", d.mean(), r.mean(), "hit", (d > .5).mean(), (r > .5).mean())


def gallery(a):
    gt = load_gt(a); ex = UP + DOWN; n = len(ex)
    fig, axs = plt.subplots(2, n, figsize=(7.1, 1.62))
    for k, (s, i) in enumerate(ex):
        img, prob, Q, pred, idx, cand, red, E = pair_info(a, gt, s, i)
        C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb")) if k == 0 else C
        p = max((p for p in C["pairs"] if p["stem"] == s and p["i"] == i), key=lambda p: abs(p["r"] - p["d"]))
        g = gt[s][p["e"]][p["g"]]; win = window([g, cand, red], pad=0.3)
        for ax, m, col, v in ((axs[0, k], cand, ORANGE, p["d"]), (axs[1, k], red, CYAN, p["r"])):
            show(ax, img, win, [(g, GREEN, "solid"), (m, col, "solid")], lw=0.9)
            ax.text(0.04, 0.96, f"{v:.2f}", transform=ax.transAxes, color="w", fontsize=6.5, va="top", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc=("#1a7f37" if v > .5 else "#b3261e"), ec="none", alpha=0.9))
            for sp in ax.spines.values():
                sp.set_edgecolor("#b3261e" if k >= len(UP) else "k"); sp.set_linewidth(1.2 if k >= len(UP) else 0.4)
    axs[0, 0].set_ylabel("draft", fontsize=7.5, color=ORANGE, weight="bold"); axs[1, 0].set_ylabel("re-drawn", fontsize=7.5, color="#0b84c6", weight="bold")
    fig.subplots_adjust(wspace=0.03, hspace=0.03, left=0.02, right=1, top=1, bottom=0); save(fig, a, "gallery")


def pipeline(a):
    gt = load_gt(a); s, i = TEASER; img, prob, Q, pred, idx = masks_of(a, s)
    C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb")); keep = C["keep"][s]
    ys, xs = np.nonzero(mu.decode(Q["cands"][i])); cy, cx = int(ys.mean()), int(xs.mean()); sd = 640
    y0, x0 = int(np.clip(cy - sd // 2, 0, 2048 - sd)), int(np.clip(cx - sd // 2, 0, 2048 - sd)); win = (y0, x0, sd)
    fig = plt.figure(figsize=(7.1, 2.9)); gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.33], hspace=0.2, wspace=0.03, left=0, right=1, top=0.94, bottom=0)
    ax = [fig.add_subplot(gs[0, k]) for k in range(4)]
    show(ax[0], img, win); ax[0].set_title("(a) H-$\\alpha$ image (640 px)")
    ax[1].imshow(np.kron(prob, np.ones((2, 2), np.uint8))[y0:y0 + sd, x0:x0 + sd], cmap="magma", vmin=0, vmax=255); ax[1].set_xticks([]); ax[1].set_yticks([]); ax[1].set_title("(b) Stage 1: probability map")
    show(ax[2], img, win, [(r, ORANGE, "solid") for r in Q["cands"]], lw=0.8); ax[2].set_title("(c) drafts (candidates)")
    show(ax[3], img, win, [(r, CYAN, "solid") for r, k in zip(pred, keep) if k] + [(r, "#ff5c5c", "dotted") for r, k in zip(pred, keep) if not k], lw=0.8)
    ax[3].set_title("(d) re-drawn, scored, kept")
    from matplotlib.lines import Line2D
    lk = dict(loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=6.3, frameon=False, handlelength=1.4, borderaxespad=0.15, ncol=2, columnspacing=0.8, handletextpad=0.4)
    ax[2].legend([Line2D([], [], color=ORANGE, lw=1.4)], ["outline of a candidate (draft)"], **lk)
    nrej = sum(1 for r, k in zip(pred, keep) if not k and mu.decode(r)[y0:y0 + sd, x0:x0 + sd].any()); print("rejected in view", nrej)
    ax[3].legend([Line2D([], [], color=CYAN, lw=1.4)] + ([Line2D([], [], color="#ff5c5c", lw=1.4, ls=":")] if nrej else []), ["outline of a kept re-drawn mask"] + (["rejected by scorer"] if nrej else []), **lk)
    ax[1].text(0.03, 0.03, "bright = filament likely\ndark = background", transform=ax[1].transAxes, color="w", fontsize=6, va="bottom",
               bbox=dict(boxstyle="round,pad=0.25", fc="k", ec="none", alpha=0.65))
    ax[0].text(0.03, 0.03, "filaments = dark bands", transform=ax[0].transAxes, color="w", fontsize=6, va="bottom",
               bbox=dict(boxstyle="round,pad=0.25", fc="k", ec="none", alpha=0.65))
    bx = fig.add_subplot(gs[1, :]); bx.set_xlim(0, 100); bx.set_ylim(0, 10); bx.axis("off")
    boxes = [("Stage 1: detector", "U-Net ResNet-34 at 2048 px\nFocal+Dice, 5 folds, 8-way TTA"),
             ("Candidates (drafts)", "hysteresis 0.15 / 0.8, mask p$\\geq$0.6\ngap bridging $\\leq$30 px, $\\leq$30$^\\circ$"),
             ("Stage 2: re-drawing", "256 px crop per draft, 4 channels\nUNet++ + ConvNeXt-T, thr. 0.45"),
             ("Instance scorer", "hurdle GBDT $\\hat h\\cdot\\hat m$, 20+12 features\nkeep if $\\geq$0.18 and $\\bar p\\geq$0.6"),
             ("Output", "erode 0.5 px, no overlaps\nCOCO RLE per instance")]
    wd = 18.2
    for k, (t, b) in enumerate(boxes):
        x = 0.6 + k * 20
        bx.add_patch(FancyBboxPatch((x, 0.8), wd, 8.2, boxstyle="round,pad=0.2,rounding_size=0.8", fc=("#e8f6fd" if k == 2 else "#f4f5f7"),
                                    ec=("#0b84c6" if k == 2 else "#5b6b7f"), lw=(1.4 if k == 2 else 0.8)))
        bx.text(x + wd / 2, 7.0, t, ha="center", va="center", fontsize=7, weight="bold"); bx.text(x + wd / 2, 3.4, b, ha="center", va="center", fontsize=5.2)
        if k < 4:
            bx.annotate("", xy=(x + 20 - 0.1, 5), xytext=(x + wd + 0.3, 5), arrowprops=dict(arrowstyle="-|>", color="#5b6b7f", lw=0.9))
    save(fig, a, "pipeline")


def evald(a):
    C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb")); gt = load_gt(a)
    mi, gcnt, pcnt, xi, xg, xp = [], [], [], [], [], []
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
                    xi += J[J > 0].tolist(); xg += (J > 0).sum(0).tolist(); xp += (J > 0).sum(1).tolist()
    mi, xi = np.array(mi), np.array(xi); dice = lambda v: 2 * v / (1 + v)
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 1.85), gridspec_kw=dict(width_ratios=[1, 1, 1.25])); b = np.linspace(0, 1, 51)
    for ax, f, nm in ((axs[0], lambda v: v, "IoU"), (axs[1], dice, "Dice")):
        ax.hist(f(xi), bins=b, weights=np.full(len(xi), 1 / len(xi)), color="#b8bcc2", label=f"expert vs. expert, mean {f(xi).mean():.2f}")
        ax.hist(f(mi), bins=b, weights=np.full(len(mi), 1 / len(mi)), color="#0b84c6", alpha=0.65, label=f"ours (OOF), mean {f(mi).mean():.2f}")
        ax.axvline(f(xi).mean(), color="#6b7178", ls="--", lw=0.8); ax.axvline(f(mi).mean(), color="#0b84c6", ls="--", lw=0.8)
        ax.set_xlabel(f"{nm} of overlapping (GT, prediction) pairs", fontsize=7); ax.set_ylabel("fraction"); ax.legend(fontsize=6, frameon=False, loc="upper left")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3); ax.spines[["top", "right"]].set_visible(False)
    frac = lambda v: np.array([(np.minimum(np.array(v), 3) == k).mean() for k in range(4)])
    ax = axs[2]; x = np.arange(4); wd = 0.2
    for k, (v, col, lab) in enumerate(((gcnt, "#0b84c6", "per GT, ours"), (xg, "#b8bcc2", "per GT, experts"), (pcnt, CYAN, "per prediction, ours"), (xp, "#dfe2e6", "per prediction, experts"))):
        ax.bar(x + (k - 1.5) * wd, frac(v), wd, color=col, label=lab, ec="k", lw=0.3)
    ax.set_xlim(-0.5, 5.2); ax.set_xticks(x); ax.set_xticklabels(["0", "1", "2", "3+"]); ax.set_xlabel("number of overlapping partners"); ax.set_ylabel("fraction")
    ax.legend(fontsize=6, frameon=False, loc="upper right"); ax.set_ylim(0, 1.0); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.5); save(fig, a, "eval_dist")
    print("mean IoU / Dice ours", mi.mean(), dice(mi).mean(), "experts", xi.mean(), dice(xi).mean())
    for nm, v in (("GT ours", gcnt), ("GT experts", xg), ("pred ours", pcnt), ("pred experts", xp)):
        print(nm, np.round(frac(v), 4))


def all(a):
    for f in (concept, why, gallery, pipeline, evald):
        f(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd")
    for k in ("team", "root", "oof", "work", "eda", "fig"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args(); sys.path.insert(0, a.team)
    globals()[a.cmd](a)
