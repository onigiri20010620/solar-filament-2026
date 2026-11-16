"""Report figures, fourth version: the expert-agreement story.
sheet : contact sheet of 3-expert windows (to choose the teaser example)
agree : teaser (figures/concept): three experts' outlines, our prediction, hit rate by expert agreement
Usage: python report_figs4.py <sheet|agree> --team .. --root .. --oof .. --work .. --eda .. --fig ..
"""
import argparse, json, os, pickle, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pycocotools import mask as mu

from report_figs import CYAN, GREEN, load_gt, masks_of, show
from report_figs2 import save

BLUE, RED, OKG, GREY = "#0072B2", "#D55E00", "#009E73", "#999999"
CYAN, ORANGE, GREEN = "#56B4E9", "#E69F00", "#009E73"
ECOL = ["#009E73", "#F0E442", "#CC79A7"]          # the three experts
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "font.family": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42})
EXAMPLE = None                                     # (stem, y0, x0, side); set after looking at the sheet


def info(a, gt, s, C):
    """per expert filament of image s: bbox centre, area, fraction of other experts matching, any overlap by others, our best IoU."""
    w = pickle.load(open(os.path.join(a.work, "why", s + ".pkl"), "rb")); keep = C["keep"][s]; es = list(gt[s]); rows = []
    for e in es:
        G = gt[s][e]; ng = len(G)
        if not ng:
            continue
        K = w["ious_r"][e][keep]; bk = K.max(0) if len(K) else np.zeros(ng); ag = np.zeros(ng); ov = np.zeros(ng)
        for o in es:
            if o != e and gt[s][o]:
                X = np.asarray(mu.iou(gt[s][o], G, [0] * ng)).reshape(len(gt[s][o]), ng); ag += X.max(0) > .5; ov = np.maximum(ov, X.max(0))
        bb = mu.toBbox(G)
        for g in range(ng):
            rows.append((bb[g][1] + bb[g][3] / 2, bb[g][0] + bb[g][2] / 2, float(mu.area(G[g])), ag[g] / (len(es) - 1), ov[g], bk[g]))
    return np.array(rows)


def panel(a, gt, s, win, axs, C, lw=1.1):
    img, prob, Q, pred, idx = masks_of(a, s); keep = C["keep"][s]; es = list(gt[s])
    show(axs[0], img, win, [(r, ECOL[k], "solid") for k, e in enumerate(es) for r in gt[s][e]], lw=lw)
    show(axs[1], img, win, [(r, CYAN, "solid") for r, k in zip(pred, keep) if k], lw=lw)


def sheet(a):
    gt = load_gt(a); C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb")); sd = 560; best = []
    for s in C["keep"]:
        if len(gt[s]) != 3:
            continue
        R = info(a, gt, s, C)
        if not len(R):
            continue
        top = None
        for y0 in range(0, 2048 - sd + 1, 140):
            for x0 in range(0, 2048 - sd + 1, 140):
                m = (R[:, 0] > y0 + 40) & (R[:, 0] < y0 + sd - 40) & (R[:, 1] > x0 + 40) & (R[:, 1] < x0 + sd - 40) & (R[:, 2] > 250)
                na = (m & (R[:, 3] == 1) & (R[:, 5] > .5)).sum() / 3; ns = (m & (R[:, 4] < 0.05) & (R[:, 5] <= .5)).sum(); nd = (m & (R[:, 3] == 0) & (R[:, 4] >= 0.05) & (R[:, 5] <= .5)).sum()
                sc = min(na, 3) + 1.5 * min(ns, 3) + 0.5 * min(nd, 2) - 0.15 * m.sum() / 3
                if na >= 2 and ns >= 2 and (top is None or sc > top[0]):
                    top = (sc, s, y0, x0)
        if top:
            best.append(top)
    best.sort(reverse=True); best = best[:16]; print(best)
    fig, axs = plt.subplots(len(best), 2, figsize=(7, 3.5 * len(best)))
    for k, (sc, s, y0, x0) in enumerate(best):
        panel(a, gt, s, (y0, x0, sd), axs[k], C); axs[k, 0].set_title(f"{k}: {s} y{y0} x{x0}  score {sc:.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(a.eda, "agree_sheet.png"), dpi=110); plt.close(fig)


def agree(a):
    gt = load_gt(a); C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb")); J = json.load(open(os.path.join(a.eda, "report_agree.json")))
    s, y0, x0, sd = EXAMPLE
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1.12]))
    panel(a, gt, s, (y0, x0, sd), axs, C, lw=1.3)
    axs[0].set_title("Three experts, one image", weight="bold", fontsize=10)
    axs[0].legend([Line2D([], [], color=c, lw=2.2) for c in ECOL], ["expert A", "expert B", "expert C"], loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=8, frameon=False,
                  handlelength=1.3, columnspacing=1.0, handletextpad=0.4)
    axs[1].set_xlabel("drawn by all three: predicted", fontsize=8, labelpad=5)
    axs[1].set_title("Our prediction", weight="bold", fontsize=10, color=BLUE)
    ax = axs[2]; keys = ("all other experts drew it", "some but not all others", "no other expert drew it (IoU > 0.5)"); lab = ("all", "some", "none")
    v = [100 * J[k]["our hit rate"] for k in keys]; sh = [100 * J[k]["share"] for k in keys]; col = [BLUE, BLUE, BLUE]
    ax.bar(range(3), v, color=col, width=0.7)
    for k in range(3):
        ax.text(k, v[k] + 2, f"{v[k]:.0f}%", ha="center", fontsize=12, weight="bold", color="k")
        ax.text(k, 3, f"{sh[k]:.0f}% of\nfilaments", ha="center", va="bottom", fontsize=7, color="w")
    ax.set_xticks(range(3)); ax.set_xticklabels(lab, fontsize=9.5); ax.set_xlabel("other experts who drew the same filament", fontsize=8.5)
    ax.set_ylim(0, 108); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("Filaments we hit", weight="bold", fontsize=10, color=BLUE)
    fig.subplots_adjust(wspace=0.06, left=0, right=1, top=0.9, bottom=0.14)
    p = ax.get_position(); ax.set_position([p.x0 + 0.02, p.y0, p.width - 0.02, p.height])
    save(fig, a, "concept")


def evid(a):
    J = json.load(open(os.path.join(a.eda, "report_agree.json"))); K = json.load(open(os.path.join(a.eda, "report_agree2.json")))
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 1.8), gridspec_kw=dict(width_ratios=[0.8, 1.0, 1.25]))
    # (a) confirmed filaments: third expert vs ours
    ax = axs[0]; v = [100 * K["B third expert: hit rate"], 100 * K["B ours: hit rate"]]
    ax.bar([0, 1], v, color=["#BBBBBB", BLUE], width=0.66)
    for k in range(2):
        ax.text(k, v[k] + 2, f"{v[k]:.0f}%", ha="center", fontsize=11, weight="bold", color=("#555", BLUE)[k])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["a third\nexpert", "ours"], fontsize=8.5); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 112); ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False); ax.set_title("(a) Hits on filaments\ntwo experts agree on", weight="bold", loc="left")
    # (b) our errors against one expert
    ax = axs[1]; v = [100 * J["our misses: share that no other expert matched either"], 100 * K["D share matching another expert of the same image"]]
    for k in range(2):
        ax.barh(1.55 * (1 - k), 100, color="#e3e6ea", height=0.5); ax.barh(1.55 * (1 - k), v[k], color=RED, height=0.5, alpha=0.85)
        ax.text(v[k] / 2, 1.55 * (1 - k), f"{v[k]:.0f}%", ha="center", va="center_baseline", color="w", fontsize=8.5, weight="bold")
    ax.text(0, 1.9, "misses no other expert matched", fontsize=7.5, va="bottom"); ax.text(0, 0.35, "false alarms another expert drew", fontsize=7.5, va="bottom")
    ax.set_xlim(0, 100); ax.set_ylim(-0.4, 2.75); ax.set_xticks([]); ax.set_yticks([]); ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_title("(b) Our errors against\none expert", weight="bold", loc="left")
    # (c) what selection keeps
    ax = axs[2]; cols = ["#CCCCCC", "#8FCDBA", OKG]
    for yy, nm in ((1, "kept"), (0, "rejected")):
        d = K[f"C {nm}"]; f = [100 * d["no expert matches"], 100 * d["some"], 100 * d["all experts match"]]; left = 0
        for k in range(3):
            ax.barh(yy, f[k], left=left, color=cols[k], height=0.5)
            if f[k] > 9:
                ax.text(left + f[k] / 2, yy, f"{f[k]:.0f}%", ha="center", va="center", fontsize=8, weight="bold", color=("w" if k == 2 else "#222"))
            left += f[k]
    ax.set_yticks([1, 0]); ax.set_yticklabels(["kept", "rejected"], fontsize=9); ax.set_xlim(0, 100); ax.set_ylim(-0.4, 2.05); ax.set_xticks([]); ax.tick_params(length=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    for x, t, c, ha in ((0, "no expert", "#6b7178", "left"), (50, "some", "#5d9a62", "center"), (100, "all experts", OKG, "right")):
        ax.text(x, 1.38, t, ha=ha, va="bottom", fontsize=8.5, color=c, weight="bold")
    ax.set_title("(c) Candidates: experts who\ndrew the same filament", weight="bold", loc="left")
    fig.tight_layout(w_pad=1.6); save(fig, a, "agree")


def gallery1(a):          # single-column gallery: 4 gains, 1 loss
    from report_figs import ORANGE, window
    from report_figs2 import DOWN, UP, pair_info
    gt = load_gt(a); ex = UP[:4] + DOWN[:1]; n = len(ex); C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb"))
    fig, axs = plt.subplots(2, n, figsize=(3.4, 1.5))
    for k, (s, i) in enumerate(ex):
        img, prob, Q, pred, idx, cand, red, E = pair_info(a, gt, s, i)
        p = max((p for p in C["pairs"] if p["stem"] == s and p["i"] == i), key=lambda p: abs(p["r"] - p["d"]))
        g = gt[s][p["e"]][p["g"]]; win = window([g, cand, red], pad=0.3)
        for ax, m, col, v in ((axs[0, k], cand, ORANGE, p["d"]), (axs[1, k], red, CYAN, p["r"])):
            show(ax, img, win, [(g, GREEN, "solid"), (m, col, "solid")], lw=1.0)
            ax.text(0.05, 0.95, f"{v:.2f}", transform=ax.transAxes, color="w", fontsize=6.5, va="top", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc=(OKG if v > .5 else RED), ec="none", alpha=0.95))
    axs[0, 0].set_ylabel("detector", fontsize=8, color="#d97700", weight="bold"); axs[1, 0].set_ylabel("re-drawn", fontsize=8, color=BLUE, weight="bold")
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.06, right=1, top=1, bottom=0)
    save(fig, a, "gallery")


def fulldisk(a):          # one full-disk training image (out of fold): input and the prediction against each of its experts
    gt = load_gt(a); C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb")); s = EXAMPLE[0]
    img, prob, Q, pred, idx = masks_of(a, s); keep = C["keep"][s]; es = list(gt[s]); w = pickle.load(open(os.path.join(a.work, "why", s + ".pkl"), "rb"))
    P = [r for r, k in zip(pred, keep) if k]; win = (0, 0, 2048)
    fig, axs = plt.subplots(1, 4, figsize=(7.1, 1.95))
    show(axs[0], img, win); axs[0].set_title("Image (full disk)", weight="bold", fontsize=8.5)
    y0, x0, sd = EXAMPLE[1:]
    axs[0].add_patch(plt.Rectangle((x0, y0), sd, sd, fill=False, ec="w", lw=0.8))
    for k, e in enumerate(es):
        J = w["ious_r"][e][keep]; G = gt[s][e]; hit = J > .5; t = hit.sum(); pq = J[hit].sum() / (t + .5 * (len(J) - t) + .5 * (len(G) - t))
        ph = hit.any(1) if len(G) else np.zeros(len(P), bool)
        show(axs[k + 1], img, win, [(r, (CYAN if h else RED), "solid") for r, h in zip(P, ph)], [(r, ECOL[k], 0.75) for r in G], lw=0.7)
        axs[k + 1].set_title(f"vs. expert {'ABC'[k]}:  PQ {pq:.2f}", weight="bold", fontsize=8.5)
        axs[k + 1].set_xlabel(f"{len(G)} drawn, {int(t)} hit", fontsize=7.5, labelpad=2)
    axs[0].set_xlabel("box: region of Fig. 1", fontsize=7.5, labelpad=2)
    fig.legend([Line2D([], [], color=CYAN, lw=1.6), Line2D([], [], color=RED, lw=1.6)], ["our instance, hit", "our instance, miss"], loc="lower center", ncol=2, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.62, -0.09), handlelength=1.4)
    fig.subplots_adjust(wspace=0.04, left=0, right=1, top=0.9, bottom=0.1)
    save(fig, a, "fulldisk")


# ---------------------------------------------------------------- single-column versions (balanced page layout)
def pipe2(a):             # pipeline as a 2 x 2 block
    from report_figs2 import TEASER
    s, i = TEASER; img, prob, Q, pred, idx = masks_of(a, s)
    C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb")); keep = C["keep"][s]
    ys, xs = np.nonzero(mu.decode(Q["cands"][i])); cy, cx = int(ys.mean()), int(xs.mean()); sd = 640
    y0, x0 = int(np.clip(cy - sd // 2, 0, 2048 - sd)), int(np.clip(cx - sd // 2, 0, 2048 - sd)); win = (y0, x0, sd)
    fig, axs = plt.subplots(2, 2, figsize=(3.4, 3.75)); ax = axs.ravel()
    show(ax[0], img, win)
    ax[1].imshow(np.kron(prob, np.ones((2, 2), np.uint8))[y0:y0 + sd, x0:x0 + sd], cmap="magma", vmin=0, vmax=255); ax[1].set_xticks([]); ax[1].set_yticks([])
    show(ax[2], img, win, [(r, ORANGE, "solid") for r in Q["cands"]], lw=0.9)
    show(ax[3], img, win, [(r, CYAN, "solid") for r, k in zip(pred, keep) if k], lw=0.9)
    for x, t, u in zip(ax, ("Input image", "1  Detect", "2  Cut candidates", "3  Re-draw and select"),
                       ("H-alpha, 2048 px", "U-Net: filament probability", "threshold, join fragments", "UNet++ / ConvNeXt, then scorer")):
        x.set_title(t, fontsize=8.5, pad=3); x.set_xlabel(u, fontsize=7, labelpad=2)
    fig.subplots_adjust(wspace=0.05, hspace=0.26, left=0.005, right=0.995, top=0.95, bottom=0.05)
    save(fig, a, "pipeline")


def disk2(a):             # full-disk example as a 2 x 2 block
    gt = load_gt(a); C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb")); s = EXAMPLE[0]
    img, prob, Q, pred, idx = masks_of(a, s); keep = C["keep"][s]; es = list(gt[s]); w = pickle.load(open(os.path.join(a.work, "why", s + ".pkl"), "rb"))
    P = [r for r, k in zip(pred, keep) if k]; win = (0, 0, 2048)
    fig, axs = plt.subplots(2, 2, figsize=(3.4, 3.95)); ax = axs.ravel()
    show(ax[0], img, win); ax[0].set_title("Input image", fontsize=8.5, pad=3)
    y0, x0, sd = EXAMPLE[1:]; ax[0].add_patch(plt.Rectangle((x0, y0), sd, sd, fill=False, ec="w", lw=0.8)); ax[0].set_xlabel("box: region of Fig. 1", fontsize=7, labelpad=2)
    for k, e in enumerate(es):
        J = w["ious_r"][e][keep]; G = gt[s][e]; hit = J > .5; t = hit.sum(); pq = J[hit].sum() / (t + .5 * (len(J) - t) + .5 * (len(G) - t))
        ph = hit.any(1) if len(G) else np.zeros(len(P), bool)
        show(ax[k + 1], img, win, [(r, (CYAN if h else RED), "solid") for r, h in zip(P, ph)], [(r, ECOL[k], 0.8) for r in G], lw=0.7)
        ax[k + 1].set_title(f"vs. expert {'ABC'[k]}: PQ {pq:.2f}", fontsize=8.5, pad=3)
        ax[k + 1].set_xlabel(f"{len(G)} drawn by the expert, {int(t)} hit", fontsize=7, labelpad=2)
    fig.legend([Line2D([], [], color=CYAN, lw=1.6), Line2D([], [], color=RED, lw=1.6)], ["our instance, hit", "our instance, miss"], loc="lower center", ncol=2, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.005), handlelength=1.4)
    fig.subplots_adjust(wspace=0.05, hspace=0.24, left=0.005, right=0.995, top=0.955, bottom=0.095)
    save(fig, a, "fulldisk")


def evid2(a):             # agreement evidence as one column of horizontal bars on a common 0-100 % axis
    J = json.load(open(os.path.join(a.eda, "report_agree.json"))); K = json.load(open(os.path.join(a.eda, "report_agree2.json")))
    fig, axs = plt.subplots(3, 1, figsize=(3.4, 2.75), gridspec_kw=dict(height_ratios=[1, 1, 1.25]))
    def bars(ax, rows, title):
        for y, (lab, v, col) in enumerate(rows[::-1]):
            ax.barh(y, 100, color="#EEEEEE", height=0.62); ax.barh(y, v, color=col, height=0.62)
            ax.text(v + 1.5, y, f"{v:.0f}%", va="center", fontsize=8)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=7.5)
        ax.set_title(title, fontsize=8, loc="left", pad=3, x=-0.62)
    bars(axs[0], [("a third expert", 100 * K["B third expert: hit rate"], "#BBBBBB"), ("ours", 100 * K["B ours: hit rate"], BLUE)], "(a) Filaments two experts agree on: share hit by")
    bars(axs[1], [("misses that no other\nexpert matched", 100 * J["our misses: share that no other expert matched either"], RED),
                  ("false positives that\nanother expert drew", 100 * K["D share matching another expert of the same image"], RED)], "(b) Our errors against one expert: share of")
    ax = axs[2]; cols = ["#CCCCCC", "#8FCDBA", OKG]; labs = ["no expert", "some", "all experts"]
    for y, nm in ((1, "kept"), (0, "rejected")):
        d = K[f"C {nm}"]; f = [100 * d["no expert matches"], 100 * d["some"], 100 * d["all experts match"]]; left = 0
        for k in range(3):
            ax.barh(y, f[k], left=left, color=cols[k], height=0.62, label=(labs[k] if y == 1 else None))
            if f[k] > 9:
                ax.text(left + f[k] / 2, y, f"{f[k]:.0f}%", ha="center", va="center", fontsize=7.5, color=("w" if k == 2 else "#222222"))
            left += f[k]
    ax.set_yticks([1, 0]); ax.set_yticklabels(["kept", "rejected"], fontsize=7.5)
    ax.set_title("(c) Candidates: how many experts drew the same filament", fontsize=8, loc="left", pad=3, x=-0.62)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=7.5, frameon=False, handlelength=1.0, columnspacing=1.2, handletextpad=0.4)
    for ax in axs:
        ax.set_xlim(0, 112 if ax is not axs[2] else 100); ax.set_xticks([]); ax.tick_params(length=0); ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    axs[2].set_xlim(0, 112)
    fig.subplots_adjust(left=0.37, right=0.995, top=0.93, bottom=0.1, hspace=0.75)
    save(fig, a, "agree")


def eval2(a):             # organizers' diagnostics as one column: two histograms on top, partner shares below
    C = pickle.load(open(os.path.join(a.eda, "why_cache.pkl"), "rb")); gt = load_gt(a)
    mi, gcnt, pcnt, xi, xg = [], [], [], [], []
    for s, keep in C["keep"].items():
        w = pickle.load(open(os.path.join(a.work, "why", s + ".pkl"), "rb"))
        for e, J in w["ious_r"].items():
            Kk = J[keep]; mi += Kk[Kk > 0].tolist(); gcnt += (Kk > 0).sum(0).tolist(); pcnt += (Kk > 0).sum(1).tolist()
        es = list(gt[s])
        for ea in es:
            for eb in es:
                if ea != eb:
                    A, B = gt[s][ea], gt[s][eb]
                    J = np.asarray(mu.iou(B, A, [0] * len(A))).reshape(len(B), len(A)) if A and B else np.zeros((len(B), len(A)))
                    xi += J[J > 0].tolist(); xg += (J > 0).sum(0).tolist()
    mi, xi = np.array(mi), np.array(xi); dice = lambda v: 2 * v / (1 + v)
    fig = plt.figure(figsize=(3.4, 3.0)); gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0]); b = np.linspace(0, 1, 41)
    for k, (f, nm, tag) in enumerate(((lambda v: v, "IoU", "a"), (dice, "Dice", "b"))):
        ax = fig.add_subplot(gs[0, k])
        ax.hist(f(mi), bins=b, weights=np.full(len(mi), 1 / len(mi)), color=BLUE, alpha=0.75, label=f"ours ({f(mi).mean():.2f})")
        ax.hist(f(xi), bins=b, weights=np.full(len(xi), 1 / len(xi)), histtype="step", color="#444444", lw=1.0, label=f"two experts ({f(xi).mean():.2f})")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.5); ax.legend(fontsize=7, frameon=False, loc="upper left", borderaxespad=0.0, handlelength=1.1, labelspacing=0.2)
        ax.set_xlabel(f"{nm} of overlapping pairs", fontsize=7.5, labelpad=2); ax.set_yticks([]); ax.set_xticks([0, 0.5, 1]); ax.tick_params(labelsize=7.5)
        ax.set_title(f"({tag}) {nm} (mean in brackets)", fontsize=8, loc="left", pad=3); ax.spines[["top", "right", "left"]].set_visible(False)
    frac = lambda v: np.array([(np.array(v) == 0).mean(), (np.array(v) == 1).mean(), (np.array(v) >= 2).mean()]) * 100
    ax = fig.add_subplot(gs[1, :]); cols = ["#CCCCCC", BLUE, RED]; labs = ["none", "exactly one", "two or more (split / merge)"]
    for nm, v, yy in (("expert's filaments\nvs. our prediction", gcnt, 2), ("our filaments\nvs. an expert", pcnt, 1), ("expert's filaments\nvs. another expert", xg, 0)):
        f = frac(v); left = 0
        for k in range(3):
            ax.barh(yy, f[k], left=left, color=cols[k], height=0.64, label=(labs[k] if yy == 2 else None))
            if f[k] > 8:
                ax.text(left + f[k] / 2, yy, f"{f[k]:.0f}%", ha="center", va="center", fontsize=7.5, color=("w" if k == 1 else "#222222"))
            left += f[k]
        ax.text(101, yy, f"{f[2]:.1f}%", va="center", fontsize=7.5)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels(["expert's filaments\nvs. our prediction", "our filaments\nvs. an expert", "expert's filaments\nvs. another expert"], fontsize=7)
    ax.set_xlim(0, 112); ax.set_xticks([]); ax.tick_params(length=0); ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_title("(c) Overlapping partners", fontsize=8, loc="left", pad=3, x=-0.43)
    ax.legend(loc="upper center", bbox_to_anchor=(0.32, -0.02), ncol=3, fontsize=7, frameon=False, handlelength=1.0, columnspacing=1.0, handletextpad=0.4)
    fig.subplots_adjust(left=0.02, right=0.995, top=0.94, bottom=0.07, hspace=0.62, wspace=0.12)
    p = ax.get_position(); ax.set_position([0.3, p.y0, 0.695, p.height])
    save(fig, a, "eval_dist")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd")
    for k in ("team", "root", "oof", "work", "eda", "fig", "ex"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args(); sys.path.insert(0, a.team)
    if a.ex:
        t = a.ex.split(","); EXAMPLE = (t[0], int(t[1]), int(t[2]), int(t[3]))
    globals()[a.cmd](a)
