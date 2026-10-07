# -*- coding: utf-8 -*-
"""Study-design schematic (Fig. 1) and the Elsevier graphical abstract.

Fig. 1 follows the user's draft layout (2026-09-18): panels (a) inputs,
(b) common processing, (c) three numbered evaluation axes, (d) robustness
checks; keyword phrases instead of sentences; the observed association is
linked to axis 3 as its reference. Small plots inside the panels are the
paper's data, read from results/paper_stats_v1.json (nothing sketched by
hand). Style: scripts/pubstyle.py (Okabe-Ito palette, Arial, 5.15 in design
width, editable PDF text).

Graphical abstract (Elsevier): >= 1328 x 531 px at 300 dpi, 500:200 aspect,
Arial/Times, all text inside the image, no heading, read left to right.
Produced at 6.5 x 2.6 in, 300 dpi PNG, 600 dpi TIFF and PDF.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve(); FIGDIR = HERE.parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import pubstyle as ps  # noqa: E402
ps.apply()
C = ps.C
COL_OBS, COL_TEB, COL_CLMU, COL_FLAG = C["dark"], C["hero"], C["orange"], C["verm"]
NAVY = "#1F4E79"
FILL_METHOD, FILL_AXES = "#F7F7F7", "#FFFFFF"          # graphical abstract (v24 values, unchanged)
EDGE = "#8f8f8f"
F1_DATA, F1_METHOD, F1_AXES, F1_ROBUST, F1_EDGE = "#DCEAF5", "#EFEFEF", "#F4F4F4", "#F3F3EE", "#7f7f7f"  # Figure 1

S = json.load(open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
PS = S["per_site"]; SITES = list(PS)
HMISS = set(S["meta"]["height_mismatch_sites"])
alb = np.array([PS[s]["albedo"] for s in SITES]); obs = np.array([PS[s]["dtsa_std"] for s in SITES])
teb = np.array([PS[s]["teb_dtsa"] for s in SITES]); clmu = np.array([PS[s]["clmu_dtsa"] for s in SITES])
flag = np.array([(s in HMISS) or s == "PL-Lipowa" for s in SITES])
AOD = S["amplitude_ordering_decomposition"]["all19"]

def ols(x, y):
    b = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1); return b, y.mean() - b * x.mean()
# v27 (supervisor A-5): the conservative core (16 unflagged records) is the
# headline set, so every fitted line and printed statistic uses the core;
# the flagged records stay in the plots as open markers.
AODC = S["amplitude_ordering_decomposition"]["core16"]
core = ~flag
b_alb, a_alb = ols(alb[core], obs[core]); r_alb = np.corrcoef(alb[core], obs[core])[0, 1]
b_teb, a_teb = ols(obs[core], teb[core]); b_clmu, a_clmu = ols(obs[core], clmu[core])
assert core.sum() == 16
assert abs(b_teb - AODC["TEB"]["slope"]) < 0.01 and abs(b_clmu - AODC["CLMU5"]["slope"]) < 0.01, (b_teb, b_clmu)
assert abs(r_alb - (-0.77)) < 0.01 and abs(b_alb / 10 - (-2.7)) < 0.1, (r_alb, b_alb)

# ----------------------------------------------------------------------------- helpers
def box(ax, x, y, w, h, fill, edge=EDGE, lw=0.6, r=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fill, ec=edge, lw=lw, zorder=z))
def arrow(ax, p, q, color=NAVY, lw=0.9, style="-|>", z=3, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0, linestyle=ls))
def txt(ax, x, y, s, size=5.8, weight="normal", ha="left", va="top", color=C["dark"], **kw):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va, color=color, zorder=5, **kw)
def panel(ax, x, y, letter, label, size=6.8):
    t = txt(ax, x, y, f"({letter})", size, "bold", va="center")
    txt(ax, x + 4.2, y, label.upper(), size - 0.6, "bold", va="center", color="#444444")
def badge(ax, x, y, n, s=4.2):
    ax.add_patch(Rectangle((x, y - s), s, s, fc=NAVY, ec="none", zorder=4))
    txt(ax, x + s / 2, y - s / 2, str(n), 7.0, "bold", ha="center", va="center", color="white")
def thin(axi, lw=0.4):
    for sp in axi.spines.values(): sp.set_linewidth(lw)
    axi.spines[["top", "right"]].set_visible(False)

def scatter_obs_albedo(axi, label_size=5.2, ms=9, ticks=True):
    axi.scatter(alb[~flag], obs[~flag], s=ms, color=COL_OBS, zorder=3, lw=0)
    axi.scatter(alb[flag], obs[flag], s=ms, facecolors="none", edgecolors=COL_FLAG, lw=0.7, zorder=3)
    xx = np.array([alb.min() - 0.005, alb.max() + 0.005]); axi.plot(xx, a_alb + b_alb * xx, color=COL_TEB, lw=0.9, zorder=2)
    axi.axhline(0, color=EDGE, lw=0.4, zorder=1)
    axi.set_xlabel("observed midday albedo", fontsize=label_size, labelpad=1.5)
    axi.set_ylabel(r"$\overline{\Delta T_{s-a}}$ (K)", fontsize=label_size, labelpad=1.5)
    axi.tick_params(labelsize=label_size - 0.4, length=1.5, width=0.4, pad=1)
    if not ticks: axi.set_xticks([]); axi.set_yticks([])
    thin(axi)

def scatter_model_obs(axi, label_size=5.2, ms=9, lines=True):
    lim = (-3.6, 6.6)
    axi.plot(lim, lim, ls="--", color=C["grey"], lw=0.6, zorder=1)
    for arr, col, mk in ((teb, COL_TEB, "o"), (clmu, COL_CLMU, "^")):
        axi.scatter(obs[~flag], arr[~flag], s=ms, marker=mk, color=col, lw=0, zorder=3)
        axi.scatter(obs[flag], arr[flag], s=ms, marker=mk, facecolors="none", edgecolors=col, lw=0.6, zorder=3)
    if lines:
        xx = np.array(lim)
        axi.plot(xx, a_teb + b_teb * xx, color=COL_TEB, lw=0.8, zorder=2)
        axi.plot(xx, a_clmu + b_clmu * xx, color=COL_CLMU, lw=0.8, zorder=2)
    axi.set_xlim(lim); axi.set_ylim(lim); axi.set_aspect("equal")
    axi.set_xlabel("observed (K)", fontsize=label_size, labelpad=1.5)
    axi.set_ylabel("modelled (K)", fontsize=label_size, labelpad=1.5)
    axi.tick_params(labelsize=label_size - 0.4, length=1.5, width=0.4, pad=1)
    thin(axi)

# pictograms for the three axes (schematic, no ticks) ---------------------------
def pict_timeseries(axi):
    t = np.linspace(0, 6, 200); o = np.sin(t) + 0.25 * np.sin(3.1 * t); m = o + 0.35 + 0.12 * np.sin(2.2 * t + 0.4)
    axi.plot(t, o, color=COL_OBS, lw=0.9); axi.plot(t, m, color=COL_TEB, lw=0.9); axi.axis("off")
def pict_bars(axi):
    axi.axhline(0, color=C["dark"], lw=0.5)
    axi.bar([0], [1.0], width=0.6, color=COL_TEB); axi.bar([1], [-0.35], width=0.6, color=COL_CLMU)
    axi.axhline(0.45, color=C["grey"], lw=0.6, ls="--"); axi.set_xlim(-0.6, 1.6); axi.set_ylim(-0.7, 1.3); axi.axis("off")
def pict_crosssite(axi):
    """Unflagged records only (a pictogram of the compression), 1:1 dashed."""
    lim = (-3.0, 3.0); axi.plot(lim, lim, ls="--", color=C["grey"], lw=0.6, zorder=1)
    for arr, col, mk in ((teb, COL_TEB, "o"), (clmu, COL_CLMU, "^")):
        axi.scatter(obs[~flag], arr[~flag], s=5, marker=mk, color=col, lw=0, zorder=3)
    axi.set_xlim(lim); axi.set_ylim(lim); axi.set_aspect("equal"); axi.set_xticks([]); axi.set_yticks([]); thin(axi)
    axi.text(0.97, 0.03, "obs", transform=axi.transAxes, fontsize=4.2, ha="right", va="bottom", color=C["grey"])
    axi.text(0.03, 0.97, "model", transform=axi.transAxes, fontsize=4.2, ha="left", va="top", color=C["grey"])

# ============================================================================= Figure 1
def elbow(ax, pts, color=NAVY, lw=0.9, ls="-", z=3):
    """Orthogonal connector through the listed points; arrowhead on the last leg."""
    for p, q in zip(pts[:-2], pts[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, z=z)

def fig_design():
    fig = plt.figure(figsize=(ps.W_DOUBLE, 3.35))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    H, B, SM, LS = 6.8, 5.6, 5.0, 1.32
    XA, WA = 2, 27.5          # column (a)
    XB, WB = 34, 27           # column (b)
    XC, WC = 66, 32.5         # column (c)

    # -- (a) inputs ---------------------------------------------------------------
    panel(ax, XA, 96.5, "a", "Inputs")
    box(ax, XA, 71, WA, 21.5, F1_DATA, edge=NAVY, r=0.8)
    txt(ax, XA + 1.8, 90.5, "Observations", H, "bold")
    txt(ax, XA + 1.8, 85, "Urban-PLUMBER\n19 records · 18 towers · 16 cities\n30-min LW↑, LW↓, $T_a$\nalbedo · geometry", B, linespacing=LS)
    box(ax, XA, 48, WA, 19.5, F1_DATA, edge=NAVY, r=0.8)
    txt(ax, XA + 1.8, 65.5, "Simulations", H, "bold")
    txt(ax, XA + 1.8, 60, "TEB 4.1.2 · CLM-Urban (CLM5.0)\noffline · shared forcing\ndefault materials", B, linespacing=LS)
    box(ax, XA, 21, WA, 23, F1_METHOD, edge=F1_EDGE, r=0.8)
    txt(ax, XA + 1.8, 42, "Observed association", H, "bold")
    axi = fig.add_axes([0.04, 0.235, 0.105, 0.13]); scatter_obs_albedo(axi, label_size=4.3, ms=5, ticks=False)
    txt(ax, 16.5, 37, f"albedo vs. mean $\\Delta T_{{s-a}}$\n$r={r_alb:.2f}$ (core 16)\n{b_alb/10:.1f} K per 0.1 albedo\nassociation only", SM, linespacing=LS)

    # -- (b) common processing ------------------------------------------------------
    panel(ax, XB, 96.5, "b", "Common processing")
    box(ax, XB, 74, WB, 18.5, F1_METHOD, edge=F1_EDGE, r=0.8)
    txt(ax, XB + 1.8, 90.5, "Matched nocturnal data", H, "bold")
    txt(ax, XB + 1.8, 85, "observed & simulated LW↑\none sampling mask", B, linespacing=LS)
    box(ax, XB, 30, WB, 37, F1_METHOD, edge=F1_EDGE, r=0.8)
    xm = XB + WB / 2
    txt(ax, XB + 1.8, 65, "Radiometric diagnostic", H, "bold")
    txt(ax, xm, 56.5, r"$T_s=\left[\dfrac{\mathrm{LW}_\uparrow-(1-\varepsilon)\,\mathrm{LW}_\downarrow}{\varepsilon\sigma}\right]^{1/4}$",
        5.8, ha="center", va="center")
    txt(ax, xm, 48.5, r"$\varepsilon=0.95$", 5.8, ha="center", va="center")
    ax.plot([XB + 2.5, XB + WB - 2.5], [45, 45], color=F1_EDGE, lw=0.5, zorder=2)
    txt(ax, XB + 1.8, 42.5, "Record-mean offset", B, "bold")
    txt(ax, xm, 37.5, r"$\overline{\Delta T_{s-a}}=\overline{T_s}-\overline{T_a}$", 6.2, ha="center", va="center")
    txt(ax, xm, 32.5, "19 means per dataset", SM, ha="center", va="center")

    # -- (c) evaluation --------------------------------------------------------------
    panel(ax, XC, 96.5, "c", "Evaluation")
    rows = [(74, 1, "Temporal skill", "LW↑ time series\n$r$ · cRMSE · agreement", pict_timeseries),
            (52, 2, "Bias & benchmarks", "mean bias · MAE\nREG2 · KM3\nleave-one-city-out", pict_bars),
            (30, 3, "Cross-site gradient", "spread $s_m/s_o$ · ordering $r$\nslope $\\hat\\beta=r\\,(s_m/s_o)$\n1:1 reference", pict_crosssite)]
    for y0, n, head, body, pict in rows:
        box(ax, XC, y0, WC, 18.5, F1_AXES, edge=F1_EDGE, r=0.8)
        badge(ax, XC + 1.7, y0 + 16.8, n)
        txt(ax, XC + 7.5, y0 + 16.5, head, H, "bold")
        txt(ax, XC + 7.5, y0 + 11.5, body, SM, linespacing=LS)
        if pict is pict_crosssite:
            axi = fig.add_axes([0.895, (y0 + 1.2) / 100, 0.085, 0.112])
        else:
            axi = fig.add_axes([0.895, (y0 + 2.5) / 100, 0.085, 0.10])
        pict(axi)

    # -- (d) robustness checks -------------------------------------------------------
    box(ax, XA, 2, 96.5, 15.5, F1_ROBUST, edge=F1_EDGE, r=0.8)
    panel(ax, XA + 1.8, 15, "d", "Targeted robustness checks", size=6.4)
    cols = [(XA + 1.8, "Observation uncertainty", "random · placement\nalbedo-aligned · 208 cells"),
            (37, "Sampling", "city clusters · seasons\nepochs · record sets"),
            (70, "Model settings", "materials · albedo\nspin-up · structural probe")]
    for x0, head, body in cols:
        txt(ax, x0, 11, head, B, "bold"); txt(ax, x0, 7.6, body, SM, linespacing=LS)
    for xd in (34.5, 67.5): ax.plot([xd, xd], [4, 12.5], color=F1_EDGE, lw=0.5)

    # -- connectors --------------------------------------------------------------------
    xa1, xb1, xc0 = XA + WA, XB + WB, XC
    xg1, xg2 = (xa1 + XB) / 2, (xb1 + XC) / 2           # gutter mid-lines
    arrow(ax, (xa1, 86), (XB, 86))                                          # observations -> matched data
    elbow(ax, [(xa1, 57.5), (xg1, 57.5), (xg1, 79), (XB, 79)])              # simulations -> matched data
    arrow(ax, (xm, 74), (xm, 67))                                           # matched data -> diagnostic
    arrow(ax, (xb1, 85.5), (xc0, 85.5))                                     # matched data -> 1
    elbow(ax, [(xb1, 79), (xg2, 79), (xg2, 61.5), (xc0, 61.5)])             # matched data -> 2
    arrow(ax, (xb1, 40), (xc0, 40))                                         # diagnostic -> 3
    elbow(ax, [(xa1, 25), (xg2, 25), (xg2, 34), (xc0, 34)], color=C["grey"], lw=0.7, ls="--")  # association -> 3
    txt(ax, (xa1 + xg2) / 2, 25.8, "reference", SM, ha="center", va="bottom", color=C["grey"], style="italic")
    return fig

# ============================================================================= graphical abstract
def graphical_abstract():
    W, Hh = 6.5, 2.6
    fig = plt.figure(figsize=(W, Hh))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    BIG, MID, SM = 9.6, 8.0, 7.2   # the image is displayed at ~13 cm (x0.79)

    # left: the observed relation (real data)
    txt(ax, 2, 97, "19 urban flux towers (Urban-PLUMBER)", BIG, "bold")
    txt(ax, 2, 87.5, "nocturnal radiometric surface–air offset", MID)
    axi = fig.add_axes([0.055, 0.26, 0.195, 0.46]); scatter_obs_albedo(axi, label_size=7.0, ms=16)
    txt(ax, 2, 5, f"organizes along site albedo (16-record core):  $r={r_alb:.2f}$,  {b_alb/10:.1f} K per 0.1 albedo", SM, va="bottom")

    # middle: the inversion and the two schemes
    box(ax, 31.5, 30, 17.5, 44, FILL_METHOD, r=1.5)
    txt(ax, 40.25, 68, "same radiometric\ninversion", MID, "bold", ha="center", linespacing=1.3)
    txt(ax, 40.25, 52, "TEB and CLM-Urban\nrun offline at every\nsite, default materials", SM, ha="center", linespacing=1.35)
    arrow(ax, (27.5, 52), (31.5, 52), lw=1.0, color=C["dark"]); arrow(ax, (49, 52), (52.5, 52), lw=1.0, color=C["dark"])

    # right: modelled versus observed (real data, fitted slopes)
    txt(ax, 66.5, 97, "both schemes compress the cross-site gradient", BIG, "bold", ha="center")
    axi = fig.add_axes([0.578, 0.20, 0.205, 0.60]); scatter_model_obs(axi, label_size=7.0, ms=16)
    axi.text(0.05, 0.93, f"TEB slope {b_teb:.2f}", transform=axi.transAxes, fontsize=SM, color=COL_TEB, va="top")
    axi.text(0.05, 0.82, f"CLM-Urban {b_clmu:.2f}", transform=axi.transAxes, fontsize=SM, color=COL_CLMU, va="top")
    axi.text(0.97, 0.06, "1:1", transform=axi.transAxes, fontsize=SM, color=C["grey"], ha="right")

    # far right: the three-axis verdict
    box(ax, 80.5, 6, 18.5, 84, FILL_AXES, r=1.5)
    txt(ax, 89.75, 87, "same output,\nthree verdicts", MID, "bold", ha="center", linespacing=1.3)
    items = [(C["teal"], "temporal skill\nhigh for both"), (C["grey"], "mean bias and\nbenchmarks\nfavour CLM-Urban"),
             (COL_FLAG, "cross-site gradient\nneither scheme")]
    y = 70
    for col, s in items:
        ax.scatter([83], [y - 2.8], s=26, color=col, zorder=4)
        txt(ax, 85.5, y, s, SM, linespacing=1.25); y -= 18
    txt(ax, 89.75, 9, "per-site scores do not\nsee the cross-site gap", SM, "bold", ha="center", va="bottom", linespacing=1.25)
    return fig

if __name__ == "__main__":
    f = fig_design()
    f.savefig(FIGDIR / "fig0_design.pdf", bbox_inches="tight"); f.savefig(FIGDIR / "fig0_design.png", dpi=300, bbox_inches="tight")
    g = graphical_abstract()
    g.savefig(FIGDIR / "graphical_abstract.pdf"); g.savefig(FIGDIR / "graphical_abstract.png", dpi=300)
    g.savefig(FIGDIR / "graphical_abstract_600dpi.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    from PIL import Image
    im = Image.open(FIGDIR / "graphical_abstract.png"); print("GA png px:", im.size, "(min 1328x531)")
    print("wrote fig0_design.{pdf,png}, graphical_abstract.{pdf,png,tiff}")
