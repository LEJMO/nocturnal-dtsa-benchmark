"""Deterministic generator for the six dTsa manuscript figures (npj CAS).

ONE script, ONE style module (scripts/pubstyle.py), ONE data source
(results/paper_stats_v1.json + paper/manuscript_dtsa/tables/site_master_table.csv).

INVARIANTS enforced here:
  * Every plotted number is a DIRECT READ from paper_stats_v1.json (or the
    site_master_table.csv, which is itself derived from the same run). No
    statistic (r, slope, span, envelope, LOO R2) is recomputed in this script.
    The only in-script arithmetic is geometric line placement: an OLS line is
    drawn through the data centroid (mean albedo, mean dTsa) using the slope
    that is READ from the stats file, and the slope-CI fan uses the READ CI
    bounds. Means are a plotting convenience, not a re-estimated statistic.
  * NO titles and NO subplot letters are drawn inside any image (npj / project
    rule). Panels are distinguished by position only; captions live in LaTeX.
    figures_README.md maps every panel to its intended caption bullets.
  * The word "first" never appears. US-Minneapolis1/2 carry the HEIGHT-MISMATCH
    flag (radiometer 2 m vs Tair 40 m) and are visually flagged everywhere.
  * PL-Lipowa (+6.1 K, roof-canyon FOV composition caveat) is annotated, never folded
    into an unqualified headline span.

Output: F1..F7 as {stem}.pdf and {stem}.png at 300+ dpi into this directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
FIGDIR = HERE.parent                       # paper/manuscript_dtsa/figures
ROOT = HERE.parents[3]                     # project root
sys.path.insert(0, str(ROOT / "scripts"))
import pubstyle as ps                       # noqa: E402
import matplotlib.pyplot as _plt
_plt.rcParams.update({"font.size":8.0,"axes.labelsize":8.0,"xtick.labelsize":7.2,"ytick.labelsize":7.2,"legend.fontsize":7.0})

STATS_PATH = ROOT / "results" / "paper_stats_v1.json"

# ----------------------------------------------------------------------------
# Load the single source of truth
# ----------------------------------------------------------------------------
with open(STATS_PATH, "r", encoding="utf-8") as fh:
    S = json.load(fh)

PS = S["per_site"]
META = S["meta"]
HMISS = set(META["height_mismatch_sites"])          # {US-Minneapolis1, US-Minneapolis2}
K_PER_WM2 = META["K_per_Wm2"]

# Deterministic site ordering helpers -----------------------------------------
SITES = list(PS.keys())


def sorted_by(key, reverse=False):
    return sorted(SITES, key=lambda s: PS[s][key], reverse=reverse)


def short(site: str) -> str:
    """Compact axis label; keep country-code so sites stay identifiable."""
    return site


def save(fig, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


ps.apply()

C = ps.C
COL_OBS = C["dark"]
COL_TEB = C["hero"]        # TEB always hero blue
COL_CLMU = C["orange"]     # CLMU5 always orange
COL_BAND = C["light"]
COL_FLAG = C["verm"]       # flag accent (Minneapolis / thresholds)


# ============================================================================
# F1 - per-site nocturnal dTsa, ranked (19 sites)
# ============================================================================
def fig1():
    order = sorted_by("dtsa_std")                 # ascending: coolest surface first
    vals = [PS[s]["dtsa_std"] for s in order]
    y = np.arange(len(order))

    core = S["headline_obs"]["variants"]["excl_mpls_lipowa"]
    core_lo, core_hi = core["span_min"], core["span_max"]   # -2.17 .. +1.19 K

    fig, ax = plt.subplots(figsize=(ps.W_DOUBLE, 5.0))

    # conservative-core span band (excl. both Minneapolis + Lipowa, n=16)
    ax.axvspan(core_lo, core_hi, color=COL_BAND, zorder=0)
    ax.axvline(0.0, color=C["dark"], lw=0.6, zorder=1)

    for yi, s in zip(y, order):
        v = PS[s]["dtsa_std"]
        flagged = s in HMISS
        ax.barh(yi, v, height=0.68,
                color=("none" if flagged else COL_TEB),
                edgecolor=(COL_FLAG if flagged else COL_TEB),
                hatch=("////" if flagged else None),
                linewidth=(0.9 if flagged else 0.0), zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([short(s) for s in order])
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("Nocturnal surface–air temperature offset  "
                  r"$\overline{\Delta T_{s-a}}$  (K)")

    # annotations
    li = order.index("PL-Lipowa")
    ax.annotate("PL-Lipowa: roof-canyon FOV composition caveat",
                xy=(PS["PL-Lipowa"]["dtsa_std"], li),
                xytext=(3.1, li - 2.2), fontsize=7.4, va="center", color=C["dark"],
                arrowprops=dict(arrowstyle="-", color=C["dark"], lw=0.5))
    m1 = order.index("US-Minneapolis1")
    ax.annotate("Minneapolis pair: radiometer 2 m vs $T_{air}$ 40 m (height-mismatch flag)",
                xy=(PS["US-Minneapolis1"]["dtsa_std"], m1),
                xytext=(-3.0, m1 + 3.0), fontsize=7.4, va="center", color=COL_FLAG,
                arrowprops=dict(arrowstyle="-", color=COL_FLAG, lw=0.5))
    ax.text(3.1, len(order) * 0.45,
            f"conservative core span\n{core_lo:+.2f} to {core_hi:+.2f} K (n=16)",
            fontsize=7.4, va="center", ha="left", color=C["dark"])

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig1_dtsa_ranked")


# ============================================================================
# F2 - CORE: obs dTsa vs midday albedo, all-19 and excl-Mpls
# ============================================================================
def _fan(ax, xs, ys, slope0p1, ci0p1, color):
    """Draw OLS line (slope READ from stats, per 0.1 albedo) through the
    centroid, plus a slope-CI fan using the READ CI bounds."""
    xbar, ybar = float(np.mean(xs)), float(np.mean(ys))
    xg = np.linspace(min(xs) - 0.01, max(xs) + 0.01, 50)

    def line(sl_per_0p1):
        m = sl_per_0p1 / 0.1
        return ybar + m * (xg - xbar)

    lo, hi = line(ci0p1[0]), line(ci0p1[1])
    ax.fill_between(xg, lo, hi, color=color, alpha=0.15, lw=0, zorder=1)
    ax.plot(xg, line(slope0p1), color=color, lw=1.4, zorder=4)


def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(ps.W_DOUBLE, 2.7), sharey=True)

    variants = [
        ("all", axes[0], SITES),
        ("excl_mpls", axes[1], [s for s in SITES if s not in HMISS]),
    ]
    for vkey, ax, site_list in variants:
        V = S["headline_obs"]["variants"][vkey]
        xs = np.array([PS[s]["albedo"] for s in site_list])
        ys = np.array([PS[s]["dtsa_std"] for s in site_list])
        _fan(ax, xs, ys,
             V["slope_K_per_0p1_albedo"], V["slope_ci95"], COL_TEB)

        for s in site_list:
            x, yv = PS[s]["albedo"], PS[s]["dtsa_std"]
            if s in HMISS:
                ax.scatter(x, yv, s=34, facecolors="none",
                           edgecolors=COL_FLAG, linewidths=1.1, zorder=5)
                if s == "US-Minneapolis1":
                    ax.annotate("×2", xy=(x, yv), xytext=(4, -2),
                                textcoords="offset points", fontsize=7.0,
                                color=COL_FLAG, va="top")
            elif s == "PL-Lipowa":
                ax.scatter(x, yv, s=30, facecolors="none",
                           edgecolors=C["dark"], linewidths=1.0, zorder=5)
            else:
                ax.scatter(x, yv, s=26, color=C["dark"], zorder=5)

        ax.axhline(0.0, color=C["dark"], lw=0.5, zorder=2)
        ax.set_xlabel("Observed midday albedo")
        r = V["pearson_r"]
        pperm = S["cluster_permutation"].get(vkey, {}).get("p_perm")
        txt = f"r = {r:+.2f}"
        if pperm is not None:
            txt += f"\n$p_{{perm}}$ = {pperm:.1e}"
        txt += f"\nslope {V['slope_K_per_0p1_albedo']:.2f} K per 0.1 albedo"
        ax.text(0.96, 0.96, txt, transform=ax.transAxes, fontsize=7.4,
                va="top", ha="right", color=C["dark"])
        n = V["n_sites"]
        cl = V["n_clusters"]
        ax.text(0.04, 0.04, f"n={n} records / {cl} city clusters",
                transform=ax.transAxes, fontsize=7.2, va="bottom", ha="left",
                color=C["grey"])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    axes[0].set_ylabel(r"$\overline{\Delta T_{s-a}}$  (K)")
    # legend for markers (shared)
    handles = [
        plt.Line2D([], [], marker="o", ls="none", color=C["dark"], ms=5,
                   label="unflagged record"),
        plt.Line2D([], [], marker="o", ls="none", mfc="none",
                   mec=COL_FLAG, mew=1.1, ms=5,
                   label="Minneapolis (height-mismatch flag)"),
        plt.Line2D([], [], marker="o", ls="none", mfc="none",
                   mec=C["dark"], mew=1.0, ms=5, label="PL-Lipowa (FOV caveat)"),
    ]
    axes[0].legend(handles=handles, loc="lower left", fontsize=7.0,
                   bbox_to_anchor=(0.0, 1.02), ncol=1)
    fig.tight_layout()
    save(fig, "fig2_core_albedo")


# ============================================================================
# F3 - model vs obs dTsa (1:1), TEB + CLMU5 clouds
# ============================================================================
def fig3():
    fig, ax = plt.subplots(figsize=(ps.W_SINGLE + 0.4, ps.W_SINGLE + 0.2))

    obs = np.array([PS[s]["dtsa_std"] for s in SITES])
    teb = np.array([PS[s]["teb_dtsa"] for s in SITES])
    clmu = np.array([PS[s]["clmu_dtsa"] for s in SITES])

    lims = [-3.6, 6.6]
    ax.plot(lims, lims, ls="--", color=C["grey"], lw=0.9, zorder=1)

    # flagged records (Minneapolis height mismatch, Lipowa FOV) = open markers;
    # marker shape distinguishes the schemes in addition to colour
    for arr, col, mk in ((teb, COL_TEB, "o"), (clmu, COL_CLMU, "^")):
        for s, xv, yv in zip(SITES, obs, arr):
            flagged = (s in HMISS) or (s == "PL-Lipowa")
            ax.scatter(xv, yv, s=30, marker=mk,
                       facecolors=("none" if flagged else col),
                       edgecolors=col, linewidths=(1.0 if flagged else 0.0),
                       zorder=4)
    # legend proxies
    # v29 (supervisor): the flagged records are drawn as open markers in each
    # scheme's own colour, so the legend shows exactly those markers.
    ax.scatter([], [], s=30, marker="o", color=COL_TEB, label="TEB")
    ax.scatter([], [], s=30, marker="^", color=COL_CLMU, label="CLMU5")
    ax.scatter([], [], s=30, marker="o", facecolors="none",
               edgecolors=COL_TEB, linewidths=1.0,
               label="TEB, flagged record (Sect. 2.1)")
    ax.scatter([], [], s=30, marker="^", facecolors="none",
               edgecolors=COL_CLMU, linewidths=1.0,
               label="CLMU5, flagged record")

    # Report the slope in ALL THREE record sets, not just the full sample: the
    # full-sample CLM-Urban value is the one driven by the flagged Minneapolis
    # pair, and foregrounding it alone overstates the scheme difference.
    VAR = S["model_evaluation"]["variants"]
    # v27: the conservative core leads (supervisor A-5); SUEWS removed (A-4).
    AOD = S["amplitude_ordering_decomposition"]
    lines = ["slope (r)        TEB      CLMU5"]
    for lab, key in (("core (16)", "core16"), ("excl. Mpls (17)", "excl_mpls17"), ("all 19", "all19")):
        t, c = AOD[key]["TEB"], AOD[key]["CLMU5"]
        lines.append(f"{lab:<16s}{t['slope']:.2f} ({t['r']:.2f})"
                     f"  {c['slope']:.2f} ({c['r']:.2f})")
    ax.text(0.04, 0.96, "\n".join(lines),
            transform=ax.transAxes, fontsize=6.4, va="top", ha="left",
            color=C["dark"], family="monospace")
    ax.text(0.04, 0.70, "both schemes compressed\nrelative to 1:1 in every\nrecord set",
            transform=ax.transAxes, fontsize=7.2, va="top", ha="left",
            color=C["grey"])

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Observed $\overline{\Delta T_{s-a}}$  (K)")
    ax.set_ylabel(r"Model $\overline{\Delta T_{s-a}}$  (K)")
    ax.legend(loc="lower right", fontsize=7.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig3_model_vs_obs")


# ============================================================================
# F4 - per-site nocturnal LWup bias, TEB vs CLMU5 paired, ordered by TEB bias
# ============================================================================
def fig4():
    order = sorted_by("teb_lwup_bias")           # ascending TEB bias
    y = np.arange(len(order))
    h = 0.38

    fig, ax = plt.subplots(figsize=(ps.W_DOUBLE, 5.0))
    ax.axvline(0.0, color=C["dark"], lw=0.6, zorder=2)
    for yi, s in zip(y, order):
        flag = s in HMISS
        fov = (s == "PL-Lipowa")
        edge = COL_FLAG if flag else (C["dark"] if fov else "none")
        lw = 0.9 if (flag or fov) else 0.0
        ax.barh(yi + h / 2, PS[s]["teb_lwup_bias"], height=h,
                color=COL_TEB, edgecolor=edge, linewidth=lw,
                hatch=("////" if flag else None), zorder=3)
        ax.barh(yi - h / 2, PS[s]["clmu_lwup_bias"], height=h,
                color=COL_CLMU, edgecolor=edge, linewidth=lw,
                hatch=("////" if flag else None), zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([short(s) for s in order])
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlabel(r"Nocturnal LW$\uparrow$ bias  (model $-$ obs, W m$^{-2}$)")
    handles = [
        mpatches.Patch(color=COL_TEB, label="TEB"),
        mpatches.Patch(color=COL_CLMU, label="CLMU5"),
        mpatches.Patch(facecolor="white", edgecolor=COL_FLAG, hatch="////",
                       label="height-mismatch flag"),
        mpatches.Patch(facecolor="white", edgecolor=C["dark"],
                       label="PL-Lipowa (FOV caveat)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig4_lwup_bias_paired")


# ============================================================================
# F5 - elimination: material envelopes (both models) + structural surgery
# ============================================================================
def fig5():
    """v27: material envelopes only; the structural-probe panel of v26 is
    described in Supplementary Sect. S1 (supervisor B-3)."""
    fig, axL = plt.subplots(figsize=(ps.W_SINGLE + 0.9, 2.9))
    teb_env = S["elimination"]["teb_material_envelope"]["per_site"]
    clmu_env = S["elimination"]["clmu_material_envelope"]["per_site"]
    sites = list(dict.fromkeys(list(teb_env.keys()) + list(clmu_env.keys())))
    def base_of(s):
        if s in teb_env:
            return teb_env[s]["base"]
        return clmu_env[s]["base"]
    sites = sorted(sites, key=base_of)
    y = np.arange(len(sites))
    off = 0.16
    for yi, s in zip(y, sites):
        if s in teb_env:
            e = teb_env[s]
            axL.plot([e["vmin"], e["vmax"]], [yi + off, yi + off], color=COL_TEB, lw=1.2, zorder=3)
            axL.scatter(e["base"], yi + off, s=22, color=COL_TEB, zorder=4)
        if s in clmu_env:
            e = clmu_env[s]
            axL.plot([e["vmin"], e["vmax"]], [yi - off, yi - off], color=COL_CLMU, lw=1.2, zorder=3)
            axL.scatter(e["base"], yi - off, s=22, color=COL_CLMU, zorder=4)
    axL.axvline(0.0, color=C["dark"], lw=0.6, zorder=2)
    axL.set_yticks(y)
    axL.set_yticklabels(sites)
    axL.set_ylim(-0.7, len(sites) - 0.3)
    axL.set_xlabel(r"Nocturnal LW$\uparrow$ bias  (W m$^{-2}$)")
    tr = S["elimination"]["teb_material_envelope"]["range_Wm2"]
    cr = S["elimination"]["clmu_material_envelope"]["range_Wm2"]
    axL.text(0.03, 0.62,
             f"material envelope (peak-to-peak)\nTEB {tr[0]:.1f}–{tr[1]:.1f} "
             f"| CLMU5 {cr[0]:.1f}–{cr[1]:.1f} W m$^{{-2}}$\n"
             "no sign reversal among tested\none-at-a-time perturbations",
             transform=axL.transAxes, fontsize=7.0, va="top", ha="left", color=C["dark"])
    handles = [mpatches.Patch(color=COL_TEB, label="TEB envelope"),
               mpatches.Patch(color=COL_CLMU, label="CLMU5 envelope")]
    axL.legend(handles=handles, loc="upper left", fontsize=7.2)
    for sp in ("top", "right"):
        axL.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig5_elimination")


# ============================================================================
# F6 - unified LOCO correction analysis (exact dTsa target, city folds)
# ============================================================================
def fig6():
    U = S["review4_additions"]["unified_correction_LOCO_dtsa"]["variants"]
    fig, axes = plt.subplots(1, 2, figsize=(ps.W_DOUBLE, 3.0), sharey=True)
    for ax, (vkey, vlab) in zip(
            axes, [("all19", "all records (n=19, 16 city folds)"),
                   ("excl_mpls17", "excl. Minneapolis (n=17, 15 folds)")]):
        row = U[vkey]
        clim = row["obs_albedo_climatology"]["locomae_K"]
        x = np.arange(2)
        w = 0.26
        for k, (mkey, col, lab) in enumerate([
                ("raw_mae_K", C["grey"], "no correction"),
                ("intercept_only_locomae_K", "#b8b8b8", "intercept-only LOCO"),
                ("albedo_locomae_K", COL_TEB, "albedo-conditioned LOCO")]):
            vals = [row["TEB"][mkey], row["CLMU5"][mkey]]
            ax.bar(x + (k - 1) * w, vals, width=w, color=col, zorder=3,
                   label=lab if vkey == "all19" else None)
        for i, sch in enumerate(("TEB", "CLMU5")):
            r = row[sch]
            lo, hi = r["mae_improvement_ci95_cluster"]
            ax.text(i, max(r["raw_mae_K"], 2.0) + 0.10,
                    f"$R^2_{{LOCO}}$={r['r2_loco']:.2f}\n"
                    f"$\\Delta$MAE CI [{lo:.2f},{hi:.2f}]",
                    ha="center", va="bottom", fontsize=7.4, color=C["dark"])
        ax.axhline(clim, color=COL_FLAG, lw=1.2, ls="--", zorder=4)
        ax.text(1.45, clim + 0.03, "no-model albedo\nclimatology",
                fontsize=7.4, color=COL_FLAG, va="bottom", ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels(["TEB", "CLMU5"])
        ps.ygrid(ax)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(r"LOCO mean |$\overline{\Delta T_{s-a}}$ error|  (K)")
    axes[0].set_ylim(0, 2.75)
    hnd, lab = axes[0].get_legend_handles_labels()
    fig.legend(hnd, lab, loc="upper center", ncol=3, fontsize=7.2,
               frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig6_prescription")


# ============================================================================
# F7 - observation-error scenarios on the model slope (v27, supervisor A-1)
# ============================================================================
def fig7():
    OE = S["review5_additions"]["obs_error_propagated_to_model"]
    base = OE["baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(ps.W_DOUBLE, 2.8), sharey=True)
    SETS = [("core16", "core (16)", 1.0, "-"), ("all19", "all 19", 0.45, "--")]
    # left: albedo-aligned +/-B step field (deterministic; 100 000 cluster resamples)
    ax = axes[0]
    al = OE["scenarios"]["aligned_step"]["variants"]
    for vk, vlab, alpha, ls in SETS:
        amps = al[vk]["amplitudes"]
        Bs = [0.0] + [amps[k]["B_K"] for k in amps]
        for sch, col, mk in (("TEB", COL_TEB, "o"), ("CLMU5", COL_CLMU, "^")):
            med = [base[vk][sch]["bootstrap_100k"]["beta_median"]] + [amps[k][sch]["bootstrap_100k"]["beta_median"] for k in amps]
            lo = [base[vk][sch]["bootstrap_100k"]["beta_ci95"][0]] + [amps[k][sch]["bootstrap_100k"]["beta_ci95"][0] for k in amps]
            hi = [base[vk][sch]["bootstrap_100k"]["beta_ci95"][1]] + [amps[k][sch]["bootstrap_100k"]["beta_ci95"][1] for k in amps]
            ax.plot(Bs, med, color=col, marker=mk, ms=3.5, lw=1.1, ls=ls, alpha=alpha,
                    label=f"{'TEB' if sch=='TEB' else 'CLMU5'}, {vlab}", zorder=3)
            if vk == "core16":
                ax.fill_between(Bs, lo, hi, color=col, alpha=0.13, lw=0, zorder=2)
    ax.axhline(1.0, color=C["dark"], lw=0.7, ls=":", zorder=1)
    ax.text(1.0, 1.03, "unity slope (1:1)", fontsize=6.8, color=C["dark"], va="bottom", ha="right")
    ax.set_xlabel("albedo-aligned bias amplitude $B$  (K)")
    ax.set_ylabel(r"observed-aligned slope $\hat\beta_{m|o}$")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(fontsize=6.6, loc="lower left", frameon=False, ncol=2)
    # right: independent random error (error-only draws; medians of 50 000)
    ax = axes[1]
    rn = OE["scenarios"]["random_independent"]["variants"]
    for vk, vlab, alpha, ls in SETS:
        levs = rn[vk]
        sig = [0.0] + [levs[k]["TEB"]["error_only"].get("sigma_Wm2", levs[k].get("sigma_Wm2")) for k in levs]
        for sch, col, mk in (("TEB", COL_TEB, "o"), ("CLMU5", COL_CLMU, "^")):
            med = [base[vk][sch]["point"]["beta"]] + [levs[k][sch]["error_only"]["beta"]["median"] for k in levs]
            lo = [base[vk][sch]["point"]["beta"]] + [levs[k][sch]["error_only"]["beta"]["ci95"][0] for k in levs]
            hi = [base[vk][sch]["point"]["beta"]] + [levs[k][sch]["error_only"]["beta"]["ci95"][1] for k in levs]
            ax.plot(sig, med, color=col, marker=mk, ms=3.5, lw=1.1, ls=ls, alpha=alpha, zorder=3)
            if vk == "core16":
                ax.fill_between(sig, lo, hi, color=col, alpha=0.13, lw=0, zorder=2)
    ax.axhline(1.0, color=C["dark"], lw=0.7, ls=":", zorder=1)
    ax.set_xlabel(r"random LW$\uparrow$ error $\sigma_E$  (W m$^{-2}$)")
    ax.set_xticks([0, 2, 4, 5, 10])
    for a_ in axes:
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        ps.ygrid(a_)
    axes[0].set_ylim(-0.6, 1.4)
    fig.tight_layout()
    save(fig, "fig7_scenarios")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    print("wrote:", sorted(p.name for p in FIGDIR.glob("fig*.p*")))
