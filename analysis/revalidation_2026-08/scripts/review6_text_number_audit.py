# -*- coding: utf-8 -*-
"""review6_text_number_audit.py -- recompute every number the v12 body text
prints, straight from results/paper_stats_v1.json (plus the two unmerged
companions it legitimately depends on), and emit the authoritative value with
its key path.

READ-ONLY with respect to the manuscript and to paper_stats_v1.json.  It writes
one new companion, results/review6_text_number_audit.json, and nothing else.

Why this exists
---------------
Round 5 regenerated the CLMU output (alignment fix), the constrained
experiments, the paired scheme bootstrap and the LOCO correction.  Several body
sentences still print pre-regeneration values, and three aggregate claims were
never recomputed after Table 1 was rebuilt.  This script is the mechanical
check: for each audited quantity it stores {claim_in_v12, authoritative,
printed_form, key_path, verdict} so the next revision can be diffed against it.

Usage:  python review6_text_number_audit.py
"""
import io
import itertools
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STATS = os.path.join(ROOT, "results", "paper_stats_v1.json")
CLMU_REBUILD = os.path.join(ROOT, "results", "clmu_v11_rebuild.json")
SURGERY = os.path.join(ROOT, "external", "dslucm_surgery", "results.json")
OUT = os.path.join(ROOT, "results", "review6_text_number_audit.json")


def load(p):
    return json.load(io.open(p, encoding="utf-8"))


def rec(bag, name, line, claim, auth, printed, key, verdict, note=""):
    bag[name] = dict(main_tex_line=line, claim_in_v12=claim,
                     authoritative=auth, printed_form=printed,
                     key_path=key, verdict=verdict, note=note)


def main():
    S = load(STATS)
    ps = S["per_site"]
    sites = list(ps.keys())
    n = np.array([ps[s]["n_night"] for s in sites], float)
    teb_b = np.array([ps[s]["teb_lwup_bias"] for s in sites])
    clm_b = np.array([ps[s]["clmu_lwup_bias"] for s in sites])
    alb = np.array([ps[s]["albedo"] for s in sites])
    dt = np.array([ps[s]["dtsa_std"] for s in sites])
    qh = np.array([ps[s]["obs_qh_qc0"] for s in sites])
    cl = np.array([ps[s]["cluster"] for s in sites])

    A = {}

    # ---- (i) Table 1 bias aggregates -----------------------------------
    rec(A, "clmu_record_equal_mean_bias_Wm2", 811, -2.8,
        float(clm_b.mean()), "-2.9",
        "review3_additions.bias_aggregates_Wm2.record_equal_mean.CLMU5",
        "TEXT_STALE", "equals mean(per_site.*.clmu_lwup_bias) exactly")
    rec(A, "clmu_sample_weighted_mean_bias_Wm2", 811, -3.8,
        float((clm_b * n).sum() / n.sum()), "-3.9",
        "review3_additions.bias_aggregates_Wm2.sample_weighted_mean.CLMU5",
        "TEXT_STALE")
    rec(A, "clmu_mean_abs_bias_Wm2", 814, 7.6,
        float(np.abs(clm_b).mean()), "7.7",
        "level_vs_gradient.CLMU5.mean_abs_LWup_bias_Wm2", "TEXT_STALE")
    rec(A, "teb_mean_abs_bias_Wm2", 814, 10.8,
        float(np.abs(teb_b).mean()), "10.8",
        "level_vs_gradient.TEB.mean_abs_LWup_bias_Wm2", "TEXT_OK")
    rec(A, "teb_record_equal_mean_bias_Wm2", 810, 9.2,
        float(teb_b.mean()), "+9.2",
        "review3_additions.bias_aggregates_Wm2.record_equal_mean.TEB", "TEXT_OK")
    rec(A, "teb_sample_weighted_mean_bias_Wm2", 810, 6.8,
        float((teb_b * n).sum() / n.sum()), "+6.8",
        "review3_additions.bias_aggregates_Wm2.sample_weighted_mean.TEB",
        "TEXT_OK")

    # ---- (ii) Fig. 6 / Sect. 3.4 LOCO numbers --------------------------
    U = S["review4_additions"]["unified_correction_LOCO_dtsa"]["variants"]
    for vs, line, claim in (("all19", 997, [-0.17, 0.73]),
                            ("excl_mpls17", 999, [-0.28, 0.61])):
        ci = U[vs]["CLMU5"]["mae_improvement_ci95_cluster"]
        rec(A, "loco_clmu_mae_improvement_ci95_" + vs, line, claim,
            [float(ci[0]), float(ci[1])], "[%.2f,%.2f]" % (ci[0], ci[1]),
            "review4_additions.unified_correction_LOCO_dtsa.variants.%s."
            "CLMU5.mae_improvement_ci95_cluster" % vs,
            "TEXT_STALE_FIGURE_RIGHT",
            "figures/make_figures.py:fig6 formats this same leaf with :.2f")
    for vs, line, claim, key in (
            ("all19", 995, 1.55, "raw_mae_K"),
            ("all19", 995, 1.23, "albedo_locomae_K"),
            ("excl_mpls17", 1010, 1.19, "albedo_locomae_K"),
            ("excl_mpls17", 1010, 1.14, "intercept_only_locomae_K")):
        v = U[vs]["CLMU5"][key]
        rec(A, "loco_clmu_%s_%s" % (key, vs), line, claim, float(v),
            "%.2f" % v,
            "review4_additions.unified_correction_LOCO_dtsa.variants.%s."
            "CLMU5.%s" % (vs, key),
            "TEXT_OK" if abs(round(v, 2) - claim) < 5e-3 else "TEXT_STALE",
            "v12 value traces to the superseded review3b_additions."
            "correction_baselines block")

    # ---- (iii) P(beta>=1) over EVERY scheme x record set x analysis -----
    cells = {}
    for blk, node0 in (
            ("scheme_ordering_paired",
             S["review5_additions"]["scheme_ordering_paired"]),
            ("core15_sensitivity.model_side",
             S["review5_additions"]["core15_sensitivity"]["model_side"])):
        for rs, node in node0.items():
            if not isinstance(node, dict) or "one_sided" not in node:
                continue
            for scheme in ("TEB", "CLMU5"):
                cells["%s.%s.%s" % (blk, rs, scheme)] = \
                    node["one_sided"]["%s_P_beta_ge_1" % scheme]
    worst_key = max(cells, key=cells.get)
    six = {k: v for k, v in cells.items()
           if k.startswith("scheme_ordering_paired")}
    worst_six = max(six, key=six.get)
    A["P_beta_ge_1_all_cells"] = dict(
        main_tex_line=[51, 806, 1238],
        claim_in_v12="<=0.002 (abstract 'everywhere', Conclusions 'throughout');"
                     " Table 2 caption says <=0.001 'in all six cells'",
        cells=cells,
        max_over_all_cells=dict(key=worst_key, value=cells[worst_key]),
        max_over_six_headline_cells=dict(key=worst_six, value=six[worst_six]),
        printed_form="<=0.004 everywhere; <=0.0014 over the six headline cells",
        key_path="review5_additions.core15_sensitivity.model_side.core15."
                 "one_sided.TEB_P_beta_ge_1",
        verdict="BOUND_TOO_TIGHT: 0.00382 > 0.002 in core15; 0.00131 > 0.001 "
                "in core16, so Table 2's <=0.001 also fails")

    # ---- (v) clear-calm: the DIFFERENCE is what is invariant ------------
    ccb = (S["review5_additions"]["obs_uncertainty_v11"]
            ["static_bias_cancellation"]["clear_calm_contrast"])
    neg = [s for s in sites if ps[s]["dtsa_std"] < 0]
    cc = np.array([ccb["per_record"][s]["dtsa_clear_calm_K"] for s in neg])
    cw = np.array([ccb["per_record"][s]["dtsa_cloudy_windy_K"] for s in neg])
    A["clear_calm_invariant_quantity"] = dict(
        main_tex_line=[595, 757],
        claim_in_v12="'-2.60 K ... against -0.90 K ... at every bias amplitude'",
        n_negative_offset_records=len(neg),
        mean_clear_calm_K=float(cc.mean()),
        mean_cloudy_windy_K=float(cw.mean()),
        invariant_contrast_K=float(cc.mean() - cw.mean()),
        key_path="review5_additions.obs_uncertainty_v11."
                 "static_bias_cancellation.clear_calm_contrast.summary."
                 "negative_offset_group.mean_contrast_K",
        verdict="WORDING: only the difference (-1.6986 K) is bias-invariant; "
                "the two composite means are not")
    frac9 = np.array([abs(ccb["per_record"][s]["residual_K_if_bias_is_5Wm2"]
                          / ccb["per_record"][s]["contrast_K"]) for s in neg])
    frac19 = np.array([abs(ccb["per_record"][s]["residual_K_if_bias_is_5Wm2"]
                           / ccb["per_record"][s]["contrast_K"]) for s in sites])
    A["clear_calm_residual_fraction_pct"] = dict(
        main_tex_line=600,
        claim_in_v12="median 2.0% ... at most 4.9% among those nine",
        median_over_nine_pct=float(np.median(frac9) * 100),
        max_over_nine_pct=float(frac9.max() * 100),
        median_over_all19_pct=float(np.median(frac19) * 100),
        key_path="review5_additions.obs_uncertainty_v11."
                 "static_bias_cancellation.clear_calm_contrast.per_record."
                 "<SITE>.residual_K_if_bias_is_5Wm2",
        verdict="SCOPE: the 2.0% median is the all-19 median; over the nine it "
                "is 1.8%")

    # ---- (vi) Lipowa vs Narutowicza, by regime -------------------------
    nr = (S["review5_additions"]["temporal_sampling"]
           ["narutowicza_regime_step"]["magnitude"])
    A["lipowa_minus_narutowicza_by_regime_K"] = dict(
        main_tex_line=[1196, 1199],
        claim_in_v12="'+1.19 vs +6.12 K, confirming that the warm extreme is a "
                     "site/footprint feature'",
        published_full_record=float(ps["PL-Lipowa"]["dtsa_std"]
                                    - nr["dtsa_full_record"]),
        narutowicza_early_regime=float(nr["lipowa_early"]
                                       - nr["dtsa_early_regime"]),
        narutowicza_late_regime=float(nr["lipowa_late"]
                                      - nr["dtsa_late_regime"]),
        narutowicza_dtsa_early_K=float(nr["dtsa_early_regime"]),
        narutowicza_dtsa_late_K=float(nr["dtsa_late_regime"]),
        lipowa_step_across_same_date_K=float(nr["lipowa_step_K"]),
        n_early_steps=nr["n_early_steps"], n_late_steps=nr["n_late_steps"],
        key_path="review5_additions.temporal_sampling."
                 "narutowicza_regime_step.magnitude",
        verdict="OVERCLAIM: the sign survives in both regimes but the gap is "
                "7.54 K on the early regime and only 1.43 K on the late "
                "(post-break) regime, which the package's own attribution "
                "argument calls the instrumentally credible one")

    # ---- sweep ---------------------------------------------------------
    sw = {}
    CR = load(CLMU_REBUILD)
    dd = {s: v["d_dtsa"] for s, v in CR["control"].items()}
    big = {s: v for s, v in dd.items() if abs(v) > 0.05}
    sw["clmu_baseline_reproduction_spread"] = dict(
        main_tex_line=364,
        claim_in_v12="within 0.05 K at 18 records and 0.26 K at one",
        n_within_0p05=int(sum(1 for v in dd.values() if abs(v) <= 0.05)),
        records_outside_0p05={k: float(v) for k, v in big.items()},
        max_abs_K=float(max(abs(v) for v in dd.values())),
        key_path="clmu_constrained_experiments.baseline_reproduction."
                 "changed_records (= results/clmu_v11_rebuild.json "
                 "control.*.d_dtsa)",
        verdict="WRONG: 17 within 0.05 K, TWO outside (UK-KingsCollege "
                "0.1425 K, GR-HECKOR 0.2467 K); the single-record framing and "
                "the 0.26 figure are both incorrect")
    sw["clmu_archived_vs_v11_slope"] = dict(
        main_tex_line=365,
        claim_in_v12="fresh baseline slope of 0.11 indistinguishable from the "
                     "archived 0.10",
        slope_archived=CR["slopes"]["all19"]["slope_archived"],
        slope_v11=CR["slopes"]["all19"]["slope_v11"],
        key_path="results/clmu_v11_rebuild.json slopes.all19",
        verdict="TEXT_OK (but the source leaf is NOT merged into "
                "paper_stats_v1.json)")
    sw["clmu_model_dtsa_span_K"] = dict(
        main_tex_line=789,
        claim_in_v12="span -2.5 to +0.7 K",
        span=[float(min(ps[s]["clmu_dtsa"] for s in sites)),
              float(max(ps[s]["clmu_dtsa"] for s in sites))],
        argmin=min(sites, key=lambda s: ps[s]["clmu_dtsa"]),
        printed_form="-2.8 to +0.7",
        key_path="model_evaluation.variants.all.CLMU5.model_span_min/max "
                 "(= per_site.*.clmu_dtsa)",
        verdict="STALE PRE-ALIGNMENT VALUE: -2.5048 is GR-HECKOR's ARCHIVED "
                "dTsa; the alignment-corrected value is -2.7515")
    i13 = [k for k, s in enumerate(sites) if ps[s]["in_g2"]]
    sw["cross_scheme_bias_corr_13_non_gate"] = dict(
        main_tex_line=829,
        claim_in_v12="r=-0.11, 95% cluster-bootstrap CI [-0.71,+0.51]",
        r=float(np.corrcoef(teb_b[i13], clm_b[i13])[0, 1]),
        ci95=(S["review2_additions"]["cross_scheme_decorrelation"]
               ["unseen13"]["r_ci95_cluster_bootstrap"]),
        printed_form="r=-0.14, CI [-0.73,+0.48]",
        key_path="review2_additions.cross_scheme_decorrelation.unseen13",
        verdict="TEXT_STALE")
    sw["clmu_bias_albedo_corr"] = dict(
        main_tex_line=827, claim_in_v12="+0.72",
        value=float(np.corrcoef(clm_b, alb)[0, 1]), printed_form="+0.71",
        key_path="model_evaluation.variants.all.CLMU5.r_bias_albedo",
        verdict="TEXT_STALE")
    fc = S["review3_additions"]["forcing_covariates"]["covariates"]
    sw["tair_covariate_spearman"] = dict(
        main_tex_line=703, claim_in_v12="+0.02",
        value=fc["tair"]["spearman_rho"], printed_form="+0.01",
        key_path="review3_additions.forcing_covariates.covariates.tair."
                 "spearman_rho",
        verdict="TEXT_STALE")
    ii = S["independent_instrument"]["all"]
    sw["dtsa_qh_spearman_p"] = dict(
        main_tex_line=762, claim_in_v12=0.028,
        value=ii["spearman_p"], printed_form="0.027",
        key_path="independent_instrument.all.spearman_p", verdict="TEXT_STALE")

    # Qh under the adversarial field (recomputed: not stored as a leaf)
    sgn = np.sign(alb - alb.mean())
    rq = stats.rankdata(qh)
    best, worst = -9.0, 9.0
    verts = np.array(list(itertools.product([-1.0, 1.0], repeat=len(sites))))
    for blk in np.array_split(verts, 32):
        pert = dt[None, :] + blk
        rk = np.apply_along_axis(stats.rankdata, 1, pert)
        rkc = rk - rk.mean(axis=1, keepdims=True)
        rqc = rq - rq.mean()
        r = (rkc @ rqc) / np.sqrt((rkc ** 2).sum(axis=1) * (rqc ** 2).sum())
        best = max(best, float(r.max()))
        worst = min(worst, float(r.min()))
    sw["dtsa_qh_spearman_under_aligned_1K"] = dict(
        main_tex_line=612,
        claim_in_v12="from +0.51 to +0.33; box range [-0.06,+0.80]",
        baseline=float(stats.spearmanr(dt, qh).statistic),
        aligned_1K=float(stats.spearmanr(dt + sgn, qh).statistic),
        box_range_1K=[worst, best],
        key_path="recomputed here from per_site.*.dtsa_std/obs_qh_qc0/albedo "
                 "(no stored leaf)",
        verdict="TEXT_OK")

    SG = load(SURGERY)["summary"]
    lip = {k: v["lipowa"] for k, v in SG.items()}
    sw["dslucm_best_lipowa_K"] = dict(
        main_tex_line=910, claim_in_v12=-0.56,
        per_configuration=lip,
        best_over_configurations=dict(config=max(lip, key=lip.get),
                                      value=lip[max(lip, key=lip.get)]),
        value_of_best_overall_config=lip["S-AB"],
        key_path="external/dslucm_surgery/results.json summary.<CONFIG>.lipowa",
        verdict="AMBIGUOUS: -0.56 is S-AB's value, but the closest ANY "
                "configuration comes to +6.12 K is S-A1 at -0.15 K")
    loco = {}
    for c in sorted(set(cl)):
        m = cl != c
        loco[str(c)] = float(np.corrcoef(alb[m], dt[m])[0, 1])
    sw["leave_one_city_out_r"] = dict(
        main_tex_line=533, claim_in_v12="r <= -0.71",
        per_city=loco, least_negative=max(loco.values()),
        least_negative_city=max(loco, key=loco.get),
        key_path="recomputed from per_site.*.albedo/dtsa_std",
        verdict="INEQUALITY_FAILS_BY_0.0007: least negative is -0.70934, which "
                "is NOT <= -0.71 although it rounds to it")

    payload = dict(
        meta=dict(
            script="analysis/revalidation_2026-08/scripts/"
                   "review6_text_number_audit.py",
            purpose="authoritative value + key path for every v12 body number "
                    "an external reviewer flagged, plus a full sweep",
            reads_only=["results/paper_stats_v1.json",
                        "results/clmu_v11_rebuild.json",
                        "external/dslucm_surgery/results.json"],
            writes=["results/review6_text_number_audit.json"],
            constants=dict(SIG=S["meta"]["sigma"], EPS=S["meta"]["eps"],
                           K_per_Wm2=S["meta"]["K_per_Wm2"]),
            note="paper_stats_v1.json is NOT modified; this is a companion."),
        reviewer_items_i_to_vii=A,
        sweep=sw)
    io.open(OUT, "w", encoding="utf-8").write(
        json.dumps(payload, indent=1, ensure_ascii=False))
    print("wrote", OUT)
    for name, blk in list(A.items()) + list(sw.items()):
        print("  %-42s %s" % (name, blk.get("verdict", "")))


if __name__ == "__main__":
    main()
