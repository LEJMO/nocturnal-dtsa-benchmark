# -*- coding: utf-8 -*-
"""Round-5 additions, task A: literature-calibrated observation-uncertainty
propagation for the albedo--dTsa association (external review of v10, 2026-09-16).

REPLACES `obs_uncertainty_sensitivity` (written by v10_tables.py). That block was
defective in four ways:
  (1) it called 5 W/m^2 "a representative field pyrgeometer accuracy" without
      disclosing that 5 W/m^2 = 1.00 K at the paper's own 0.2005 K/(W/m^2);
  (2) its "systematic_FOV_*" scenario was a ONE-SIDED ramp, bias_i =
      B*(alb_max - alb_i)/(alb_max - alb_min) >= 0, SUBTRACTED from the offset,
      i.e. it COOLED the low-albedo records; the manuscript prose described it as
      "warming the dense, low-albedo records";
  (3) it was computed for all19 only -- never for the conservative core, which is
      the record set the paper's own framing treats as primary;
  (4) it licensed the claim that the association "is not an artefact of plausible
      instrument or composite-footprint error", which the albedo-ALIGNED error
      result refutes (core16 reaches r = +0.03 at an aligned amplitude of 1 K).

Everything here is OBSERVATION-SIDE (dtsa_std, albedo, obs_qh_qc0, raw corpus
LWup/Tair/LWdown/Wind) or literature. NOTHING depends on per_site.clmu_dtsa or
per_site.clmu_lwup_bias, so this block is unaffected by the concurrent CLMU
output-alignment fix.

Output: results/review5_adderley-propagation.json  (NOT paper_stats_v1.json; the
lead merges it under review5_additions.obs_uncertainty_v11 and supersedes
`obs_uncertainty_sensitivity`).

Scenario families
  (i)   random independent Gaussian per-record LW_up error, sigma_E in
        {2,4,5,10} W/m^2, 50 000 draws, INDEPENDENTLY SEEDED per (scenario,
        record set) so no result depends on evaluation order;
  (ii)  as (i) but US-Minneapolis1/2 SHARE one error draw (one radiometer, one
        tower, identical dtsa_std to 16 significant figures);
  (iii) Adderley et al. (2015) placement error: per-record sigma from the record's
        own z_rad/H by an exponential fit to their published 11.2/6.3/2.0 W/m^2 at
        z/H = 2/3/5, run CAPPED (at 11.2) and UNCAPPED (Minneapolis gets the value
        its z_rad/H = 0.396 implies), plus flat 11.2 and flat 9.4 W/m^2;
  (iiib) common-mode multiplicative calibration-scale error p% of each record's own
        mean nocturnal LW_up (the form that manufacturer "% of reading" specs
        actually take), same-sign and random-sign;
  (iv)  albedo-ALIGNED step family delta_i = B*sign(alb_i - mean(alb));
  (v)   UNRESTRICTED worst case over the box |delta_i| <= B (exhaustive 2^n vertex
        enumeration + bounded interior search to confirm no interior optimum);
  (vi)  breakdown amplitude B*: smallest amplitude pushing r above -0.5/-0.4/-0.3/0.0
        for both families, in K, in W/m^2, and as peak-to-peak.

Defensive analysis (section `static_bias_cancellation`)
  (a) a static per-record bias cancels EXACTLY in the within-site clear--calm vs
      cloudy--windy contrast (the split is a per-site median split on eps_sky and
      wind only -- verified against conditioning_analysis.py:119-127), and the
      residual from specifying the bias in W/m^2 rather than K is computed exactly
      from the raw corpus;
  (b) the eddy-covariance Q_H values are produced by a different instrument and are
      therefore untouched by delta_i, so the adversarial field must PAY for
      flattening the albedo relation by degrading the dTsa--Q_H rank agreement;
      that cost is quantified at each breakdown amplitude.

Run:  python \
        analysis/revalidation_2026-08/scripts/review5_adderley_propagation.py
The slow part (raw-corpus pass, ~2 min) is cached in
results/.cache_review5_obs_extras.json; delete that file to force recomputation.
"""
import io
import json
import sys
import zlib
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[3]
STATS = ROOT / "results" / "paper_stats_v1.json"
OUT = ROOT / "results" / "review5_adderley-propagation.json"
CACHE = ROOT / "results" / ".cache_review5_obs_extras.json"

# ---- FROZEN CONSTANTS (must match paper_stats.py exactly) -------------------
SIG = 5.67e-8          # NOT the exact Stefan-Boltzmann value; paper_stats.py convention
EPS = 0.95
TREF = 285.0
K_PER_WM2 = 1.0 / (4.0 * EPS * SIG * TREF ** 3)      # 0.2004927923369955 K per W/m2

SEED_BASE = 20260916
NDRAW = 50_000

# ---- Adderley et al. 2015 (AMT 8, 2699-2714, doi:10.5194/amt-8-2699-2015) ---
# "The root mean squared error (RMSE) between different horizontal positions in
#  retrieving outgoing longwave emittance L_up decreased exponentially with height,
#  and was 11.2, 6.3 and 2.0 W m-2 at 2, 3, and 5 times the mean building height z_b."
ADD_ZH = np.array([2.0, 3.0, 5.0])
ADD_SIG = np.array([11.2, 6.3, 2.0])
ADD_T0_RMSE_WM2 = 9.4   # their T0,h vs T0,C RMSE: 1.7 K (9.4 W/m2)
ADD_T0_RMSE_K = 1.7

d = json.load(open(STATS, encoding="utf-8"))
P = d["per_site"]
META = d["obs_metadata"]
SITES = sorted(P)

RECORD_SETS = {
    "all19": SITES,
    "excl_mpls17": [s for s in SITES if not s.startswith("US-Minneapolis")],
    "core16": [s for s in SITES
               if not s.startswith("US-Minneapolis") and s != "PL-Lipowa"],
}
MPLS = [s for s in SITES if s.startswith("US-Minneapolis")]


def rng_for(*parts):
    """Independent, order-independent RNG stream per (scenario, record set)."""
    tag = "|".join(str(p) for p in parts).encode("utf-8")
    return np.random.default_rng(SEED_BASE + int(zlib.crc32(tag)))


def alb_off(ss):
    return (np.array([P[s]["albedo"] for s in ss], float),
            np.array([P[s]["dtsa_std"] for s in ss], float))


# =============================================================================
# 0.  raw-corpus extras (cached): mean nocturnal LW_up, clear--calm composites
# =============================================================================
def build_obs_extras():
    """Per-record mean nocturnal LW_up, and the per-site clear--calm /
    cloudy--windy dTsa composites under the VERIFIED definition
    (conditioning_analysis.py:119-127: per-site percentile thresholds on
    eps_sky = LWdown/(sigma Ta^4) and on wind; clear--calm = both below the site
    median, cloudy--windy = both above)."""
    import xarray as xr
    sys.path.insert(0, str(ROOT))
    from src.training.corpus_loader import load_all_sites   # noqa: E402
    recs = {r.site: r for r in load_all_sites()}
    ex = {}
    for site in SITES:
        rec = recs[site]
        ds = xr.open_dataset(ROOT / f"data/urban-plumber/corpus/{site}.nc")
        # xarray mask_and_scale=True (default) turns the _FillValue = -999.0
        # sentinel into NaN; the finite-mask below is what actually removes it.
        olw = np.asarray(ds["obs_LWup"].values, float)
        ta = np.asarray(ds["forcing_Tair"].values, float)
        ld = np.asarray(ds["forcing_LWdown"].values, float)
        # wind is stored as components, exactly as conditioning_analysis.py:99 and
        # review3_additions.py:64 do it
        wd = np.hypot(np.asarray(ds["forcing_Wind_N"].values, float),
                      np.asarray(ds["forcing_Wind_E"].values, float))
        ds.close()
        night = (np.asarray(rec.night_mask, bool)
                 & ~np.asarray(rec.pre_spinup_flag, bool))
        m = night & np.isfinite(olw) & np.isfinite(ta) & np.isfinite(ld)
        # belt-and-braces guard: if a file ever lost its _FillValue attribute the
        # -999.0 sentinel would survive as a finite value (the PL-Lipowa trap).
        m &= (olw > 0.0) & (olw < 900.0)
        # m is the FROZEN dTsa nocturnal mask; it must reproduce per_site.dtsa_std
        ts = ((olw[m] - (1 - EPS) * ld[m]) / (EPS * SIG)) ** 0.25
        dts = ts - ta[m]
        row = {"n_night": int(m.sum()),
               "mean_lwup_Wm2": float(olw[m].mean()),
               "mean_ts_K": float(ts.mean()),
               "dtsa_recheck": float(dts.mean())}
        # the conditioning split additionally needs finite wind
        mw = m & np.isfinite(wd)
        if mw.sum() > 100:
            tsw = ((olw[mw] - (1 - EPS) * ld[mw]) / (EPS * SIG)) ** 0.25
            dtw = tsw - ta[mw]
            eps_sky = ld[mw] / (SIG * ta[mw] ** 4)
            wind = wd[mw]
            e50, w50 = np.median(eps_sky), np.median(wind)
            cc = (eps_sky <= e50) & (wind <= w50)
            cw = (eps_sky >= e50) & (wind >= w50)
            row.update(n_split=int(mw.sum()), n_cc=int(cc.sum()), n_cw=int(cw.sum()),
                       dtsa_cc=float(dtw[cc].mean()),
                       dtsa_cw=float(dtw[cw].mean()))
            # EXACT residual of the cc-minus-cw contrast when the static bias is
            # specified in W/m^2 (constant E) rather than in K: the induced Ts
            # shift is E/(4 eps sig Ts^3) and therefore varies slightly with Ts.
            for E in (5.0, 11.2):
                ts2 = ((olw[mw] + E - (1 - EPS) * ld[mw]) / (EPS * SIG)) ** 0.25
                sh = ts2 - tsw
                row[f"contrast_residual_K_at_{E:g}Wm2"] = float(
                    sh[cc].mean() - sh[cw].mean())
                row[f"mean_induced_shift_K_at_{E:g}Wm2"] = float(sh.mean())
        ex[site] = row
        print(f"  {site:18s} n={row['n_night']:6d} LWup={row['mean_lwup_Wm2']:6.1f} "
              f"dTsa recheck={row['dtsa_recheck']:+.4f}", flush=True)
    return ex


if CACHE.exists():
    obs_extras = json.load(open(CACHE, encoding="utf-8"))
    print(f"[cache] obs extras loaded from {CACHE.name}")
else:
    print("Building raw-corpus extras (slow, ~2 min) ...", flush=True)
    obs_extras = build_obs_extras()
    json.dump(obs_extras, open(CACHE, "w", encoding="utf-8"), indent=1)
    print(f"[cache] written {CACHE.name}")

# integrity: our recomputed dTsa must reproduce per_site.dtsa_std
recheck = {s: abs(obs_extras[s]["dtsa_recheck"] - P[s]["dtsa_std"]) for s in SITES}
assert max(recheck.values()) < 1e-9, recheck

out = {
    "_merge_target": "review5_additions.obs_uncertainty_v11",
    "_supersedes": ["obs_uncertainty_sensitivity"],
    "meta": {
        "script": "analysis/revalidation_2026-08/scripts/review5_adderley_propagation.py",
        "date": "2026-09-16",
        "task": "review-5 task A: literature-calibrated observation-uncertainty "
                "propagation for the albedo--dTsa association",
        "SIG": SIG, "EPS": EPS, "TREF_K": TREF, "K_per_Wm2": K_PER_WM2,
        "seed_base": SEED_BASE,
        "seeding": "one independent np.random.default_rng per (scenario, record "
                   "set), seed = SEED_BASE + crc32('scenario|recordset'); no RNG "
                   "stream is shared across record sets, so no reported number "
                   "depends on evaluation order",
        "n_draws": NDRAW,
        "perturbation_convention": "delta_i is ADDED to the record's observed "
                                   "nocturnal mean offset dTsa_i (K). A positive "
                                   "LW_up error E_i gives delta_i = +E_i*K_per_Wm2, "
                                   "i.e. an apparently warmer surface.",
        "clmu_independence": "This block uses only observation-side quantities "
                             "(dtsa_std, albedo, obs_qh_qc0, raw corpus "
                             "LWup/Tair/LWdown/Wind) and published literature. It "
                             "contains no CLMU- or TEB-derived statistic and is "
                             "unaffected by the CLMU output-alignment fix.",
        "why_supersede": [
            "obs_uncertainty_sensitivity used sigma = 3/5/10 W/m2 with only 2000 "
            "draws and reported 5 W/m2 as 'a representative field pyrgeometer "
            "accuracy' without disclosing that it equals 1.00 K at the paper's own "
            "0.2005 K/(W/m2).",
            "Its systematic_FOV_0.5K / systematic_FOV_1.0K entries are a ONE-SIDED "
            "ramp bias_i = B*(alb_max-alb_i)/(alb_max-alb_min) >= 0 SUBTRACTED from "
            "the offset, i.e. the low-albedo records are COOLED by 0..B K and the "
            "high-albedo records are untouched; the manuscript prose describes the "
            "same scenario as 'warming the dense, low-albedo records'. The entries "
            "are therefore mislabelled relative to the prose and must be replaced, "
            "not merely supplemented.",
            "It was computed for all19 only, never for the conservative core.",
            "It licensed an exclusion claim ('not an artefact of plausible "
            "instrument or composite-footprint error') that the albedo-aligned "
            "result refutes.",
        ],
    },
    "record_sets": {k: list(v) for k, v in RECORD_SETS.items()},
}

# =============================================================================
# 1.  baseline
# =============================================================================
base = {}
for v, ss in RECORD_SETS.items():
    a, y = alb_off(ss)
    r, p = pearsonr(a, y)
    rho, pp = spearmanr(a, y)
    base[v] = {"n": len(ss),
               "n_clusters": len({P[s]["cluster"] for s in ss}),
               "pearson_r": float(r), "pearson_p": float(p),
               "spearman_rho": float(rho), "spearman_p": float(pp)}
out["baseline"] = base

# =============================================================================
# 2.  literature-calibrated error budget
# =============================================================================
lw = np.array([obs_extras[s]["mean_lwup_Wm2"] for s in SITES])
alb_all = np.array([P[s]["albedo"] for s in SITES])
off_all = np.array([P[s]["dtsa_std"] for s in SITES])
r_lw_alb, p_lw_alb = pearsonr(alb_all, lw)
rho_lw_alb, prho_lw_alb = spearmanr(alb_all, lw)

pct_rows = {}
for pct in (1.0, 2.0, 3.0, 7.0, 15.0):
    pct_rows[f"{pct:g}pct"] = {
        "at_300Wm2": {"Wm2": 300.0 * pct / 100, "K": 300.0 * pct / 100 * K_PER_WM2},
        "at_400Wm2": {"Wm2": 400.0 * pct / 100, "K": 400.0 * pct / 100 * K_PER_WM2},
        "at_corpus_mean_LWup": {
            "mean_LWup_Wm2": float(lw.mean()),
            "Wm2": float(lw.mean() * pct / 100),
            "K": float(lw.mean() * pct / 100 * K_PER_WM2)},
    }

out["literature_error_budget"] = {
    "sources": [
        {"id": "CGR4_manual",
         "citation": "Kipp & Zonen, CGR 4 Pyrgeometer Instruction Manual, manual "
                     "version 0806 (2006).",
         "url": "https://bsrn.aemet.es/manuales/kipp_manual_cgr4_1781.pdf",
         "fetched": "2026-09-16",
         "verbatim": [
             "Kipp & Zonen expects maximum uncertainty of 2% for hourly totals and "
             "1% for daily totals for the CGR 4 pyrgeometer. (Sect. 2.5, p. 12)",
             "Performance specifications (Sect. 6.2, p. 25): Uncertainty in daily "
             "total < 3 % at 95 % confidence level; Uncertainty in hourly total "
             "'Not defined'; Non-linearity < 1 % from -250 to +250 W/m2; Window "
             "heating offset < 4 W/m2 for 0 to 1000 W/m2 solar irradiance; Zero "
             "offset B < 2 W/m2 at 5 K/h temperature change; Non-stability < 1 %; "
             "Tilt error < 1 % (deviation when facing downwards); Field of view "
             "180 deg; Directional error 'Not defined - irrelevant to isotropic IR "
             "source'.",
         ],
         "relevance": "Reference-class instrument; the 1-3 % daily-total figures are "
                      "MULTIPLICATIVE, i.e. % of reading."},
        {"id": "WMO_CIMO",
         "citation": "WMO, Guide to Instruments and Methods of Observation "
                     "(WMO-No. 8), Part I, Chapter 7 'Measurement of radiation', "
                     "Sect. 7.4.3, p. I.7-22.",
         "url": "https://www.weather.gov/media/epz/mesonet/CWOP-WMO8.pdf",
         "fetched": "2026-09-16",
         "verbatim": [
             "Several recent comparisons have been made using instruments of similar "
             "manufacture in a variety of measurement configurations. These studies "
             "have indicated that, following careful calibration, fluxes measured at "
             "night agree to within 2 per cent, but in periods of high solar energy "
             "the difference between instruments may reach 13 per cent.",
             "By shading the instrument, ventilating it as recommended by ISO "
             "(1990a), and measuring the temperature of the dome and the instrument "
             "case, this discrepancy can be reduced to less than 5 per cent of the "
             "thermopile signal (approximately 15 W m-2).",
             "Sect. 7.4.1, p. I.7-20 (pyrradiometers): In situ comparisons at "
             "different sites between different designs of pyrradiometer yield "
             "results manifesting differences of up to 5 to 10 per cent under the "
             "best conditions.",
         ],
         "relevance": "The governing normative statement for our case: NIGHTTIME "
                      "agreement within 2 per cent of reading after careful "
                      "calibration. Also MULTIPLICATIVE."},
        {"id": "Philipona2001",
         "citation": "Philipona, R., Dutton, E. G., Stoffel, T., Michalsky, J., Reda, "
                     "I., Stifter, A., Wendling, P., Wood, N., Clough, S. A., Mlawer, "
                     "E. J., Anderson, G., Revercomb, H. E., and Shippert, T. R.: "
                     "Atmospheric longwave irradiance uncertainty: Pyrgeometers "
                     "compared to an absolute sky-scanning radiometer, atmospheric "
                     "emitted radiance interferometer, and radiative transfer model "
                     "calculations, J. Geophys. Res., 106(D22), 28129-28141, 2001 "
                     "(paper number 2000JD000196; doi:10.1029/2000JD000196).",
         "url": "https://www.patarnott.com/atms749/pdf/LongWaveIrradianceMeas.pdf",
         "fetched": "2026-09-16",
         "verbatim": [
             "a group of modified and field-calibrated pyrgeometers is capable of "
             "measuring hourly averages of nighttime longwave irradiance with "
             "|Max - Min| differences of ~2 W m-2, which corresponds to a precision "
             "of ~+/-1 W m-2 compared to the mean irradiance value of the group. "
             "(Sect. 8.1)",
             "Daytime measurements show average |Max - Min| differences of ~2 W m-2 "
             "larger than nighttime measurements. (Sect. 8.1)",
             "IPASRC-I shows that atmospheric longwave irradiance measurements and "
             "calculations agree very well, which indicates that state-of-the-art "
             "longwave radiometers and radiative transfer models are able to "
             "determine nighttime atmospheric longwave irradiance with an absolute "
             "uncertainty of ~+/-1.5 W m-2. (Sect. 8.2)",
             "In the early 1990s, precision and uncertainties of longwave "
             "measurements were of the order of 10%. (Sect. 9)",
             "with blackbody calibration factors determined at NOAA/CMDL and "
             "PMOD/WRC the mean irradiance measured by a group of seven or eight "
             "pyrgeometers is within ~1 W m-2 to the measurement of the absolute "
             "sky-scanning radiometer. (Sect. 8.2)",
         ],
         "relevance": "Published FIELD INTERCOMPARISON (IPASRC-I, ARM SGP, 1999; 15 "
                      "calibration-traced pyrgeometers; L_down 260-420 W/m2). Gives "
                      "the BEST ACHIEVABLE nocturnal figure, ~1-2 W/m2, for "
                      "UPWARD-facing, shaded, ventilated, field-calibrated reference "
                      "instruments at a dedicated radiation site. It is a floor, not "
                      "a realistic value for a downward-facing radiometer on a "
                      "routine urban flux tower."},
        {"id": "Hukx_IR02",
         "citation": "Hukx (formerly Hukseflux; www.hukseflux.com 301-redirects to "
                     "www.hukx.com, verified 2026-09-16), IR02 pyrgeometer user "
                     "manual (pyrgeometer with heater).",
         "url": "https://www.hukx.com/uploads/Hukx-IR02-user-manual.pdf",
         "fetched": "2026-09-16",
         "verbatim": [
             "calibration uncertainty < 7 % (k = 2) (p. 16)",
             "achievable uncertainty (95 % confidence level) daily totals +/- 15 % "
             "(Hukx's own estimate) (p. 16)",
             "Calibration uncertainty, which is larger for other than upfacing "
             "instruments; for downfacing instruments a blackbody calibration seems "
             "preferable. (p. 26)",
             "Errors due to water deposition at clear nights; these completely block "
             "the longwave irradiance exchange between pyrgeometer and may cause the "
             "signal U/S to change from a large negative value (-100 W/m2) to around "
             "0 W/m2. Water deposition at clear nights may largely be avoided by "
             "using the on-board heater of IR02. (p. 26)",
             "Errors due to instrument non-stability. This is now estimated at "
             "< +/-1 % change per year. (p. 26)",
             "Errors due to the temperature measurement T ... Required accuracy of "
             "the read-out is +/-0.2 degC, which results in around 1 W/m2 uncertainty "
             "of the irradiance measurement. (p. 26)",
             "working standard IR20 calibration at PMOD/WRC Davos ... A typical "
             "uncertainty of S is 4.2 % (k = 2). (p. 39)",
         ],
         "relevance": "Mid-class heated instrument, the looser end of the range used "
                      "at urban flux towers. Two points bear directly on our case: "
                      "(a) the manufacturer states calibration uncertainty is LARGER "
                      "for DOWNFACING instruments, which is exactly our LW_up "
                      "geometry; (b) dew/frost on clear nights is a large, "
                      "clear-night-specific error, i.e. it is CORRELATED with the "
                      "very conditions the clear--calm compositing selects."},
        {"id": "Adderley2015",
         "citation": "Adderley, C., Christen, A., and Voogt, J. A.: The effect of "
                     "radiometer placement and view on inferred directional and "
                     "hemispheric radiometric temperatures of an urban canopy, "
                     "Atmos. Meas. Tech., 8, 2699-2714, 2015, "
                     "doi:10.5194/amt-8-2699-2015.",
         "url": "https://amt.copernicus.org/articles/8/2699/2015/",
         "fetched": "2026-09-16",
         "verbatim": [
             "The root mean squared error (RMSE) between different horizontal "
             "positions in retrieving outgoing longwave emittance L_up decreased "
             "exponentially with height, and was 11.2, 6.3 and 2.0 W m-2 at 2, 3, and "
             "5 times the mean building height z_b.",
             "Generally, above 3.5 z_b the horizontal positional error is less than "
             "the typical accuracy of common pyrgeometers.",
             "However, over the course of the day, the difference between T_0,h and "
             "T_0,C shows an RMSE of 1.7 K (9.4 W m-2).",
         ],
         "relevance": "The only published, urban, height-resolved quantification of "
                      "the composite-footprint (placement) error for a downward-"
                      "facing hemispherical radiometer. Supplies both the "
                      "height-dependent sigma and the flat 9.4 W/m2 "
                      "facet-composition term."},
    ],
    "multiplicative_to_additive": {
        "note": "Every manufacturer and CIMO figure above is a PERCENTAGE OF READING, "
                "not an absolute W/m2. At nocturnal LW_up levels the conversion is "
                "linear in the record's own mean LW_up.",
        "corpus_nocturnal_mean_LWup_Wm2": {
            "per_record": {s: obs_extras[s]["mean_lwup_Wm2"] for s in SITES},
            "min": float(lw.min()), "max": float(lw.max()),
            "mean": float(lw.mean()), "median": float(np.median(lw)),
            "argmin": SITES[int(np.argmin(lw))], "argmax": SITES[int(np.argmax(lw))]},
        "percent_of_reading_table": pct_rows,
        "multiplicand_vs_albedo": {
            "question": "Does the multiplicand (each record's own mean nocturnal "
                        "LW_up) correlate with albedo? If it does, a common-mode "
                        "'% of reading' calibration-scale error is not a random "
                        "error but an albedo-ALIGNED one.",
            "pearson_r": float(r_lw_alb), "pearson_p": float(p_lw_alb),
            "spearman_rho": float(rho_lw_alb), "spearman_p": float(prho_lw_alb),
            "n": int(len(SITES)),
            "lwup_spread_Wm2": float(lw.max() - lw.min()),
            "implied_max_differential_at_2pct_K": float(
                0.02 * (lw.max() - lw.min()) * K_PER_WM2)},
        "recommended_budget": {
            "instrument_random_1sigma_Wm2": [2.0, 5.0],
            "instrument_random_1sigma_K": [2.0 * K_PER_WM2, 5.0 * K_PER_WM2],
            "basis": "CIMO 2 % of a 300-400 W/m2 nocturnal LW_up is 6-8 W/m2 at 95 % "
                     "coverage, i.e. ~3-4 W/m2 as a 1-sigma; CGR4 1-3 % daily is "
                     "3-12 W/m2 at 95 %; Philipona's ~1-2 W/m2 is the reference-site "
                     "floor. sigma_E = 2 W/m2 is therefore the optimistic floor and "
                     "sigma_E = 5 W/m2 (= 1.00 K) the realistic upper 1-sigma for a "
                     "routinely-calibrated downward-facing tower radiometer; 10 W/m2 "
                     "is retained as a stress case.",
            "placement_systematic_Wm2": "record-specific, 0.8-11.2 (Adderley fit)",
            "facet_composition_Wm2": ADD_T0_RMSE_WM2},
    },
}

# =============================================================================
# 3.  scenario machinery
# =============================================================================
def r_of(a, y, delta):
    return float(pearsonr(a, y + delta)[0])


def summarise(rs, a=None):
    rs = np.asarray(rs, float)
    return {"r_mean": float(rs.mean()), "r_median": float(np.median(rs)),
            "r_ci95": [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))],
            "r_min": float(rs.min()), "r_max": float(rs.max()),
            "frac_above_minus0p5": float((rs > -0.5).mean()),
            "frac_above_zero": float((rs > 0.0).mean()),
            "n_draws": int(rs.size)}


def mc_random(ss, sigma_vec_Wm2, tag, share_pairs=()):
    """Monte-Carlo independent Gaussian LW_up error with per-record sigma.
    share_pairs: tuples of site names that share ONE error draw."""
    a, y = alb_off(ss)
    n = len(ss)
    g = rng_for(tag, ",".join(ss))
    E = g.normal(0.0, 1.0, size=(NDRAW, n)) * sigma_vec_Wm2[None, :]
    for grp in share_pairs:
        ix = [ss.index(s) for s in grp if s in ss]
        if len(ix) > 1:
            E[:, ix] = E[:, ix[0]][:, None]
    yp = y[None, :] + E * K_PER_WM2
    ac = a - a.mean()
    ycen = yp - yp.mean(axis=1, keepdims=True)
    den = np.sqrt((ac ** 2).sum() * (ycen ** 2).sum(axis=1))
    rs = (ycen @ ac) / den
    # self-check against scipy on the first draw
    chk = abs(rs[0] - pearsonr(a, yp[0])[0])
    assert chk < 1e-10, chk
    res = summarise(rs)
    res["sigma_Wm2"] = (float(sigma_vec_Wm2[0])
                        if np.allclose(sigma_vec_Wm2, sigma_vec_Wm2[0])
                        else {s: float(v) for s, v in zip(ss, sigma_vec_Wm2)})
    res["sigma_K"] = (float(sigma_vec_Wm2[0] * K_PER_WM2)
                      if np.allclose(sigma_vec_Wm2, sigma_vec_Wm2[0])
                      else {s: float(v * K_PER_WM2) for s, v in zip(ss, sigma_vec_Wm2)})
    return res


# ---- (i) random independent ------------------------------------------------
rand = {}
for v, ss in RECORD_SETS.items():
    rand[v] = {}
    for sg in (2.0, 4.0, 5.0, 10.0):
        rand[v][f"sigma_{sg:g}Wm2"] = mc_random(
            ss, np.full(len(ss), sg), f"random|{sg:g}")
out["random_independent"] = {
    "spec": "delta_i = E_i*K_per_Wm2, E_i ~ N(0, sigma_E^2) independent across "
            "records; sigma_E in {2,4,5,10} W/m2; 50 000 draws per (sigma, record "
            "set) from an independently seeded stream.",
    "sigma_K_equivalents": {f"sigma_{sg:g}Wm2": sg * K_PER_WM2
                            for sg in (2.0, 4.0, 5.0, 10.0)},
    "variants": rand}

# ---- (ii) shared Minneapolis draw ------------------------------------------
shared = {}
for sg in (2.0, 4.0, 5.0, 10.0):
    shared[f"sigma_{sg:g}Wm2"] = mc_random(
        RECORD_SETS["all19"], np.full(19, sg), f"shared|{sg:g}",
        share_pairs=(tuple(MPLS),))
out["random_shared_minneapolis"] = {
    "spec": "as random_independent, but US-Minneapolis1 and US-Minneapolis2 draw "
            "ONE common E: they share a tower and a radiometer and their dtsa_std "
            "agree to 16 significant figures "
            f"({P['US-Minneapolis1']['dtsa_std']!r} vs "
            f"{P['US-Minneapolis2']['dtsa_std']!r}).",
    "applies_to": "all19 only -- excl_mpls17 and core16 contain no Minneapolis "
                  "record, so the variant is identical to random_independent there.",
    "dtsa_std_identical": bool(P["US-Minneapolis1"]["dtsa_std"]
                               == P["US-Minneapolis2"]["dtsa_std"]),
    "all19": shared}

# ---- (iii) Adderley-calibrated placement error -----------------------------
# Exponential fit ln(sigma) = c0 + c1*(z/H). Justified by the authors' own wording:
# the RMSE "decreased exponentially with height".
c1, c0 = np.polyfit(ADD_ZH, np.log(ADD_SIG), 1)
fit = lambda zh: float(np.exp(c0 + c1 * np.asarray(zh, float)))
fit_v = lambda zh: np.exp(c0 + c1 * np.asarray(zh, float))
pred = fit_v(ADD_ZH)
ss_res = float(((np.log(ADD_SIG) - np.log(pred)) ** 2).sum())
ss_tot = float(((np.log(ADD_SIG) - np.log(ADD_SIG).mean()) ** 2).sum())

zoh = {s: META[s]["rad_height"] / META[s]["H"] for s in SITES}
sig_unc = {s: fit(zoh[s]) for s in SITES}
sig_cap = {s: min(sig_unc[s], float(ADD_SIG.max())) for s in SITES}

add = {"fit": {
    "form": "sigma(z/H) = exp(c0 + c1*(z/H))  [W/m2]",
    "c0": float(c0), "c1": float(c1),
    "e_folding_zh": float(-1.0 / c1),
    "fitted_at_2_3_5": [float(x) for x in pred],
    "published_at_2_3_5": [float(x) for x in ADD_SIG],
    "log_residuals": [float(x) for x in (np.log(ADD_SIG) - np.log(pred))],
    "r2_in_log": float(1 - ss_res / ss_tot),
    "justification": "Adderley et al. state the RMSE 'decreased exponentially with "
                     "height', so a straight line in log(sigma) vs z/H is the "
                     "functional form the authors themselves assert. The three "
                     "published points are almost exactly collinear in log space "
                     f"(R2_log = {1 - ss_res / ss_tot:.6f}, e-folding "
                     f"{-1.0 / c1:.3f} z_b), so the interpolation adds essentially "
                     "no assumption of our own.",
    "beyond_zh_5": "The same exponential is continued for z/H > 5 (no floor). This "
                   "is an extrapolation, but an immaterial one: Adderley et al. "
                   "state that above 3.5 z_b the positional error already falls "
                   "below typical pyrgeometer accuracy, and the fit puts "
                   f"sigma(3.5) = {fit(3.5):.2f} W/m2, which is "
                   f"{100 * fit(3.5) / float(lw.mean()):.2f} % of our corpus mean "
                   "nocturnal LW_up -- i.e. inside the 1-2 % instrument band. Any "
                   "concern that extrapolating the decay makes the placement term "
                   "too small at tall towers is answered by the flat 11.2 and flat "
                   "9.4 W/m2 scenarios, which bound every record from above.",
    "implied_typical_pyrgeometer_accuracy_Wm2": fit(3.5),
    "implied_typical_pyrgeometer_accuracy_pct_of_mean_LWup": float(
        100 * fit(3.5) / lw.mean()),
    "extrapolation_warning": "US-Minneapolis1/2 have z_rad/H = "
                             f"{zoh['US-Minneapolis1']:.4f}, five times BELOW the "
                             "lowest published height (z/H = 2). The uncapped sigma "
                             f"there, {sig_unc['US-Minneapolis1']:.2f} W/m2 "
                             f"(= {sig_unc['US-Minneapolis1'] * K_PER_WM2:.2f} K), is "
                             "an extrapolation far outside the calibration range and "
                             "must be reported as such, not used as an estimate."},
    "z_rad_over_H": {s: float(zoh[s]) for s in SITES},
    "per_record_sigma_Wm2": {
        "uncapped": {s: float(sig_unc[s]) for s in SITES},
        "capped_at_11p2": {s: float(sig_cap[s]) for s in SITES}},
    "per_record_sigma_K": {
        "uncapped": {s: float(sig_unc[s] * K_PER_WM2) for s in SITES},
        "capped_at_11p2": {s: float(sig_cap[s] * K_PER_WM2) for s in SITES}},
    "variants": {}}

for v, ss in RECORD_SETS.items():
    add["variants"][v] = {
        "capped_at_11p2": mc_random(
            ss, np.array([sig_cap[s] for s in ss]), "add_cap"),
        "uncapped": mc_random(
            ss, np.array([sig_unc[s] for s in ss]), "add_unc"),
        "flat_11p2Wm2": mc_random(ss, np.full(len(ss), 11.2), "add_flat112"),
        "flat_9p4Wm2": mc_random(ss, np.full(len(ss), ADD_T0_RMSE_WM2), "add_flat94"),
    }
add["cap_is_load_bearing"] = (
    "Compare variants.all19.capped_at_11p2 with variants.all19.uncapped: the cap "
    "changes only the two Minneapolis records, and it is those two records that "
    "carry the all19 headline r (baseline -0.793 vs -0.709 without them). The "
    "uncapped result must therefore be reported alongside the capped one, not "
    "hidden behind it.")
add["flat_9p4_meaning"] = (
    f"flat_9p4Wm2 uses Adderley et al.'s T_0,h vs T_0,C RMSE, {ADD_T0_RMSE_K} K "
    f"({ADD_T0_RMSE_WM2} W/m2). That term is NOT a positional error: it is the "
    "mismatch between what a hemispherical radiometer's field of view weights and "
    "the true facet fractions, and it does not decay with height. It is applied "
    "flat to every record for that reason.")
out["adderley_placement"] = add

# ---- (iiib) common-mode multiplicative scale error -------------------------
scale = {}
for v, ss in RECORD_SETS.items():
    a, y = alb_off(ss)
    lwv = np.array([obs_extras[s]["mean_lwup_Wm2"] for s in ss])
    row = {}
    for pct in (1.0, 2.0, 3.0):
        row[f"common_mode_plus_{pct:g}pct"] = {
            "r": r_of(a, y, +pct / 100 * lwv * K_PER_WM2),
            "max_delta_K": float((pct / 100 * lwv * K_PER_WM2).max()),
            "delta_spread_K": float(np.ptp(pct / 100 * lwv * K_PER_WM2))}
        row[f"common_mode_minus_{pct:g}pct"] = {
            "r": r_of(a, y, -pct / 100 * lwv * K_PER_WM2),
            "max_delta_K": float((pct / 100 * lwv * K_PER_WM2).max()),
            "delta_spread_K": float(np.ptp(pct / 100 * lwv * K_PER_WM2))}
        g = rng_for("scale_randsign", f"{pct:g}", ",".join(ss))
        s_sign = g.choice([-1.0, 1.0], size=(NDRAW, len(ss)))
        yp = y[None, :] + s_sign * (pct / 100 * lwv * K_PER_WM2)[None, :]
        ac = a - a.mean()
        ycen = yp - yp.mean(axis=1, keepdims=True)
        rs = (ycen @ ac) / np.sqrt((ac ** 2).sum() * (ycen ** 2).sum(axis=1))
        row[f"random_sign_{pct:g}pct"] = summarise(rs)
    scale[v] = row
out["common_mode_scale_error"] = {
    "spec": "delta_i = s * (p/100) * <LW_up>_i * K_per_Wm2, with <LW_up>_i the "
            "record's own mean nocturnal LW_up. This is the form the manufacturer "
            "and CIMO '% of reading' specifications actually take. 'common_mode' "
            "applies one sign to every record (a shared calibration-scale error); "
            "'random_sign' draws the sign per record.",
    "why_this_matters": "Because the multiplicand correlates with albedo at "
                        f"r = {r_lw_alb:+.4f} (p = {p_lw_alb:.4f}), a common-mode "
                        "percentage error is NOT a random error with respect to the "
                        "tested association -- it carries a weak albedo-aligned "
                        "component. Its magnitude is nonetheless small: the full "
                        "spread of 2 %-of-reading across the corpus is "
                        f"{0.02 * (lw.max() - lw.min()) * K_PER_WM2:.4f} K.",
    "variants": scale}

# =============================================================================
# 4.  (iv) aligned step family, (v) unrestricted box, (vi) breakdown amplitude
# =============================================================================
def vertex_tables(a, y):
    """Precompute, for every vertex s in {-1,+1}^n, the three scalars that make
    r(B) closed-form:  u = ac.s, v = yc.s, w = sum(s).
      r(B) = (C0 + B*u) / sqrt(Saa * (Syy + 2*B*v + B^2*(n - w^2/n)))
    Derivation: mean(y+Bs) = ybar + B*sbar, so (y+Bs) - mean = yc + B(s - sbar);
    sum(ac*(yc + B(s-sbar))) = C0 + B*u  because sum(ac)=0;
    sum((yc + B(s-sbar))^2) = Syy + 2B*v + B^2*(n - w^2/n)  because sum(s^2)=n."""
    n = len(a)
    ac = a - a.mean()
    yc = y - y.mean()
    C0 = float(ac @ yc)
    Saa = float(ac @ ac)
    Syy = float(yc @ yc)
    N = 1 << n
    u = np.empty(N); v = np.empty(N); w = np.empty(N, dtype=np.float64)
    bits = np.arange(n, dtype=np.int64)
    step = 1 << 16
    for lo in range(0, N, step):
        idx = np.arange(lo, min(lo + step, N), dtype=np.int64)
        S = (((idx[:, None] >> bits) & 1) * 2.0 - 1.0)
        u[lo:lo + len(idx)] = S @ ac
        v[lo:lo + len(idx)] = S @ yc
        w[lo:lo + len(idx)] = S.sum(axis=1)
    quad = n - w * w / n
    return dict(n=n, ac=ac, yc=yc, C0=C0, Saa=Saa, Syy=Syy,
                u=u, v=v, w=w, quad=quad, bits=bits)


def sup_r(T, B):
    """max over box vertices of r at amplitude B, and the argmax index."""
    num = T["C0"] + B * T["u"]
    den = np.sqrt(T["Saa"] * (T["Syy"] + 2 * B * T["v"] + B * B * T["quad"]))
    rr = num / den
    k = int(np.argmax(rr))
    return float(rr[k]), k


def sign_of(k, n):
    return np.array([1.0 if (k >> i) & 1 else -1.0 for i in range(n)])


def aligned_delta(a, B):
    return B * np.sign(a - a.mean())


TAB = {}
for v, ss in RECORD_SETS.items():
    a, y = alb_off(ss)
    TAB[v] = vertex_tables(a, y)

BGRID = (0.25, 0.5, 0.75, 1.0)

# ---- (iv) aligned step -----------------------------------------------------
aligned = {}
for v, ss in RECORD_SETS.items():
    a, y = alb_off(ss)
    sgn = np.sign(a - a.mean())
    row = {"sign_vector": {s: float(x) for s, x in zip(ss, sgn)},
            "n_plus": int((sgn > 0).sum()), "n_minus": int((sgn < 0).sum()),
            "n_zero": int((sgn == 0).sum()),
            "min_abs_albedo_minus_mean": float(np.abs(a - a.mean()).min()),
            "amplitudes": {}}
    for B in BGRID:
        dl = aligned_delta(a, B)
        row["amplitudes"][f"B_{B:g}K"] = {
            "B_K": B, "B_Wm2": B / K_PER_WM2,
            "peak_to_peak_K": float(np.ptp(dl)),
            "peak_to_peak_Wm2": float(np.ptp(dl) / K_PER_WM2),
            "r": r_of(a, y, dl),
            "delta_r_from_baseline": r_of(a, y, dl) - base[v]["pearson_r"]}
    aligned[v] = row
out["aligned_step_family"] = {
    "spec": "delta_i = B*sign(albedo_i - mean(albedo)): the high-albedo records are "
            "warmed by B and the low-albedo records cooled by B, i.e. the sign "
            "pattern that most efficiently flattens a negative albedo--offset slope. "
            "This is a saturated field: every record is displaced by exactly B, so "
            "the peak-to-peak displacement is 2B.",
    "note_vs_superseded_block": "The superseded systematic_FOV_* entries instead used "
                                "a one-sided ramp in [0, B] applied with a single "
                                "sign, which is both weaker and mislabelled.",
    "variants": aligned}

# ---- (v) unrestricted box worst case --------------------------------------
def neg_r_and_grad(delta, ac, y, sqrtSaa):
    """-r and its exact gradient. With N = ac.zc, L = ||zc||, zc the centred
    y+delta:  r = N/(sqrt(Saa)*L);  dN/ddelta_i = ac_i (because sum(ac)=0) and
    dL/ddelta_i = zc_i/L, so dr/ddelta_i = ac_i/(sqrt(Saa) L) - N zc_i/(sqrt(Saa) L^3)."""
    z = y + delta
    zc = z - z.mean()
    L = float(np.sqrt(zc @ zc))
    N = float(ac @ zc)
    r = N / (sqrtSaa * L)
    g = ac / (sqrtSaa * L) - N * zc / (sqrtSaa * L ** 3)
    return -r, -g


unres = {}
for v, ss in RECORD_SETS.items():
    a, y = alb_off(ss)
    T = TAB[v]
    row = {"n_vertices": int(1 << T["n"]), "amplitudes": {}}
    for B in BGRID:
        rv, k = sup_r(T, B)
        sv = sign_of(k, T["n"])
        dlv = B * sv
        # bounded interior search from many starts, to confirm the vertex optimum
        g = rng_for("interior", v, f"{B:g}")
        bnds = [(-B, B)] * T["n"]
        ac = a - a.mean()
        sqrtSaa = float(np.sqrt(ac @ ac))
        best_int, best_x = -np.inf, None
        starts = [np.zeros(T["n"]), dlv * 0.5, aligned_delta(a, B) * 0.5]
        starts += [g.uniform(-B, B, T["n"]) for _ in range(60)]
        for x0 in starts:
            res = minimize(neg_r_and_grad, x0, args=(ac, y, sqrtSaa), jac=True,
                           method="L-BFGS-B", bounds=bnds,
                           options={"maxiter": 5000, "ftol": 1e-16, "gtol": 1e-14})
            if -res.fun > best_int:
                best_int, best_x = float(-res.fun), res.x
        row["amplitudes"][f"B_{B:g}K"] = {
            "B_K": B, "B_Wm2": B / K_PER_WM2,
            "r_sup_vertex": rv,
            "argmax_sign_vector": {s: float(x) for s, x in zip(ss, sv)},
            "matches_albedo_aligned_sign": bool(
                np.array_equal(sv, np.sign(a - a.mean()))),
            "peak_to_peak_K": float(np.ptp(dlv)),
            "peak_to_peak_Wm2": float(np.ptp(dlv) / K_PER_WM2),
            "interior_search_best_r": best_int,
            "interior_beats_vertex": bool(best_int > rv + 1e-9),
            "interior_minus_vertex": float(best_int - rv),
            "interior_max_abs_delta_over_B": float(np.abs(best_x).max() / B),
            "n_starts": len(starts)}
    unres[v] = row
out["unrestricted_box_worstcase"] = {
    "spec": "sup of r over the box |delta_i| <= B, by exhaustive enumeration of all "
            "2^n vertices using the closed form r(B) = (C0 + B*u)/sqrt(Saa*(Syy + "
            "2B*v + B^2*(n - w^2/n))) with u = ac.s, v = yc.s, w = sum(s). Because r "
            "is a linear form over the square root of a convex quadratic, a vertex "
            "optimum is not guaranteed a priori, so a bounded interior search "
            "(L-BFGS-B, 63 starts) is run at every B and reported.",
    "interior_check_verdict": (
        "At every amplitude and every record set the bounded interior search fails "
        "to beat the best vertex; the largest excess is at the level of double "
        "rounding (<= 1.2e-16 in r). The box supremum is attained at a vertex."),
    "finding_vs_prior_audit": (
        "CORRECTION TO A BRIEFED 'GROUND TRUTH' ITEM. The prior audit recorded that "
        "the albedo-aligned step values at B = 1 K are the box supremum. This "
        "exhaustive enumeration confirms that for excl_mpls17 and core16 at B = 1 K "
        "(the argmax sign vector equals sign(albedo - mean albedo) exactly, and the "
        "two r values agree to all printed digits) but REFUTES it for all19 at "
        "B = 1 K: the supremum there is -0.4543, not the aligned -0.4825, and it is "
        "attained by flipping AU-Preston to -1 and PL-Lipowa to +1 relative to the "
        "aligned pattern. More generally the aligned step is NOT the box supremum at "
        "sub-breakdown amplitudes in any record set; the gap is largest in core16 at "
        "small B (aligned -0.7015 vs supremum -0.6374 at B = 0.25 K). The mechanism "
        "is that r = cov/(sd*sd) can be reduced either by attacking the covariance, "
        "which is what the aligned sign pattern does, or by INFLATING var(offset) at "
        "records that lie far from the fitted line, which is why the supremum warms "
        "PL-Lipowa (+6.12 K offset, below-mean albedo) instead of cooling it. The "
        "two families coincide only at the amplitude where the aligned field has "
        "driven r to the threshold on its own. Consequence for the manuscript: quote "
        "the ALIGNED B* when describing a physically interpretable bias pattern, and "
        "the UNRESTRICTED B* when stating what cannot be excluded at all; the "
        "unrestricted one is the smaller and therefore the conservative number."),
    "variants": unres}

# ---- (vi) breakdown amplitude B* ------------------------------------------
THRESH = (-0.5, -0.4, -0.3, 0.0)
BMAX = 20.0


def b_star_aligned(a, y, thr):
    """Smallest aligned amplitude with r > thr, on a 0.0002 K grid then bisection."""
    f = lambda B: r_of(a, y, aligned_delta(a, B))
    if f(0.0) > thr:
        return 0.0, True
    grid = np.arange(0.0, BMAX + 1e-9, 0.01)
    vals = np.array([f(B) for B in grid])
    mono = bool(np.all(np.diff(vals) >= -1e-12))
    hit = np.nonzero(vals > thr)[0]
    if hit.size == 0:
        return None, mono
    hi = grid[hit[0]]; lo = grid[hit[0] - 1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) > thr:
            hi = mid
        else:
            lo = mid
    return float(hi), mono


def b_star_unres(T, thr):
    f = lambda B: sup_r(T, B)[0]
    if f(0.0) > thr:
        return 0.0, True
    grid = np.arange(0.0, BMAX + 1e-9, 0.02)
    vals = np.array([f(B) for B in grid])
    mono = bool(np.all(np.diff(vals) >= -1e-12))
    hit = np.nonzero(vals > thr)[0]
    if hit.size == 0:
        return None, mono
    hi = grid[hit[0]]; lo = grid[hit[0] - 1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) > thr:
            hi = mid
        else:
            lo = mid
    return float(hi), mono


def bstar_row(B, ptp_K):
    if B is None:
        return {"B_K": None, "note": f"no amplitude below {BMAX:g} K reaches it"}
    return {"B_K": B, "B_Wm2": B / K_PER_WM2,
            "peak_to_peak_K": ptp_K, "peak_to_peak_Wm2": ptp_K / K_PER_WM2,
            "pct_of_corpus_mean_LWup": 100.0 * (B / K_PER_WM2) / float(lw.mean())}


bd = {}
for v, ss in RECORD_SETS.items():
    a, y = alb_off(ss)
    T = TAB[v]
    row = {"baseline_r": base[v]["pearson_r"], "aligned": {}, "unrestricted": {}}
    for thr in THRESH:
        Ba, mono_a = b_star_aligned(a, y, thr)
        row["aligned"][f"r_above_{thr:+.1f}"] = bstar_row(
            Ba, 2 * Ba if Ba is not None else None)
        row["aligned"][f"r_above_{thr:+.1f}"]["r_monotone_in_B"] = mono_a
        Bu, mono_u = b_star_unres(T, thr)
        if Bu is not None:
            _, k = sup_r(T, Bu)
            ptp = float(np.ptp(Bu * sign_of(k, T["n"])))
        else:
            ptp = None
        row["unrestricted"][f"r_above_{thr:+.1f}"] = bstar_row(Bu, ptp)
        row["unrestricted"][f"r_above_{thr:+.1f}"]["r_sup_monotone_in_B"] = mono_u
    bd[v] = row
out["breakdown_amplitude"] = {
    "spec": "B* = smallest amplitude at which r exceeds the threshold. Reported in K, "
            "in W/m2 (B/K_per_Wm2), as peak-to-peak (2B for a saturated step field), "
            "and as a percentage of the corpus mean nocturnal LW_up. Peak-to-peak is "
            "the quantity a reviewer will compare against an instrument "
            "specification, because a saturated step field displaces the two ends of "
            "the albedo range in opposite directions.",
    "thresholds": list(THRESH), "search_ceiling_K": BMAX,
    "variants": bd}

# =============================================================================
# 5.  defensive analysis: what a static per-record bias CANNOT touch
# =============================================================================
cc_sites = [s for s in SITES if "dtsa_cc" in obs_extras[s]]
cancel = {
    "clear_calm_contrast": {
        "definition_verified": True,
        "definition_source": "review2_additions.clear_calm_definition and "
                             "analysis/revalidation_2026-08/scripts/conditioning_analysis.py:119-127 "
                             "(eps_q = np.percentile(eps_v,[25,50,75]); wnd_q = "
                             "np.percentile(wind_v,[25,50,75]); clear_calm = "
                             "(eps_sky <= eps_q[1]) & (wind <= wnd_q[1]); cloudy_windy "
                             "= (eps_sky >= eps_q[1]) & (wind >= wnd_q[1])).",
        "definition_quoted": d["review2_additions"]["clear_calm_definition"],
        "analytic_statement":
            "The two composites are means of the SAME record's nocturnal samples over "
            "subsets selected by thresholds on eps_sky = LW_down/(sigma Ta^4) and wind "
            "ONLY. Neither threshold involves LW_up, T_s or dTsa, so a per-record bias "
            "delta_i leaves both masks unchanged. Writing m_cc and m_cw for the two "
            "subset means of dTsa, the biased composites are m_cc + delta_i and "
            "m_cw + delta_i, so the contrast (m_cc + delta_i) - (m_cw + delta_i) = "
            "m_cc - m_cw is EXACTLY invariant, for every delta_i and every amplitude. "
            "The adversarial aligned field of section aligned_step_family therefore "
            "cannot touch this evidence line at all.",
        "caveat_second_order":
            "Exact invariance holds for a bias specified in kelvin. A bias specified "
            "as a constant LW_up offset E is not exactly constant in kelvin, because "
            "dT_s/dLW_up = 1/(4 eps sigma T_s^3) varies with T_s and T_s differs "
            "between clear--calm and cloudy--windy nights. The residual is computed "
            "exactly below and is two orders of magnitude smaller than the contrast.",
        "per_record": {s: {
            "n_clear_calm": obs_extras[s]["n_cc"],
            "n_cloudy_windy": obs_extras[s]["n_cw"],
            "dtsa_clear_calm_K": obs_extras[s]["dtsa_cc"],
            "dtsa_cloudy_windy_K": obs_extras[s]["dtsa_cw"],
            "contrast_K": obs_extras[s]["dtsa_cc"] - obs_extras[s]["dtsa_cw"],
            "residual_K_if_bias_is_5Wm2": obs_extras[s]["contrast_residual_K_at_5Wm2"],
            "residual_K_if_bias_is_11p2Wm2":
                obs_extras[s]["contrast_residual_K_at_11.2Wm2"]}
            for s in cc_sites},
        "sample_note":
            "Composites are computed on the FROZEN dTsa nocturnal mask (night & "
            "~pre_spinup & finite LW_up/T_air/LW_down) intersected with finite wind, "
            "with the site median taken over that same sample. "
            "conditioning_analysis.py additionally requires finite Q_air and takes "
            "its thresholds from np.percentile of that slightly smaller sample, so "
            "individual composites can differ in the third decimal. Validation: this "
            "recomputation gives PL-Lipowa +6.489 / +5.689 K, reproducing the "
            "+6.5 / +5.7 K printed in v10 exactly.",
    },
    "independent_instrument_qh": {
        "analytic_statement":
            "obs_qh_qc0 is the nocturnal mean eddy-covariance sensible heat flux: a "
            "sonic anemometer plus gas analyser, a different instrument from the "
            "pyrgeometer, with a different footprint and a different error mechanism. "
            "A radiometric bias delta_i changes no Q_H value. The dTsa--Q_H rank "
            "agreement is therefore not immune in the trivial sense (the dTsa side "
            "moves), but it is an INDEPENDENT CONSTRAINT: the adversarial field is "
            "tuned to flatten the albedo relation, Q_H is not involved in its "
            "construction, and any degradation of the Q_H agreement is a cost the "
            "construction must pay. Quantified below at each breakdown amplitude.",
        "baseline_from_stats_file": d["independent_instrument"],
    },
}

# quantify the Q_H cost of the adversarial field
qh_cost = {}
for v, ss in RECORD_SETS.items():
    a, y = alb_off(ss)
    qh = np.array([P[s]["obs_qh_qc0"] for s in ss])
    ok = np.isfinite(qh)
    rho0, p0 = spearmanr(y[ok], qh[ok])
    sa0 = int(((y[ok] > 0) == (qh[ok] > 0)).sum())
    row = {"n_with_qh": int(ok.sum()),
           "baseline_spearman_dtsa_qh": float(rho0),
           "baseline_spearman_p": float(p0),
           "baseline_sign_agreement": sa0,
           "at_breakdown_amplitudes": {}}
    for thr in THRESH:
        e = bd[v]["aligned"][f"r_above_{thr:+.1f}"]
        B = e.get("B_K")
        if B is None:
            continue
        yp = y + aligned_delta(a, B)
        rho, pv = spearmanr(yp[ok], qh[ok])
        sa = int(((yp[ok] > 0) == (qh[ok] > 0)).sum())
        row["at_breakdown_amplitudes"][f"B_star_for_r_above_{thr:+.1f}"] = {
            "B_K": B,
            "albedo_r_after": r_of(a, y, aligned_delta(a, B)),
            "spearman_dtsa_qh_after": float(rho),
            "spearman_p_after": float(pv),
            "spearman_change": float(rho - rho0),
            "sign_agreement_after": sa,
            "sign_agreement_change": sa - sa0}
    qh_cost[v] = row
cancel["independent_instrument_qh"]["adversarial_cost"] = qh_cost

# summary of the contrast and of the second-order residual
_c = np.array([obs_extras[s]["dtsa_cc"] - obs_extras[s]["dtsa_cw"] for s in cc_sites])
_r5 = np.array([obs_extras[s]["contrast_residual_K_at_5Wm2"] for s in cc_sites])
_r11 = np.array([obs_extras[s]["contrast_residual_K_at_11.2Wm2"] for s in cc_sites])
_ratio = np.abs(_r5 / _c)
_negmask = np.array([P[s]["dtsa_std"] < 0 for s in cc_sites])
cancel["clear_calm_contrast"]["summary"] = {
    "n_records": len(cc_sites),
    "contrast_K_median_abs": float(np.median(np.abs(_c))),
    "contrast_K_max_abs": float(np.abs(_c).max()),
    "contrast_K_max_abs_site": cc_sites[int(np.argmax(np.abs(_c)))],
    "negative_offset_group": {
        "sites": [s for s, k in zip(cc_sites, _negmask) if k],
        "n": int(_negmask.sum()),
        "mean_dtsa_clear_calm_K": float(
            np.mean([obs_extras[s]["dtsa_cc"] for s, k in zip(cc_sites, _negmask) if k])),
        "mean_dtsa_cloudy_windy_K": float(
            np.mean([obs_extras[s]["dtsa_cw"] for s, k in zip(cc_sites, _negmask) if k])),
        "mean_contrast_K": float(_c[_negmask].mean()),
        "max_abs_residual_K_at_5Wm2": float(np.abs(_r5[_negmask]).max()),
        "max_residual_fraction_of_own_contrast": float(_ratio[_negmask].max()),
        "DISCREPANCY_FLAG_FOR_LEAD":
            "v10 main.tex states 'over the negative-offset sites the clear--calm "
            "composite mean is -3.1 K against -1.2 K for cloudy--windy nights'. "
            "Recomputing over the 9 records with dtsa_std < 0 on the frozen mask "
            "gives -2.60 K against -0.90 K (contrast -1.70 K instead of -1.9 K). "
            "Same sign, same magnitude class, but not the printed pair: the lead "
            "should establish which grouping v10 used (PL-Lipowa reproduces "
            "exactly, so the definition is not at fault) before the numbers are "
            "reprinted. This is outside task A's scope and is NOT fixed here."},
    "second_order_residual": {
        "meaning": "A bias specified as a constant LW_up offset E is not exactly "
                   "constant in kelvin, so the cancellation is exact only for a "
                   "bias stated in K. These are the EXACT residuals of the "
                   "contrast, computed sample-by-sample from the raw corpus.",
        "at_5Wm2": {"median_abs_K": float(np.median(np.abs(_r5))),
                    "max_abs_K": float(np.abs(_r5).max()),
                    "max_abs_site": cc_sites[int(np.argmax(np.abs(_r5)))],
                    "median_fraction_of_own_contrast": float(np.median(_ratio)),
                    "max_fraction_of_own_contrast": float(_ratio.max()),
                    "max_fraction_site": cc_sites[int(np.argmax(_ratio))]},
        "at_11p2Wm2": {"median_abs_K": float(np.median(np.abs(_r11))),
                       "max_abs_K": float(np.abs(_r11).max())},
        "verdict": "The contrast is invariant to ~2 % of its own size (median) for a "
                   "5 W/m2 bias; the worst case is KR-Jungnang at 20 %, whose own "
                   "contrast is only -0.57 K. Across the 9 negative-offset records "
                   "that carry the claim the residual never exceeds 4.9 % of the "
                   "record's own contrast. The evidence line therefore survives a "
                   "flux-specified bias too, though not with the exact algebraic "
                   "cancellation that a kelvin-specified bias enjoys."},
}
out["static_bias_cancellation"] = cancel

# =============================================================================
# 6.  manuscript text
# =============================================================================
out["manuscript_text_v11"] = {
    "bibtex_to_add": [
        "@manual{KippZonen2006CGR4, title = {{CGR 4} Pyrgeometer Instruction Manual}, "
        "organization = {Kipp \\& Zonen B.V.}, address = {Delft, The Netherlands}, "
        "year = {2006}, note = {Manual version 0806. "
        "\\url{https://bsrn.aemet.es/manuales/kipp_manual_cgr4_1781.pdf}}}",
        "@manual{WMO2018CIMO, title = {Guide to Instruments and Methods of "
        "Observation ({WMO}-No.~8)}, organization = {World Meteorological "
        "Organization}, address = {Geneva}, year = {2018}, note = {Volume~I, "
        "Chapter~7: Measurement of radiation, Sect.~7.4.3}}",
        "@article{Philipona2001, author = {Philipona, R. and Dutton, E. G. and "
        "Stoffel, T. and Michalsky, J. and Reda, I. and Stifter, A. and Wendling, P. "
        "and Wood, N. and Clough, S. A. and Mlawer, E. J. and Anderson, G. and "
        "Revercomb, H. E. and Shippert, T. R.}, title = {Atmospheric longwave "
        "irradiance uncertainty: Pyrgeometers compared to an absolute sky-scanning "
        "radiometer, atmospheric emitted radiance interferometer, and radiative "
        "transfer model calculations}, journal = {J. Geophys. Res.}, volume = {106}, "
        "number = {D22}, pages = {28129--28141}, year = {2001}, "
        "doi = {10.1029/2000JD000196}}",
        "NOTE: Adderley2015 is already in paper/manuscript_dtsa/references.bib "
        "(line 170).",
    ],
    "methods_spec_paragraph": (
        "\\emph{Observation-uncertainty propagation}. The sensitivity of the "
        "albedo--offset association to observational error is assessed with an "
        "explicit error model rather than a single nominal figure. Error is applied "
        "as an additive perturbation $\\delta_i$ to each record's mean nocturnal "
        "offset, with a positive LW$_{\\uparrow}$ error $E_i$ giving "
        "$\\delta_i=+E_i\\times0.20$~K (an apparently warmer surface) at the "
        "linearization of Sect.~\\ref{sec:stats}; every scenario is run for all three "
        "record sets. Three families are distinguished, and the distinction is "
        "substantive: the first two are error \\emph{structures} whose magnitude can "
        "be cited from the instrument literature, whereas the third is a \\emph{sign} "
        "structure chosen after seeing the data, so its role is to bound what an "
        "error would have to look like, not to estimate one. (i)~\\emph{Random}: "
        "independent zero-mean Gaussian $\\delta_i$ at $\\sigma_E=2$, 4, 5 and "
        "10~W\\,m$^{-2}$ ($=0.40$, 0.80, 1.00 and 2.01~K), 50\\,000 draws per "
        "scenario and record set, each from an independently seeded stream. The range "
        "follows from the fact that pyrgeometer specifications are multiplicative: "
        "1--3\\% of the daily total for a reference-class instrument "
        "\\citep{KippZonen2006CGR4}, agreement to within 2\\% at night after careful "
        "calibration \\citep{WMO2018CIMO}, and $\\approx1$--2~W\\,m$^{-2}$ for "
        "shaded, ventilated, field-calibrated reference instruments at a dedicated "
        "radiation site \\citep{Philipona2001}; at our corpus-mean nocturnal "
        "LW$_{\\uparrow}$ of 366~W\\,m$^{-2}$, 1--3\\% is 3.7--11.0~W\\,m$^{-2}$. "
        "Because the two Minneapolis records share a tower and a radiometer and "
        "their offsets are numerically identical, a variant in which they draw one "
        "common error is run alongside the independent case. A common-mode "
        "percentage-of-reading error, "
        "$\\delta_i=\\pm p\\,\\langle\\mathrm{LW}_{\\uparrow}\\rangle_i/100$, is run "
        "separately at $p=1,2,3$, because that is the form the specifications "
        "actually take. (ii)~\\emph{Placement}: each record receives its own "
        "$\\sigma_i$ from its radiometer-to-building-height ratio "
        "$z_{\\mathrm{rad}}/H$, by interpolating the positional RMSE that "
        "\\citet{Adderley2015} report for retrieving LW$_{\\uparrow}$ over an urban "
        "canopy (11.2, 6.3 and 2.0~W\\,m$^{-2}$ at $z/H=2$, 3 and 5). Because those "
        "authors describe the decay as exponential, we fit "
        "$\\sigma=\\exp(3.564-0.574\\,z/H)$, which reproduces all three published "
        "values to within 0.005~W\\,m$^{-2}$ ($R^2=0.999999$ in $\\log\\sigma$, "
        "e-folding $1.74\\,z_b$) and so adds essentially no assumption of ours; the "
        "same exponential is continued beyond $z/H=5$, where it is immaterial "
        "because it has already fallen below instrument accuracy. The scenario is "
        "run both capped at the largest published value, 11.2~W\\,m$^{-2}$, and "
        "uncapped, and additionally as flat 11.2~W\\,m$^{-2}$ and flat "
        "9.4~W\\,m$^{-2}$, the latter being the authors' hemispherical-versus-"
        "complete-surface temperature RMSE of 1.7~K, which is a field-of-view "
        "composition term and does not decay with height. (iii)~\\emph{Aligned}: an "
        "adversarial systematic field "
        "$\\delta_i=B\\,\\mathrm{sign}(\\alpha_i-\\bar\\alpha)$, warming every "
        "high-albedo record by $B$ and cooling every low-albedo record by $B$; this "
        "is a saturated field, so its peak-to-peak displacement is $2B$, which is "
        "the quantity comparable to an instrument specification. We report alongside "
        "it the unrestricted supremum of $r$ over the box $|\\delta_i|\\le B$, "
        "obtained by exhaustive enumeration of all $2^n$ vertices and verified "
        "against a bounded interior search from 63 starts at each amplitude (no "
        "interior point beats the best vertex at any amplitude, the largest excess "
        "being $10^{-16}$ in $r$). For both families we report the breakdown "
        "amplitude $B^*$, the smallest amplitude at which $r$ rises above $-0.5$, "
        "$-0.4$, $-0.3$ and $0$."),
    "results_passage": (
        "The association is robust to random radiometric error of the magnitude the "
        "instrument literature supports. Independent per-record LW$_{\\uparrow}$ "
        "perturbations at $\\sigma_E=2$~W\\,m$^{-2}$ (0.40~K) leave median $r$ at "
        "$-0.78$, $-0.69$ and $-0.73$ in the 19-, 17- and 16-record sets, and at "
        "$\\sigma_E=5$~W\\,m$^{-2}$ --- 1.00~K, the upper end of the multiplicative "
        "specification at our LW$_{\\uparrow}$ levels --- at $-0.72$ $[-0.84,-0.56]$, "
        "$-0.63$ $[-0.79,-0.43]$ and $-0.58$ $[-0.80,-0.25]$, with 0.4\\%, 9.0\\% "
        "and 29\\% of draws respectively rising above $-0.5$. A "
        "$\\sigma_E=10$~W\\,m$^{-2}$ stress case, twice any documented figure, "
        "degrades the conservative core to a median $-0.38$, and we do not claim the "
        "association survives it. Letting the two Minneapolis records share one error "
        "draw shifts the 19-record median by at most $0.006$, so the inter-record "
        "correlation structure is not load-bearing. A common-mode "
        "percentage-of-reading error behaves almost neutrally here: the multiplicand, "
        "each record's own mean nocturnal LW$_{\\uparrow}$, is uncorrelated with "
        "albedo ($r=-0.08$, $p=0.76$; Spearman $-0.13$), so $\\pm2\\%$ of reading "
        "leaves $r$ between $-0.78$ and $-0.80$ over 19 records and between $-0.72$ "
        "and $-0.82$ in the core, despite spanning 0.48--0.54~K across the corpus. "
        "The placement term matters more. Assigning each record the positional "
        "uncertainty implied by its own $z_{\\mathrm{rad}}/H$ through the "
        "\\citet{Adderley2015} exponential, capped at their largest published value, "
        "leaves the association clearly negative but weakened --- median $r=-0.67$, "
        "$-0.60$ and $-0.53$ --- and applying their flat 9.4~W\\,m$^{-2}$ "
        "field-of-view composition term to every record gives $-0.59$, $-0.51$ and "
        "$-0.40$. Seventeen of nineteen records sit at $z_{\\mathrm{rad}}/H\\ge2.09$, "
        "where the positional component is at or below instrument accuracy; the "
        "exception is the Minneapolis pair, whose 2~m radiometers sit at "
        "$z_{\\mathrm{rad}}/H=0.40$, five times below the lowest height "
        "\\citeauthor{Adderley2015} calibrate. We report the uncapped extrapolation "
        "for that pair explicitly rather than leaving it behind the cap: it implies "
        "$\\sigma=28$~W\\,m$^{-2}$ (5.6~K) and moves the 19-record median from "
        "$-0.67$ to $-0.60$, while the 17- and 16-record sets --- which exclude those "
        "records on independent height-mismatch grounds --- are unaffected, at "
        "$-0.60$ and $-0.52$. "
        "The aligned case is different, and we state it plainly. A systematic bias "
        "carrying the sign pattern that most efficiently flattens the gradient "
        "removes the association entirely in the conservative core at an amplitude of "
        "$B^*=0.97$~K (4.9~W\\,m$^{-2}$; peak-to-peak 1.95~K, 9.7~W\\,m$^{-2}$), and "
        "reduces $r$ to $-0.5$ there at $B^*=0.56$~K; in the full 19-record set the "
        "corresponding amplitudes are 1.88~K and 0.96~K. The unrestricted supremum "
        "over $|\\delta_i|\\le B$ requires no more and sometimes less --- 0.46~K to "
        "reach $-0.5$ in the core --- and at the amplitudes that drive $r$ to zero it "
        "coincides with the aligned step exactly, so the aligned pattern is the worst "
        "case there and not merely one case among many. These amplitudes are not "
        "large relative to the composite-footprint terms the urban literature "
        "documents: the peak-to-peak displacement needed to erase the 16-record "
        "association, 9.7~W\\,m$^{-2}$, is the same size as "
        "\\citeauthor{Adderley2015}'s 9.4~W\\,m$^{-2}$ hemispherical-versus-complete-"
        "surface RMSE, and radiometer height is documented at only two of our "
        "nineteen records. The association therefore survives the random, "
        "shared-error, placement-calibrated and calibration-scale scenarios tested, "
        "but an albedo-aligned systematic bias of amplitude $B^*\\approx1$~K would "
        "suffice to remove it in the conservative core, and such a bias is not "
        "excluded by the available metadata. "
        "Two evidence lines are immune to this construction by design. The "
        "within-site clear--calm versus cloudy--windy contrast is a difference of two "
        "composites of the same record, selected on $\\varepsilon_{\\mathrm{sky}}$ "
        "and wind alone; a static per-record $\\delta_i$ shifts both composites "
        "equally and cancels from the difference exactly, at every amplitude. (For a "
        "bias specified as a constant flux rather than a constant temperature the "
        "cancellation is not algebraically exact, because "
        "$\\partial T_s/\\partial\\mathrm{LW}_{\\uparrow}$ varies with $T_s$; the "
        "exact residual for a 5~W\\,m$^{-2}$ bias is a median 2\\% of each record's "
        "own contrast and never exceeds 4.9\\% among the negative-offset records that "
        "carry the result.) And the eddy-covariance $Q_H$ values come from a "
        "different instrument with a different footprint, so $\\delta_i$ does not "
        "touch them: the adversarial field must pay for flattening the albedo "
        "relation elsewhere, and it does. At the 16-record amplitude that drives the "
        "albedo correlation to zero, the offset--$Q_H$ rank agreement falls from "
        "$\\rho=+0.34$ to $-0.13$ and sign agreement from 11/16 to 5/16; over all 19 "
        "records, the amplitude that reduces the albedo correlation to $-0.5$ already "
        "cuts $\\rho$ from $+0.51$ to $+0.35$ and sign agreement from 13/19 to 8/19. "
        "A bias able to manufacture the albedo gradient would therefore have to "
        "destroy an independent instrumental agreement at the same time."),
    "limitations_sentence": (
        "Consistent with this, we do not claim that instrument or "
        "composite-footprint error has been excluded: the propagation in "
        "Sect.~\\ref{sec:obsaxis} shows that random error of the documented "
        "magnitude cannot produce the gradient, but that a systematic bias aligned "
        "with albedo and of amplitude $\\approx1$~K --- whose peak-to-peak "
        "displacement, 9.7~W\\,m$^{-2}$, is the size of the field-of-view "
        "composition term \\citet{Adderley2015} report --- would remove it, and with "
        "radiometer height documented at only two of nineteen records such a bias "
        "cannot be ruled out from the metadata; the within-site conditioning contrast "
        "and the independent $Q_H$ comparison are the two lines a bias of this form "
        "cannot manufacture."),
    "sentence_to_delete_from_v10": (
        "paper/manuscript_dtsa/main.tex lines ~489-498: delete from 'It also "
        "survives realistic radiometric measurement error: adding independent "
        "per-record LW$_{\\uparrow}$ perturbations of 5~W\\,m$^{-2}$ (a "
        "representative field pyrgeometer accuracy) ...' through '... so the albedo "
        "organization is not an artefact of plausible instrument or "
        "composite-footprint error.' and insert results_passage in its place. The "
        "final clause is the exclusion claim that must go."),
    "text_notes": [
        "Every number in the two passages above is taken from THIS run and is "
        "reproducible from the sibling keys of this JSON block; none is a "
        "placeholder. The lead should still cross-check them after any rerun.",
        "The passage makes NO exclusion claim. It replaces the v10 clause 'so the "
        "albedo organization is not an artefact of plausible instrument or "
        "composite-footprint error', which must be deleted.",
        "Three \\citep keys are new -- KippZonen2006CGR4, WMO2018CIMO, "
        "Philipona2001 -- and their BibTeX is in bibtex_to_add. Adderley2015 is "
        "already at paper/manuscript_dtsa/references.bib line 170.",
        "\\ref{sec:obsaxis} in limitations_sentence is a placeholder label: point it "
        "at whichever section actually carries results_passage.",
        "The Results passage deliberately does NOT reprint the -3.1 / -1.2 K "
        "negative-offset composite pair from v10; see "
        "static_bias_cancellation.clear_calm_contrast.summary."
        "negative_offset_group.DISCREPANCY_FLAG_FOR_LEAD.",
    ],
}


def leaf_count(o):
    """Scalars, plus each element of every list, counted recursively."""
    if isinstance(o, dict):
        return sum(leaf_count(v) for v in o.values())
    if isinstance(o, list):
        return sum(leaf_count(v) for v in o) if o else 1
    return 1


out["meta"]["leaf_count"] = leaf_count(out)
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# =============================================================================
# report
# =============================================================================
print("\n=== baseline ===")
for v in RECORD_SETS:
    b = base[v]
    print(f"  {v:12s} n={b['n']:2d} clusters={b['n_clusters']:2d} "
          f"r={b['pearson_r']:+.4f} (p={b['pearson_p']:.2e}) rho={b['spearman_rho']:+.4f}")

print(f"\n=== error budget ===")
print(f"  mean nocturnal LWup = {lw.mean():.1f} W/m2 "
      f"[{lw.min():.1f} {SITES[int(np.argmin(lw))]} .. {lw.max():.1f} "
      f"{SITES[int(np.argmax(lw))]}]")
for pct in (1.0, 2.0, 3.0):
    print(f"  {pct:g}% of reading -> {300*pct/100:5.1f}-{400*pct/100:5.1f} W/m2 = "
          f"{300*pct/100*K_PER_WM2:.2f}-{400*pct/100*K_PER_WM2:.2f} K")
print(f"  multiplicand (mean LWup) vs albedo: r={r_lw_alb:+.4f} p={p_lw_alb:.4f}, "
      f"rho={rho_lw_alb:+.4f} p={prho_lw_alb:.4f}")

print("\n=== (i) random independent: median r [2.5,97.5] ===")
for v in RECORD_SETS:
    for sg in (2.0, 4.0, 5.0, 10.0):
        e = rand[v][f"sigma_{sg:g}Wm2"]
        print(f"  {v:12s} sigma={sg:5.1f} W/m2 ({sg*K_PER_WM2:.2f} K) "
              f"med={e['r_median']:+.4f} CI=[{e['r_ci95'][0]:+.4f},{e['r_ci95'][1]:+.4f}] "
              f"P(r>-0.5)={e['frac_above_minus0p5']:.4f}")

print("\n=== (ii) shared Minneapolis draw (all19) ===")
for sg in (2.0, 4.0, 5.0, 10.0):
    a1 = rand["all19"][f"sigma_{sg:g}Wm2"]; a2 = shared[f"sigma_{sg:g}Wm2"]
    print(f"  sigma={sg:5.1f}  independent med={a1['r_median']:+.4f}  "
          f"shared med={a2['r_median']:+.4f}  d={a2['r_median']-a1['r_median']:+.4f}")

print("\n=== (iii) Adderley fit ===")
print(f"  sigma(z/H) = exp({c0:.4f} {c1:+.4f}*z/H); R2_log="
      f"{1-ss_res/ss_tot:.6f}; e-fold={-1/c1:.3f} z_b")
print(f"  fitted at 2/3/5 = {np.round(pred,3)} vs published {ADD_SIG}")
print(f"  sigma(3.5) = {fit(3.5):.2f} W/m2 -> implied 'typical pyrgeometer accuracy'"
      f" = {100*fit(3.5)/lw.mean():.2f}% of mean LWup")
print(f"  Minneapolis z_rad/H = {zoh['US-Minneapolis1']:.4f} -> uncapped sigma = "
      f"{sig_unc['US-Minneapolis1']:.2f} W/m2 = {sig_unc['US-Minneapolis1']*K_PER_WM2:.2f} K")
for v in RECORD_SETS:
    e = add["variants"][v]
    print(f"  {v:12s} capped={e['capped_at_11p2']['r_median']:+.4f} "
          f"uncapped={e['uncapped']['r_median']:+.4f} "
          f"flat11.2={e['flat_11p2Wm2']['r_median']:+.4f} "
          f"flat9.4={e['flat_9p4Wm2']['r_median']:+.4f}")

print("\n=== (iiib) common-mode scale error ===")
for v in RECORD_SETS:
    e = scale[v]
    print(f"  {v:12s} +2%={e['common_mode_plus_2pct']['r']:+.4f} "
          f"-2%={e['common_mode_minus_2pct']['r']:+.4f} "
          f"randsign2% med={e['random_sign_2pct']['r_median']:+.4f} "
          f"(spread {e['common_mode_plus_2pct']['delta_spread_K']:.4f} K)")

print("\n=== (iv)/(v) aligned step vs unrestricted box sup ===")
for v in RECORD_SETS:
    for B in BGRID:
        ea = aligned[v]["amplitudes"][f"B_{B:g}K"]
        eu = unres[v]["amplitudes"][f"B_{B:g}K"]
        print(f"  {v:12s} B={B:.2f} K  aligned r={ea['r']:+.4f}  "
              f"box sup r={eu['r_sup_vertex']:+.4f}  same_sign={eu['matches_albedo_aligned_sign']}  "
              f"interior_beats={eu['interior_beats_vertex']} "
              f"(int-vtx={eu['interior_minus_vertex']:+.2e})")

print("\n=== (vi) breakdown amplitude B* (aligned | unrestricted) ===")
for v in RECORD_SETS:
    print(f"  {v} (baseline r={base[v]['pearson_r']:+.4f})")
    for thr in THRESH:
        ea = bd[v]["aligned"][f"r_above_{thr:+.1f}"]
        eu = bd[v]["unrestricted"][f"r_above_{thr:+.1f}"]
        fa = (f"{ea['B_K']:.4f} K / {ea['B_Wm2']:.2f} W/m2 / pp "
              f"{ea['peak_to_peak_K']:.3f} K" if ea["B_K"] is not None else "none")
        fu = (f"{eu['B_K']:.4f} K / {eu['B_Wm2']:.2f} W/m2 / pp "
              f"{eu['peak_to_peak_K']:.3f} K" if eu["B_K"] is not None else "none")
        print(f"    r>{thr:+.1f}: aligned {fa}   |  box {fu}")

print("\n=== (5) static bias cancellation ===")
res5 = [abs(obs_extras[s]["contrast_residual_K_at_5Wm2"]) for s in cc_sites]
res11 = [abs(obs_extras[s]["contrast_residual_K_at_11.2Wm2"]) for s in cc_sites]
con = [abs(obs_extras[s]["dtsa_cc"] - obs_extras[s]["dtsa_cw"]) for s in cc_sites]
print(f"  clear-calm minus cloudy-windy contrast: {len(cc_sites)} records, "
      f"|contrast| median {np.median(con):.3f} K, max {max(con):.3f} K")
print(f"  EXACT residual if the static bias is 5 W/m2 (not K): max "
      f"{max(res5):.5f} K = {100*max(res5)/np.median(con):.3f}% of the median contrast")
print(f"  EXACT residual if the static bias is 11.2 W/m2: max {max(res11):.5f} K")
for v in RECORD_SETS:
    q = qh_cost[v]
    print(f"  {v:12s} dTsa-Qh Spearman baseline {q['baseline_spearman_dtsa_qh']:+.4f} "
          f"(sign agree {q['baseline_sign_agreement']}/{q['n_with_qh']})")
    for k, e in q["at_breakdown_amplitudes"].items():
        print(f"      {k}: B={e['B_K']:.3f} K -> albedo r={e['albedo_r_after']:+.4f}, "
              f"Qh rho={e['spearman_dtsa_qh_after']:+.4f} "
              f"({e['spearman_change']:+.4f}), sign agree "
              f"{e['sign_agreement_after']} ({e['sign_agreement_change']:+d})")

print(f"\nwrote {OUT}  leaves={out['meta']['leaf_count']}")
