"""STEP-3 MINI-TEST: dSLUCM structural surgery (Category-B, standalone).

Question: does a GLOBAL structural modification to the differentiable SLUCM
break its nocturnal dTsa = <Ts - Ta> ceiling -- reaching the observed negative
end (open sites) AND raising the dense end (Lipowa) -- WITHOUT any site-specific
tuning (identical global constants for every site; only per-site levers are the
meteorological forcing and, for the pervious-tile configs, f_perv from site
input files)?

Category-B compliance: src/ is NOT modified. The base model
(src/models/dslucm_forward.py) is imported only for cross-checking constants;
every stepper here is re-implemented locally so all structural variants live in
this one file.

Machinery pattern copied from
  analysis/revalidation_2026-08/scripts/energy_conserving_probe.py
(rollout via jax.lax.scan over corpus forcing; corpus_loader.load_all_sites;
frozen night mask = night_mask & ~pre_spinup_flag).

FROZEN EVALUATION (identical across ALL configs; only internal structure varies)
--------------------------------------------------------------------------------
MODEL LWup aggregation (fixed):
  plan-view LWup_urban =
      f_roof * [eps_r*sig*T_roof^4 + (1-eps_r)*LWdown]
    + (1-f_roof) * LWup_canyon_top
  LWup_canyon_top =
      w_g * [eps_g*sig*T_road^4 + (1-eps_g)*LWdown*svf_road]
    + w_w * [eps_w*sig*T_wall^4 + (1-eps_w)*LWdown*svf_wall]
  w_g = svf_road (road seen through canyon mouth), w_w = 1 - svf_road.
  f_roof: SLUCMParams carries no roof-plan-fraction, so f_roof is set to a
  GLOBAL constant 0.5 (equal roof/road plan widths -- the WRF-SLUCM default when
  the plan-area building fraction lambda_p is unspecified). Because f_roof is
  identical across every config and site, it only shifts the absolute dTsa level
  uniformly; it does not affect the between-config slope/r comparison.

If a pervious tile is present (S-B / S-AB / S-ABD):
  LWup_total = f_perv * LWup_perv + (1-f_perv) * LWup_urban
  LWup_perv  = eps_p*sig*T_perv^4 + (1-eps_p)*LWdown,  eps_p = 0.95
  f_perv = 1 - impervious_area_fraction from
    data/urban-plumber/FullCollection/<SITE>/<SITE>_sitedata_v1.csv (site INPUT).

Surface-temperature inversion (IDENTICAL to the obs convention):
  Ts = ((LWup_model - (1-0.95)*LWdown) / (0.95*sig))^0.25
  dTsa_model = nocturnal mean of (Ts - T_atm) over night_mask & ~pre_spinup.

CONFIGS (each = ONE global structural change; identical global constants)
--------------------------------------------------------------------------------
  A0    baseline dSLUCM as-is (Louis-1979 MOST a=5; other-facet-mean LW trapping).
  S-A1  stability-damped exchange: every sensible exchange coefficient (roof->atm,
        wall->canyon, road->canyon, canyon->atm) multiplied by
        f = 1/(1+a*Ri_b)^2 for Ri_b>0 else 1, floored at 0.05, a=10.
        Ri_b is a bulk Richardson from LOCAL (T_atm - T_surface), forcing-level
        wind. (The roof->atmosphere exchange is a surface-to-atmosphere sensible
        exchange too, so it is damped as well; this is the physical clear-calm
        turbulent-shutdown mechanism and is required for the open-site
        decoupling.)  Replaces (does not stack on) the baseline Louis factor.
  S-A2  same as S-A1 with a=30.
  S-B   add a parallel pervious tile: lumped slab of areal heat capacity
        C_eff = mu / sqrt(omega_diurnal) chosen so the slab's dynamic admittance
        |Q/dT| = C_eff*omega equals the target admittance mu*sqrt(omega) at the
        diurnal frequency (mu = 800 J m^-2 K^-1 s^-1/2, omega = 2*pi/86400 s^-1
        -> C_eff = 9.38e4 J m^-2 K^-1). Free longwave cooling (eps=0.95, full
        sky), shortwave albedo 0.20, NEUTRAL sensible exchange with ambient air
        (same neutral bulk form as the other facets, no stability factor).
        Urban canyon runs exactly as A0; outputs aggregated by f_perv. No
        separate storage term -- C_eff IS the tile's thermal inertia.
  S-AB  S-A2 (a=30 damped urban) + pervious tile whose exchange is ALSO
        stability-damped (a=30, same local-Ri rule).
  S-D   longwave-trapping enhancement: for wall & road, replace the crude
        other-facet-mean LW term with a cavity effective exchange
        LW_net = eps*(LWdown - sig*T^4) * tau,  tau = svf/(1-(1-eps)*(1-svf)).
        Roof (svf=1 -> tau=1) is unchanged, reducing to full-sky exchange.
        Urban sensible stays baseline Louis a=5. Global; strongest where svf is
        low. (svf is a global constant here, so S-D acts uniformly.)
  S-ABD S-A2 + S-B + S-D combined.

PASS per config: (>=4 of the obs-negative sites come out model-negative) AND
(Lipowa >= +4.0) AND (slope_vs_obs >= 0.5) AND (r >= 0.6).

NOTE on the count criterion: the brief wrote ">=4/7 negative sites" but the
frozen obs table has 9 sites with dTsa<0. We report the count among ALL 9
obs-negative sites and apply the >=4 threshold (the binding number).
"""
import sys
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import yaml

jax.config.update("jax_enable_x64", True)

from src.training.corpus_loader import load_all_sites, FORCING_CHANNELS

# ---------------------------------------------------------------------------
# Constants / global params (identical for every site and config)
# ---------------------------------------------------------------------------
SIG = 5.67e-8
CP = 1004.0
RD = 287.0
G = 9.80665
ZREF = 10.0
MIN_U = 0.1
DT = 1800.0

cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
d = cfg["experiment"]["slucm_params_defaults"]
HW = float(d["canyon_aspect_ratio_h_w"])
SVF_ROAD = float(d["svf_road"])
SVF_WALL = float(d["svf_wall"])
ALB_ROOF = float(d["albedo_roof"])
ALB_WALL = float(d["albedo_wall"])
ALB_ROAD = float(d["albedo_road"])
EPS_ROOF = float(d["emissivity_roof"])
EPS_WALL = float(d["emissivity_wall"])
EPS_ROAD = float(d["emissivity_road"])
CH = float(d["heat_transfer_coefficient"])
TM_ROOF = float(d["thermal_mass_roof"])
TM_WALL = float(d["thermal_mass_wall"])
TM_ROAD = float(d["thermal_mass_road"])
A1 = 0.25  # OHM storage a1 per facet (a3 = 0 in config)

F_ROOF = 0.5  # global roof plan fraction (see module docstring)
EPS_PERV = 0.95
ALB_PERV = 0.20
MU_PERV = 800.0
OMEGA_DIURNAL = 2.0 * np.pi / 86400.0
C_PERV = MU_PERV / np.sqrt(OMEGA_DIURNAL)  # ~9.38e4 J m^-2 K^-1

# cavity trapping factors (S-D); svf global so these are constants
TAU_WALL = SVF_WALL / (1.0 - (1.0 - EPS_WALL) * (1.0 - SVF_WALL))
TAU_ROAD = SVF_ROAD / (1.0 - (1.0 - EPS_ROAD) * (1.0 - SVF_ROAD))

# ---------------------------------------------------------------------------
# Frozen obs reference (ADDENDUM1 section 4)
# ---------------------------------------------------------------------------
OBS = {
    "US-Minneapolis1": -3.04, "US-Minneapolis2": -3.04, "KR-Ochang": -2.17,
    "AU-Preston": -1.99, "GR-HECKOR": -1.67, "US-WestPhoenix": -1.36,
    "UK-Swindon": -0.63, "FI-Kumpula": -0.32, "UK-KingsCollege": -0.17,
    "SG-TelokKurau06": 0.26, "US-Baltimore": 0.39, "JP-Yoyogi": 0.43,
    "NL-Amsterdam": 0.44, "FI-Torni": 0.45, "FR-Capitole": 0.49,
    "KR-Jungnang": 0.66, "CA-Sunset": 0.98, "PL-Narutowicza": 1.19,
    "PL-Lipowa": 6.12,
}
OBS_NEG_SITES = [s for s, v in OBS.items() if v < 0]  # 9 sites

FULLCOL = ROOT / "data" / "urban-plumber" / "FullCollection"


def f_perv_of(site: str) -> float:
    p = FULLCOL / site / f"{site}_sitedata_v1.csv"
    with open(p, encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) > 2 and r[1] == "impervious_area_fraction":
                return 1.0 - float(r[2])
    raise KeyError(f"impervious_area_fraction not found for {site}")


# ---------------------------------------------------------------------------
# Steppers (all jax; state = (Troof, Twall, Troad, Tc, Tperv) scalars)
# ---------------------------------------------------------------------------
def _damp(Ri, a):
    """Stability-damped exchange factor: 1/(1+a*Ri)^2 for Ri>0, else 1; floor 0.05."""
    stable = 1.0 / jnp.square(1.0 + a * Ri)
    f = jnp.where(Ri > 0.0, jnp.maximum(stable, 0.05), 1.0)
    return f


def _louis_Fh(Ri):
    stable = 1.0 / jnp.square(1.0 + 5.0 * Ri)
    unstable = jnp.sqrt(jnp.maximum(1.0 - 15.0 * Ri, 1e-12))
    return jnp.where(Ri >= 0.0, stable, unstable)


def make_step(stab_mode, lw_mode, a_damp, has_perv, perv_damped):
    """Build a jax scan-body for one config."""

    def radiation(Troof, Twall, Troad, SW, LW):
        emit_roof = SIG * Troof ** 4
        emit_wall = SIG * Twall ** 4
        emit_road = SIG * Troad ** 4
        SWnet_roof = (1.0 - ALB_ROOF) * SW * 1.0
        SWnet_wall = (1.0 - ALB_WALL) * SW * SVF_WALL
        SWnet_road = (1.0 - ALB_ROAD) * SW * SVF_ROAD
        if lw_mode == "baseline":
            other_wall = 0.5 * (emit_roof + emit_road)
            other_road = 0.5 * (emit_roof + emit_wall)
            LWnet_roof = EPS_ROOF * (LW - emit_roof)
            LWnet_wall = EPS_WALL * (LW * SVF_WALL + other_wall * (1.0 - SVF_WALL) - emit_wall)
            LWnet_road = EPS_ROAD * (LW * SVF_ROAD + other_road * (1.0 - SVF_ROAD) - emit_road)
        else:  # trap
            LWnet_roof = EPS_ROOF * (LW - emit_roof)
            LWnet_wall = EPS_WALL * (LW - emit_wall) * TAU_WALL
            LWnet_road = EPS_ROAD * (LW - emit_road) * TAU_ROAD
        return (SWnet_roof, LWnet_roof, SWnet_wall, LWnet_wall, SWnet_road, LWnet_road)

    def body(state, row):
        Troof, Twall, Troad, Tc, Tperv = state
        SW = row[0]; LW = row[1]; Tatm = row[2]; P = row[4]
        U = jnp.sqrt(row[5] ** 2 + row[6] ** 2)
        Us = jnp.maximum(U, MIN_U)
        rho = P / (RD * Tatm)
        bulk0 = rho * CP * CH * U

        (SWr, LWr, SWw, LWw, SWg, LWg) = radiation(Troof, Twall, Troad, SW, LW)

        # storage (OHM a1=0.25, a3=0)
        Gr = A1 * (SWr + LWr)
        Gw = A1 * (SWw + LWw)
        Gg = A1 * (SWg + LWg)

        if stab_mode == "louis":
            Ri = G * ZREF * (Tatm - Tc) / (Tatm * Us * Us)
            Fh = _louis_Fh(Ri)
            bulk = bulk0 * Fh
            Hr = bulk * (Troof - Tatm)
            Hw = bulk * (Twall - Tc)
            Hg = bulk * (Troad - Tc)
        else:  # damped, per-facet local Ri (ref = T_atm)
            Ri_roof = G * ZREF * (Tatm - Troof) / (Tatm * Us * Us)
            Ri_wall = G * ZREF * (Tatm - Twall) / (Tatm * Us * Us)
            Ri_road = G * ZREF * (Tatm - Troad) / (Tatm * Us * Us)
            Ri_can = G * ZREF * (Tatm - Tc) / (Tatm * Us * Us)
            f_roof = _damp(Ri_roof, a_damp)
            f_wall = _damp(Ri_wall, a_damp)
            f_road = _damp(Ri_road, a_damp)
            f_can = _damp(Ri_can, a_damp)
            Hr = f_roof * bulk0 * (Troof - Tatm)
            Hw = f_wall * bulk0 * (Twall - Tc)
            Hg = f_road * bulk0 * (Troad - Tc)

        net_roof = SWr + LWr - Hr - Gr
        net_wall = SWw + LWw - Hw - Gw
        net_road = SWg + LWg - Hg - Gg
        Troof_n = Troof + DT / TM_ROOF * net_roof
        Twall_n = Twall + DT / TM_WALL * net_wall
        Troad_n = Troad + DT / TM_ROAD * net_road

        if stab_mode == "louis":
            Tc_n = 0.25 * (Troad_n + 2.0 * Twall_n + Tatm)
        else:
            Tc_n = (f_road * Troad_n + 2.0 * f_wall * Twall_n + f_can * Tatm) \
                / (f_road + 2.0 * f_wall + f_can)

        # pervious tile (parallel; exchanges with ambient air)
        if has_perv:
            SWp = (1.0 - ALB_PERV) * SW
            LWp = EPS_PERV * (LW - SIG * Tperv ** 4)
            if perv_damped:
                Ri_p = G * ZREF * (Tatm - Tperv) / (Tatm * Us * Us)
                bp = _damp(Ri_p, a_damp) * bulk0
            else:
                bp = bulk0  # neutral
            Hp = bp * (Tperv - Tatm)
            Tperv_n = Tperv + DT / C_PERV * (SWp + LWp - Hp)
        else:
            Tperv_n = Tperv

        return (Troof_n, Twall_n, Troad_n, Tc_n, Tperv_n), \
            jnp.stack([Troof_n, Twall_n, Troad_n, Tperv_n])

    @jax.jit
    def rollout(forcing):
        T0 = forcing[0, 2]
        init = (T0, T0, T0, T0, T0)
        _, out = jax.lax.scan(body, init, forcing)
        return out

    return rollout


CONFIGS = {
    "A0":    dict(stab_mode="louis",  lw_mode="baseline", a_damp=5.0,  has_perv=False, perv_damped=False),
    "S-A1":  dict(stab_mode="damped", lw_mode="baseline", a_damp=10.0, has_perv=False, perv_damped=False),
    "S-A2":  dict(stab_mode="damped", lw_mode="baseline", a_damp=30.0, has_perv=False, perv_damped=False),
    "S-B":   dict(stab_mode="louis",  lw_mode="baseline", a_damp=5.0,  has_perv=True,  perv_damped=False),
    "S-AB":  dict(stab_mode="damped", lw_mode="baseline", a_damp=30.0, has_perv=True,  perv_damped=True),
    "S-D":   dict(stab_mode="louis",  lw_mode="trap",     a_damp=5.0,  has_perv=False, perv_damped=False),
    "S-ABD": dict(stab_mode="damped", lw_mode="trap",     a_damp=30.0, has_perv=True,  perv_damped=True),
}


def model_dTsa(out, forcing, night, f_perv, has_perv):
    Troof = np.asarray(out[:, 0]); Twall = np.asarray(out[:, 1])
    Troad = np.asarray(out[:, 2]); Tperv = np.asarray(out[:, 3])
    LW = np.asarray(forcing[:, 1]); Tatm = np.asarray(forcing[:, 2])
    lwup_roof = EPS_ROOF * SIG * Troof ** 4 + (1.0 - EPS_ROOF) * LW
    lwup_top = (SVF_ROAD * (EPS_ROAD * SIG * Troad ** 4 + (1.0 - EPS_ROAD) * LW * SVF_ROAD)
                + (1.0 - SVF_ROAD) * (EPS_WALL * SIG * Twall ** 4 + (1.0 - EPS_WALL) * LW * SVF_WALL))
    lwup_urban = F_ROOF * lwup_roof + (1.0 - F_ROOF) * lwup_top
    if has_perv:
        lwup_perv = EPS_PERV * SIG * Tperv ** 4 + (1.0 - EPS_PERV) * LW
        lwup = f_perv * lwup_perv + (1.0 - f_perv) * lwup_urban
    else:
        lwup = lwup_urban
    Ts = ((lwup - (1.0 - 0.95) * LW) / (0.95 * SIG)) ** 0.25
    dts = Ts - Tatm
    mk = night & np.isfinite(dts)
    return float(np.mean(dts[mk])), np.isnan(Ts).any() or np.isinf(Ts).any()


def main():
    records = {}
    for rec in load_all_sites():
        if rec.site == "MX-Escandon":
            continue  # not in the frozen obs table (no obs_LWup)
        records[rec.site] = rec
    sites = [s for s in OBS if s in records]
    print(f"Sites evaluated: {len(sites)}  (missing: {[s for s in OBS if s not in records]})")

    fperv = {s: f_perv_of(s) for s in sites}
    forcings = {}
    nights = {}
    for s in sites:
        rec = records[s]
        fr = np.stack([np.asarray(rec.forcing[c], float) for c in FORCING_CHANNELS], -1)
        forcings[s] = jnp.asarray(fr)
        nights[s] = np.asarray(rec.night_mask, bool) & (~np.asarray(rec.pre_spinup_flag, bool))

    results = {}
    per_config_dtsa = {}
    for cname, kw in CONFIGS.items():
        roll = make_step(**kw)
        dts_by_site = {}
        bad = []
        for s in sites:
            out = roll(forcings[s])
            val, isbad = model_dTsa(np.asarray(out), np.asarray(forcings[s]),
                                    nights[s], fperv[s], kw["has_perv"])
            dts_by_site[s] = val
            if isbad or not np.isfinite(val):
                bad.append(s)
        per_config_dtsa[cname] = dts_by_site

        obs_v = np.array([OBS[s] for s in sites])
        mod_v = np.array([dts_by_site[s] for s in sites])
        finite = np.isfinite(mod_v)
        if finite.sum() >= 2:
            slope = float(np.polyfit(obs_v[finite], mod_v[finite], 1)[0])
            r = float(np.corrcoef(obs_v[finite], mod_v[finite])[0, 1])
        else:
            slope, r = float("nan"), float("nan")
        n_neg = int(sum(1 for s in OBS_NEG_SITES if s in dts_by_site and dts_by_site[s] < 0))
        min_site = min(sites, key=lambda s: dts_by_site[s])
        min_val = dts_by_site[min_site]
        lip = dts_by_site.get("PL-Lipowa", float("nan"))

        passes = (n_neg >= 4) and (lip >= 4.0) and (slope >= 0.5) and (r >= 0.6)
        crit = {
            "n_neg>=4": (n_neg, n_neg >= 4),
            "Lipowa>=+4.0": (round(lip, 3), lip >= 4.0),
            "slope>=0.5": (round(slope, 3), slope >= 0.5),
            "r>=0.6": (round(r, 3), r >= 0.6),
        }
        results[cname] = {
            "slope": slope, "r": r, "n_neg_of_9": n_neg,
            "min_site": min_site, "min_val": min_val,
            "lipowa": lip, "pass": bool(passes),
            "criteria": {k: {"value": v[0], "met": bool(v[1])} for k, v in crit.items()},
            "bad_sites": bad,
        }

    # ---- print tables ----
    print("\n" + "=" * 100)
    print("PER-SITE dTsa_model (K)  [obs in brackets]")
    print("=" * 100)
    hdr = f"{'site':18s} {'obs':>6s} " + " ".join(f"{c:>7s}" for c in CONFIGS)
    print(hdr)
    order = sorted(sites, key=lambda s: OBS[s])
    for s in order:
        line = f"{s:18s} {OBS[s]:>6.2f} " + " ".join(
            f"{per_config_dtsa[c][s]:>7.2f}" for c in CONFIGS)
        print(line)

    print("\n" + "=" * 100)
    print("PER-CONFIG SUMMARY")
    print("=" * 100)
    print(f"{'config':7s} {'slope':>7s} {'r':>6s} {'n_neg/9':>8s} {'min_val':>8s} "
          f"{'min_site':>18s} {'Lipowa':>8s}  {'PASS':>5s}")
    for c in CONFIGS:
        R = results[c]
        print(f"{c:7s} {R['slope']:>7.2f} {R['r']:>6.2f} {R['n_neg_of_9']:>8d} "
              f"{R['min_val']:>8.2f} {R['min_site']:>18s} {R['lipowa']:>8.2f}  "
              f"{'YES' if R['pass'] else 'no':>5s}")

    print("\nCRITERIA DETAIL (value, met):")
    for c in CONFIGS:
        print(f"  {c}: {results[c]['criteria']}  bad_sites={results[c]['bad_sites']}")

    out_path = ROOT / "external" / "dslucm_surgery" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "n_sites": len(sites), "sites": sites,
            "f_roof": F_ROOF, "C_perv": C_PERV, "mu_perv": MU_PERV,
            "tau_wall": TAU_WALL, "tau_road": TAU_ROAD,
            "obs_neg_sites": OBS_NEG_SITES,
            "pass_rule": "n_neg>=4 AND Lipowa>=+4.0 AND slope>=0.5 AND r>=0.6",
        },
        "obs": OBS, "f_perv": fperv,
        "per_site_dtsa": per_config_dtsa,
        "summary": results,
    }
    json.dump(payload, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
