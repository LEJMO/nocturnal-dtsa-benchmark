"""Week 2-3 — Build the Path C per-site training corpus.

Reads Urban-PLUMBER FullCollection (Zenodo 10.5281/zenodo.7104984, v1 652 MB)
and emits one NetCDF per site containing metforcing (ERA5 gap-filled) aligned
with the clean observations window, plus derived masks.

Output layout (Path C 9-gate convention):
  data/urban-plumber/corpus/{SITE}.nc           — per-site processed NetCDF
  data/urban-plumber/corpus_index.json           — manifest + per-site metadata

Per-site NetCDF contents (dims: time):
  * forcing_SWdown, forcing_LWdown, forcing_Tair, forcing_Qair, forcing_PSurf,
    forcing_Wind_N, forcing_Wind_E  (all 100% finite from metforcing)
  * obs_Tair, obs_Qh, obs_Qle, obs_SWup, obs_LWup, obs_Qg  (NaN where obs gap)
  * night_mask (SWdown < 1 W/m^2)
  * season (0=DJF_NH, 1=MAM_NH, 2=JJA_NH, 3=SON_NH; SH sites flipped)
  * pre_spinup_flag (True for first 6 months after the observation start; for
    training we mask loss inside this window to let the JAX dSLUCM state
    equilibrate before being scored)

Attrs: latitude, longitude, koppen_zone, timestep_s, site_name, obs_start,
       obs_end, obs_months_total, path_c_version="v1_9gate".

Design notes
------------
* We retain each site's native observation window (0.9 yr -- 5.0 yr), NOT a
  fixed 12-month cut. Rationale: Urban-PLUMBER site observation lengths are
  heterogeneous (see data/urban-plumber/FullCollection/<site>/); clipping every
  site to 12 months would discard many site-years. The Path C spec of
  "20 sites * 1-year" is a minimum per site, not a maximum. Longer records
  strengthen LOSO stability.
* AU-SurreyHills (0.4 yr obs) is excluded: below the 0.5-yr minimum required
  to cover 2 seasons after spin-up.
* Hourly sites (JP-Yoyogi, PL-Lipowa, PL-Narutowicza, US-Baltimore; dt=3600 s)
  are upsampled to the 1800-s canonical cadence by forward-fill so the whole
  corpus shares one timestep.
* Cross-UCM scope (former stop-loss #5) dropped 2026-04-24 (R11 materialisation).
  See R11_materialization_report.md.

This script is rerunnable; pass --force to overwrite.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "urban-plumber"
FULL_DIR = DATA_DIR / "FullCollection"
OUT_DIR = DATA_DIR / "corpus"
INDEX_JSON = DATA_DIR / "corpus_index.json"

CANONICAL_DT_S = 1800
MIN_OBS_MONTHS = 6  # AU-SurreyHills at 0.4 yr is below this -> skip
SPINUP_MONTHS = 6

FORCING_VARS = ["SWdown", "LWdown", "Tair", "Qair", "PSurf", "Wind_N", "Wind_E"]
OBS_VARS = ["Tair", "Qh", "Qle", "SWup", "LWup", "Qg"]

SITE_KOPPEN = {
    "AU-Preston": "Cfb",
    "AU-SurreyHills": "Cfb",
    "CA-Sunset": "Cfb",
    "FI-Kumpula": "Dfb",
    "FI-Torni": "Dfb",
    "FR-Capitole": "Cfb",
    "GR-HECKOR": "Csa",
    "JP-Yoyogi": "Cfa",
    "KR-Jungnang": "Dwa",
    "KR-Ochang": "Dwa",
    "MX-Escandon": "Cwb",
    "NL-Amsterdam": "Cfb",
    "PL-Lipowa": "Dfb",
    "PL-Narutowicza": "Dfb",
    "SG-TelokKurau06": "Af",
    "UK-KingsCollege": "Cfb",
    "UK-Swindon": "Cfb",
    "US-Baltimore": "Cfa",
    "US-Minneapolis1": "Dfa",
    "US-Minneapolis2": "Dfa",
    "US-WestPhoenix": "BWh",
}


def discover_sites() -> list[str]:
    if not FULL_DIR.exists():
        sys.stderr.write(f"[week2] {FULL_DIR} missing; unzip FullCollection first\n")
        sys.exit(2)
    return sorted(p.name for p in FULL_DIR.iterdir() if p.is_dir())


def site_paths(site: str) -> tuple[Path, Path]:
    ts = FULL_DIR / site / "timeseries"
    return (
        ts / f"{site}_metforcing_v1.nc",
        ts / f"{site}_clean_observations_v1.nc",
    )


def load_and_align_site(site: str) -> tuple[xr.Dataset, dict]:
    """Load metforcing + observations, align to obs window, resample to 1800 s."""
    mf_p, co_p = site_paths(site)
    if not mf_p.exists() or not co_p.exists():
        raise FileNotFoundError(f"{site}: metforcing or clean_observations missing")

    mf = xr.open_dataset(mf_p).squeeze(drop=True)
    co = xr.open_dataset(co_p).squeeze(drop=True)

    obs_t = pd.to_datetime(co.time.values)
    obs_start, obs_end = obs_t[0], obs_t[-1]
    obs_months = (obs_end - obs_start).total_seconds() / (30 * 24 * 3600)
    if obs_months < MIN_OBS_MONTHS:
        raise ValueError(f"{site}: obs window {obs_months:.1f} mo < min {MIN_OBS_MONTHS}")

    native_dt = int(np.median(np.diff(mf.time.values).astype("timedelta64[s]").astype(int)))

    # Align metforcing to observation window (with a small buffer for resampling)
    mf_win = mf.sel(time=slice(obs_start, obs_end))
    co_win = co.sel(time=slice(obs_start, obs_end))

    # Resample to canonical 1800 s if needed
    def resample_to_canonical(ds: xr.Dataset) -> xr.Dataset:
        if native_dt == CANONICAL_DT_S:
            return ds
        if native_dt == 2 * CANONICAL_DT_S:
            # Hourly -> 30-min by ffill
            return ds.resample(time=f"{CANONICAL_DT_S}s").ffill()
        if native_dt < CANONICAL_DT_S and CANONICAL_DT_S % native_dt == 0:
            return ds.resample(time=f"{CANONICAL_DT_S}s").mean()
        raise ValueError(f"{site}: unsupported native dt {native_dt} s")

    mf_win = resample_to_canonical(mf_win)
    co_win = resample_to_canonical(co_win)

    # Reindex obs to forcing time axis (obs has the same window; insert NaN where missing)
    co_win = co_win.reindex(time=mf_win.time, method=None)

    time_axis = mf_win.time.values
    n = len(time_axis)

    # Extract arrays
    forcing = {v: np.asarray(mf_win[v].values, dtype=np.float64) if v in mf_win.data_vars else np.full(n, np.nan) for v in FORCING_VARS}
    obs = {v: np.asarray(co_win[v].values, dtype=np.float64) if v in co_win.data_vars else np.full(n, np.nan) for v in OBS_VARS}

    # Night mask (SWdown < 1 W/m^2)
    night_mask = np.zeros(n, dtype=bool)
    sw = forcing["SWdown"]
    finite_sw = np.isfinite(sw)
    night_mask[finite_sw] = sw[finite_sw] < 1.0

    # Season encoding: pull latitude from variables first, then attrs, else 0
    lat = 0.0
    if "latitude" in mf.variables:
        try:
            lat = float(np.atleast_1d(mf["latitude"].values).ravel()[0])
        except Exception:
            pass
    elif "latitude" in mf.attrs:
        try:
            lat = float(mf.attrs["latitude"])
        except Exception:
            pass
    months = pd.to_datetime(time_axis).month
    nh_season = np.where(np.isin(months, [12, 1, 2]), 0,
                 np.where(np.isin(months, [3, 4, 5]), 1,
                 np.where(np.isin(months, [6, 7, 8]), 2, 3))).astype("int8")
    if lat < 0:
        flip = {0: 2, 1: 3, 2: 0, 3: 1}
        nh_season = np.vectorize(flip.get)(nh_season).astype("int8")

    # Spin-up flag: first SPINUP_MONTHS of the observation window
    spinup_end = pd.to_datetime(time_axis[0]) + pd.DateOffset(months=SPINUP_MONTHS)
    pre_spinup_flag = np.asarray(pd.to_datetime(time_axis) < spinup_end, dtype=bool)

    # Longitude
    lon = 0.0
    if "longitude" in mf.variables:
        try:
            lon = float(np.atleast_1d(mf["longitude"].values).ravel()[0])
        except Exception:
            pass
    elif "longitude" in mf.attrs:
        try:
            lon = float(mf.attrs["longitude"])
        except Exception:
            pass

    ds_out = xr.Dataset(
        data_vars={
            **{f"forcing_{v}": (("time",), forcing[v]) for v in FORCING_VARS},
            **{f"obs_{v}": (("time",), obs[v]) for v in OBS_VARS},
            "night_mask": (("time",), night_mask),
            "season": (("time",), nh_season),
            "pre_spinup_flag": (("time",), pre_spinup_flag),
        },
        coords={"time": time_axis},
        attrs={
            "site_name": site,
            "koppen_zone": SITE_KOPPEN.get(site, "Unknown"),
            "latitude": lat,
            "longitude": lon,
            "timestep_s": CANONICAL_DT_S,
            "obs_start": str(obs_start),
            "obs_end": str(obs_end),
            "obs_months_total": float(obs_months),
            "spin_up_months_masked": SPINUP_MONTHS,
            "native_timestep_s": native_dt,
            "path_c_version": "v1_9gate",
            "source": "Urban-PLUMBER FullCollection v1 (Zenodo 10.5281/zenodo.7104984)",
        },
    )

    mf.close()
    co.close()

    obs_finite_tair = int(np.isfinite(obs["Tair"]).sum())
    obs_finite_qh = int(np.isfinite(obs["Qh"]).sum())
    night_count = int(night_mask.sum())
    night_with_tair = int((night_mask & np.isfinite(obs["Tair"])).sum())

    meta = {
        "site": site,
        "koppen_zone": SITE_KOPPEN.get(site, "Unknown"),
        "latitude": lat,
        "longitude": lon,
        "obs_start": str(obs_start),
        "obs_end": str(obs_end),
        "obs_months_total": round(obs_months, 2),
        "native_timestep_s": native_dt,
        "canonical_timestep_s": CANONICAL_DT_S,
        "n_timesteps": n,
        "obs_finite_Tair": obs_finite_tair,
        "obs_finite_Qh": obs_finite_qh,
        "night_total": night_count,
        "night_with_Tair_obs": night_with_tair,
        "included_in_corpus": True,
    }
    return ds_out, meta


def build_corpus(force: bool) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = {
        "path_c_version": "v1_9gate",
        "built_at": pd.Timestamp.utcnow().isoformat(),
        "canonical_timestep_s": CANONICAL_DT_S,
        "sites": [],
        "excluded": [],
    }
    sites = discover_sites()
    print(f"[week2] {len(sites)} sites in FullCollection")
    built = 0
    night_total = 0
    for site in sites:
        out_nc = OUT_DIR / f"{site}.nc"
        if out_nc.exists() and not force:
            print(f"  [skip] {site}: already processed ({out_nc.name})")
            try:
                existing = xr.open_dataset(out_nc)
                nm = int(existing["night_mask"].sum().item())
                night_total += nm
                existing.close()
            except Exception:
                pass
            continue
        try:
            ds, meta = load_and_align_site(site)
        except (ValueError, FileNotFoundError) as exc:
            print(f"  [exclude] {site}: {exc}")
            index["excluded"].append({"site": site, "reason": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] {site}: {exc}")
            index["excluded"].append({"site": site, "reason": f"error: {exc}"})
            continue

        ds.to_netcdf(out_nc)
        ds.close()
        built += 1
        night_total += meta["night_total"]
        index["sites"].append(meta)
        print(
            f"  [ok] {site:<18s} obs={meta['obs_months_total']:.1f}mo "
            f"steps={meta['n_timesteps']:>6d} night={meta['night_total']:>5d} "
            f"night+obs={meta['night_with_Tair_obs']:>5d}"
        )

    index["summary"] = {
        "sites_included": len(index["sites"]),
        "sites_excluded": len(index["excluded"]),
        "night_samples_total": night_total,
        "night_samples_with_Tair_obs": sum(s["night_with_Tair_obs"] for s in index["sites"]),
    }
    with INDEX_JSON.open("w") as f:
        json.dump(index, f, indent=2)
    print(
        f"[week2] wrote {built} new site files into {OUT_DIR} + index {INDEX_JSON}\n"
        f"  sites_included={len(index['sites'])} excluded={len(index['excluded'])}\n"
        f"  night_samples_total={night_total}\n"
        f"  night_samples_with_Tair_obs={index['summary']['night_samples_with_Tair_obs']}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Overwrite existing corpus files")
    args = ap.parse_args()
    return build_corpus(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
