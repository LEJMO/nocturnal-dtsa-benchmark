# -*- coding: utf-8 -*-
"""Reconstructed CLMU5 post-processing (the lost post.py), validated to
reproduce the archived KR-Ochang metrics (dTsa_model=0.046, lwup_bias=10.79).

Reads a CLM2 history .nc (FIRE=total upward LW, FLDS=downward LW, Tair echo)
plus the corpus for the site (obs_LWup, forcing_Tair/LWdown, night mask), finds
the spin-up offset by matching the model Tair echo to the corpus forcing_Tair,
and computes nocturnal LW-up bias and dTsa via the identical inversion (eps=0.95).

Usage: clmu_post.py <SITE> <history.nc> [--json out.json]
"""
import io, sys, json
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
SIG, EPS = 5.67e-8, 0.95


def corpus(site):
    ds = xr.open_dataset(ROOT / f"data/urban-plumber/corpus/{site}.nc")
    out = dict(
        night=ds["night_mask"].values.astype(bool),
        presp=ds["pre_spinup_flag"].values.astype(bool),
        olw=ds["obs_LWup"].values.astype(float),
        ta=ds["forcing_Tair"].values.astype(float),
        ld=ds["forcing_LWdown"].values.astype(float))
    ds.close()
    return out


def metrics(site, hist_path):
    C = corpus(site)
    n = len(C["olw"])
    h = xr.open_dataset(hist_path)
    fire = np.asarray(h["FIRE"].values).reshape(-1)
    flds = np.asarray(h["FLDS"].values).reshape(-1)
    tair_m = np.asarray(h["Tair"].values).reshape(-1) if "Tair" in h else \
        np.asarray(h["TBOT"].values).reshape(-1)
    h.close()
    # find spin-up offset: the model record period (length n) whose Tair echo
    # matches the corpus forcing_Tair. Search near model_steps - n.
    best, bestrmse = None, 1e9
    for off in range(len(fire) - n, -1, -1):
        seg = tair_m[off:off + n]
        r = float(np.sqrt(np.nanmean((seg - C["ta"]) ** 2)))
        if r < bestrmse:
            bestrmse, best = r, off
        if r < 1e-3:
            break
    off = best
    m = C["night"] & ~C["presp"]
    lw = fire[off:off + n]
    ld = flds[off:off + n]
    # lwup_bias mask requires finite obs; dTsa mask needs only the model.
    okb = m & np.isfinite(C["olw"]) & np.isfinite(lw)
    okd = m & np.isfinite(lw) & np.isfinite(ld) & np.isfinite(C["ta"])
    lwup_bias = float(np.mean(lw[okb] - C["olw"][okb]))
    ts = ((lw[okd] - (1 - EPS) * ld[okd]) / (EPS * SIG)) ** 0.25
    dtsa = float(np.mean(ts - C["ta"][okd]))
    return dict(site=site, lwup_bias=lwup_bias, dTsa_model=dtsa,
                n_night=int(okb.sum()), n_dtsa=int(okd.sum()),
                align_offset=int(off), align_rmse_Tair=bestrmse,
                model_steps=int(len(fire)))


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    site = sys.argv[1]; hist = sys.argv[2]
    r = metrics(site, hist)
    print(json.dumps(r, indent=1))
    if "--json" in sys.argv:
        json.dump(r, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
