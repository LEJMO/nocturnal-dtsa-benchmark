# -*- coding: utf-8 -*-
"""CLMU5 post-processing, v11 — alignment fix only.

WHY THIS FILE EXISTS
--------------------
clmu_post.py (Category D, left untouched) selects the model record-period window
by scanning `range(len(fire) - n, -1, -1)` and taking the global arg-min of the
Tair-echo RMSE. Two defects follow:

  (1) The scan cannot reach any offset above `len(model) - n`, so a record whose
      model output is even ONE step short of the spin-up length can never find its
      true window. Diagnosed: the true offset is min(17520, n) + 1 at ALL 19
      records (= one recycled spin-up year, +1 for the pairing convention), but
      GR-HECKOR (headroom 17520) and UK-KingsCollege (headroom 17520) fall one
      step short and are silently aligned to the WRONG period -- Tair-echo RMSE
      0.174 K and 0.341 K respectively, versus exactly 0 at the true offset.
  (2) Taking a global arg-min instead of requiring an exact echo match lets such a
      mis-alignment pass without any error.

THE FIX (metric definitions are NOT touched)
--------------------------------------------
  * Candidate offsets run to len(model) - 1, with a TRUNCATED window: only the
    overlapping m = min(n, len(model) - off) steps are compared and evaluated.
  * An EXACT echo match (rmse == 0) is required. Among exact matches the
    physically grounded offset min(17520, n) + 1 is preferred; this reproduces the
    archive's choice at SG-TelokKurau06, whose window is genuinely degenerate
    (offsets 1 and 16081 both match exactly because n = 16080 <= 17520).
  * If no exact match exists the record HARD-FAILS rather than returning a
    silently mis-aligned number.

Control: at the 17 records the archive already aligned correctly, this file
reproduces clmu_post.py bit-for-bit (verified; see clmu_align_report.json).

Usage: clmu_post_v11.py <SITE> <history.nc> [--json out.json]
"""
import io, sys, json
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
SIG, EPS = 5.67e-8, 0.95
SPINUP_YEAR = 17520          # half-hourly steps in one recycled forcing year


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


def _echo_rmse(tair_m, ta, off, n):
    """RMSE of the model Tair echo against the corpus forcing Tair over the
    m = min(n, len(model) - off) overlapping steps. Returns (rmse, m)."""
    m = min(n, len(tair_m) - off)
    if m <= 0:
        return np.inf, 0
    d = tair_m[off:off + m] - ta[:m]
    return float(np.sqrt(np.nanmean(d * d))), m


def align(tair_m, ta, n):
    """Return (offset, m, rmse). Requires an EXACT Tair-echo match; allows a
    truncated final window. Raises if no exact match exists.

    The physically grounded offset min(SPINUP_YEAR, n) + 1 is tested first, so
    the common case costs one window comparison instead of a full scan."""
    preferred = min(SPINUP_YEAR, n) + 1
    if 0 <= preferred < len(tair_m):
        r, m = _echo_rmse(tair_m, ta, preferred, n)
        if r == 0.0 and m >= n * 0.5:
            return preferred, m, 0.0

    # fall back to a descending scan (archive's preference order among ties)
    exact = []
    for off in range(len(tair_m) - 1, -1, -1):
        r, m = _echo_rmse(tair_m, ta, off, n)
        if m < n * 0.5:
            continue
        if r == 0.0:
            exact.append((off, m))
            break                            # first (highest) exact match wins
    if not exact:
        raise RuntimeError(
            f"no exact Tair-echo alignment found (n={n}, model_steps={len(tair_m)}); "
            "refusing to return a silently mis-aligned metric")
    off, m = exact[0]
    return off, m, 0.0


def metrics(site, hist_path):
    C = corpus(site)
    n = len(C["olw"])
    h = xr.open_dataset(hist_path)
    fire = np.asarray(h["FIRE"].values).reshape(-1)
    flds = np.asarray(h["FLDS"].values).reshape(-1)
    tair_m = np.asarray(h["Tair"].values).reshape(-1) if "Tair" in h else \
        np.asarray(h["TBOT"].values).reshape(-1)
    h.close()

    off, m, rmse = align(tair_m, C["ta"], n)
    # every series is evaluated on the SAME m overlapping steps
    lw = fire[off:off + m]
    ld = flds[off:off + m]
    night = C["night"][:m]
    presp = C["presp"][:m]
    olw = C["olw"][:m]
    ta = C["ta"][:m]

    msk = night & ~presp
    okb = msk & np.isfinite(olw) & np.isfinite(lw)
    okd = msk & np.isfinite(lw) & np.isfinite(ld) & np.isfinite(ta)
    lwup_bias = float(np.mean(lw[okb] - olw[okb]))
    ts = ((lw[okd] - (1 - EPS) * ld[okd]) / (EPS * SIG)) ** 0.25
    dtsa = float(np.mean(ts - ta[okd]))
    return dict(site=site, lwup_bias=lwup_bias, dTsa_model=dtsa,
                n_night=int(okb.sum()), n_dtsa=int(okd.sum()),
                align_offset=int(off), align_window=int(m),
                align_rmse_Tair=rmse, truncated=bool(m < n),
                model_steps=int(len(fire)), corpus_steps=int(n))


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    site = sys.argv[1]; hist = sys.argv[2]
    r = metrics(site, hist)
    print(json.dumps(r, indent=1))
    if "--json" in sys.argv:
        json.dump(r, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
