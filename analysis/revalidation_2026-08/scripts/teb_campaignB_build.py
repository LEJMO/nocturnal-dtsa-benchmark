# -*- coding: utf-8 -*-
"""Build TEB Campaign B (equilibrium spin-up) inputs: full metforcing span
(~10-yr prepend + record period) per site, identical mappings to the baseline
builder, reusing the baseline namelist geometry and only re-timing it.

Writes external/teb_spinup10/<SITE>/{input.nml, input/Forc_*.txt, output/}.
The driver is run separately (bash), producing output/LWU.txt over the full
span; evaluation (teb_campaign_eval.py) uses only the record-period tail.
"""
import io, re, sys, json
from pathlib import Path
import numpy as np
import xarray as xr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
# reuse the exact mapping helpers from the baseline builder
from teb_make_inputs import (solar_cos_zenith, erbs_split, interp_nan,
                             write_col, set_nml, utc_offset_hours)  # noqa: E402

RUNS = ROOT / "external/teb_runs"
FULL = ROOT / "data/urban-plumber/FullCollection"
OUT = ROOT / "external/teb_spinup10"
SITES = sorted(json.load(open(ROOT / "results/paper_stats_v1.json",
                              encoding="utf-8"))["per_site"])
meta = {}
for s in SITES:
    mf = xr.open_dataset(FULL / s / "timeseries" / f"{s}_metforcing_v1.nc").squeeze(drop=True)
    lat = float(mf.attrs.get("latitude", 0.0)); lon = float(mf.attrs.get("longitude", 0.0))
    if lat == 0.0:  # some files store lat/lon as vars
        lat = float(mf["latitude"].values) if "latitude" in mf else 0.0
        lon = float(mf["longitude"].values) if "longitude" in mf else 0.0
    t = np.asarray(mf["time"].values)
    ta = interp_nan(mf["Tair"].values.astype(float), 285.0)
    qa = np.clip(interp_nan(mf["Qair"].values.astype(float), 0.005), 1e-5, 0.03)
    ps = interp_nan(mf["PSurf"].values.astype(float), 101325.0)
    lw = interp_nan(mf["LWdown"].values.astype(float), 320.0)
    sw = np.clip(interp_nan(mf["SWdown"].values.astype(float), 0.0), 0.0, None)
    wn = interp_nan(mf["Wind_N"].values.astype(float), 0.5)
    we = interp_nan(mf["Wind_E"].values.astype(float), 0.5)
    mf.close()
    u = np.clip(np.sqrt(wn ** 2 + we ** 2), 0.1, None)
    rho = ps / (287.06 * ta * (1.0 + 0.608 * qa))
    qa_vol = qa * rho
    off = utc_offset_hours(s)
    time_utc = t - np.timedelta64(int(round(off * 3600)), "s")
    cosz = solar_cos_zenith(time_utc, lat, lon)
    sw_dir, sw_sca = erbs_split(sw, cosz)

    wd = OUT / s
    (wd / "input").mkdir(parents=True, exist_ok=True)
    (wd / "output").mkdir(exist_ok=True)
    inp = wd / "input"
    write_col(inp / "Forc_TA.txt", ta); write_col(inp / "Forc_QA.txt", qa_vol)
    write_col(inp / "Forc_PS.txt", ps); write_col(inp / "Forc_LW.txt", lw)
    write_col(inp / "Forc_WIND.txt", u)
    write_col(inp / "Forc_DIR_SW.txt", sw_dir); write_col(inp / "Forc_SCA_SW.txt", sw_sca)
    write_col(inp / "Forc_RAIN.txt", np.zeros_like(ta))
    write_col(inp / "Forc_SNOW.txt", np.zeros_like(ta))
    write_col(inp / "Forc_DIR.txt", np.zeros_like(ta))
    write_col(inp / "Forc_CO2.txt", np.full_like(ta, 0.00062))

    # reuse baseline namelist geometry/materials; only re-time + re-init
    nml = (RUNS / s / "input.nml").read_text()
    t0 = time_utc[0].astype("datetime64[s]").item()
    nml = set_nml(nml, "IYEAR", str(t0.year))
    nml = set_nml(nml, "IMONTH", str(t0.month))
    nml = set_nml(nml, "IDAY", str(t0.day))
    nml = set_nml(nml, "ZTIME_START", f"{t0.hour*3600 + t0.minute*60:.0f}.")
    nml = set_nml(nml, "INB_STEP_ATM", str(len(ta) - 1))
    t_init = f"{float(ta[0]):.2f}"
    nml = re.sub(r"^(\s*ZT_(?:ROAD|ROOF|WALL|FLOOR|MASS)\(:,\d+\)\s*=\s*)\S+",
                 lambda m: m.group(1) + t_init, nml, flags=re.M)
    nml = re.sub(r"^(\s*ZT_(?:CANYON|WIN1|WIN2)\s*=\s*)\S+",
                 lambda m: m.group(1) + t_init, nml, flags=re.M)
    nml = re.sub(r"^(\s*ZQ_CANYON\s*=\s*)\S+",
                 lambda m: m.group(1) + f"{float(qa[0]):.5f}", nml, flags=re.M)
    (wd / "input.nml").write_text(nml, newline="\n")

    n_record = int(json.load(open(ROOT / "results/paper_stats_v1.json",
                                  encoding="utf-8"))["per_site"][s]["n_night"]) if False else None
    meta[s] = {"full_len": int(len(ta)), "n_prepend": int(len(ta) -
               xr.open_dataset(ROOT / f"data/urban-plumber/corpus/{s}.nc").sizes["time"])}
    print(f"[B-build] {s}: full_len={len(ta)} prepend={meta[s]['n_prepend']}")

json.dump(meta, open(OUT / "_build_meta.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print(f"[B-build] wrote {len(SITES)} sites")
