"""Build standalone-TEB driver inputs for the Urban-PLUMBER 20-site corpus.

For each site, writes external/teb_runs/<SITE>/ containing:
  input.nml     -- CAPITOUL example namelist as template, with site-specific
                   time/geometry/initial-state keys replaced (materials keep
                   the canonical example defaults; disclosed in the paper)
  input/Forc_*.txt -- single-column half-hourly forcing in TEB driver format
  output/       -- empty dir for driver outputs

Forcing mapping (corpus -> TEB):
  Tair->Forc_TA (K), |Wind|->Forc_WIND (m/s, floor 0.1), LWdown->Forc_LW,
  PSurf->Forc_PS (Pa), Qair->Forc_QA (kg/kg), SWdown -> Erbs diffuse split
  into Forc_DIR_SW + Forc_SCA_SW using solar zenith from site lat/lon and
  the LOCAL-STANDARD-TIME axis (corpus time is local standard; we convert
  to UTC via station utc_offset for TEB's internal solar geometry).
  RAIN/SNOW = 0 (nocturnal Tair target; impervious assumption as in dSLUCM),
  CO2 = 0.000620 kg/m3 (example default), wind direction Forc_DIR = 0.

Gap policy: forcing NaNs are linearly interpolated (TEB cannot ingest NaN);
evaluation later uses the corpus training_mask, which excludes non-observed
Tair -- interpolation only keeps the integration alive, mirroring the
dSLUCM pipeline's approach.

Geometry from FullCollection sitedata CSVs:
  ZBLD_HEIGHT = building_mean_height
  ZWALL_O_HOR = wall_to_plan_area_ratio
  ZGARDEN     = tree+grass+bare_soil+water fractions
  ZBLD        = roof_area_fraction / (1 - ZGARDEN)   (building fraction of
                the artificial part; TEB averages garden separately)
  ZZ0         = roughness_length_momentum
  ZZREF       = max(measurement_height_above_ground - building_mean_height, 2)
  ZH_TRAFFIC  = anthropogenic_heat_flux_mean (all assigned to traffic)
"""
from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.corpus_loader import load_all_sites  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEB_DIR = PROJECT_ROOT / "external" / "teb"
RUNS_DIR = PROJECT_ROOT / "external" / "teb_runs"
SITEDATA_DIR = PROJECT_ROOT / "data" / "urban-plumber" / "FullCollection"
TEMPLATE_NML = TEB_DIR / "examples" / "CAPITOUL" / "input.nml"

UTC_OFFSETS_NC = PROJECT_ROOT / "data" / "urban-plumber" / "obs_in_one" / \
    "UP_all_clean_observations_localstandardtime_v1.nc"


def read_sitedata(site: str) -> dict[str, float]:
    path = SITEDATA_DIR / site / f"{site}_sitedata_v1.csv"
    vals: dict[str, float] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                vals[row["parameter"]] = float(row["value"])
            except (TypeError, ValueError):
                pass
    return vals


def utc_offset_hours(site: str) -> float:
    import xarray as xr
    with xr.open_dataset(UTC_OFFSETS_NC) as ds:
        names = [n.item().decode() if hasattr(n.item(), "decode") else str(n.item())
                 for n in ds["station_name"]]
        idx = names.index(site)
        return float(ds["station_utc_offset"].values[idx])


def solar_cos_zenith(time_utc_ns: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
    """NOAA-style solar position, adequate for an Erbs split."""
    t = time_utc_ns.astype("datetime64[s]").astype("int64")
    days = t / 86400.0
    jd = days + 2440587.5
    n = jd - 2451545.0
    L = np.radians((280.460 + 0.9856474 * n) % 360.0)
    g = np.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 0.0000004 * n)
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    # equation of time (minutes)
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    eot = 4.0 * np.degrees(((L - ra) + np.pi) % (2 * np.pi) - np.pi)
    frac_day = (t % 86400) / 3600.0  # UTC hours
    tst = frac_day * 60.0 + eot + 4.0 * lon_deg  # true solar time, minutes
    ha = np.radians(tst / 4.0 - 180.0)
    lat = math.radians(lat_deg)
    cosz = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    return np.clip(cosz, -1.0, 1.0)


def erbs_split(sw: np.ndarray, cosz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Erbs et al. (1982) diffuse fraction from clearness index."""
    s0 = 1361.0
    ext = s0 * np.clip(cosz, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(ext > 1.0, sw / ext, 0.0)
    kt = np.clip(kt, 0.0, 1.2)
    fd = np.where(
        kt <= 0.22, 1.0 - 0.09 * kt,
        np.where(kt <= 0.80,
                 0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4,
                 0.165))
    fd = np.where(ext <= 1.0, 1.0, fd)  # sun below horizon: all diffuse
    sca = sw * fd
    direct = sw - sca
    return direct, sca


def interp_nan(a: np.ndarray, fill: float) -> np.ndarray:
    a = a.astype(np.float64).copy()
    bad = ~np.isfinite(a)
    if bad.all():
        a[:] = fill
        return a
    if bad.any():
        idx = np.arange(a.size)
        a[bad] = np.interp(idx[bad], idx[~bad], a[~bad])
    return a


def write_col(path: Path, arr: np.ndarray) -> None:
    with path.open("w", newline="\n") as f:
        for v in arr:
            f.write(f"{v:20.5f}\n")


def set_nml(text: str, key: str, value: str) -> str:
    pat = re.compile(rf"^(\s*{key}\s*=\s*)\S[^!\n]*", re.M)
    new, n = pat.subn(lambda m: m.group(1) + value + "  ", text, count=1)
    if n != 1:
        raise KeyError(f"namelist key not found: {key}")
    return new


def build_site(rec, template: str) -> Path:
    site = rec.site
    sd = read_sitedata(site)
    off = utc_offset_hours(site)

    out = RUNS_DIR / site
    (out / "input").mkdir(parents=True, exist_ok=True)
    (out / "output").mkdir(exist_ok=True)

    f = rec.forcing
    ta = interp_nan(f["Tair"], 285.0)
    qa = np.clip(interp_nan(f["Qair"], 0.005), 1e-5, 0.03)
    ps = interp_nan(f["PSurf"], 101325.0)
    # TEB driver expects Forc_QA in kg m-3 (volumetric); convert kg/kg -> kg/m3
    rho_air = ps / (287.06 * ta * (1.0 + 0.608 * qa))
    qa_vol = qa * rho_air
    lw = interp_nan(f["LWdown"], 320.0)
    sw = np.clip(interp_nan(f["SWdown"], 0.0), 0.0, None)
    u = np.clip(np.sqrt(interp_nan(f["Wind_N"], 0.5) ** 2 +
                        interp_nan(f["Wind_E"], 0.5) ** 2), 0.1, None)

    time_utc = rec.time_axis - np.timedelta64(int(round(off * 3600)), "s")
    cosz = solar_cos_zenith(time_utc, rec.latitude, rec.longitude)
    sw_dir, sw_sca = erbs_split(sw, cosz)

    inp = out / "input"
    write_col(inp / "Forc_TA.txt", ta)
    write_col(inp / "Forc_QA.txt", qa_vol)
    write_col(inp / "Forc_PS.txt", ps)
    write_col(inp / "Forc_LW.txt", lw)
    write_col(inp / "Forc_WIND.txt", u)
    write_col(inp / "Forc_DIR_SW.txt", sw_dir)
    write_col(inp / "Forc_SCA_SW.txt", sw_sca)
    write_col(inp / "Forc_RAIN.txt", np.zeros_like(ta))
    write_col(inp / "Forc_SNOW.txt", np.zeros_like(ta))
    write_col(inp / "Forc_DIR.txt", np.zeros_like(ta))
    write_col(inp / "Forc_CO2.txt", np.full_like(ta, 0.00062))

    # --- namelist ---------------------------------------------------------
    t0 = time_utc[0].astype("datetime64[s]").item()
    n_steps = int(rec.n_steps)

    # TEB fractions are of the WHOLE town tile: ZBLD + ZGARDEN + road = 1.
    zgarden = min(0.90, max(0.0,
        sd.get("tree_area_fraction", 0.0) + sd.get("grass_area_fraction", 0.0)
        + sd.get("bare_soil_area_fraction", 0.0) + sd.get("water_area_fraction", 0.0)))
    zbld = min(0.95, max(0.05, sd.get("roof_area_fraction", 0.4)))
    if zbld + zgarden > 0.97:  # keep road fraction >= 3%
        zgarden = max(0.0, 0.97 - zbld)
    zh = max(3.0, sd.get("building_mean_height", 10.0))
    zzref = max(2.0, sd.get("measurement_height_above_ground", 30.0) - zh)
    wall = max(0.1, sd.get("wall_to_plan_area_ratio",
                           2.0 * sd.get("canyon_height_width_ratio", 1.0) * zbld))
    zz0 = max(0.05, sd.get("roughness_length_momentum", 1.0))
    # Headline protocol: anthropogenic heat OFF, matching the dSLUCM/rTEB
    # configurations (neither contains QF). TEB injects ZH_TRAFFIC as a 24/7
    # constant, which grossly overweights nights at high-QF sites
    # (KR-Jungnang 92.7 W/m2 -> +3.7 K artefact). Pass --with-qf to write the
    # sitedata mean instead (QF-on sensitivity configuration).
    if "--with-qf" in sys.argv:
        qf = max(0.0, sd.get("anthropogenic_heat_flux_mean", 0.0))
    else:
        qf = 0.0

    nml = template
    nml = set_nml(nml, "IYEAR", str(t0.year))
    nml = set_nml(nml, "IMONTH", str(t0.month))
    nml = set_nml(nml, "IDAY", str(t0.day))
    nml = set_nml(nml, "ZTIME_START", f"{t0.hour * 3600 + t0.minute * 60:.0f}.")
    nml = set_nml(nml, "ZLON", f"{rec.longitude:.4f}")
    nml = set_nml(nml, "ZLAT", f"{rec.latitude:.4f}")
    nml = set_nml(nml, "INB_STEP_ATM", str(n_steps - 1))
    nml = set_nml(nml, "ZZREF", f"{zzref:.1f}")
    nml = set_nml(nml, "ZZ0", f"{zz0:.3f}")
    nml = set_nml(nml, "ZBLD", f"{zbld:.3f}")
    nml = set_nml(nml, "ZGARDEN", f"{zgarden:.3f}")
    nml = set_nml(nml, "ZBLD_HEIGHT", f"{zh:.1f}")
    nml = set_nml(nml, "ZWALL_O_HOR", f"{wall:.3f}")
    nml = set_nml(nml, "ZH_TRAFFIC", f"{qf:.1f}")
    # Initial state: all outdoor-facing layer temperatures start at first
    # forcing Tair (cold start; first 6 months are excluded by the corpus
    # pre-spinup mask anyway). Indoor ZTI_BLD keeps the template default.
    t_init = f"{float(ta[0]):.2f}"
    nml = re.sub(r"^(\s*ZT_(?:ROAD|ROOF|WALL|FLOOR|MASS)\(:,\d+\)\s*=\s*)\S+",
                 lambda m: m.group(1) + t_init, nml, flags=re.M)
    nml = re.sub(r"^(\s*ZT_(?:CANYON|WIN1|WIN2)\s*=\s*)\S+",
                 lambda m: m.group(1) + t_init, nml, flags=re.M)
    nml = re.sub(r"^(\s*ZQ_CANYON\s*=\s*)\S+",
                 lambda m: m.group(1) + f"{float(qa[0]):.5f}", nml, flags=re.M)
    (out / "input.nml").write_text(nml, newline="\n")
    return out


def main() -> int:
    only = sys.argv[sys.argv.index("--site") + 1] if "--site" in sys.argv else None
    template = TEMPLATE_NML.read_text()
    records = load_all_sites()
    done = []
    for rec in records:
        if only and rec.site != only:
            continue
        out = build_site(rec, template)
        done.append(rec.site)
        print(f"[teb-inputs] {rec.site}: n={rec.n_steps} -> {out}")
    print(f"[teb-inputs] wrote {len(done)} sites")
    return 0


if __name__ == "__main__":
    sys.exit(main())
