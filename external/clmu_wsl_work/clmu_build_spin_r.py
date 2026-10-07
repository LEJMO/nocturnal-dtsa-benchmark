"""Build CLMU5 forcing + per-site surfdata + eval metadata for the G1 6 sites.

Frozen protocol mirror (PREREGISTRATION_G1 9c4ee72 + ADDENDUM1 0bcc1b7):
- Forcing from data/urban-plumber/corpus/<SITE>.nc forcing_* (SI units already),
  Rainf from FullCollection metforcing aligned by timestamp.
- Surface data: pyclmuapp default (King's College, 100% MD urban) with ONLY
  per-site geometry overridden {CANYON_HWR, HT_ROOF, WTLUNIT_ROOF, WTROAD_PERV,
  WIND_HGT_CANYON} + location {LATIXY, LONGXY}. ALL materials (TK/CV/EM/ALB/THICK)
  stay at default, identical across sites. Auditable diff printed.
- Spin-up: prepend ONE recycled year (same convention as external/suews_g1
  add_spinup), discarded in evaluation. +2 pad days so datm never reads past end.
- CLM forcing schema replicates pyclmuapp/era_forcing.py exactly (var names/units/
  dims/_FillValue) so the container datm stream reads it (Zbot->z, Prectmms->precn,
  Wind->wind, LWdown->lwdn, PSurf->pbot, Qair->shum, Tair->tbot, SWdown->swdn).
"""
import os, sys, json, csv, shutil
import numpy as np, pandas as pd, xarray as xr

ROOT = "/mnt/d/JM/ResearchManager/nocturnalUHIDiscovery"
CORP = f"{ROOT}/data/urban-plumber/corpus"
FULL = f"{ROOT}/data/urban-plumber/FullCollection"
PKG  = "/root/clmu_work/venv/lib/python3.12/site-packages/pyclmuapp"
DEFAULT_SURF = f"{PKG}/usp/surfdata.nc"
DEFAULT_DOMAIN = f"{PKG}/usp/domain.nc"
OUT = "/root/clmu_work/inputs_spin"
UT = 2  # MD urban density class (the only populated one; PCT_URBAN=[0,0,100])
STEPS_PER_YEAR = int(round(365*24*3600/1800))  # 17520

SITES = ["US-Minneapolis1","US-Minneapolis2","KR-Ochang","US-WestPhoenix","PL-Lipowa","FR-Capitole"]

def read_sitedata(site):
    d={}
    with open(f"{FULL}/{site}/{site}_sitedata_v1.csv", newline='', encoding='utf-8-sig') as f:
        for r in csv.reader(f):
            if len(r)>=3 and r[1] and r[1]!='parameter':
                try: d[r[1]]=float(r[2])
                except ValueError: pass
    return d

def morphology(sd):
    R=sd['roof_area_fraction']; imp=sd['impervious_area_fraction']
    ht=sd['building_mean_height']
    return {
        'CANYON_HWR': sd['canyon_height_width_ratio'],
        'HT_ROOF': ht,
        'WTLUNIT_ROOF': R,                       # roof plan fraction of urban tile
        'WTROAD_PERV': (1.0-imp)/(1.0-R),        # pervious fraction of canyon floor (road)
        'WIND_HGT_CANYON': ht/2.0,               # preserve default HT_ROOF:WIND ratio (15:7.5)
    }

def build_forcing_arrays(site, zbot):
    ds = xr.open_dataset(f"{CORP}/{site}.nc")
    t = pd.to_datetime(ds['time'].values)
    SW = ds['forcing_SWdown'].values.astype(float)
    LW = ds['forcing_LWdown'].values.astype(float)
    Tair = ds['forcing_Tair'].values.astype(float)      # K
    q   = ds['forcing_Qair'].values.astype(float)       # kg/kg
    P   = ds['forcing_PSurf'].values.astype(float)      # Pa
    WN  = ds['forcing_Wind_N'].values.astype(float)
    WE  = ds['forcing_Wind_E'].values.astype(float)
    night = np.asarray(ds['night_mask'].values, bool)
    presp = np.asarray(ds['pre_spinup_flag'].values, bool)
    obs_LWup = ds['obs_LWup'].values.astype(float)
    obs_Qh   = ds['obs_Qh'].values.astype(float)
    ds.close()
    # Rainf from metforcing, aligned by timestamp (mm/s == kg/m2/s)
    mf = xr.open_dataset(f"{FULL}/{site}/timeseries/{site}_metforcing_v1.nc")
    mt = pd.to_datetime(mf['time'].values)
    rainf = mf['Rainf'].values.squeeze().astype(float)
    mf.close()
    rain = pd.Series(rainf, index=mt).reindex(t).values
    rain = np.nan_to_num(rain, nan=0.0)
    rain = np.clip(rain, 0.0, None)
    U = np.hypot(WN, WE)
    SW = np.clip(SW, 0.0, None)
    Z = np.full(len(t), float(zbot))
    core = dict(Tair=Tair, PSurf=P, Qair=np.clip(q,1e-9,None), Wind=U,
                SWdown=SW, LWdown=LW, Prectmms=rain, Zbot=Z)
    meta = dict(time=t, night=night, presp=presp, obs_LWup=obs_LWup, obs_Qh=obs_Qh,
                Tair_K=Tair, LWdown=LW)
    return core, meta, t

def prepend_and_pad(core, t):
    """Prepend 5 recycled years (first-year block tiled), append 2 pad days."""
    n = len(t); dt = pd.Timedelta('1800s')
    base = min(STEPS_PER_YEAR, n); NYR = 5; nsp = base*NYR
    spin_t = pd.date_range(end=t[0]-dt, periods=nsp, freq='1800s')
    pad_steps = 96
    pad_t = pd.date_range(start=t[-1]+dt, periods=pad_steps, freq='1800s')
    times_full = spin_t.append(t).append(pad_t)
    core_full={}
    for k,v in core.items():
        core_full[k] = np.concatenate([np.tile(v[:base], NYR), v, v[-pad_steps:]])
    return core_full, times_full, nsp

def write_forcing(core_full, times_full, path):
    N=len(times_full)
    units={'Tair':'K','PSurf':'Pa','Qair':'kg/kg','Wind':'m/s',
           'Zbot':'m','SWdown':'W/m^2','LWdown':'W/m^2','Prectmms':'mm/s'}
    lname={'Tair':'Air temperature','PSurf':'Surface pressure','Qair':'Specific humidity',
           'Wind':'Wind speed','Zbot':'Observational height','SWdown':'Surface solar radiation downwards',
           'LWdown':'Surface thermal radiation downwards','Prectmms':'Total precipitation'}
    data_vars={}
    for k,v in core_full.items():
        arr=np.asarray(v,dtype='f8').reshape(N,1,1)
        da=xr.DataArray(arr, dims=('time','y','x'))
        da.attrs['units']=units[k]; da.attrs['long_name']=lname[k]; da.attrs['_FillValue']=1.e36
        data_vars[k]=da
    # time as a PLAIN numeric coord (float days since a midnight reference) with the
    # units string attached verbatim -> xarray does NOT CF-re-encode it, so no ISO 'T'
    # separator (CESM shr_string_parseCFtunit rejects 'T'). Midnight ref, space sep.
    ti=pd.DatetimeIndex(times_full)
    t0=ti[0].normalize()
    tvals=((ti - t0)/pd.Timedelta('1D')).to_numpy().astype('f8')
    ds=xr.Dataset(data_vars, coords={'time':('time',tvals),
                                     'y':[np.float64(1)], 'x':[np.float64(1)]})
    ds['time'].attrs={'units':f'days since {t0.strftime("%Y-%m-%d %H:%M:%S")}',
                      'calendar':'gregorian','long_name':'time'}
    if os.path.exists(path): os.remove(path)
    ds.to_netcdf(path)
    ds.close()

def build_surfdata(site, sd, morph, path):
    ds=xr.open_dataset(DEFAULT_SURF)
    default=xr.open_dataset(DEFAULT_SURF)  # reference for diff
    lat=sd['latitude']; lon=sd['longitude']
    lon360 = lon+360.0 if lon<0 else lon
    ds['LATIXY'].values[:] = lat
    ds['LONGXY'].values[:] = lon360
    changed=['LATIXY','LONGXY']
    for var,val in morph.items():
        ds[var].values[UT,0,0]=np.float64(val)
        changed.append(var)
    if os.path.exists(path): os.remove(path)
    ds.to_netcdf(path)
    # ---- audit: prove ONLY intended vars changed vs default ----
    ds2=xr.open_dataset(path)
    diffs=[]
    for v in default.data_vars:
        a=default[v].values; b=ds2[v].values
        if a.shape!=b.shape:
            diffs.append((v,'shape')); continue
        if not np.allclose(np.nan_to_num(a),np.nan_to_num(b),atol=0,rtol=0):
            diffs.append((v,'changed'))
    ds.close(); ds2.close(); default.close()
    unexpected=[v for v,_ in diffs if v not in changed]
    return changed, [v for v,_ in diffs], unexpected

def build_domain(site, sd, path):
    """Site domain: center=site lat/lon, corners +/-0.05deg. area kept at default
    (steradians; irrelevant to per-area W/m2 fluxes). Bypasses pyclmuapp's buggy
    area recompute (numpy scalar-assign error) by passing ATM_DOM to run()."""
    d=xr.open_dataset(DEFAULT_DOMAIN)
    lat=sd['latitude']; lon=sd['longitude']
    lon360 = lon+360.0 if lon<0 else lon
    d['yc'].values[0,0]=lat
    d['xc'].values[0,0]=lon360
    d['xv'].values[0,0,:]=np.array([lon360-0.05, lon360-0.05, lon360+0.05, lon360+0.05])
    d['yv'].values[0,0,:]=np.array([lat-0.05, lat+0.05, lat+0.05, lat-0.05])
    if os.path.exists(path): os.remove(path)
    d.to_netcdf(path); d.close()

def main():
    site=sys.argv[1]
    sd=read_sitedata(site)
    zbot=sd['measurement_height_above_ground']
    morph=morphology(sd)
    outdir=f"{OUT}/{site}"; os.makedirs(outdir, exist_ok=True)
    core, meta, t = build_forcing_arrays(site, zbot)
    core_full, times_full, nsp = prepend_and_pad(core, t)
    write_forcing(core_full, times_full, f"{outdir}/forcing.nc")
    changed, alldiff, unexpected = build_surfdata(site, sd, morph, f"{outdir}/surfdata.nc")
    build_domain(site, sd, f"{outdir}/domain.nc")
    np.savez(f"{outdir}/meta.npz", time=t.values.astype('datetime64[ns]'),
             night=meta['night'], presp=meta['presp'], obs_LWup=meta['obs_LWup'],
             obs_Qh=meta['obs_Qh'], Tair_K=meta['Tair_K'], LWdown=meta['LWdown'])
    t0=times_full[0]; tN_real=t[-1]
    start_tod=int(t0.hour*3600 + t0.minute*60 + t0.second)
    stop_days=int(np.ceil((tN_real - t0)/pd.Timedelta('1D')))
    params=dict(site=site, RUN_STARTDATE=t0.strftime("%Y-%m-%d"), START_TOD=str(start_tod),
                STOP_OPTION="ndays", STOP_N=str(stop_days),
                zbot=zbot, morphology=morph, n_corpus=len(t), n_spin=int(nsp),
                n_forcing=len(times_full), forcing_span=[str(times_full[0]),str(times_full[-1])],
                eval_night_notpresp=int((meta['night']&~meta['presp']).sum()),
                surf_changed=changed, surf_alldiff=alldiff, surf_unexpected=unexpected)
    with open(f"{outdir}/run_params.json","w") as f: json.dump(params,f,indent=2)
    print(json.dumps(params, indent=2))
    if unexpected:
        print("!!! UNEXPECTED surfdata changes:", unexpected)
    else:
        print("OK surfdata audit: only", changed, "changed vs default")

if __name__=='__main__':
    main()
