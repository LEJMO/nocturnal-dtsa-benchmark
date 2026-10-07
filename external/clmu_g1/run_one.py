"""Run one G1 site through CLMU5 (pyclmuapp/docker) coldstart, single continuous
run over [1 recycled spin-up year + corpus period + 2 pad days]. Identical config;
urban_hac OFF uniformly (CLMU analog of frozen QF-off used for TEB/SUEWS)."""
import os, sys, json, time
from pyclmuapp import usp_clmu

WORK="/home/junmo/clmu_g1_work"
site=sys.argv[1]
indir=f"{WORK}/inputs/{site}"
params=json.load(open(f"{indir}/run_params.json"))
case=("g1_"+site.replace("-","_"))

pwd=f"{WORK}/runs/{site}"
os.makedirs(pwd, exist_ok=True)
usp=usp_clmu(pwd=pwd)

t0=time.time()
out=usp.run(
    case_name=case,
    SURF=f"{indir}/surfdata.nc",
    FORCING=f"{indir}/forcing.nc",
    ATM_DOM=f"{indir}/domain.nc",
    RUN_STARTDATE=params["RUN_STARTDATE"],
    START_TOD=params["START_TOD"],
    STOP_OPTION=params["STOP_OPTION"],
    STOP_N=params["STOP_N"],
    RUN_TYPE="coldstart",
    hist_type="GRID",
    hist_nhtfrq=1,
    hist_mfilt=1000000000,
    urban_hac="OFF",
    iflog=True,
    logfile=f"{pwd}/pyclmuapprun.log",
)
dt=time.time()-t0
print(f"RUN_DONE {site} in {dt/60:.1f} min")
print("OUTPUTS:", out)
with open(f"{indir}/output_paths.json","w") as f: json.dump({"site":site,"outputs":out,"minutes":dt/60}, f, indent=2)
