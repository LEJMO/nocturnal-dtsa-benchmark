"""Run one CLMU5 site for Campaign B (equilibrium spin-up) using the
inputs_spin/ (N recycled years) inputs and writing to runs_spin/. Same run
config as the frozen run_one.py; only the input/output dirs differ."""
import os, sys, json, time
from pyclmuapp import usp_clmu

WORK = "/root/clmu_work"
site = sys.argv[1]
indir = f"{WORK}/inputs_spin/{site}"
params = json.load(open(f"{indir}/run_params.json"))
case = "g1_" + site.replace("-", "_")
pwd = f"{WORK}/runs_spin/{site}"
os.makedirs(pwd, exist_ok=True)
usp = usp_clmu(pwd=pwd)
t0 = time.time()
out = usp.run(
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
print(f"RUN_DONE {site} in {(time.time()-t0)/60:.1f} min")
print("OUTPUTS:", out)
