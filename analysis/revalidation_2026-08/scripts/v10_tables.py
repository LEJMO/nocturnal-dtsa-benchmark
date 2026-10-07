# -*- coding: utf-8 -*-
"""v10: 3-variant model-comparison table (Table 2), z/H metadata, observation-
uncertainty sensitivity of the albedo correlation. Writes tables + stores stats."""
import io, sys, json, csv
from pathlib import Path
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def _pear(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); return (float(np.corrcoef(a,b)[0,1]),)
def _rank(x):
    x=np.asarray(x,float); o=x.argsort(); r=np.empty(len(x)); r[o]=np.arange(len(x))
    # average ties
    _,inv,cnt=np.unique(x,return_inverse=True,return_counts=True)
    sums=np.zeros(len(cnt)); np.add.at(sums,inv,r); avg=sums/cnt; return avg[inv]
def pearsonr(a,b): return _pear(a,b)
def spearmanr(a,b): return (float(np.corrcoef(_rank(a),_rank(b))[0,1]),)
rng = np.random.default_rng(20260915)
ROOT = Path(__file__).resolve().parents[3]
sp = json.load(open(ROOT/"results/paper_stats_v1.json", encoding="utf-8"))
d = sp["per_site"]; S = sorted(d)
obs = np.array([d[s]["dtsa_std"] for s in S])
teb = np.array([d[s]["teb_dtsa"] for s in S]); clm = np.array([d[s]["clmu_dtsa"] for s in S])
alb = np.array([d[s]["albedo"] for s in S])
ci = sp["review4_additions"]["compression_slope_inference"]
VAR = {"All 19 records":"all19","Excl.\\ Minneapolis (17)":"excl_mpls17","Conservative core (16)":"core16"}
MASK = {"all19":np.ones(len(S),bool),
        "excl_mpls17":np.array([not s.startswith("US-Minneapolis") for s in S]),
        "core16":np.array([not s.startswith("US-Minneapolis") and s!="PL-Lipowa" for s in S])}

def sdd(site,key):
    p=ROOT/f"data/urban-plumber/FullCollection/{site}/{site}_sitedata_v1.csv"
    for r in csv.DictReader(p.open(encoding="utf-8")):
        if r["parameter"]==key:
            try: return float(r["value"])
            except: return float("nan")
    return float("nan")

# ---- Table 2: 3-variant model comparison ----
rows=[]
for lab,vk in VAR.items():
    mk=MASK[vk]; o=obs[mk]
    for nm,m,key in [("TEB",teb[mk],"TEB"),("CLM-Urban",clm[mk],"CLMU5")]:
        r=pearsonr(o,m)[0]; rho=spearmanr(o,m)[0]; sdr=m.std(ddof=1)/o.std(ddof=1)
        e=ci[key][vk]; sl=e["slope"]; lo,hi=e["ci95_cluster"]
        rows.append((lab if nm=="TEB" else "", nm, f"{sdr:.2f}", f"{r:+.2f}", f"{rho:+.2f}",
                     f"{sl:.2f} [{lo:.2f}, {hi:.2f}]"))
out=[r"\begin{table}[t]", r"\centering\small",
 r"\caption{Model reproduction of the observed cross-site gradient, decomposed into "
 r"amplitude (spread ratio $s_m/s_o$), ordering (Pearson $r$, Spearman $\rho$), and the "
 r"observed-aligned slope $\hat\beta_{m|o}=r\,(s_m/s_o)$ with its cluster-bootstrap 95\% CI, "
 r"in three record sets. The low CLM-Urban all-record slope is driven by the two "
 r"height-mismatch Minneapolis records: excluding them, its slope and ordering rise to "
 r"resemble TEB's.}", r"\label{tab:modelvar}",
 r"\begin{tabular}{llcccc}", r"\toprule",
 r"Record set & Scheme & $s_m/s_o$ & $r$ & $\rho$ & slope [95\% CI] \\", r"\midrule"]
for i,(lab,nm,sdr,r,rho,sl) in enumerate(rows):
    if lab and i>0: out.append(r"\addlinespace")
    out.append(f"{lab} & {nm} & {sdr} & {r} & {rho} & {sl} \\\\")
out+= [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(ROOT/"paper/manuscript_dtsa/tables/table2_model_variants.tex").write_text("\n".join(out)+"\n",encoding="utf-8")
print("wrote table2_model_variants.tex")

# ---- z/H metadata + supplementary metadata table ----
meta={}
for s in S:
    zta=sp["review2_additions"]["tair_sensor_height_per_site"][s]
    H=sdd(s,"building_mean_height")
    radh=2.0 if s.startswith("US-Minneapolis") else zta  # radiation height = tower except Mpls
    meta[s]=dict(z_Ta=zta,H=H,z_over_H=zta/H if H and H>0 else None,rad_height=radh)
sp["obs_metadata"]=meta
zoh=np.array([meta[s]["z_over_H"] for s in S if meta[s]["z_over_H"]])
print(f"z/H range [{zoh.min():.1f},{zoh.max():.1f}] median {np.median(zoh):.1f}")

# ---- observation-uncertainty sensitivity of albedo correlation ----
SIG,EPS=5.67e-8,0.95; Kper=1/(4*EPS*SIG*285.0**3)
r0=pearsonr(alb,obs)[0]
res={}
for sig_w in [3.0,5.0,10.0]:
    rs=[]
    for _ in range(2000):
        pert=obs + rng.normal(0,sig_w*Kper,len(S))  # per-site random LWup error -> K
        rs.append(pearsonr(alb,pert)[0])
    res[f"random_{sig_w:.0f}Wm2"]=dict(r_median=float(np.median(rs)),
        r_ci=[float(np.percentile(rs,2.5)),float(np.percentile(rs,97.5))])
# worst-case systematic: albedo-correlated FOV bias (dense sites biased warm by up to X K)
for xk in [0.5,1.0]:
    # add bias proportional to (max_alb - alb): low-albedo(dense) get +xk, high-albedo get 0
    bias = xk*(alb.max()-alb)/(alb.max()-alb.min())
    rp=pearsonr(alb, obs - bias)[0]  # remove a warm bias that grows toward dense sites
    res[f"systematic_FOV_{xk}K"]=dict(r=float(rp), note="dense(low-alb) de-biased by up to xk K")
sp["obs_uncertainty_sensitivity"]={"r_baseline":float(r0),"K_per_Wm2":Kper,"variants":res}
print(f"albedo r baseline {r0:.3f}")
for k,v in res.items(): print(f"  {k}: {v}")

# ---- LWup level metric vs dTsa ordering (for same-run comparison) ----
tb=np.array([d[s]["teb_lwup_bias"] for s in S]); cb=np.array([d[s]["clmu_lwup_bias"] for s in S])
sp["level_vs_gradient"]={
 "TEB":{"mean_abs_LWup_bias_Wm2":float(np.mean(np.abs(tb))),"dtsa_ordering_r":float(pearsonr(obs,teb)[0])},
 "CLMU5":{"mean_abs_LWup_bias_Wm2":float(np.mean(np.abs(cb))),"dtsa_ordering_r":float(pearsonr(obs,clm)[0])}}
json.dump(sp,open(ROOT/"results/paper_stats_v1.json","w",encoding="utf-8"),indent=1,ensure_ascii=False)
print("stored obs_metadata, obs_uncertainty_sensitivity, level_vs_gradient")
