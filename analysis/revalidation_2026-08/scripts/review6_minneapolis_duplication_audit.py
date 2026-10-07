# -*- coding: utf-8 -*-
"""ADVERSARIAL independent duplication check -- raw + corpus, three city pairs."""
import os
for _v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(_v,"4")
import io,sys,json
from pathlib import Path
import numpy as np, xarray as xr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[3]
SIG, EPS = 5.67e-8, 0.95
PAIRS = {"Minneapolis":("US-Minneapolis1","US-Minneapolis2"),
         "Lodz":("PL-Lipowa","PL-Narutowicza"),
         "Helsinki":("FI-Kumpula","FI-Torni")}
res={}
for name,(a,b) in PAIRS.items():
    blk={}
    # ---- RAW clean_observations ----
    try:
        da=xr.open_dataset(ROOT/f"data/urban-plumber/FullCollection/{a}/timeseries/{a}_clean_observations_v1.nc")
        db=xr.open_dataset(ROOT/f"data/urban-plumber/FullCollection/{b}/timeseries/{b}_clean_observations_v1.nc")
        ta_=da["time"].values; tb_=db["time"].values
        blk["raw_nsteps"]=[int(ta_.size),int(tb_.size)]
        blk["raw_time_identical"]= bool(ta_.size==tb_.size and np.all(ta_==tb_))
        blk["raw_time_range_a"]=[str(ta_[0]),str(ta_[-1])]
        blk["raw_time_range_b"]=[str(tb_[0]),str(tb_[-1])]
        blk["raw_comment_equal"]= bool(da.attrs.get("comment")==db.attrs.get("comment"))
        blk["raw_comment_a_has_2m_clause"]= "not from tower as other variables" in str(da.attrs.get("comment",""))
        vv={}
        for v in ("LWup","LWdown","SWdown","SWup","Tair","Qh","Qle"):
            if v not in da or v not in db: vv[v]="absent"; continue
            xa=np.asarray(da[v].values,dtype=float).reshape(-1)
            xb=np.asarray(db[v].values,dtype=float).reshape(-1)
            n=min(xa.size,xb.size); xa=xa[:n]; xb=xb[:n]
            fa=np.isfinite(xa); fb=np.isfinite(xb); both=fa&fb
            d={"n_both_finite":int(both.sum()),
               "n_finite_a":int(fa.sum()),"n_finite_b":int(fb.sum()),
               "finite_mask_disagree":int(int((fa!=fb).sum()))}
            if both.sum()>0:
                diff=np.abs(xa[both]-xb[both])
                d["maxabsdiff"]=float(diff.max())
                d["n_exactly_equal"]=int(int((diff==0).sum()))
                if both.sum()>2 and np.std(xa[both])>0 and np.std(xb[both])>0:
                    d["r"]=float(np.corrcoef(xa[both],xb[both])[0,1])
            vv[v]=d
        # qc flag
        for q in ("LWup_qc",):
            if q in da and q in db:
                qa_=np.asarray(da[q].values,dtype=float).reshape(-1)
                qb_=np.asarray(db[q].values,dtype=float).reshape(-1)
                n=min(qa_.size,qb_.size)
                vv[q]={"identical":bool(np.array_equal(qa_[:n],qb_[:n],equal_nan=True)),
                       "hist_a":{str(int(k)):int(v) for k,v in zip(*np.unique(qa_[:n][np.isfinite(qa_[:n])],return_counts=True))},
                       "hist_b":{str(int(k)):int(v) for k,v in zip(*np.unique(qb_[:n][np.isfinite(qb_[:n])],return_counts=True))}}
        blk["raw_vars"]=vv
        da.close(); db.close()
    except Exception as e:
        blk["raw_error"]=repr(e)
    # ---- CORPUS ----
    ca=xr.open_dataset(ROOT/f"data/urban-plumber/corpus/{a}.nc")
    cb=xr.open_dataset(ROOT/f"data/urban-plumber/corpus/{b}.nc")
    cv={}
    for v in ("obs_LWup","forcing_LWdown","forcing_Tair","forcing_SWdown","forcing_Qair"):
        xa=np.asarray(ca[v].values,dtype=float).reshape(-1)
        xb=np.asarray(cb[v].values,dtype=float).reshape(-1)
        n=min(xa.size,xb.size)
        both=np.isfinite(xa[:n])&np.isfinite(xb[:n])
        e={"len":[int(xa.size),int(xb.size)],"n_both_finite":int(both.sum())}
        if both.sum(): 
            dd=np.abs(xa[:n][both]-xb[:n][both]); e["maxabsdiff"]=float(dd.max())
            if both.sum()>2 and np.std(xa[:n][both])>0: e["r"]=float(np.corrcoef(xa[:n][both],xb[:n][both])[0,1])
        cv[v]=e
    na=ca["night_mask"].values.astype(bool); nb=cb["night_mask"].values.astype(bool)
    pa=ca["pre_spinup_flag"].values.astype(bool); pb=cb["pre_spinup_flag"].values.astype(bool)
    n=min(na.size,nb.size)
    cv["night_mask_identical"]=bool(na.size==nb.size and np.array_equal(na,nb))
    cv["pre_spinup_identical"]=bool(pa.size==pb.size and np.array_equal(pa,pb))
    # nocturnal evaluation mask (TEB-frame convention on full axis, pre-slice)
    def evalmask(ds):
        o=np.asarray(ds["obs_LWup"].values,dtype=float).reshape(-1)
        t=np.asarray(ds["forcing_Tair"].values,dtype=float).reshape(-1)
        l=np.asarray(ds["forcing_LWdown"].values,dtype=float).reshape(-1)
        nm=ds["night_mask"].values.astype(bool); ps=ds["pre_spinup_flag"].values.astype(bool)
        return nm&~ps&np.isfinite(o)&np.isfinite(t)&np.isfinite(l)
    ma=evalmask(ca); mb=evalmask(cb)
    cv["nocturnal_mask_n"]=[int(ma.sum()),int(mb.sum())]
    cv["nocturnal_mask_identical"]=bool(ma.size==mb.size and np.array_equal(ma,mb))
    # dTsa recompute
    def dtsa(ds):
        o=np.asarray(ds["obs_LWup"].values,dtype=float).reshape(-1)
        t=np.asarray(ds["forcing_Tair"].values,dtype=float).reshape(-1)
        m=evalmask(ds)
        Ts=(o[m]/(EPS*SIG))**0.25
        return float(np.mean(Ts-t[m]))
    cv["dtsa_recomputed"]=[dtsa(ca),dtsa(cb)]
    ca.close(); cb.close()
    blk["corpus"]=cv
    res[name]=blk
json.dump(res, io.open(ROOT/"../../../..".replace("..","..") if False else Path(os.environ["SCR"])/"verify_dup.json","w",encoding="utf-8"),indent=1)
print(json.dumps(res,indent=1)[:12000])
