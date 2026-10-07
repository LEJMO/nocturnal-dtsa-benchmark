"""ADVERSARIAL VERIFIER for review-5 task A (review5_adderley_propagation.py).

Independent re-derivation of every headline number in that block, with its own
code path and its own seeds (200 000 draws vs the original's 50 000), plus two
things the original did not compute:

  * a second, netCDF4-masked-array read of the raw corpus (the original used
    xarray), as an independent check against the LWup _FillValue = -999 trap;
  * the literature-matched sigma grid (1.5 / 2 / 3 % of the corpus mean nocturnal
    LWup = 5.49 / 7.33 / 10.99 W/m2), which is what the cited specifications
    actually state, and the z_rad/H stratification against Adderley's 3.5 z_b
    threshold rather than against the corpus minimum of 2.09.

Writes NOTHING to results/paper_stats_v1.json and nothing to
results/review5_adderley-propagation.json.  Corrections are persisted separately
in results/review5_adderley-propagation_verify.json.

Constants are the frozen paper_stats.py convention: SIG = 5.67e-8 (NOT the exact
Stefan-Boltzmann value) and EPS = 0.95, giving K_per_Wm2 = 0.2004927923369955.
Uses only observation-side quantities (dtsa_std, albedo, obs_qh_qc0,
obs_metadata, raw corpus).  No CLMU- or TEB-derived statistic appears anywhere.
"""
"""ADVERSARIAL independent recompute of review-5 task A headline numbers.
Own seeds, own code path. Nothing shared with review5_adderley_propagation.py."""
import json, itertools, numpy as np
from scipy import stats

SIG = 5.67e-8
EPS = 0.95
TREF = 285.0
KPW = 1.0 / (4.0 * EPS * SIG * TREF**3)
print("K_per_Wm2 =", repr(KPW))
# also check exact SB for contrast
KPW_exact = 1.0 / (4.0 * EPS * 5.670374419e-8 * TREF**3)
print("K_per_Wm2 with exact SB =", repr(KPW_exact))

ROOT = str(Path(__file__).resolve().parents[3])
PS = json.load(open(ROOT + r"\results\paper_stats_v1.json", encoding="utf-8"))
ps = PS["per_site"]
om = PS["obs_metadata"]

ALL19 = sorted(ps.keys())
assert len(ALL19) == 19, len(ALL19)
SETS = {
    "all19": ALL19,
    "excl_mpls17": [s for s in ALL19 if not s.startswith("US-Minneapolis")],
    "core16": [s for s in ALL19 if not s.startswith("US-Minneapolis") and s != "PL-Lipowa"],
}

def vec(sites, key):
    return np.array([ps[s][key] for s in sites], float)

print("\n=== BASELINE ===")
base = {}
for name, sites in SETS.items():
    a = vec(sites, "albedo"); o = vec(sites, "dtsa_std")
    r, p = stats.pearsonr(a, o)
    rho, pp = stats.spearmanr(a, o)
    ncl = len({ps[s]["cluster"] for s in sites})
    base[name] = r
    print(f"{name:12s} n={len(sites)} clusters={ncl} r={r:.10f} p={p:.4g} rho={rho:.6f}")

# ---------- fast vectorised pearson ----------
def rmany(a, O):
    """a: (n,) fixed; O: (m,n) perturbed offsets -> (m,) r"""
    ac = a - a.mean()
    Oc = O - O.mean(axis=1, keepdims=True)
    num = Oc @ ac
    den = np.sqrt((Oc**2).sum(axis=1) * (ac**2).sum())
    return num / den

# sanity check vs scipy
_rng = np.random.default_rng(1)
_a = vec(ALL19, "albedo"); _o = vec(ALL19, "dtsa_std")
_O = _o + _rng.normal(0, 0.5, (7, 19))
assert np.allclose(rmany(_a, _O), [stats.pearsonr(_a, x)[0] for x in _O]), "rmany broken"
print("rmany validated vs scipy")

NDRAW = 200000  # 4x theirs, my own seeds
print(f"\n=== (i) RANDOM INDEPENDENT GAUSSIAN, NDRAW={NDRAW}, seed family 777 ===")
rand_out = {}
for si, sigW in enumerate([2.0, 4.0, 5.0, 10.0]):
    for ni, (name, sites) in enumerate(SETS.items()):
        a = vec(sites, "albedo"); o = vec(sites, "dtsa_std")
        rng = np.random.default_rng(777000 + si * 31 + ni)
        d = rng.normal(0.0, sigW * KPW, (NDRAW, len(sites)))
        rr = rmany(a, o + d)
        med = float(np.median(rr))
        lo, hi = np.percentile(rr, [2.5, 97.5])
        fa5 = float((rr > -0.5).mean()); fa0 = float((rr > 0).mean())
        rand_out[(sigW, name)] = (med, lo, hi, fa5, fa0)
        print(f"  sig={sigW:5.1f}W/m2 ({sigW*KPW:.4f}K) {name:12s} med={med:+.4f} ci=[{lo:+.3f},{hi:+.3f}] P(r>-0.5)={fa5:.5f} P(r>0)={fa0:.5f}")

print("\n=== (ii) SHARED MINNEAPOLIS DRAW (all19) ===")
print("  dtsa_std M1 =", repr(ps["US-Minneapolis1"]["dtsa_std"]),
      " M2 =", repr(ps["US-Minneapolis2"]["dtsa_std"]),
      " identical:", ps["US-Minneapolis1"]["dtsa_std"] == ps["US-Minneapolis2"]["dtsa_std"])
sites = SETS["all19"]
i1 = sites.index("US-Minneapolis1"); i2 = sites.index("US-Minneapolis2")
a = vec(sites, "albedo"); o = vec(sites, "dtsa_std")
for si, sigW in enumerate([2.0, 4.0, 5.0, 10.0]):
    rng = np.random.default_rng(888000 + si)
    d = rng.normal(0.0, sigW * KPW, (NDRAW, 19))
    d[:, i2] = d[:, i1]
    rr = rmany(a, o + d)
    print(f"  sig={sigW:5.1f} shared med={np.median(rr):+.4f}  (indep was {rand_out[(sigW,'all19')][0]:+.4f}, diff {np.median(rr)-rand_out[(sigW,'all19')][0]:+.4f})")

print("\n=== (iii) ADDERLEY EXPONENTIAL FIT ===")
zc = np.array([2.0, 3.0, 5.0]); sc = np.array([11.2, 6.3, 2.0])
A = np.vstack([np.ones(3), zc]).T
coef, res, *_ = np.linalg.lstsq(A, np.log(sc), rcond=None)
c0, c1 = coef
pred = np.exp(c0 + c1 * zc)
ss_tot = ((np.log(sc) - np.log(sc).mean())**2).sum()
ss_res = ((np.log(sc) - (c0 + c1 * zc))**2).sum()
print(f"  sigma(z/H) = exp({c0:.4f} {c1:+.4f} z/H);  R2_log = {1-ss_res/ss_tot:.8f}")
print(f"  fitted at 2/3/5: {pred[0]:.4f} {pred[1]:.4f} {pred[2]:.4f}  max|err|={np.abs(pred-sc).max():.5f}")
print(f"  e-folding = {-1/c1:.4f} z_b ;  sigma(3.5) = {np.exp(c0+c1*3.5):.4f} W/m2")

def sigma_z(z, cap=None):
    s = np.exp(c0 + c1 * np.asarray(z, float))
    return np.minimum(s, cap) if cap is not None else s

zoH = {s: om[s]["rad_height"] / om[s]["H"] for s in ALL19}
nonm = [s for s in ALL19 if not s.startswith("US-Minneapolis")]
print("  z_rad/H non-Mpls range: %.4f (%s) .. %.4f (%s)" % (
    min(zoH[s] for s in nonm), min(nonm, key=lambda s: zoH[s]),
    max(zoH[s] for s in nonm), max(nonm, key=lambda s: zoH[s])))
print("  sigma range non-Mpls: %.4f .. %.4f W/m2" % (
    sigma_z(max(zoH[s] for s in nonm)), sigma_z(min(zoH[s] for s in nonm))))
zm = zoH["US-Minneapolis1"]
print(f"  Minneapolis z_rad/H = {zm:.4f} -> UNCAPPED sigma = {sigma_z(zm):.4f} W/m2 = {sigma_z(zm)*KPW:.4f} K")

print("\n=== (iii) PLACEMENT SCENARIOS, medians ===")
for label, mk in [("capped11.2", lambda sts: sigma_z([zoH[s] for s in sts], cap=11.2)),
                  ("UNCAPPED  ", lambda sts: sigma_z([zoH[s] for s in sts])),
                  ("flat 11.2 ", lambda sts: np.full(len(sts), 11.2)),
                  ("flat 9.4  ", lambda sts: np.full(len(sts), 9.4))]:
    row = []
    for ni, (name, sts) in enumerate(SETS.items()):
        a = vec(sts, "albedo"); o = vec(sts, "dtsa_std")
        sg = np.asarray(mk(sts), float) * KPW
        rng = np.random.default_rng(999000 + ni + 7 * hash(label) % 1000)
        d = rng.normal(0.0, 1.0, (NDRAW, len(sts))) * sg
        row.append(float(np.median(rmany(a, o + d))))
    print(f"  {label}: " + " / ".join(f"{v:+.4f}" for v in row))

print("\n=== (iv) ALIGNED STEP delta_i = B*sign(alb - mean alb) ===")
for name, sts in SETS.items():
    a = vec(sts, "albedo")
    sgn = np.sign(a - a.mean())
    print(f"  {name}: +1 count={int((sgn>0).sum())} -1 count={int((sgn<0).sum())} "
          f"min|alb-mean|={np.abs(a-a.mean()).min():.6f}")
for B in [0.25, 0.5, 0.75, 1.0]:
    row = []
    for name, sts in SETS.items():
        a = vec(sts, "albedo"); o = vec(sts, "dtsa_std")
        d = B * np.sign(a - a.mean())
        row.append(stats.pearsonr(a, o + d)[0])
    print(f"  B={B:.2f}K: " + " / ".join(f"{v:+.4f}" for v in row))

print("\n=== (v) UNRESTRICTED BOX SUPREMUM, exhaustive 2^n ===")
def box_sup(a, o, B):
    n = len(a)
    V = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    rr = rmany(a, o + B * V)
    k = int(np.argmax(rr))
    return float(rr[k]), V[k]
sup_store = {}
for B in [0.25, 0.5, 0.75, 1.0]:
    row = []
    for name, sts in SETS.items():
        a = vec(sts, "albedo"); o = vec(sts, "dtsa_std")
        v, vv = box_sup(a, o, B)
        sup_store[(B, name)] = (v, vv, sts, a, o)
        row.append(v)
    print(f"  B={B:.2f}: " + " / ".join(f"{v:+.4f}" for v in row))

print("\n  -- all19 B=1.0 argmax vs aligned sign pattern --")
v, vv, sts, a, o = sup_store[(1.0, "all19")]
sgn = np.sign(a - a.mean())
diff = [(sts[i], sgn[i], vv[i]) for i in range(len(sts)) if sgn[i] != vv[i]]
print("   sup r =", f"{v:+.6f}", " aligned r =", f"{stats.pearsonr(a,o+1.0*sgn)[0]:+.6f}")
print("   flips (site, aligned, sup):", diff)
for name in ("excl_mpls17", "core16"):
    v, vv, sts, a, o = sup_store[(1.0, name)]
    sgn = np.sign(a - a.mean())
    print(f"   {name}: sup={v:+.6f} aligned={stats.pearsonr(a,o+sgn)[0]:+.6f} "
          f"argmax==aligned: {np.array_equal(vv,sgn)}")

print("\n=== (vi) BREAKDOWN AMPLITUDE B* (bisection) ===")
def r_aligned(a, o, B):
    return stats.pearsonr(a, o + B * np.sign(a - a.mean()))[0]
def r_sup(a, o, B):
    return box_sup(a, o, B)[0]
for name, sts in SETS.items():
    a = vec(sts, "albedo"); o = vec(sts, "dtsa_std")
    print(f"  {name} (base {stats.pearsonr(a,o)[0]:+.4f})")
    for th in [-0.5, -0.4, -0.3, 0.0]:
        out = []
        for fn in (r_aligned, r_sup):
            lo, hi = 0.0, 6.0
            if fn(a, o, hi) <= th:
                out.append(float("nan")); continue
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if fn(a, o, mid) > th: hi = mid
                else: lo = mid
            out.append(hi)
        print(f"    r>{th:+.1f}: aligned {out[0]:.4f}K / {out[0]/KPW:.2f}W/m2 / p2p {2*out[0]:.3f}K "
              f"| unrestr {out[1]:.4f}K / {out[1]/KPW:.2f}W/m2 / p2p {2*out[1]:.3f}K")

print("\n=== (5b) Q_H SPEARMAN COST ===")
for name, sts in SETS.items():
    a = vec(sts, "albedo"); o = vec(sts, "dtsa_std"); q = vec(sts, "obs_qh_qc0")
    rho = stats.spearmanr(o, q)[0]
    agree = int(((o > 0) == (q > 0)).sum())
    print(f"  {name}: baseline rho(dtsa,qh)={rho:+.4f} sign-agree {agree}/{len(sts)}")
print("  paper_stats independent_instrument:",
      json.dumps({k: v for k, v in PS["independent_instrument"].items()
                  if not isinstance(v, (dict, list))})[:400])
# at the aligned B* that zeroes core16 albedo r
for name in ("core16", "all19"):
    sts = SETS[name]; a = vec(sts, "albedo"); o = vec(sts, "dtsa_std"); q = vec(sts, "obs_qh_qc0")
    lo, hi = 0.0, 6.0
    for _ in range(80):
        mid = .5*(lo+hi)
        if r_aligned(a, o, mid) > 0.0: hi = mid
        else: lo = mid
    d = hi * np.sign(a - a.mean())
    rho2 = stats.spearmanr(o + d, q)[0]
    ag2 = int((((o+d) > 0) == (q > 0)).sum())
    print(f"  {name} at B*(r->0)={hi:.4f}K : rho {stats.spearmanr(o,q)[0]:+.4f} -> {rho2:+.4f}, "
          f"sign-agree {int(((o>0)==(q>0)).sum())}/{len(sts)} -> {ag2}/{len(sts)}")
sts = SETS["all19"]; a = vec(sts,"albedo"); o = vec(sts,"dtsa_std"); q = vec(sts,"obs_qh_qc0")
lo, hi = 0.0, 6.0
for _ in range(80):
    mid=.5*(lo+hi)
    if r_aligned(a,o,mid) > -0.5: hi=mid
    else: lo=mid
d = hi*np.sign(a-a.mean())
print(f"  all19 at B*(r>-0.5)={hi:.4f}K : rho -> {stats.spearmanr(o+d,q)[0]:+.4f}, "
      f"sign-agree -> {int((((o+d)>0)==(q>0)).sum())}/19")
"""ADVERSARIAL independent recompute of the RAW-CORPUS parts of task A:
mean nocturnal LWup, multiplicand-vs-albedo, common-mode %, clear-calm contrast,
flux-vs-kelvin residual. Uses netCDF4 masked reads (NOT xarray) as a second
independent code path against the -999 _FillValue trap."""
import json, sys, numpy as np
import netCDF4 as nc
from scipy import stats
from pathlib import Path

SIG = 5.67e-8; EPS = 0.95; TREF = 285.0
KPW = 1.0 / (4 * EPS * SIG * TREF**3)
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
P = json.load(open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))["per_site"]
SITES = sorted(P)
from src.training.corpus_loader import load_all_sites
recs = {r.site: r for r in load_all_sites()}

out = {}
for site in SITES:
    rec = recs[site]
    ds = nc.Dataset(ROOT / f"data/urban-plumber/corpus/{site}.nc")
    ds.set_auto_maskandscale(True)          # masked-array-aware
    def g(v):
        x = ds.variables[v][:]
        return np.ma.filled(np.ma.masked_invalid(x).astype(float), np.nan)
    olw, ta, ld = g("obs_LWup"), g("forcing_Tair"), g("forcing_LWdown")
    wn, we = g("forcing_Wind_N"), g("forcing_Wind_E")
    fv = ds.variables["obs_LWup"].getncattr("_FillValue") if "_FillValue" in ds.variables["obs_LWup"].ncattrs() else None
    ds.close()
    wd = np.hypot(wn, we)
    night = np.asarray(rec.night_mask, bool) & ~np.asarray(rec.pre_spinup_flag, bool)
    m = night & np.isfinite(olw) & np.isfinite(ta) & np.isfinite(ld)
    ts = ((olw[m] - (1 - EPS) * ld[m]) / (EPS * SIG)) ** 0.25
    dts = ts - ta[m]
    row = dict(fill=fv, n=int(m.sum()), lwup=float(olw[m].mean()),
               dtsa=float(dts.mean()), d_vs_frozen=float(dts.mean() - P[site]["dtsa_std"]),
               lwup_raw_min=float(np.nanmin(olw[night])) if np.isfinite(olw[night]).any() else None)
    mw = m & np.isfinite(wd)
    tsw = ((olw[mw] - (1 - EPS) * ld[mw]) / (EPS * SIG)) ** 0.25
    dtw = tsw - ta[mw]
    epsky = ld[mw] / (SIG * ta[mw] ** 4); wind = wd[mw]
    e50, w50 = np.median(epsky), np.median(wind)
    cc = (epsky <= e50) & (wind <= w50); cw = (epsky >= e50) & (wind >= w50)
    row.update(cc=float(dtw[cc].mean()), cw=float(dtw[cw].mean()),
               contrast=float(dtw[cc].mean() - dtw[cw].mean()),
               n_cc=int(cc.sum()), n_cw=int(cw.sum()))
    for E in (5.0, 11.2):
        sh = ((olw[mw] + E - (1 - EPS) * ld[mw]) / (EPS * SIG)) ** 0.25 - tsw
        row[f"resid_{E:g}"] = float(sh[cc].mean() - sh[cw].mean())
    out[site] = row
    print(f"{site:18s} fill={fv} n={row['n']:6d} LWup={row['lwup']:7.2f} "
          f"dTsa={row['dtsa']:+.6f} (frozen delta {row['d_vs_frozen']:+.2e}) "
          f"cc={row['cc']:+.3f} cw={row['cw']:+.3f} contrast={row['contrast']:+.4f}")

lw = np.array([out[s]["lwup"] for s in SITES]); alb = np.array([P[s]["albedo"] for s in SITES])
off = np.array([P[s]["dtsa_std"] for s in SITES])
print("\n=== corpus mean nocturnal LWup ===")
print(f"  mean over 19 records = {lw.mean():.4f} W/m2; min {lw.min():.2f} ({SITES[int(lw.argmin())]}) "
      f"max {lw.max():.2f} ({SITES[int(lw.argmax())]})")
print(f"  PL-Lipowa = {out['PL-Lipowa']['lwup']:.2f} (trap check: must NOT be ~160)")
print(f"  1% of mean = {0.01*lw.mean():.3f}  3% = {0.03*lw.mean():.3f} W/m2")
print("\n=== multiplicand vs albedo ===")
print(f"  Pearson r={stats.pearsonr(alb,lw)[0]:+.4f} p={stats.pearsonr(alb,lw)[1]:.4f}; "
      f"Spearman rho={stats.spearmanr(alb,lw)[0]:+.4f} p={stats.spearmanr(alb,lw)[1]:.4f}")

SETS = {"all19": SITES,
        "excl_mpls17": [s for s in SITES if not s.startswith("US-Minneapolis")],
        "core16": [s for s in SITES if not s.startswith("US-Minneapolis") and s != "PL-Lipowa"]}
print("\n=== common-mode % of reading ===")
for p in (1, 2, 3):
    for sgn, lab in ((+1, "+"), (-1, "-")):
        row = []
        for nm, ss in SETS.items():
            a = np.array([P[s]["albedo"] for s in ss]); o = np.array([P[s]["dtsa_std"] for s in ss])
            L = np.array([out[s]["lwup"] for s in ss])
            d = sgn * (p / 100.0) * L * KPW
            row.append(stats.pearsonr(a, o + d)[0])
        sp = None
        if p == 2:
            L19 = lw * (p/100.0) * KPW; sp = L19.max() - L19.min()
        print(f"  {lab}{p}%: " + " / ".join(f"{v:+.4f}" for v in row)
              + (f"   [spread {sp:.4f} K over 19]" if sp else ""))
for nm, ss in SETS.items():
    L = np.array([out[s]["lwup"] for s in ss]) * 0.02 * KPW
    print(f"  2% delta magnitude {nm}: min {L.min():.4f} max {L.max():.4f} K, spread {L.max()-L.min():.4f} K")
# random-sign 2%
print("  random-sign 2% medians (200k, seed 4242):")
for ni,(nm, ss) in enumerate(SETS.items()):
    a = np.array([P[s]["albedo"] for s in ss]); o = np.array([P[s]["dtsa_std"] for s in ss])
    L = np.array([out[s]["lwup"] for s in ss]) * 0.02 * KPW
    rng = np.random.default_rng(4242 + ni)
    S = rng.choice([-1.0, 1.0], (200000, len(ss)))
    O = o + S * L
    ac = a - a.mean(); Oc = O - O.mean(1, keepdims=True)
    rr = (Oc @ ac) / np.sqrt((Oc**2).sum(1) * (ac**2).sum())
    print(f"    {nm}: {np.median(rr):+.4f}")

print("\n=== clear-calm contrast summary ===")
con = np.array([abs(out[s]["contrast"]) for s in SITES])
print(f"  |contrast| median {np.median(con):.4f} K, max {con.max():.4f} K ({SITES[int(con.argmax())]})")
print(f"  PL-Lipowa cc={out['PL-Lipowa']['cc']:+.4f} cw={out['PL-Lipowa']['cw']:+.4f}")
neg = [s for s in SITES if P[s]["dtsa_std"] < 0]
print(f"  negative-offset records n={len(neg)}: {neg}")
ccm = np.mean([out[s]["cc"] for s in neg]); cwm = np.mean([out[s]["cw"] for s in neg])
print(f"  mean cc={ccm:+.4f} cw={cwm:+.4f} contrast={ccm-cwm:+.4f}")
print("\n=== flux-vs-Kelvin residual of the contrast ===")
for E in (5.0, 11.2):
    fr = np.array([abs(out[s][f"resid_{E:g}"]) / abs(out[s]["contrast"]) for s in SITES])
    ab = np.array([abs(out[s][f"resid_{E:g}"]) for s in SITES])
    k = int(fr.argmax())
    print(f"  E={E}: |resid| max {ab.max():.4f} K; frac median {np.median(fr)*100:.2f}%, "
          f"max {fr[k]*100:.2f}% ({SITES[k]}, own contrast {out[SITES[k]]['contrast']:+.4f} K)")
    frn = np.array([abs(out[s][f"resid_{E:g}"]) / abs(out[s]["contrast"]) for s in neg])
    print(f"      among {len(neg)} negative-offset records: max {frn.max()*100:.2f}%")
json.dump(out, open(ROOT / "results/review5_adderley_verify_rawcache.json", "w"), indent=1)

print("\n=== z_rad/H vs Adderley 3.5 z_b threshold ===")
om = json.load(open(ROOT/"results/paper_stats_v1.json", encoding="utf-8"))["obs_metadata"]
c0, c1 = 3.5637975036470765, -0.5742058802082072
zs = sorted(((om[s]["rad_height"]/om[s]["H"], s) for s in SITES))
for z, s in zs:
    print(f"  {s:18s} z_rad/H={z:.4f}  sigma={np.exp(c0+c1*z):8.3f} W/m2  {'ABOVE 3.5' if z>=3.5 else 'below 3.5'}")
print(f"  count >= 3.5 z_b: {sum(1 for z,_ in zs if z>=3.5)} / 19")
print(f"  count >= 2.09   : {sum(1 for z,_ in zs if z>=2.0909)} / 19")


# =============================================================================
#  SUPPLEMENT the original did not run: the literature-matched sigma grid.
#  The cited percentage specifications are 95 %-confidence (k = 2) figures, so a
#  one-sigma per-record Gaussian is HALF the quoted percentage.  Both readings
#  are printed, because which one is used decides whether the random-error claim
#  survives in the conservative core.
# =============================================================================
LWBAR = float(np.mean([out[s]["lwup"] for s in SITES]))
print(f"\n=== literature-matched sigma grid (corpus mean nocturnal LWup = {LWBAR:.4f}) ===")
for pct, lab in [(1.0, "1 % (CGR4 expected daily)"),
                 (1.5, "1.5 % = one-sigma of the 3 % k=2 daily spec"),
                 (2.0, "2 % (WMO night agreement; CGR4 max hourly)"),
                 (3.0, "3 % (CGR4 daily-total, 95 % confidence)")]:
    w = pct / 100.0 * LWBAR
    print(f"  {pct:4.1f}% -> {w:6.3f} W/m2 = {w*KPW:.3f} K   [{lab}]")
for si, sw in enumerate([5.4945, 7.3260, 10.9890, 11.2]):
    row = []
    for ni, (nm, ss) in enumerate(SETS.items()):
        a = np.array([P[s]["albedo"] for s in ss]); o = np.array([P[s]["dtsa_std"] for s in ss])
        rg = np.random.default_rng(31337 + si * 11 + ni)
        D = o + rg.normal(0, sw * KPW, (200000, len(ss)))
        ac = a - a.mean(); Dc = D - D.mean(1, keepdims=True)
        rr = (Dc @ ac) / np.sqrt((Dc**2).sum(1) * (ac**2).sum())
        row.append((float(np.median(rr)), float((rr > -0.5).mean())))
    print(f"  sigma={sw:7.3f} W/m2 ({sw*KPW:.3f} K, {sw/LWBAR*100:.2f}%): "
          + " / ".join(f"{m:+.4f} (P(r>-0.5)={p:.3f})" for m, p in row))
