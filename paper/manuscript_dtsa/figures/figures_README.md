# Manuscript figures — dTsa nocturnal UHI study

All six figures are produced by a single deterministic script:

```
python paper/manuscript_dtsa/figures/make_figures.py
```

- **Single data source**: every plotted number is a direct read from
  `results/paper_stats_v1.json` (the single-source-of-truth statistics package)
  or the sibling `../tables/site_master_table.csv` (same run). **No statistic**
  (r, slope, span, envelope, LOO R²) is recomputed in the plotting script. The
  only in-script arithmetic is *geometric line placement* in F2: the OLS fit
  line is drawn through the data centroid (mean albedo, mean dTsa) using the
  slope **read** from the stats file, and the shaded band is a slope-CI fan
  built from the **read** `slope_ci95` bounds. Means are a drawing convenience,
  not a re-estimated statistic.
- **Style**: `scripts/pubstyle.py` (Okabe–Ito colorblind-safe palette, Arial
  6.5–8 pt, ≥300 dpi, RGB, editable PDF text `pdf.fonttype=42`).
- **No in-image titles and no subplot letters** (npj / project rule). Panels are
  identified by position only; all lettering and captions live in LaTeX. This
  README maps every panel to its intended caption bullets.
- **Framing guards baked in**: the word "first" never appears; US-Minneapolis1/2
  carry a visible HEIGHT-MISMATCH flag (hatch + vermilion edge, "radiometer 2 m
  vs T_air 40 m") in every figure they appear in; PL-Lipowa is annotated with
  its 37-m FOV composition caveat and never folded into an unqualified span.
- **Semantic colour constants** (same meaning in every figure): TEB = hero blue
  `#0072B2`; CLMU5 = orange `#E69F00`; SUEWS-OHM = grey `#8a8a8a` (auxiliary /
  void-instrument); before-correction = grey; flag/threshold accent = vermilion
  `#D55E00`.

## Width / column choices (production width ≤ 190 mm)

| File | Design width | Column | Notes |
|------|--------------|--------|-------|
| `fig1_dtsa_ranked` | 5.15 in ≈ 131 mm | two-column | tall (19 site rows) |
| `fig2_core_albedo` | 5.15 in ≈ 131 mm | two-column | two panels share y-axis |
| `fig3_model_vs_obs` | 3.9 in ≈ 99 mm | single-column | square, aspect 1:1 |
| `fig4_lwup_bias_paired` | 5.15 in ≈ 131 mm | two-column | tall (19 site rows × 2 bars) |
| `fig5_elimination` | 5.15 in ≈ 131 mm | two-column | 2 panels, width ratio 1.55:1 |
| `fig6_prescription` | 5.15 in ≈ 131 mm | two-column | grouped bars |

`pubstyle.W_DOUBLE = 5.15 in` targets the review manuscript (`sn-jnl`
`\textwidth = 371 pt`), keeping 6.5–8 pt fonts at true size in the compiled PDF.
All widths are ≤ 190 mm, so any figure can also be scaled to the 183 mm
production double-column without upscaling.

---

## Caption bullet points

### F1 — `fig1_dtsa_ranked` (nocturnal surface–air offset, ranked)
- Per-site nocturnal mean surface–air temperature offset
  \(\overline{\Delta T_{s-a}}\) (K) for the 19 evaluable Urban-PLUMBER records,
  ranked coolest-surface to warmest.
- Shaded band = conservative core span (excl. both Minneapolis + Lipowa, n = 16):
  −2.17 … +1.19 K [KR-Ochang … PL-Narutowicza].
- Hatched vermilion-outlined bars = US-Minneapolis1/2: HEIGHT-MISMATCH flag
  (radiation radiometer at 2 m vs \(T_{air}\) at 40 m); reported but excluded
  from the core span.
- PL-Lipowa (+6.12 K) annotated separately: 37-m FOV composition caveat.
- Numbers: `per_site[*].dtsa_std`; band from
  `headline_obs.variants.excl_mpls_lipowa.span_min/max`.

### F2 — `fig2_core_albedo` (CORE: offset vs midday albedo)
- Observed \(\overline{\Delta T_{s-a}}\) against observed midday albedo — the
  central organizing relationship: high-albedo sites run cooler-than-air at night.
- **Left panel** all sites (n = 19 / 16 clusters): r = −0.79, cluster-permutation
  \(p = 2.0\times10^{-4}\), slope −4.00 K per 0.1 albedo.
- **Right panel** Minneapolis excluded (n = 17 / 15 clusters): r = −0.71,
  \(p = 7.0\times10^{-4}\), slope −4.03 K per 0.1 albedo — the gradient survives
  removal of the height-mismatch pair.
- Line = OLS through centroid at the stats-file slope; shaded band = slope-CI fan
  from `slope_ci95`. Open vermilion marker = Minneapolis (flag); open black
  marker = PL-Lipowa (FOV caveat).
- Numbers: `headline_obs.variants.{all,excl_mpls}`,
  `cluster_permutation.{all,excl_mpls}.p_perm`.

### F3 — `fig3_model_vs_obs` (model vs obs, 1:1)
- Modelled vs observed \(\overline{\Delta T_{s-a}}\); dashed line = 1:1.
- TEB (blue) and CLMU5 (orange) point clouds both collapse toward a narrow
  vertical band: model spread is compressed relative to observations
  (TEB slope 0.23, r = 0.63; CLMU5 slope 0.10, r = 0.24; 19-site values).
- SUEWS-OHM shown as faint grey crosses — auxiliary/void-instrument (supy
  EHC/ESTM defective, OHM facet fallback), slope 1.28; legend flags its status.
- Open markers = Minneapolis height-mismatch flag.
- Numbers: `model_evaluation.variants.all.{TEB,CLMU5}`,
  `model_evaluation.suews_auxiliary`.

### F4 — `fig4_lwup_bias_paired` (nocturnal LW↑ bias divergence)
- Per-site nocturnal LW↑ bias (model − obs, W m⁻²), TEB vs CLMU5 paired bars,
  ordered by TEB bias.
- The two structurally independent UCMs diverge site-by-site (e.g. TEB warm where
  CLMU5 is cold), motivating the cross-model treatment; hatched bars flag the
  Minneapolis pair.
- Numbers: `per_site[*].teb_lwup_bias`, `per_site[*].clmu_lwup_bias`.

### F5 — `fig5_elimination` (elimination of material and structural explanations)
- **Left panel**: per-site material envelopes — TEB (blue, 5 sites from
  `probe_rows.csv`) and CLMU5 (orange, 5 sites from `clmu_probe/summary.json`),
  point = base bias, whisker = min–max over material configs. Peak-to-peak
  envelope (TEB 6.2–11.0, CLMU5 2.2–7.3 W m⁻²) with 0 sign reversals is far
  smaller than the bias itself → surface-material uncertainty cannot explain it.
- **Right panel**: structural surgery of dSLUCM — observed span vs best config
  (S-AB). S-AB (slope 0.46, r 0.72) cannot reproduce the PL-Lipowa positive
  extreme (obs +6.1 K → model −0.6 K); no config passes the gate.
- Numbers: `elimination.teb_material_envelope`, `elimination.clmu_material_envelope`,
  `elimination.structural_surgery`, `headline_obs.variants.all.span_min`.

### F6 — `fig6_prescription` (albedo-conditioned LOO correction)
- Mean absolute per-site LW↑ bias (K) before (grey) and after (blue) an
  albedo-conditioned leave-one-out correction, for TEB and CLMU5, each in the
  all-sites (n = 19) and Minneapolis-excluded (n = 17) variants.
- \(R^2_{LOO}\) annotated per group (standard LOO 1 − SS_res/SS_tot). TEB
  reduces from 2.16→0.90 K (all) / 1.97→1.02 K (excl-Mpls); CLMU5 from
  1.52→1.19 K (all), while excl-Mpls CLMU5 (\(R^2_{LOO}=-0.06\)) shows the
  correction does not generalize once the flagged pair is removed.
- Numbers: `prescription.{all,excl_mpls}.{TEB,CLMU5}` (K fields).
