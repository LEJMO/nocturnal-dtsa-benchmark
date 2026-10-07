# nocturnal-dtsa-benchmark

Analysis code, model-run configurations and the complete statistics package
behind

> Lee, J. and Kang, J. (2026). *An albedo-organized gradient in nocturnal
> surface–air temperature offsets across cities: a flux-tower benchmark for
> urban canopy schemes.* Submitted to *Urban Climate*.

The paper derives the nocturnal apparent radiometric surface temperature from
tower upward and downward longwave radiation at 19 records of the harmonized
Urban-PLUMBER collection, shows that its offset from tower air temperature
forms an albedo-organized cross-site gradient, and tests whether two urban
canopy schemes run offline with default materials, TEB v4.1.2 and CLM-Urban
(CLM5.0/CLMU), reproduce that gradient. Every number, table and figure in the
manuscript and its Supplementary Material is produced by the code in this
repository from `results/paper_stats_v1.json`, which the code also regenerates
from the observations and the archived model output.

Archived release v1.0.0: <https://doi.org/10.5281/zenodo.23205916>
(concept DOI for all versions: <https://doi.org/10.5281/zenodo.23205915>).
Software citation: `CITATION.cff`.

## Contents

```
analysis/revalidation_2026-08/
  scripts/        statistics generators (chain), table generators, verification
                  scripts, probe, and the TEB / CLM-Urban run and post-processing tooling
  evidence/       small CSV inputs read by the chain (material probe rows, sign audit,
                  within-site conditioning) and the probe run notes
results/          paper_stats_v1.json (single source of truth) and the companion
                  packages that are merged into it
external/         archived model-run configurations and the outputs the chain reads:
                  TEB namelists and baseline LW-up output per record, TEB/CLM-Urban
                  campaign summaries, CLM-Urban per-record results and surface data,
                  probe summaries, the CLM-Urban runner scripts
paper/manuscript_dtsa/figures/   figure generators and the figures as submitted
paper/manuscript_dtsa/tables/    table generators' output (LaTeX) and the site master table
scripts/          corpus builder, TEB input builder, shared figure style
src/              corpus loader and the differentiable single-layer probe model
patches/          local modifications to TEB v4.1.2 (CeCILL-licensed, see patches/README.md)
data/             where the Urban-PLUMBER collection goes (not redistributed; data/README.md)
RELEASE_MANIFEST.json   SHA-256 of every file, source path in the research project,
                  and the unified diff of every line changed for the release
```

Frozen conventions used throughout (see the header of `paper_stats.py`):
Stefan–Boltzmann constant `SIG = 5.67e-8` W m⁻² K⁻⁴ and surface emissivity
`EPS = 0.95` in the inversion
`Ts = ((LWup − (1 − EPS)·LWdown) / (EPS·SIG))^0.25`; nocturnal mask
`night_mask & ~pre_spinup_flag`; `ΔT_s−a` = nocturnal mean of `Ts − Tair`;
TEB output row *i* pairs with forcing step *i + 1*.

## Environment

Python 3.12 with the packages in `requirements.txt` (the published numbers
were produced with the bracketed versions, CPU only). The figure generators
expect the Arial font (they fall back to Helvetica or DejaVu Sans).

```
python -m venv .venv && . .venv/bin/activate     # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

The model runs need, in addition, a Linux environment (WSL Ubuntu 24.04 was
used) with gfortran and CMake for TEB, and Docker plus `pyclmuapp==0.0.2`
with the image `envdes/clmu-app:1.1` for CLM-Urban.

## Reproducing the paper

### Level 1: tables and figures from the shipped statistics (no data needed)

```
python paper/manuscript_dtsa/figures/make_figures.py       # Figs. 2-7 and S1
python paper/manuscript_dtsa/figures/make_schematics.py    # Fig. 1 and the graphical abstract
python analysis/revalidation_2026-08/scripts/review5_table1_rebuild.py          # Table 1 (table1_sites.tex)
python analysis/revalidation_2026-08/scripts/v10_metatable.py                   # Table 2 (table3_geometry.tex; needs data/, see below)
python analysis/revalidation_2026-08/scripts/review5_provenance_table.py        # Table 3 (table4_provenance.tex)
python analysis/revalidation_2026-08/scripts/review5_table_modelvar.py          # Table 4 (table2_model_variants.tex)
python analysis/revalidation_2026-08/scripts/review7_table_scenario_cells.py    # Table 5 (table6_scenarios.tex) and Table S2
python analysis/revalidation_2026-08/scripts/review7_table_metric_families.py   # Table 6 (table5_metric_families.tex)
python analysis/revalidation_2026-08/scripts/review7_table_benchmark_protocols.py  # Table S1 (tableA1_benchmark_protocols.tex)
python analysis/revalidation_2026-08/scripts/review8_table_b1.py                # Table S3
python analysis/revalidation_2026-08/scripts/review8_table_material_K.py        # Table S4
```

Table numbers follow the order of appearance in the manuscript, which differs
from the historical file names given in parentheses. Figure files map to the
manuscript as follows: `fig0_design` → Fig. 1, `fig1_dtsa_ranked` → Fig. 2,
`fig2_core_albedo` → Fig. 3, `fig3_model_vs_obs` → Fig. 4,
`fig4_lwup_bias_paired` → Fig. 5, `fig7_scenarios` → Fig. 6,
`fig5_elimination` → Fig. 7, `fig6_prescription` → Fig. S1,
`graphical_abstract` → graphical abstract.

Checked for this release: regenerating all nine PNG figures inside this tree
gives files identical pixel for pixel to the submitted figures, and eight of
the nine table generators rewrite the shipped LaTeX tables byte for byte.
`v10_metatable.py` (Table 2) additionally reads the site-metadata CSVs of the
raw collection and therefore needs step 1 of Level 2.

### Level 2: the statistics package from observations and archived model output

1. Obtain the Urban-PLUMBER collection and build the corpus (`data/README.md`).
2. Run the generator chain **in this order** from the repository root. The
   first script rewrites `results/paper_stats_v1.json` wholesale; the others
   append sections in place or write companion packages under `results/`
   that `review5_merge.py` folds in. The merge runs three times because the
   round-6 audits, `review7_debiased_mae.py` and `review8_core16_and_b1.py`
   read blocks that only exist after a merge; `review5_merge.py` is
   idempotent, so repeated runs are safe.

```
S=analysis/revalidation_2026-08/scripts
# statistics package
python $S/paper_stats.py                      # observational gradient, TEB/CLM-Urban dTsa, elimination numbers
python $S/attribution_covariates.py           # morphology covariates and partial correlations
python $S/review2_additions.py
python $S/review3_additions.py
python $S/review3b_additions.py
python $S/review4_additions.py
python $S/v10_tables.py                       # obs metadata, level-vs-gradient (run exactly once, here)
python $S/review5_adderley_propagation.py     # observation-uncertainty propagation (obs side)
python $S/review5_temporal_seasonal.py        # temporal and seasonal sampling
python $S/review5_permutation_rebuild.py      # city-block permutation inference
python $S/review5_scheme_ordering_paired.py   # paired TEB-minus-CLM-Urban bootstrap
python $S/review5_benchmark_skill.py          # per-record skill scores and benchmarks
python $S/review5_core15.py                   # PL-Narutowicza sensitivity
python $S/review5_obs_error_into_model.py     # observation-error scenarios on the model side (Table 5 / S2)
python $S/review5_obs_error_into_model_ordering.py   # same scenarios carried into the ordering claim
python $S/review5_benchmark_cityholdout.py    # leave-one-city-cluster-out benchmark refit
python $S/review5_run_provenance.py           # needs raw model output (Level 3); its result is shipped
python $S/review5_constrained_experiments.py  # needs raw model output (Level 3); its result is shipped
python $S/review5_merge.py                    # merge pass 1
python $S/review5_benchmark_decomposition.py  # mean-bias vs temporal-structure error budget
python $S/review5_merge.py
# audits that read merged blocks
python $S/review6_text_number_audit.py
python $S/review6_adversarial_mc_bounds.py
export SCR=./scratch; mkdir -p $SCR
python $S/review6_benchmark_holdout_verify.py 2>&1 | tee $SCR/adv.log   # the merge below parses this log
python $S/review6_benchmark_seed_fragility.py
python $S/review6_benchmark_dedup_pool.py
python $S/review6_benchmark_dedup_control.py
python $S/review6_minneapolis_duplication_audit.py
python $S/review6_benchmark_verification_merge.py
python $S/review5_merge.py                    # merge pass 2
python $S/review7_debiased_mae.py             # own-mean-bias-removed benchmark counts
python $S/review5_merge.py
python $S/review8_core16_and_b1.py            # core-16 metric families, albedo + sky emissivity, material perturbations in K
python $S/review5_merge.py                    # merge pass 3 (final)
```

Then run the table generators of Level 1 (`review5_table_modelvar.py`
overwrites the `table2_model_variants.tex` that `v10_tables.py` writes) and
the figure generators. Do not re-run `v10_tables.py` after a merge: it would
resurrect numbers the merge has superseded (its docstring says so).

Inputs that these scripts read besides the corpus are all shipped:
`external/teb_runs/<SITE>/output/LWU_base.txt` (TEB baseline upward longwave,
19 evaluable records plus MX-Escandon), `external/clmu_g1/g2_results.json`
(CLM-Urban per-record results after the alignment correction, with the
pre-correction file kept for audit; aggregated from the per-record
`external/clmu_g1/results/<SITE>.json`), the probe summaries under
`external/clmu_probe/`, `external/dslucm_surgery/` and
`analysis/revalidation_2026-08/evidence/`, and the auxiliary SUEWS facet
values under `external/suews_g1/` (read by `paper_stats.py`; not used in the
paper). The resampling packages (`review5_*`, `review6_*`) are the slow part
of the chain.

Shipped as archived artifacts because no committed script writes them:
`results/clmu_v11_rebuild.json` (the 19-record aggregation of
`clmu_post_v11.py`, which is a per-record tool), `results/geometry_v11.json`
(documented radiometer heights and z/H), `results/teb_persite_warming.json`
(per-record warming of the albedo campaign, used only as a cross-check), and
`evidence/probe_rows.csv` (TEB material-perturbation envelope, produced from
the runs described in `evidence/probe_invocation_notes.md` whose namelists are
under `external/teb_probe/`). Two further results files are regenerated only
from raw model output that is not redistributed (Level 3):
`results/review5_constrained.json` and `results/review5_run_provenance.json`.
`results/review5_obs_error_ordering.json` is written by
`review5_obs_error_into_model_ordering.py` and is quoted in the Supplement
directly; it is not folded into `paper_stats_v1.json`.

`evidence/sign_audit_table.csv` is produced by `sign_audit.py`;
`evidence/conditioning_table.csv` by `conditioning_analysis.py` (exploratory;
not read by the chain). `flux_direction_check.py` is the script from which
the masking and alignment convention was copied verbatim into the others.
`external/teb_campaign_results.json` and `external/clmu_campaign_results.json`
are the outputs of the two campaign evaluators and are kept for reference;
the chain takes the campaign numbers from `review5_constrained_experiments.py`.

The differentiable single-layer probe of Supplementary Sect. S1 is
`src/models/dslucm_forward.py` with the structural variants in
`analysis/revalidation_2026-08/scripts/dslucm_surgery.py`, which writes
`external/dslucm_surgery/results.json` (requires JAX and Equinox).

### Level 3: the model runs

**TEB.** Build TEB as described in `patches/README.md`. Per-record driver
inputs are produced by `scripts/teb_make_inputs.py` from the corpus and the
site metadata (the namelists it wrote are shipped as
`external/teb_runs/<SITE>/input.nml`; the forcing files are regenerated). The
baseline run is the driver executed in each record folder; the
albedo-constrained campaign and its baseline-reproduction check are
`teb_rebuild_and_campaignA.sh`, the spin-up campaign is
`teb_campaignB_build.py` plus `teb_run_campaignB.sh`, and `teb_campaign_eval.py`
turns the outputs into `external/teb_campaign_results.json`. All runs were made
with the environment variables `TEB_DH_CLOSURE` and `TEB_DH_DIAG` unset.

**CLM-Urban.** `external/clmu_g1/build_inputs.py` builds the forcing and the
per-record surface dataset (pyclmuapp's packaged King's College surface data
with only geometry and location overridden; the exact files are shipped as
`external/clmu_g1/<SITE>/surfdata.nc` and `run_params.json`), and
`external/clmu_g1/run_one.py` runs one record in the container. The scripts in
`analysis/revalidation_2026-08/scripts/clmu_*.sh` copy these two files into a
Linux working directory and run the baseline, albedo-constrained and spin-up
campaigns over the 19 records; the copies that were actually executed are
preserved in `external/clmu_wsl_work/`. `clmu_post.py` and `clmu_post_v11.py`
compute the record-period metrics from the history files (the latter fixes
the window alignment at two records, documented in its header), and
`clmu_campaign_eval.py` writes `external/clmu_campaign_results.json`. The
constants `WORK` and `ROOT` at the top of `build_inputs.py`, `run_one.py` and
the `clmu_wsl_work` copies, and `W=/root/clmu_work` in the shell scripts, are
the paths of the machine the runs were made on; set them to your own.

Raw model output is large (about 14 GB for all campaigns) and is not part of
this repository; the files the analysis reads from it are shipped as listed
under Level 2.

## Changes relative to the research project

The code was copied from the authors' research project without touching any
computation. `RELEASE_MANIFEST.json` records, for every file, its SHA-256 and
its path in the project, and for every edited file the unified diff. The
edits are limited to:

* root-path lines: absolute `D:\JM\...` or `/mnt/d/JM/...` paths replaced by
  paths resolved relative to the file (`Path(__file__).resolve().parents[3]`);
* the two Korean folder names of the project renamed to
  `analysis/revalidation_2026-08` and `evidence` (also in provenance strings
  that the generators write into their JSON output);
* `review5_table_modelvar.py`: its cross-check of two printed strings against
  the manuscript source is skipped when `main.tex` is absent;
* `adv_cityholdout.py`: a new two-line alias module so that three
  verification scripts can import the consolidated
  `review6_benchmark_holdout_verify.py` under the name they were written with.

## Not included

The Urban-PLUMBER observations and the derived corpus (CC-BY-4.0; obtain from
Zenodo), the TEB source (CeCILL v2.1; obtain from its archive), the CLM-Urban
container, raw model output, and the manuscript source.

## License and citation

Our code is released under the MIT license (`LICENSE`); third-party
components keep their own terms (`THIRD_PARTY_NOTICES.md`). Please cite the
article and the archived software release (`CITATION.cff`).

## Funding

National Research Foundation of Korea (NRF) grant funded by the Korea
government (MSIT), RS-2023-00259403; Knowledge-based Environmental
Specialized Graduate Program through the Korea Environment Industry &
Technology Institute (KEITI), funded by the Ministry of Climate, Energy, and
Environment (MCEE).

## Contact

Junsuk Kang, Department of Civil and Environmental Engineering, Seoul
National University (junkang@snu.ac.kr).
