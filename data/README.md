# Observational data

The observations are **not redistributed** in this repository. They come from
the harmonized Urban-PLUMBER collection (Lipson et al., 2022, *Earth System
Science Data*, CC-BY-4.0), versioned data record
<https://doi.org/10.5281/zenodo.7104984>.

## 1. Download and unpack

Download `Urban-PLUMBER_FullCollection_v1.zip` (about 650 MB) from the Zenodo
record and unpack it so that the per-site folders sit under
`data/urban-plumber/FullCollection/`:

```
data/urban-plumber/FullCollection/
  AU-Preston/
    AU-Preston_sitedata_v1.csv
    timeseries/
      AU-Preston_metforcing_v1.nc
      AU-Preston_clean_observations_v1.nc
      ...
  CA-Sunset/
  ...
```

## 2. Build the processed corpus

```
python scripts/week2_build_multisite_corpus.py
```

writes one NetCDF per site to `data/urban-plumber/corpus/<SITE>.nc` (20 files,
about 100 MB in total) plus `data/urban-plumber/corpus_index.json`. Every
analysis script reads the corpus through `src/training/corpus_loader.py`.

Conventions baked into the corpus (see the script docstring): native
observation window per site, hourly sites forward-filled to the 1800 s
cadence, `night_mask` = SWdown < 1 W m-2, `pre_spinup_flag` = first six
months. AU-SurreyHills (0.4 yr) is excluded by the builder; MX-Escandon has no
upward-longwave observations, which leaves the 19 evaluable records used in
the paper.

## 3. What else is read from the raw collection

`<SITE>_sitedata_v1.csv` (site metadata: albedo, geometry, cover fractions,
measurement heights) is read directly from `FullCollection/` by
`paper_stats.py`, `review5_run_provenance.py` and the TEB/CLM-Urban input
builders; `timeseries/<SITE>_metforcing_v1.nc` is read by the constrained
experiments and the campaign input builders.
