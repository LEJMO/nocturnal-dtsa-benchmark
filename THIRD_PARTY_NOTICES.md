# Third-party notices

The code in this repository that we wrote is MIT-licensed (see `LICENSE`).
The components below are not ours and are not relicensed by us. None of them
is redistributed here except the TEB-derived patch noted in the first row.

| Component | How it is used | Terms | Where to obtain |
|---|---|---|---|
| Town Energy Balance (TEB) v4.1.2, upstream commit `b685f5f` | offline runs at 19 flux-tower records; built from source with a local patch | CeCILL Free Software License Agreement v2.1 | <https://github.com/TEB-model/teb>, archive <https://doi.org/10.5281/zenodo.5775962> |
| `patches/teb_v4.1.2_local_modifications.patch` (ours, derived from TEB) | gfortran-13 build fix plus dormant hooks, see `patches/README.md` | CeCILL v2.1 (derived work); **not** MIT | this repository |
| CLM5.0 / CLMU (Community Land Model, urban parameterization) | offline runs through the containerized workflow below | CESM/CTSM license (NCAR) | <https://github.com/ESCOMP/CTSM> |
| `pyclmuapp` 0.0.2 and container image `envdes/clmu-app:1.1` (Yu et al., 2025, *Environmental Modelling & Software* 188, 106391) | drives the CLM-Urban runs inside Docker | upstream terms | <https://github.com/envdes/pyclmuapp>, Docker Hub `envdes/clmu-app` |
| Urban-PLUMBER harmonized collection v1 (Lipson et al., 2022, *ESSD* 14, 5157-5178) | observations and forcing at 20 sites | CC-BY-4.0 | <https://doi.org/10.5281/zenodo.7104984> (see `data/README.md`) |
| SUEWS (OHM facet configuration, six sites) | auxiliary input kept only because the statistics generator reads it; not used in the paper | upstream terms | <https://github.com/UMEP-dev/SUEWS> |
| Python packages listed in `requirements.txt` | analysis, figures, probe | their own licenses | PyPI |

## What this means for reuse

* The analysis code may be reused under MIT.
* Anyone who rebuilds TEB with our patch is bound by CeCILL v2.1 for the
  resulting program.
* The observational data must be obtained from Zenodo and attributed to
  Lipson et al. (2022) under CC-BY-4.0; the processed corpus built by
  `scripts/week2_build_multisite_corpus.py` is a derived work of that data and
  is therefore not distributed here either.
* Model output files that we do ship (`external/teb_runs/*/output/LWU_base.txt`
  and the JSON summaries under `external/`) are outputs of our own runs and are
  provided under MIT for reproducibility; they do not contain TEB or CLM source.
