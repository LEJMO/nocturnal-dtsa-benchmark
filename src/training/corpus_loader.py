"""Corpus loader for Path C training.

Reads data/urban-plumber/corpus/<SITE>.nc (written by
scripts/week2_build_multisite_corpus.py) and turns each site into a training-
ready `CorpusRecord`: JAX-ready numpy arrays (float64) for forcing/obs plus
boolean masks, with per-site normalisation statistics computed from pooled
training data.

Design notes
------------
* We keep float64 throughout (CPU, jax_enable_x64 already set in conftest).
* Night mask + finite-Tair mask are AND-combined into `training_mask` so the
  training loss only activates where observations exist and SWdown < 1 W/m^2.
* Pre-spinup flag zeroes the first 6 months of each site (model state
  equilibration before loss scoring).
* Pooled statistics (mean/std for each forcing channel and Tair obs) are
  computed across the training fold only; the held-out site uses those same
  statistics to avoid leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = PROJECT_ROOT / "data" / "urban-plumber" / "corpus"

FORCING_CHANNELS = ("SWdown", "LWdown", "Tair", "Qair", "PSurf", "Wind_N", "Wind_E")


@dataclass(frozen=True)
class CorpusRecord:
    """Per-site training record (unnormalised)."""

    site: str
    koppen_zone: str
    latitude: float
    longitude: float
    time_axis: np.ndarray  # datetime64[ns]
    n_steps: int
    timestep_s: int
    forcing: dict[str, np.ndarray]  # keys match FORCING_CHANNELS
    obs_Tair: np.ndarray
    night_mask: np.ndarray
    season: np.ndarray
    pre_spinup_flag: np.ndarray
    training_mask: np.ndarray  # night_mask & finite(obs_Tair) & ~pre_spinup_flag
    temporal_holdout_mask: np.ndarray  # bool; True = held-out test window (default all-False)


@dataclass
class SiteBatch:
    """Normalised site batch consumed by the trainer."""

    site: str
    koppen_zone: str
    forcing: np.ndarray  # shape (n_steps, 7) in the FORCING_CHANNELS order, normalised
    forcing_raw: np.ndarray  # shape (n_steps, 7), physical units (for dSLUCM forward)
    obs_Tair_raw: np.ndarray  # K
    training_mask: np.ndarray  # bool
    pre_spinup_flag: np.ndarray  # bool
    season: np.ndarray
    night_mask: np.ndarray
    temporal_holdout_mask: np.ndarray  # bool, same length as forcing
    n_steps: int


def _load_nc(path: Path) -> xr.Dataset:
    return xr.open_dataset(path)


def list_sites(corpus_dir: Path = CORPUS_DIR) -> list[str]:
    return sorted(p.stem for p in corpus_dir.glob("*.nc"))


def load_site_corpus(site: str, corpus_dir: Path = CORPUS_DIR,
                     temporal_holdout_mask: np.ndarray | None = None) -> CorpusRecord:
    nc_path = corpus_dir / f"{site}.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"corpus missing for site {site}: {nc_path}")

    ds = _load_nc(nc_path)
    try:
        forcing = {ch: np.asarray(ds[f"forcing_{ch}"].values, dtype=np.float64) for ch in FORCING_CHANNELS}
        obs_Tair = np.asarray(ds["obs_Tair"].values, dtype=np.float64)
        night_mask = np.asarray(ds["night_mask"].values, dtype=bool)
        season = np.asarray(ds["season"].values, dtype=np.int8)
        pre_spinup_flag = np.asarray(ds["pre_spinup_flag"].values, dtype=bool)
        finite_obs = np.isfinite(obs_Tair)
        n_steps = int(ds.sizes["time"])
        if temporal_holdout_mask is None:
            hmask = np.zeros(n_steps, dtype=bool)
        else:
            hmask = np.asarray(temporal_holdout_mask, dtype=bool)
            if hmask.shape[0] != n_steps:
                raise ValueError(f"{site}: holdout mask len {hmask.shape[0]} != n_steps {n_steps}")
        training_mask = night_mask & finite_obs & (~pre_spinup_flag) & (~hmask)

        record = CorpusRecord(
            site=site,
            koppen_zone=str(ds.attrs.get("koppen_zone", "Unknown")),
            latitude=float(ds.attrs.get("latitude", 0.0)),
            longitude=float(ds.attrs.get("longitude", 0.0)),
            time_axis=np.asarray(ds["time"].values),
            n_steps=n_steps,
            timestep_s=int(ds.attrs.get("timestep_s", 1800)),
            forcing=forcing,
            obs_Tair=obs_Tair,
            night_mask=night_mask,
            season=season,
            pre_spinup_flag=pre_spinup_flag,
            training_mask=training_mask,
            temporal_holdout_mask=hmask,
        )
    finally:
        ds.close()
    return record


def load_all_sites(corpus_dir: Path = CORPUS_DIR) -> list[CorpusRecord]:
    return [load_site_corpus(s, corpus_dir) for s in list_sites(corpus_dir)]


def load_all_sites_with_holdout(masks: dict[str, np.ndarray],
                                corpus_dir: Path = CORPUS_DIR) -> list[CorpusRecord]:
    """Load every site, attaching its temporal holdout mask from `masks`."""
    out = []
    for s in list_sites(corpus_dir):
        out.append(load_site_corpus(s, corpus_dir, temporal_holdout_mask=masks.get(s)))
    return out


def compute_pooled_stats(records: Sequence[CorpusRecord]) -> dict[str, tuple[float, float]]:
    """Compute pooled (mean, std) per forcing channel from training fold records.

    Uses all timesteps where forcing is finite (which is ~100% for metforcing).
    Separately computes obs_Tair statistics on training_mask==True positions
    so that the residual loss can be normalised.
    """
    stats: dict[str, tuple[float, float]] = {}
    for ch in FORCING_CHANNELS:
        arrs = [r.forcing[ch][~r.temporal_holdout_mask] for r in records]
        concat = np.concatenate(arrs)
        finite = concat[np.isfinite(concat)]
        if finite.size == 0:
            stats[ch] = (0.0, 1.0)
        else:
            stats[ch] = (float(np.mean(finite)), float(np.std(finite) or 1.0))
    # Tair observed (on training mask)
    obs_collect: list[np.ndarray] = []
    for r in records:
        obs_collect.append(r.obs_Tair[r.training_mask])
    obs_concat = np.concatenate(obs_collect) if obs_collect else np.array([], dtype=np.float64)
    if obs_concat.size > 0:
        stats["obs_Tair"] = (float(np.mean(obs_concat)), float(np.std(obs_concat) or 1.0))
    else:
        stats["obs_Tair"] = (288.0, 10.0)  # fall-back typical urban Tair
    return stats


def normalise_site(record: CorpusRecord, stats: dict[str, tuple[float, float]]) -> SiteBatch:
    """Apply pooled stats to produce a normalised SiteBatch for the trainer."""
    normalised_cols: list[np.ndarray] = []
    raw_cols: list[np.ndarray] = []
    for ch in FORCING_CHANNELS:
        mu, sigma = stats[ch]
        raw = record.forcing[ch]
        norm = (raw - mu) / sigma
        normalised_cols.append(norm)
        raw_cols.append(raw)
    forcing_norm = np.stack(normalised_cols, axis=-1)  # (n_steps, 7)
    forcing_raw = np.stack(raw_cols, axis=-1)

    return SiteBatch(
        site=record.site,
        koppen_zone=record.koppen_zone,
        forcing=forcing_norm,
        forcing_raw=forcing_raw,
        obs_Tair_raw=record.obs_Tair,
        training_mask=record.training_mask,
        pre_spinup_flag=record.pre_spinup_flag,
        season=record.season,
        night_mask=record.night_mask,
        temporal_holdout_mask=record.temporal_holdout_mask,
        n_steps=record.n_steps,
    )
