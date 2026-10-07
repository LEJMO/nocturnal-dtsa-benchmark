"""Path C training package.

Loads the multi-site corpus (data/urban-plumber/corpus/*.nc) and drives LOSO
training of dSLUCM + residual operator. Cross-UCM work is out of scope
(R11 materialised 2026-04-24).
"""
from src.training.corpus_loader import (  # noqa: F401
    CorpusRecord,
    SiteBatch,
    load_site_corpus,
    load_all_sites,
    compute_pooled_stats,
    normalise_site,
)
