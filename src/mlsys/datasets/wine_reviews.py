"""spawn99/wine-reviews loader.

The generic loader in :mod:`mlsys.datasets` already handles this dataset — every
column the template needs is plain HF metadata. This module exists as the
documented home for any future wine-specific preprocessing (e.g. filtering rows
with missing prices) so peers know where to put it.
"""

from __future__ import annotations

from mlsys.datasets import LoadedDataset, load_dataset

DATASET_NAME = "wine_reviews"


def load() -> LoadedDataset:
    return load_dataset(DATASET_NAME)
