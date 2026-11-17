# Copyright (C) 2026 Bartłomiej Kózka
"""Load the MFCC dataset and split it for training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from music_genre_classification.data_preprocessing import DatasetError

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

REQUIRED_KEYS = ("genres", "labels", "tracks", "mfcc")
VALIDATION_FOLDS = 5  # one fold, i.e. 20 %, is held out for validation
RANDOM_SEED = 42
# Genres of GTZAN in the alphabetical order the preprocessing step labels them in.
GTZAN_GENRES = (
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
)
DEFAULT_FEATURE_DIM = 13


@dataclass(frozen=True, slots=True)
class Dataset:
    """MFCC dataset loaded from JSON.

    Attributes:
        genres: Genre names; a label is an index into this tuple.
        features: MFCC array of shape (segments, frames, coefficients).
        labels: Genre index of every segment.
        tracks: Source track index of every segment.
    """

    genres: tuple[str, ...]
    features: NDArray[np.float64]
    labels: NDArray[np.int64]
    tracks: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Split:
    """Train and validation parts of a dataset.

    Attributes:
        train: Segments used to fit the model.
        validation: Segments held out to measure accuracy.
    """

    train: Dataset
    validation: Dataset


def load_dataset(path: Path) -> Dataset:
    """Load the dataset written by the preprocessing step.

    Args:
        path: JSON file produced by ``data_preprocessing``.

    Returns:
        The parsed dataset.

    Raises:
        DatasetError: If the file is missing, malformed or empty.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Could not read dataset: {path}"
        raise DatasetError(msg) from error
    if not isinstance(raw, dict) or any(key not in raw for key in REQUIRED_KEYS):
        msg = f"Dataset must contain: {', '.join(REQUIRED_KEYS)}"
        raise DatasetError(msg)
    labels = np.asarray(raw["labels"], dtype=np.int64)
    if len(labels) == 0 or not len(labels) == len(raw["mfcc"]) == len(raw["tracks"]):
        msg = "Dataset is empty or its columns differ in length."
        raise DatasetError(msg)
    return Dataset(
        genres=tuple(raw["genres"]),
        features=np.asarray(raw["mfcc"], dtype=np.float64),
        labels=labels,
        tracks=np.asarray(raw["tracks"], dtype=np.int64),
    )


def split_dataset(dataset: Dataset, *, seed: int = RANDOM_SEED) -> Split:
    """Split into train and validation sets keeping genres balanced and tracks apart.

    About a fifth of the segments is held out. Every genre is represented in both parts and the
    segments of one track never end up in both.

    Args:
        dataset: Dataset to split.
        seed: Random seed for a reproducible split.

    Returns:
        The train and validation datasets.
    """
    splitter = StratifiedGroupKFold(n_splits=VALIDATION_FOLDS, shuffle=True, random_state=seed)
    train_index, validation_index = next(
        splitter.split(dataset.features, dataset.labels, groups=dataset.tracks)
    )
    return Split(train=_subset(dataset, train_index), validation=_subset(dataset, validation_index))


def _subset(dataset: Dataset, index: NDArray[np.intp]) -> Dataset:
    """Select segments of a dataset by index.

    Args:
        dataset: Dataset to select from.
        index: Positions of the segments to keep.

    Returns:
        A dataset containing only the selected segments.
    """
    return Dataset(
        genres=dataset.genres,
        features=dataset.features[index],
        labels=dataset.labels[index],
        tracks=dataset.tracks[index],
    )
