# Copyright (C) 2026 Bartłomiej Kózka
"""Train a baseline genre classifier on the MFCC dataset."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from music_genre_classification.data_preprocessing import DEFAULT_OUTPUT_PATH, DatasetError

if TYPE_CHECKING:
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/baseline.joblib")
REQUIRED_KEYS = ("genres", "labels", "tracks", "mfcc")
VALIDATION_FRACTION = 0.2
RANDOM_SEED = 42


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
    """Train and validation parts of a dataset."""

    train: Dataset
    validation: Dataset


@dataclass(frozen=True, slots=True)
class TrainResult:
    """A trained model together with its validation accuracy."""

    model: Pipeline
    accuracy: float


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


def summarize(features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Describe each segment by the mean and standard deviation of every MFCC over time.

    Args:
        features: Array of shape (segments, frames, coefficients).

    Returns:
        Array of shape (segments, 2 * coefficients).
    """
    return np.concatenate([features.mean(axis=1), features.std(axis=1)], axis=1)


def split_dataset(dataset: Dataset, *, seed: int = RANDOM_SEED) -> Split:
    """Split into train and validation sets without sharing a track between them.

    Args:
        dataset: Dataset to split.
        seed: Random seed for a reproducible split.

    Returns:
        The train and validation datasets.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_FRACTION, random_state=seed)
    train_index, validation_index = next(splitter.split(dataset.features, groups=dataset.tracks))
    return Split(train=_subset(dataset, train_index), validation=_subset(dataset, validation_index))


def train_baseline(split: Split) -> TrainResult:
    """Fit a scaled RBF support-vector classifier on summarized MFCCs.

    Args:
        split: Train and validation datasets.

    Returns:
        The fitted pipeline and its accuracy on the validation set.
    """
    model = make_pipeline(StandardScaler(), SVC())
    model.fit(summarize(split.train.features), split.train.labels)
    accuracy = model.score(summarize(split.validation.features), split.validation.labels)
    return TrainResult(model=model, accuracy=float(accuracy))


def save_model(result: TrainResult, genres: tuple[str, ...], path: Path) -> None:
    """Persist the model and its genre names, creating parent directories when needed.

    Args:
        result: Trained model.
        genres: Genre names matching the model's labels.
        path: Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model": result.model, "genres": genres}
    joblib.dump(payload, path)


def main() -> None:
    """Train the baseline on the default dataset and save it."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dataset = load_dataset(DEFAULT_OUTPUT_PATH)
    result = train_baseline(split_dataset(dataset))
    LOGGER.info("Validation accuracy: %.3f", result.accuracy)
    save_model(result, dataset.genres, DEFAULT_MODEL_PATH)
    LOGGER.info("Saved model to %s", DEFAULT_MODEL_PATH)


def _subset(dataset: Dataset, index: NDArray[np.intp]) -> Dataset:
    return Dataset(
        genres=dataset.genres,
        features=dataset.features[index],
        labels=dataset.labels[index],
        tracks=dataset.tracks[index],
    )


if __name__ == "__main__":
    main()
