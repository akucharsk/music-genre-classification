# Copyright (C) 2026 Bartłomiej Kózka
"""Tests for dataset loading and splitting."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from music_genre_classification.data_preprocessing import DatasetError
from music_genre_classification.dataset import Dataset, load_dataset, split_dataset

if TYPE_CHECKING:
    from pathlib import Path

TRACKS_PER_GENRE = 10
SEGMENTS_PER_TRACK = 3
FRAMES = 16
COEFFICIENTS = 8


def synthetic_dataset() -> Dataset:
    """Build two genres that differ in the direction of a ramp over time.

    Returns:
        A small, easily separable dataset.
    """
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], TRACKS_PER_GENRE * SEGMENTS_PER_TRACK)
    tracks = np.repeat(np.arange(2 * TRACKS_PER_GENRE), SEGMENTS_PER_TRACK)
    ramp = np.linspace(-1.0, 1.0, FRAMES)[None, :, None] * (2.0 * labels[:, None, None] - 1.0) * 3
    noise = rng.normal(scale=1.0, size=(len(labels), FRAMES, COEFFICIENTS))
    return Dataset(genres=("a", "b"), features=ramp + noise, labels=labels, tracks=tracks)


def write_json(path: Path, dataset: Dataset) -> None:
    """Write a dataset in the format produced by the preprocessing step.

    Args:
        path: Destination file.
        dataset: Dataset to write.
    """
    payload = {
        "genres": list(dataset.genres),
        "labels": dataset.labels.tolist(),
        "tracks": dataset.tracks.tolist(),
        "mfcc": dataset.features.tolist(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_dataset_round_trips(tmp_path: Path) -> None:
    """A written dataset is loaded back with the same shapes."""
    path = tmp_path / "data.json"
    original = synthetic_dataset()
    write_json(path, original)

    loaded = load_dataset(path)

    assert loaded.genres == ("a", "b")
    assert loaded.features.shape == original.features.shape
    assert np.array_equal(loaded.labels, original.labels)


@pytest.mark.parametrize("content", ["not json", "[]", '{"genres": []}'])
def test_load_dataset_rejects_malformed_files(tmp_path: Path, content: str) -> None:
    """Invalid or incomplete JSON raises DatasetError."""
    path = tmp_path / "data.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(DatasetError):
        load_dataset(path)


def test_load_dataset_rejects_missing_file(tmp_path: Path) -> None:
    """A missing file raises DatasetError."""
    with pytest.raises(DatasetError, match="Could not read"):
        load_dataset(tmp_path / "missing.json")


def test_split_keeps_tracks_apart() -> None:
    """No track appears in both train and validation."""
    split = split_dataset(synthetic_dataset())

    assert not set(split.train.tracks) & set(split.validation.tracks)
    assert len(split.train.labels) + len(split.validation.labels) == 60
