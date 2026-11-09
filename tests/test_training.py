# Copyright (C) 2026 Bartłomiej Kózka
"""Tests for baseline training."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pytest

from music_genre_classification.data_preprocessing import DatasetError
from music_genre_classification.training import (
    Dataset,
    TrainResult,
    load_dataset,
    save_model,
    split_dataset,
    summarize,
    train_baseline,
)

if TYPE_CHECKING:
    from pathlib import Path

TRACKS_PER_GENRE = 10
SEGMENTS_PER_TRACK = 3


def _synthetic_dataset() -> Dataset:
    """Build two genres whose MFCC means are far apart.

    Returns:
        A synthetic, easily separable dataset.
    """
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], TRACKS_PER_GENRE * SEGMENTS_PER_TRACK)
    tracks = np.repeat(np.arange(2 * TRACKS_PER_GENRE), SEGMENTS_PER_TRACK)
    features = rng.normal(loc=labels[:, None, None] * 5.0, scale=1.0, size=(len(labels), 8, 4))
    return Dataset(genres=("a", "b"), features=features, labels=labels, tracks=tracks)


def _write_json(path: Path, dataset: Dataset) -> None:
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
    original = _synthetic_dataset()
    _write_json(path, original)

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


def test_summarize_reduces_time_axis() -> None:
    """Each segment becomes its means followed by its standard deviations."""
    assert summarize(_synthetic_dataset().features).shape == (60, 8)


def test_split_keeps_tracks_apart() -> None:
    """No track appears in both train and validation."""
    split = split_dataset(_synthetic_dataset())

    assert not set(split.train.tracks) & set(split.validation.tracks)
    assert len(split.train.labels) + len(split.validation.labels) == 60


def test_baseline_learns_separable_genres(tmp_path: Path) -> None:
    """The baseline separates two well-separated genres and can be saved and reloaded."""
    result = train_baseline(split_dataset(_synthetic_dataset()))

    assert result.accuracy > 0.9
    path = tmp_path / "models" / "baseline.joblib"
    save_model(result, ("a", "b"), path)
    saved = joblib.load(path)
    assert saved["genres"] == ("a", "b")
    assert isinstance(result, TrainResult)
