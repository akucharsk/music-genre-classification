# Copyright (C) 2026 Bartłomiej Kózka
"""Tests for MFCC preprocessing."""

from __future__ import annotations

import json
import wave
from typing import TYPE_CHECKING

import numpy as np
import pytest

from music_genre_classification.data_preprocessing import (
    SAMPLE_RATE,
    SAMPLES_PER_TRACK,
    DatasetError,
    MfccConfig,
    build_dataset,
    find_genres,
    save_dataset,
    split_segments,
)

if TYPE_CHECKING:
    from pathlib import Path

CONFIG = MfccConfig(num_segments=3)


def _write_track(path: Path, samples: int = SAMPLES_PER_TRACK) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pcm = rng.integers(-16000, 16000, samples, dtype=np.int16)
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(pcm.tobytes())


def test_config_derives_segment_sizes() -> None:
    """Segment length and frame count follow from the configuration."""
    config = MfccConfig(num_segments=10, hop_length=512)

    assert config.samples_per_segment == SAMPLES_PER_TRACK // 10
    assert config.frames_per_segment == 130


@pytest.mark.parametrize("field", ["num_mfcc", "n_fft", "hop_length", "num_segments"])
def test_config_rejects_non_positive_values(field: str) -> None:
    """Zero or negative parameters are refused."""
    with pytest.raises(ValueError, match="positive"):
        MfccConfig(**{field: 0})


def test_split_segments_drops_incomplete_tail() -> None:
    """A signal shorter than the full track yields only complete segments."""
    size = CONFIG.samples_per_segment
    signal = np.zeros(size * 2 + size // 2)

    segments = list(split_segments(signal, CONFIG))

    assert len(segments) == 2
    assert all(len(segment) == size for segment in segments)


def test_find_genres_is_sorted(tmp_path: Path) -> None:
    """Genre directories are returned alphabetically, files are ignored."""
    for name in ("rock", "blues", "jazz"):
        (tmp_path / name).mkdir()
    (tmp_path / "notes.txt").write_text("ignored")

    assert [path.name for path in find_genres(tmp_path)] == ["blues", "jazz", "rock"]


def test_find_genres_requires_existing_directory(tmp_path: Path) -> None:
    """A missing dataset directory raises DatasetError."""
    with pytest.raises(DatasetError, match="not found"):
        find_genres(tmp_path / "missing")


def test_find_genres_requires_genres(tmp_path: Path) -> None:
    """An empty dataset directory raises DatasetError."""
    with pytest.raises(DatasetError, match="No genre"):
        find_genres(tmp_path)


def test_build_dataset_labels_segments_and_skips_corrupt_tracks(tmp_path: Path) -> None:
    """Every segment is labelled by genre index and unreadable files are skipped."""
    _write_track(tmp_path / "blues" / "blues.00000.wav")
    _write_track(tmp_path / "jazz" / "jazz.00000.wav")
    (tmp_path / "jazz" / "jazz.00001.wav").write_bytes(b"not audio")

    data = build_dataset(tmp_path, CONFIG)

    assert data["genres"] == ["blues", "jazz"]
    assert data["labels"] == [0, 0, 0, 1, 1, 1]
    assert len(data["mfcc"]) == 6
    assert len(data["mfcc"][0]) == CONFIG.frames_per_segment
    assert len(data["mfcc"][0][0]) == CONFIG.num_mfcc


def test_save_dataset_round_trips(tmp_path: Path) -> None:
    """The JSON file contains the dataset and parent directories are created."""
    output = tmp_path / "nested" / "data.json"
    data = {"genres": ["blues"], "labels": [0], "mfcc": [[[1.0, 2.0]]]}

    save_dataset(data, output)  # type: ignore[arg-type]

    assert json.loads(output.read_text(encoding="utf-8")) == data
