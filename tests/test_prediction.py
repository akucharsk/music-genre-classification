# Copyright (C) 2026 Bartłomiej Kózka
"""Tests for track-level prediction."""

from __future__ import annotations

import numpy as np
import pytest

from music_genre_classification.prediction import SEGMENT_SECONDS, Prediction, predict_track

GENRES = ("a", "b", "c")


class _FixedClassifier:
    """Classifier that returns preset probabilities, one row per segment."""

    name = "fixed"
    genres = GENRES

    def __init__(self, probabilities: list[list[float]]) -> None:
        """Remember the probabilities to return."""
        self._probabilities = np.asarray(probabilities)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return the preset probabilities for as many rows as there are segments."""
        return self._probabilities[: len(features)]


def _prediction() -> Prediction:
    rows = [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]]
    return Prediction(GENRES, np.asarray(rows))


def test_shares_count_winning_segments() -> None:
    """Each genre's share is the fraction of segments it wins."""
    assert _prediction().shares.tolist() == [0.5, 0.25, 0.25]


def test_top_orders_by_share_then_mean_probability() -> None:
    """The best genre comes first and ties are broken by mean probability."""
    top = _prediction().top(2)

    assert top[0] == ("a", 0.5)
    assert top[1][0] == "b"


def test_times_follow_segment_length() -> None:
    """Segments start every ``segment_seconds``."""
    assert _prediction().times.tolist() == [0.0, 3.0, 6.0, 9.0]
    assert pytest.approx(3.0) == SEGMENT_SECONDS


def test_predict_track_uses_the_classifier() -> None:
    """The prediction carries the classifier's genres and probabilities."""
    classifier = _FixedClassifier([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])

    prediction = predict_track(classifier, [[[0.0] * 13] * 130] * 2)  # type: ignore[arg-type]

    assert prediction.genres == GENRES
    assert prediction.probabilities.shape == (2, 3)
    assert prediction.top(1) == [("c", 0.5)]


def test_predict_track_needs_segments() -> None:
    """An empty track is rejected."""
    with pytest.raises(ValueError, match="segment"):
        predict_track(_FixedClassifier([]), [])  # type: ignore[arg-type]


def test_binned_averages_probabilities_over_longer_steps() -> None:
    """Segments are grouped by the step they start in and their probabilities averaged."""
    binned = _prediction().binned(6.0)

    assert binned.segment_seconds == pytest.approx(6.0)
    assert binned.times.tolist() == [0.0, 6.0]
    assert binned.probabilities[0].tolist() == pytest.approx([0.65, 0.25, 0.1])
    assert binned.probabilities.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_binned_keeps_segments_when_the_step_is_not_longer() -> None:
    """A step of at most one segment leaves the prediction unchanged."""
    prediction = _prediction()

    assert prediction.binned(3.0) is prediction
    assert prediction.duration == pytest.approx(12.0)
