# Copyright (C) 2026 Bartłomiej Kózka
"""Classify a track by predicting the genre of each of its segments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from music_genre_classification.data_preprocessing import SAMPLE_RATE, MfccConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from music_genre_classification.classifiers.base import GenreClassifier

SEGMENT_SECONDS = MfccConfig().samples_per_segment / SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class Prediction:
    """Per-segment genre probabilities of one track.

    Attributes:
        genres: Genre names; the columns of ``probabilities`` follow this order.
        probabilities: Array of shape (segments, genres) whose rows sum to one.
        segment_seconds: Duration of one segment.
    """

    genres: tuple[str, ...]
    probabilities: NDArray[np.float64]
    segment_seconds: float = SEGMENT_SECONDS

    @property
    def shares(self) -> NDArray[np.float64]:
        """Fraction of segments in which each genre is the most probable one.

        Returns:
            Array of shape (genres,) that sums to one.
        """
        winners = np.bincount(self.probabilities.argmax(axis=1), minlength=len(self.genres))
        return winners / len(self.probabilities)

    @property
    def times(self) -> NDArray[np.float64]:
        """Start time of every segment in seconds.

        Returns:
            Array of shape (segments,).
        """
        return np.arange(len(self.probabilities), dtype=np.float64) * self.segment_seconds

    @property
    def duration(self) -> float:
        """Length of the analyzed audio in seconds.

        Returns:
            Number of segments times the segment duration.
        """
        return len(self.probabilities) * self.segment_seconds

    def binned(self, seconds: float) -> Prediction:
        """Average the probabilities over longer time steps.

        Args:
            seconds: Width of one time step; steps not longer than a segment change nothing.

        Returns:
            A prediction with one row per time step whose rows still sum to one.
        """
        if seconds <= self.segment_seconds:
            return self
        step_of_segment = (self.times // seconds).astype(np.intp)
        means = np.stack([
            self.probabilities[step_of_segment == step].mean(axis=0)
            for step in np.unique(step_of_segment)
        ])
        return Prediction(self.genres, means, seconds)

    def top(self, count: int = 3) -> list[tuple[str, float]]:
        """Genres with the largest share of segments.

        Ties are broken by the mean probability over the whole track.

        Args:
            count: Number of genres to return.

        Returns:
            Genre names with their shares, best first.
        """
        order = np.lexsort((-self.probabilities.mean(axis=0), -self.shares))[:count]
        return [(self.genres[index], float(self.shares[index])) for index in order]


def predict_track(
    classifier: GenreClassifier, segments: Sequence[Sequence[Sequence[float]]]
) -> Prediction:
    """Predict the genre of every segment of a track.

    Args:
        classifier: Trained classifier.
        segments: One MFCC matrix (frames by coefficients) per segment, non-empty.

    Returns:
        The per-segment probabilities.

    Raises:
        ValueError: If there are no segments.
    """
    if not segments:
        msg = "At least one segment is required."
        raise ValueError(msg)
    features = np.asarray(segments, dtype=np.float64)
    return Prediction(classifier.genres, classifier.predict_proba(features))
