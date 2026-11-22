# Copyright (C) 2026 Bartłomiej Kózka
"""Measure how well a classifier predicts genres."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from music_genre_classification.classifiers.base import GenreClassifier
    from music_genre_classification.dataset import Dataset


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Quality of a classifier on one dataset.

    Attributes:
        accuracy: Fraction of segments classified correctly.
        report: Per-genre precision, recall and F1 as printable text.
        confusion: Matrix whose rows are true genres and columns predicted genres.
    """

    accuracy: float
    report: str
    confusion: NDArray[np.int64]


def evaluate(classifier: GenreClassifier, dataset: Dataset) -> Evaluation:
    """Predict every segment of a dataset and compare with the true genres.

    Args:
        classifier: Trained classifier.
        dataset: Segments with their true labels.

    Returns:
        Accuracy, per-genre report and confusion matrix.
    """
    predicted = classifier.predict_proba(dataset.features).argmax(axis=1)
    labels = list(range(len(dataset.genres)))
    return Evaluation(
        accuracy=float(accuracy_score(dataset.labels, predicted)),
        report=str(
            classification_report(
                dataset.labels,
                predicted,
                labels=labels,
                target_names=dataset.genres,
                zero_division=0,
            )
        ),
        confusion=confusion_matrix(dataset.labels, predicted, labels=labels),
    )
