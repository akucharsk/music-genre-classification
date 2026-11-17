# Copyright (C) 2026 Bartłomiej Kózka
"""Support-vector machine on PCA-reduced MFCCs."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC

from music_genre_classification.dataset import RANDOM_SEED

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from music_genre_classification.dataset import Split

PCA_COMPONENTS = 50
SVM_C = 5
SVM_GAMMA = "scale"  # a fixed 0.001 predicted a single genre on unscaled MFCCs


class SvmClassifier:
    """RBF support-vector classifier on flattened MFCCs reduced with PCA.

    Attributes:
        name: Short identifier used on the command line and in file names.
    """

    name: ClassVar[str] = "svm"

    def __init__(self) -> None:
        """Create an untrained classifier."""
        self._pipeline: Pipeline | None = None
        self._genres: tuple[str, ...] = ()

    @property
    def genres(self) -> tuple[str, ...]:
        """Genre names; the columns of ``predict_proba`` follow this order."""
        return self._genres

    def fit(self, split: Split) -> None:
        """Fit PCA and the SVM on the training segments.

        Args:
            split: Train and validation datasets; only the training part is used.
        """
        self._genres = split.train.genres
        flat = _flatten(split.train.features)
        components = min(PCA_COMPONENTS, *flat.shape)
        self._pipeline = make_pipeline(
            PCA(n_components=components, random_state=RANDOM_SEED),
            CalibratedClassifierCV(
                SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA, random_state=RANDOM_SEED),
                ensemble=False,
            ),
        )
        self._pipeline.fit(flat, split.train.labels)

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Estimate genre probabilities for MFCC segments.

        Args:
            features: Array of shape (segments, frames, coefficients).

        Returns:
            Array of shape (segments, genres) whose rows sum to one.

        Raises:
            RuntimeError: If the model has not been trained or loaded.
        """
        if self._pipeline is None:
            msg = "The classifier is not trained."
            raise RuntimeError(msg)
        return np.asarray(self._pipeline.predict_proba(_flatten(features)), dtype=np.float64)

    def save(self, path: Path) -> None:
        """Write the fitted pipeline to disk.

        Args:
            path: Destination file; parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self._pipeline, "genres": list(self._genres)}, path)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Restore a classifier written by ``save``.

        Args:
            path: File written by ``save``.

        Returns:
            The restored classifier.
        """
        payload = joblib.load(path)
        classifier = cls()
        classifier._genres = tuple(payload["genres"])
        classifier._pipeline = payload["pipeline"]
        return classifier


def _flatten(features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Turn every segment's MFCC matrix into one vector.

    Args:
        features: Array of shape (segments, frames, coefficients).

    Returns:
        Array of shape (segments, frames * coefficients).
    """
    return features.reshape(features.shape[0], -1)
