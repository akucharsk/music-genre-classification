# Copyright (C) 2026 Bartłomiej Kózka
"""Interface shared by every genre classifier."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, Self

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from music_genre_classification.dataset import Split


class GenreClassifier(Protocol):
    """A model that learns to predict a genre from MFCC segments.

    Attributes:
        name: Short identifier used on the command line and in file names.
    """

    name: ClassVar[str]

    def fit(self, split: Split) -> None:
        """Train on the training part of a split, using the validation part to monitor progress.

        Args:
            split: Train and validation datasets.
        """
        ...

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Estimate genre probabilities for MFCC segments.

        Args:
            features: Array of shape (segments, frames, coefficients).

        Returns:
            Array of shape (segments, genres) whose rows sum to one.
        """
        ...

    def save(self, path: Path) -> None:
        """Write the trained model to disk, creating parent directories when needed.

        Args:
            path: Destination file.
        """
        ...

    @classmethod
    def load(cls, path: Path) -> Self:
        """Restore a model written by ``save``.

        Args:
            path: File written by ``save``.

        Returns:
            The restored model.
        """
        ...
