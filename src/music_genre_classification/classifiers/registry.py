# Copyright (C) 2026 Bartłomiej Kózka
"""Registry of the available genre classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from music_genre_classification.classifiers.cnn import CnnClassifier
from music_genre_classification.classifiers.svm import SvmClassifier
from music_genre_classification.classifiers.transformer import TransformerClassifier

if TYPE_CHECKING:
    from music_genre_classification.classifiers.base import GenreClassifier

CLASSIFIERS: dict[str, type[GenreClassifier]] = {
    SvmClassifier.name: SvmClassifier,
    CnnClassifier.name: CnnClassifier,
    TransformerClassifier.name: TransformerClassifier,
}

DEFAULT_MODEL_DIR = Path("models")
MODEL_SUFFIXES = {"svm": ".joblib", "cnn": ".pt", "transformer": ".pt"}


def model_path(name: str, directory: Path = DEFAULT_MODEL_DIR) -> Path:
    """Return where a trained model of the given kind is stored.

    Args:
        name: Classifier name, one of ``CLASSIFIERS``.
        directory: Directory holding trained models.

    Returns:
        Path of the model file.
    """
    return directory / f"{name}{MODEL_SUFFIXES[name]}"


def available_models(directory: Path = DEFAULT_MODEL_DIR) -> list[str]:
    """List the classifiers that have a trained model file.

    Args:
        directory: Directory holding trained models.

    Returns:
        Names of the classifiers whose model file exists, in registry order.
    """
    return [name for name in CLASSIFIERS if model_path(name, directory).is_file()]
