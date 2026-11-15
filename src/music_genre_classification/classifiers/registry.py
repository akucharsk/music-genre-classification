# Copyright (C) 2026 Bartłomiej Kózka
"""Registry of the available genre classifiers."""

from __future__ import annotations

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
