# Copyright (C) 2026 Bartłomiej Kózka
"""Train a genre classifier from the command line."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from music_genre_classification.classifiers.registry import CLASSIFIERS
from music_genre_classification.classifiers.torch_base import TorchClassifier
from music_genre_classification.data_preprocessing import DEFAULT_OUTPUT_PATH
from music_genre_classification.dataset import load_dataset, split_dataset
from music_genre_classification.evaluation import Evaluation, evaluate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from music_genre_classification.classifiers.base import GenreClassifier

LOGGER = logging.getLogger(__name__)

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


def train(
    name: str,
    data_path: Path,
    model_dir: Path,
    epochs: int | None = None,
) -> Evaluation:
    """Train one classifier, evaluate it on the validation split and save it.

    Args:
        name: Classifier name, one of ``CLASSIFIERS``.
        data_path: JSON file produced by the preprocessing step.
        model_dir: Directory to save the trained model in.
        epochs: Number of epochs for neural networks; the classifier's default when omitted.

    Returns:
        Validation-set evaluation of the trained classifier.
    """
    dataset = load_dataset(data_path)
    split = split_dataset(dataset)
    classifier = _create(name, epochs)
    classifier.fit(split)
    evaluation = evaluate(classifier, split.validation)
    classifier.save(model_path(name, model_dir))
    return evaluation


def main(argv: Sequence[str] | None = None) -> None:
    """Train the chosen classifier on the default dataset and print its quality.

    Args:
        argv: Command-line arguments; ``sys.argv`` when omitted.
    """
    parser = argparse.ArgumentParser(description="Train a genre classifier.")
    parser.add_argument("--model", choices=sorted(CLASSIFIERS), default="svm")
    parser.add_argument("--data", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--epochs", type=int, help="epochs for neural networks")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    evaluation = train(args.model, args.data, args.model_dir, args.epochs)
    LOGGER.info("Validation accuracy: %.3f\n%s", evaluation.accuracy, evaluation.report)
    LOGGER.info("Saved model to %s", model_path(args.model, args.model_dir))


def _create(name: str, epochs: int | None) -> GenreClassifier:
    """Instantiate a classifier, overriding the epoch count when requested.

    Args:
        name: Classifier name, one of ``CLASSIFIERS``.
        epochs: Number of epochs for neural networks, or None for the default.

    Returns:
        The untrained classifier.
    """
    cls = CLASSIFIERS[name]
    if epochs is not None and issubclass(cls, TorchClassifier):
        return cls(replace(cls.default_config, epochs=epochs))
    return cls()


if __name__ == "__main__":
    main()
