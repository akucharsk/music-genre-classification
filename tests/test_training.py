# Copyright (C) 2026 Bartłomiej Kózka
"""Tests for the classifiers, evaluation and the training command."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from music_genre_classification.classifiers.cnn import CnnClassifier
from music_genre_classification.classifiers.registry import CLASSIFIERS, model_path
from music_genre_classification.classifiers.svm import SvmClassifier
from music_genre_classification.classifiers.torch_base import TorchClassifier
from music_genre_classification.classifiers.transformer import TransformerClassifier
from music_genre_classification.dataset import GTZAN_GENRES, split_dataset
from music_genre_classification.evaluation import evaluate
from music_genre_classification.training import main, train
from tests.test_dataset import synthetic_dataset, write_json

if TYPE_CHECKING:
    from pathlib import Path

    from music_genre_classification.classifiers.base import GenreClassifier

EPOCHS = 2


def _classifier(name: str) -> GenreClassifier:
    cls = CLASSIFIERS[name]
    if issubclass(cls, TorchClassifier):
        return cls(replace(cls.default_config, epochs=EPOCHS))
    return cls()


def test_registry_lists_all_classifiers() -> None:
    """The registry exposes the three models under their names."""
    assert set(CLASSIFIERS) == {"svm", "cnn", "transformer"}
    assert CLASSIFIERS["svm"] is SvmClassifier
    assert CLASSIFIERS["cnn"] is CnnClassifier
    assert CLASSIFIERS["transformer"] is TransformerClassifier


@pytest.mark.parametrize("name", ["svm", "cnn", "transformer"])
def test_classifier_predicts_probabilities_and_survives_saving(name: str, tmp_path: Path) -> None:
    """Trained models output valid probabilities and give the same answers after reloading."""
    split = split_dataset(synthetic_dataset())
    classifier = _classifier(name)
    classifier.fit(split)

    probabilities = classifier.predict_proba(split.validation.features)
    path = tmp_path / "nested" / f"{name}.bin"
    classifier.save(path)
    restored = type(classifier).load(path)

    assert probabilities.shape == (len(split.validation.labels), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(restored.predict_proba(split.validation.features), probabilities, atol=1e-5)


def test_svm_separates_the_genres() -> None:
    """The SVM reaches high accuracy on well-separated genres."""
    split = split_dataset(synthetic_dataset())
    classifier = SvmClassifier()
    classifier.fit(split)

    assert evaluate(classifier, split.validation).accuracy > 0.9


@pytest.mark.parametrize("cls", [CnnClassifier, TransformerClassifier])
def test_neural_networks_record_history(cls: type[TorchClassifier]) -> None:
    """Training records one entry per epoch for loss and accuracy."""
    classifier = cls(replace(cls.default_config, epochs=EPOCHS))
    classifier.fit(split_dataset(synthetic_dataset()))

    assert len(classifier.history.train_loss) == EPOCHS
    assert len(classifier.history.val_acc) == EPOCHS
    assert all(0.0 <= value <= 1.0 for value in classifier.history.val_acc)


@pytest.mark.parametrize("cls", [SvmClassifier, CnnClassifier])
def test_untrained_classifier_cannot_predict_or_save(
    cls: type[GenreClassifier], tmp_path: Path
) -> None:
    """Using a classifier before fitting raises a clear error."""
    classifier = cls()
    features = synthetic_dataset().features

    with pytest.raises(RuntimeError, match="not trained"):
        classifier.predict_proba(features)
    if cls is CnnClassifier:
        with pytest.raises(RuntimeError, match="not trained"):
            classifier.save(tmp_path / "model.pt")


def test_evaluate_reports_accuracy_and_confusion() -> None:
    """Evaluation has an accuracy, a per-genre report and a square confusion matrix."""
    split = split_dataset(synthetic_dataset())
    classifier = SvmClassifier()
    classifier.fit(split)

    result = evaluate(classifier, split.validation)

    assert 0.0 <= result.accuracy <= 1.0
    assert "precision" in result.report
    assert result.confusion.shape == (2, 2)
    assert result.confusion.sum() == len(split.validation.labels)


def test_train_saves_the_model(tmp_path: Path) -> None:
    """Training from a data file evaluates the model and writes it to the model directory."""
    data = tmp_path / "data.json"
    write_json(data, synthetic_dataset())

    result = train("svm", data, tmp_path / "models")

    assert result.accuracy > 0.9
    assert model_path("svm", tmp_path / "models").exists()


def test_main_trains_a_neural_network_with_epoch_override(tmp_path: Path) -> None:
    """The command line trains the chosen model and saves it."""
    data = tmp_path / "data.json"
    write_json(data, synthetic_dataset())
    models = tmp_path / "models"

    main([
        "--model",
        "transformer",
        "--data",
        str(data),
        "--model-dir",
        str(models),
        "--epochs",
        "1",
    ])

    assert model_path("transformer", models).exists()


@pytest.mark.parametrize(
    ("cls", "layer_name"),
    [(CnnClassifier, "conv1.0.weight"), (TransformerClassifier, "input_proj.weight")],
)
def test_weights_of_the_original_project_load_unchanged(
    cls: type[TorchClassifier], layer_name: str, tmp_path: Path
) -> None:
    """A bare state_dict with the original layer names is accepted for the GTZAN genres."""
    network = cls().build_network(len(GTZAN_GENRES), 13)
    state = network.state_dict()
    assert layer_name in state
    path = tmp_path / "original.pth"
    torch.save(state, path)

    restored = cls.load(path)

    assert restored.genres == GTZAN_GENRES
    features = np.random.default_rng(0).normal(size=(3, 130, 13))
    assert restored.predict_proba(features).shape == (3, len(GTZAN_GENRES))
