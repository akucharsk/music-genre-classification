# Copyright (C) 2026 Bartłomiej Kózka
"""Shared training and prediction code for the PyTorch classifiers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from music_genre_classification.dataset import DEFAULT_FEATURE_DIM, GTZAN_GENRES, RANDOM_SEED

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from music_genre_classification.dataset import Dataset, Split

LOGGER = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREDICT_BATCH_SIZE = 256
LR_REDUCTION_FACTOR = 0.5


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyper-parameters of the training loop.

    Attributes:
        epochs: Number of passes over the training data.
        batch_size: Segments per optimization step.
        learning_rate: Initial Adam learning rate.
        weight_decay: L2 penalty of the optimizer.
        lr_patience: Epochs without validation improvement before the learning rate is halved,
            or None to keep it constant.
    """

    epochs: int
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lr_patience: int | None = None


@dataclass(slots=True)
class History:
    """Loss and accuracy after every epoch.

    Attributes:
        train_loss: Mean training loss per epoch.
        val_loss: Mean validation loss per epoch.
        train_acc: Training accuracy per epoch.
        val_acc: Validation accuracy per epoch.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


class TorchClassifier(ABC):
    """Base class of the neural-network classifiers.

    Subclasses define the network, how raw MFCC arrays become model input, and the default
    hyper-parameters; this class provides the training loop, prediction and persistence. After
    ``fit``, the ``history`` instance attribute holds the per-epoch metrics.

    Attributes:
        name: Short identifier used on the command line and in file names.
        default_config: Hyper-parameters used when none are given.
    """

    name: ClassVar[str]
    default_config: ClassVar[TrainingConfig]

    def __init__(self, config: TrainingConfig | None = None) -> None:
        """Create an untrained classifier.

        Args:
            config: Training hyper-parameters; the class default when omitted.
        """
        self.config = self.default_config if config is None else config
        self.history = History()
        self._network: nn.Module | None = None
        self._genres: tuple[str, ...] = ()
        self._feature_dim = 0

    @property
    def genres(self) -> tuple[str, ...]:
        """Genre names; the columns of ``predict_proba`` follow this order."""
        return self._genres

    @abstractmethod
    def build_network(self, num_classes: int, feature_dim: int) -> nn.Module:
        """Create the untrained network.

        Args:
            num_classes: Number of genres.
            feature_dim: Number of MFCC coefficients per frame.

        Returns:
            The network, mapping prepared input to class logits.
        """

    @abstractmethod
    def prepare(self, features: NDArray[np.float64]) -> torch.Tensor:
        """Convert raw MFCC segments into network input.

        Args:
            features: Array of shape (segments, frames, coefficients).

        Returns:
            Float tensor in the shape the network expects.
        """

    def augment(  # ruff: ignore[no-self-use]
        self,
        features: NDArray[np.float64],
        rng: np.random.Generator,  # ruff: ignore[unused-method-argument] - subclasses may use it
    ) -> NDArray[np.float64]:
        """Create a randomly modified copy of the training segments.

        Args:
            features: Array of shape (segments, frames, coefficients).
            rng: Random generator to draw from.

        Returns:
            The segments to train on; unchanged unless a subclass overrides this.
        """
        return features

    def fit(self, split: Split) -> None:
        """Train the network, recording metrics in ``history``.

        Args:
            split: Train and validation datasets.
        """
        torch.manual_seed(RANDOM_SEED)
        rng = np.random.default_rng(RANDOM_SEED)
        self._genres = split.train.genres
        self._feature_dim = split.train.features.shape[2]
        network = self.build_network(len(self._genres), self._feature_dim).to(DEVICE)
        self._network = network
        train_x = self.prepare(self.augment(split.train.features, rng))
        val_x = self.prepare(split.validation.features)
        train_loader = self._loader(train_x, split.train, shuffle=True)
        val_loader = self._loader(val_x, split.validation, shuffle=False)
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = self._scheduler(optimizer)
        self.history = History()
        for epoch in range(self.config.epochs):
            self._run_epoch(network, train_loader, val_loader, optimizer, scheduler)
            LOGGER.info(
                "Epoch %d/%d train loss %.4f acc %.4f | val loss %.4f acc %.4f",
                epoch + 1,
                self.config.epochs,
                self.history.train_loss[-1],
                self.history.train_acc[-1],
                self.history.val_loss[-1],
                self.history.val_acc[-1],
            )

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Estimate genre probabilities for MFCC segments.

        Args:
            features: Array of shape (segments, frames, coefficients).

        Returns:
            Array of shape (segments, genres) whose rows sum to one.

        Raises:
            RuntimeError: If the model has not been trained or loaded.
        """
        if self._network is None:
            msg = "The classifier is not trained."
            raise RuntimeError(msg)
        inputs = self.prepare(features)
        self._network.eval()
        batches = []
        with torch.no_grad():
            for start in range(0, len(inputs), PREDICT_BATCH_SIZE):
                logits = self._network(inputs[start : start + PREDICT_BATCH_SIZE].to(DEVICE))
                batches.append(functional.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(batches).astype(np.float64)

    def save(self, path: Path) -> None:
        """Write the weights, genre names and input size to disk.

        Args:
            path: Destination file; parent directories are created.

        Raises:
            RuntimeError: If the model has not been trained or loaded.
        """
        if self._network is None:
            msg = "The classifier is not trained."
            raise RuntimeError(msg)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self._network.state_dict(),
            "genres": list(self._genres),
            "feature_dim": self._feature_dim,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Restore a classifier from a file written by ``save`` or by the original project.

        The original project stored a bare ``state_dict``; such weights are assumed to be for the
        ten GTZAN genres and 13 MFCC coefficients.

        Args:
            path: File to load.

        Returns:
            The restored classifier, ready to predict.
        """
        payload = torch.load(path, map_location=DEVICE, weights_only=True)
        if "state_dict" in payload:
            weights = payload["state_dict"]
            genres = tuple(payload["genres"])
            feature_dim = int(payload["feature_dim"])
        else:
            weights, genres, feature_dim = payload, GTZAN_GENRES, DEFAULT_FEATURE_DIM
        classifier = cls()
        classifier._genres = genres
        classifier._feature_dim = feature_dim
        network = classifier.build_network(len(genres), feature_dim)
        network.load_state_dict(weights)
        classifier._network = network.to(DEVICE)
        return classifier

    def _loader(
        self, inputs: torch.Tensor, dataset: Dataset, *, shuffle: bool
    ) -> DataLoader[tuple[torch.Tensor, ...]]:
        """Wrap prepared inputs and labels in a batching data loader.

        Args:
            inputs: Prepared network input.
            dataset: Dataset providing the labels.
            shuffle: Whether to reshuffle every epoch.

        Returns:
            The data loader.
        """
        labels = torch.as_tensor(dataset.labels, dtype=torch.long)
        return DataLoader(
            TensorDataset(inputs, labels), batch_size=self.config.batch_size, shuffle=shuffle
        )

    def _scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
        """Create the learning-rate scheduler when the config asks for one.

        Args:
            optimizer: Optimizer whose learning rate is adjusted.

        Returns:
            The scheduler, or None when ``lr_patience`` is not set.
        """
        if self.config.lr_patience is None:
            return None
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=LR_REDUCTION_FACTOR, patience=self.config.lr_patience
        )

    def _run_epoch(
        self,
        network: nn.Module,
        train_loader: DataLoader[tuple[torch.Tensor, ...]],
        val_loader: DataLoader[tuple[torch.Tensor, ...]],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    ) -> None:
        """Run one training pass and one validation pass, recording the metrics.

        Args:
            network: Network being trained.
            train_loader: Batches of training data.
            val_loader: Batches of validation data.
            optimizer: Optimizer updating the network.
            scheduler: Learning-rate scheduler, or None.
        """
        train_loss, train_acc = _epoch(network, train_loader, optimizer)
        val_loss, val_acc = _epoch(network, val_loader, None)
        if scheduler is not None:
            scheduler.step(val_loss)
        self.history.train_loss.append(train_loss)
        self.history.train_acc.append(train_acc)
        self.history.val_loss.append(val_loss)
        self.history.val_acc.append(val_acc)


def _epoch(
    network: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """Pass over a data loader, training when an optimizer is given.

    Args:
        network: Network to run.
        loader: Batches of data.
        optimizer: Optimizer to step, or None to only evaluate.

    Returns:
        Mean loss and accuracy over the data.
    """
    training = optimizer is not None
    network.train(mode=training)
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)  # ruff: ignore[redefined-loop-name]
            logits = network(inputs)
            loss = functional.cross_entropy(logits, targets)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            total_loss += loss.item() * len(targets)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            count += len(targets)
    return total_loss / count, correct / count
