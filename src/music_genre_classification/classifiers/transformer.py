# Copyright (C) 2026 Bartłomiej Kózka
"""Transformer encoder over the MFCC frames of a segment."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np
import torch
from torch import nn

from music_genre_classification.classifiers.torch_base import TorchClassifier, TrainingConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 4
FEEDFORWARD_DIM = 256
DROPOUT = 0.4
HEAD_HIDDEN = 128
MAX_LENGTH = 5000
POSITION_BASE = 10000.0
NOISE_STD = 0.005
MAX_SHIFT = 5
FREQUENCY_MASK_DIVISOR = 5
TIME_MASK_DIVISOR = 10


class PositionalEncoding(nn.Module):
    """Adds fixed sine and cosine position information to a sequence.

    Attributes:
        table: Precomputed encodings of shape (1, max_length, d_model).
    """

    table: torch.Tensor

    def __init__(self, d_model: int, max_length: int = MAX_LENGTH) -> None:
        """Precompute the encoding table.

        Args:
            d_model: Size of the model's embeddings.
            max_length: Longest supported sequence.
        """
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(POSITION_BASE) / d_model)
        )
        table = torch.zeros(max_length, d_model)
        table[:, 0::2] = torch.sin(position * scale)
        table[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("table", table.unsqueeze(0), persistent=False)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the position encoding.

        Args:
            x: Tensor of shape (batch, length, d_model).

        Returns:
            Tensor of the same shape with positions added.
        """
        return x + self.table[:, : x.size(1), :]


class TransformerNetwork(nn.Module):
    """Linear projection, positional encoding, encoder layers, mean pooling and a head."""

    def __init__(self, num_classes: int, feature_dim: int) -> None:
        """Build the layers.

        Args:
            num_classes: Number of genres to predict.
            feature_dim: Number of MFCC coefficients per frame.
        """
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, D_MODEL)
        self.positional_encoding = PositionalEncoding(D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            dim_feedforward=FEEDFORWARD_DIM,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS)
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HEAD_HIDDEN, num_classes),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        Args:
            x: Tensor of shape (batch, frames, coefficients).

        Returns:
            Tensor of shape (batch, classes).
        """
        encoded = self.encoder(self.positional_encoding(self.input_projection(x)))
        logits: torch.Tensor = self.head(encoded.mean(dim=1))
        return logits


class TransformerClassifier(TorchClassifier):
    """Transformer genre classifier with noise, shift and masking augmentation.

    Attributes:
        name: Short identifier used on the command line and in file names.
        default_config: Hyper-parameters of the original experiments.
    """

    name: ClassVar[str] = "transformer"
    default_config: ClassVar[TrainingConfig] = TrainingConfig(
        epochs=150, learning_rate=5e-4, weight_decay=1e-3, lr_patience=5
    )

    @override
    def build_network(self, num_classes: int, feature_dim: int) -> nn.Module:
        """Create the untrained network.

        Args:
            num_classes: Number of genres.
            feature_dim: Number of MFCC coefficients per frame.

        Returns:
            The Transformer.
        """
        return TransformerNetwork(num_classes, feature_dim)

    @override
    def prepare(self, features: NDArray[np.float64]) -> torch.Tensor:
        """Convert segments to a float tensor of shape (segments, frames, coefficients).

        Args:
            features: Array of shape (segments, frames, coefficients).

        Returns:
            The float tensor.
        """
        return torch.as_tensor(features, dtype=torch.float32)

    @override
    def augment(
        self, features: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Add noise, roll in time and mask a random time-frequency block of every segment.

        Args:
            features: Array of shape (segments, frames, coefficients).
            rng: Random generator to draw from.

        Returns:
            Augmented copy of the segments.
        """
        return np.stack([_augment_segment(segment, rng) for segment in features])


def _augment_segment(segment: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
    """Augment one MFCC matrix.

    Args:
        segment: Array of shape (frames, coefficients).
        rng: Random generator to draw from.

    Returns:
        Augmented copy of the matrix.
    """
    frames, coefficients = segment.shape
    noisy = segment + rng.normal(0.0, NOISE_STD, segment.shape)
    shifted = np.roll(noisy, int(rng.integers(-MAX_SHIFT, MAX_SHIFT)), axis=0)
    frequency_width = int(rng.integers(0, max(coefficients // FREQUENCY_MASK_DIVISOR, 1)))
    time_width = int(rng.integers(0, max(frames // TIME_MASK_DIVISOR, 1)))
    frequency_start = int(rng.integers(0, coefficients - frequency_width + 1))
    time_start = int(rng.integers(0, frames - time_width + 1))
    shifted[
        time_start : time_start + time_width, frequency_start : frequency_start + frequency_width
    ] = 0
    return shifted
