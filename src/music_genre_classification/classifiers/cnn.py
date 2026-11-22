# Copyright (C) 2026 Bartłomiej Kózka
"""Convolutional network on MFCC matrices treated as one-channel images."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

import torch
from torch import nn

from music_genre_classification.classifiers.torch_base import TorchClassifier, TrainingConfig

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

CHANNELS = (32, 64, 128)
CONV_DROPOUT = 0.25
HEAD_HIDDEN = 64
HEAD_DROPOUT = 0.3
NORMALIZATION_EPSILON = 1e-9


class CnnNetwork(nn.Module):
    """Three convolution blocks, global average pooling and a small classifier head."""

    def __init__(self, num_classes: int) -> None:
        """Build the layers.

        Args:
            num_classes: Number of genres to predict.
        """
        super().__init__()
        # Layer names match the weights of the original project so they load unchanged.
        self.conv1 = _conv_block(1, CHANNELS[0])
        self.conv2 = _conv_block(CHANNELS[0], CHANNELS[1])
        self.conv3 = _conv_block(CHANNELS[1], CHANNELS[2])
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(CHANNELS[-1], HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(HEAD_HIDDEN, num_classes),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        Args:
            x: Tensor of shape (batch, 1, frames, coefficients).

        Returns:
            Tensor of shape (batch, classes).
        """
        pooled = self.global_avg(self.conv3(self.conv2(self.conv1(x))))
        logits: torch.Tensor = self.fc(torch.flatten(pooled, 1))
        return logits


class CnnClassifier(TorchClassifier):
    """CNN genre classifier.

    Attributes:
        name: Short identifier used on the command line and in file names.
        default_config: Hyper-parameters of the original experiments.
    """

    name: ClassVar[str] = "cnn"
    default_config: ClassVar[TrainingConfig] = TrainingConfig(
        epochs=500, learning_rate=1e-4, weight_decay=1e-2
    )

    @override
    def build_network(self, num_classes: int, feature_dim: int) -> nn.Module:
        """Create the untrained network.

        Args:
            num_classes: Number of genres.
            feature_dim: Number of MFCC coefficients per frame; unused by this network.

        Returns:
            The CNN.
        """
        del feature_dim
        return CnnNetwork(num_classes)

    @override
    def prepare(self, features: NDArray[np.float64]) -> torch.Tensor:
        """Normalize every segment and add a channel dimension.

        Args:
            features: Array of shape (segments, frames, coefficients).

        Returns:
            Tensor of shape (segments, 1, frames, coefficients).
        """
        mean = features.mean(axis=(1, 2), keepdims=True)
        std = features.std(axis=(1, 2), keepdims=True)
        normalized = (features - mean) / (std + NORMALIZATION_EPSILON)
        return torch.as_tensor(normalized, dtype=torch.float32).unsqueeze(1)


def _conv_block(in_channels: int, out_channels: int) -> nn.Module:
    """Create a convolution, normalization, activation, pooling and dropout block.

    Args:
        in_channels: Channels of the block input.
        out_channels: Channels of the block output.

    Returns:
        The block.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(CONV_DROPOUT),
    )
