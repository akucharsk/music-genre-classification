# Copyright (C) 2026 Bartłomiej Kózka
"""Tests for the prediction display helpers."""

from __future__ import annotations

import numpy as np

from music_genre_classification.results import MFCC_MAX_COLUMNS, MFCC_ROW_HEIGHT, mfcc_image

COEFFICIENTS = 13
FRAMES = 130


def _segments(count: int) -> list[list[list[float]]]:
    rng = np.random.default_rng(0)
    scale = np.linspace(1.0, 300.0, COEFFICIENTS)
    return [(rng.normal(size=(FRAMES, COEFFICIENTS)) * scale).tolist() for _ in range(count)]


def test_mfcc_image_scales_every_coefficient_on_its_own() -> None:
    """A coefficient with huge values does not flatten the others."""
    image = mfcc_image(_segments(2))

    assert image.min() >= 0.0
    assert image.max() <= 1.0
    row_spreads = image.reshape(COEFFICIENTS, MFCC_ROW_HEIGHT, -1)[:, 0, :].std(axis=1)
    assert row_spreads.min() > 0.1


def test_mfcc_image_keeps_short_tracks_unchanged_in_width() -> None:
    """Tracks that fit are not pooled."""
    image = mfcc_image(_segments(2))

    assert image.shape == (COEFFICIENTS * MFCC_ROW_HEIGHT, 2 * FRAMES)


def test_mfcc_image_limits_the_width_of_long_tracks() -> None:
    """Long tracks are shortened so the image is not a thin strip."""
    image = mfcc_image(_segments(60))

    assert image.shape[0] == COEFFICIENTS * MFCC_ROW_HEIGHT
    assert image.shape[1] <= MFCC_MAX_COLUMNS
    assert image.shape[1] > MFCC_MAX_COLUMNS // 2
