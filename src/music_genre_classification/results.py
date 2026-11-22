# Copyright (C) 2026 Bartłomiej Kózka
"""Streamlit display of a track's genre prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import streamlit as st

from music_genre_classification.charts import CHARTS, TIME_STEPS, TimeResolved
from music_genre_classification.prediction import SEGMENT_SECONDS

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from music_genre_classification.charts import PredictionChart
    from music_genre_classification.prediction import Prediction

TOP_COUNT = 3
MFCC_ROW_HEIGHT = 12
MFCC_MAX_COLUMNS = 600
MFCC_CLIP = 2.0
STD_EPSILON = 1e-9


def render_prediction(prediction: Prediction, tags: tuple[str, ...] = (), key: str = "") -> None:
    """Show the predicted genre followed by every chart.

    Args:
        prediction: Per-segment probabilities of the track.
        tags: Genre tags the track's source gives for it, compared with the prediction.
        key: Suffix that keeps widget keys unique when several predictions are on the page.
    """
    _render_summary(prediction, tags)
    for index, chart in enumerate(CHARTS):
        _render_chart(chart, prediction, f"{key}-chart-{index}")


@st.fragment
def _render_chart(chart: PredictionChart, prediction: Prediction, key: str) -> None:
    """Show one chart with its heading and, when it supports it, a time-step selector.

    This is a fragment: changing the time step reruns only this function, so the other charts
    and the analysis stay on the page.

    Args:
        chart: Chart to show.
        prediction: Per-segment probabilities of the track.
        key: Unique key of this chart's widgets.
    """
    st.subheader(chart.title)
    if chart.caption:
        st.caption(chart.caption)
    if isinstance(chart, TimeResolved):
        chart = chart.with_step(_pick_step(prediction, key))
    st.altair_chart(chart.build(prediction), width="stretch")


def _pick_step(prediction: Prediction, key: str) -> float:
    """Let the viewer choose how many seconds one chart column covers.

    Args:
        prediction: Per-segment probabilities of the track.
        key: Unique key of the selector.

    Returns:
        The chosen time step in seconds.
    """
    steps = [step for step in TIME_STEPS if step <= max(prediction.duration, SEGMENT_SECONDS)]
    return float(
        st.selectbox(
            "Time step",
            steps,
            format_func=lambda step: "Every segment" if step == SEGMENT_SECONDS else f"{step:g} s",
            key=f"{key}-step",
        )
    )


def _render_summary(prediction: Prediction, tags: tuple[str, ...]) -> None:
    """Show the most likely genres and compare the best one with the source's tags.

    Args:
        prediction: Per-segment probabilities of the track.
        tags: Genre tags the track's source gives for it.
    """
    top = prediction.top(TOP_COUNT)
    genre, share = top[0]
    st.metric("Predicted genre", genre.capitalize(), f"{share:.0%} of segments", delta_color="off")
    st.caption("Runner-up: " + ", ".join(f"{name} ({value:.0%})" for name, value in top[1:]))
    if not tags:
        return
    listed = ", ".join(tags)
    if genre in {tag.lower() for tag in tags}:
        st.success(f"Jamendo tags: {listed}. The prediction matches.")
    else:
        st.info(f"Jamendo tags: {listed}. The prediction differs.")


def mfcc_image(segments: list[list[list[float]]]) -> NDArray[np.float64]:
    """Join segment MFCCs along time and turn them into a grayscale image in [0, 1].

    Every coefficient is standardized on its own, because the first coefficient is an order of
    magnitude larger than the rest and would otherwise hide them. Long tracks are shortened by
    averaging neighbouring frames so the image keeps a readable aspect ratio.

    Args:
        segments: One MFCC matrix (frames by coefficients) per audio segment, non-empty.

    Returns:
        Array with ``MFCC_ROW_HEIGHT`` rows per coefficient and at most ``MFCC_MAX_COLUMNS``
        columns.
    """
    matrix = np.concatenate([np.asarray(segment) for segment in segments]).T
    standardized = (matrix - matrix.mean(axis=1, keepdims=True)) / (
        matrix.std(axis=1, keepdims=True) + STD_EPSILON
    )
    scaled = (np.clip(standardized, -MFCC_CLIP, MFCC_CLIP) + MFCC_CLIP) / (2 * MFCC_CLIP)
    factor = -(-scaled.shape[1] // MFCC_MAX_COLUMNS)
    usable = scaled.shape[1] - scaled.shape[1] % factor
    pooled = scaled[:, :usable].reshape(scaled.shape[0], -1, factor).mean(axis=2)
    return np.repeat(pooled, MFCC_ROW_HEIGHT, axis=0)
