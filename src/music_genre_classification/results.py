# Copyright (C) 2026 Bartłomiej Kózka
"""Streamlit display of a track's genre prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from music_genre_classification.prediction import Prediction

SHARE_COLUMN = "Share of segments (%)"
GENRE_COLUMN = "Genre"
TIME_COLUMN = "Time (s)"
PROBABILITY_COLUMN = "Probability (%)"
PERCENT = 100
TOP_COUNT = 3
CHART_HEIGHT = 300
MFCC_ROW_HEIGHT = 12
MFCC_MAX_COLUMNS = 600
MFCC_CLIP = 2.0
STD_EPSILON = 1e-9


def render_prediction(prediction: Prediction, tags: tuple[str, ...] = ()) -> None:
    """Show the predicted genre, genre shares, and probabilities over time.

    Args:
        prediction: Per-segment probabilities of the track.
        tags: Genre tags the track's source gives for it, compared with the prediction.
    """
    _render_summary(prediction, tags)
    st.subheader("Share of segments per genre")
    _chart(_shares_chart(prediction))
    st.subheader("Probability over time")
    st.caption("Click a genre in the legend to highlight it.")
    _chart(_timeline_chart(prediction))
    st.subheader("Genre activation over time")
    _chart(_heatmap_chart(prediction))


def _chart(chart: alt.Chart | alt.LayerChart) -> None:
    """Show an Altair chart at the full container width.

    Args:
        chart: Chart to show.
    """
    st.altair_chart(chart, width="stretch")


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


def _shares_chart(prediction: Prediction) -> alt.Chart:
    """Build a bar chart of the share of segments each genre wins.

    Args:
        prediction: Per-segment probabilities of the track.

    Returns:
        Bar chart with the largest share first.
    """
    rows = [
        {GENRE_COLUMN: genre, SHARE_COLUMN: round(float(share) * PERCENT, 1)}
        for genre, share in zip(prediction.genres, prediction.shares, strict=True)
    ]
    chart: alt.Chart = (
        alt
        .Chart(pd.DataFrame(rows))
        .mark_bar()
        .encode(
            x=alt.X(f"{GENRE_COLUMN}:N", sort="-y"),
            y=alt.Y(f"{SHARE_COLUMN}:Q"),
            tooltip=[GENRE_COLUMN, SHARE_COLUMN],
        )
        .properties(height=CHART_HEIGHT)
    )
    return chart


def _timeline_chart(prediction: Prediction) -> alt.Chart:
    """Build a line chart of each genre's probability over time.

    Args:
        prediction: Per-segment probabilities of the track.

    Returns:
        Line chart whose legend highlights a genre when clicked.
    """
    selection = alt.selection_point(fields=[GENRE_COLUMN], bind="legend")
    chart: alt.Chart = (
        alt
        .Chart(pd.DataFrame(_long_rows(prediction)))
        .mark_line()
        .encode(
            x=alt.X(f"{TIME_COLUMN}:Q"),
            y=alt.Y(f"{PROBABILITY_COLUMN}:Q"),
            color=f"{GENRE_COLUMN}:N",
            opacity=alt.when(selection).then(alt.value(1.0)).otherwise(alt.value(0.15)),
            tooltip=[GENRE_COLUMN, TIME_COLUMN, PROBABILITY_COLUMN],
        )
        .add_params(selection)
        .properties(height=CHART_HEIGHT)
    )
    return chart


def _heatmap_chart(prediction: Prediction) -> alt.Chart:
    """Build a heatmap with genres as rows and time as columns.

    Args:
        prediction: Per-segment probabilities of the track.

    Returns:
        Heatmap coloured by probability.
    """
    chart: alt.Chart = (
        alt
        .Chart(pd.DataFrame(_long_rows(prediction)))
        .mark_rect()
        .encode(
            x=alt.X(f"{TIME_COLUMN}:O"),
            y=alt.Y(f"{GENRE_COLUMN}:N"),
            color=alt.Color(f"{PROBABILITY_COLUMN}:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=[GENRE_COLUMN, TIME_COLUMN, PROBABILITY_COLUMN],
        )
        .properties(height=CHART_HEIGHT)
    )
    return chart


def _long_rows(prediction: Prediction) -> list[dict[str, object]]:
    """Flatten the probabilities to one row per segment and genre.

    Args:
        prediction: Per-segment probabilities of the track.

    Returns:
        Rows with the genre, the start time and the probability in percent.
    """
    return [
        {
            GENRE_COLUMN: genre,
            TIME_COLUMN: round(float(seconds), 1),
            PROBABILITY_COLUMN: round(float(probability) * PERCENT, 1),
        }
        for seconds, row in zip(prediction.times, prediction.probabilities, strict=True)
        for genre, probability in zip(prediction.genres, row, strict=True)
    ]


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
