# Copyright (C) 2026 Bartłomiej Kózka
"""Altair charts of a genre prediction, interchangeable through the ``PredictionChart`` protocol."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

import altair as alt
import pandas as pd

from music_genre_classification.prediction import SEGMENT_SECONDS

if TYPE_CHECKING:
    from music_genre_classification.prediction import Prediction

GENRE_COLUMN = "Genre"
SHARE_COLUMN = "Share of segments (%)"
TIME_COLUMN = "Time (s)"
PROBABILITY_COLUMN = "Probability (%)"
PERCENT = 100
CHART_HEIGHT = 300
INACTIVE_OPACITY = 0.15
TIME_STEPS = (SEGMENT_SECONDS, 10.0, 30.0, 60.0)


class PredictionChart(Protocol):
    """Anything that can draw one chart of a prediction.

    The protocol is structural: a class qualifies by having these members, without inheriting
    from anything. The page renders every chart in the same way, so a new chart only has to
    provide them.
    """

    @property
    def title(self) -> str:
        """Heading shown above the chart."""
        ...

    @property
    def caption(self) -> str:
        """One-line hint shown under the heading, empty for none."""
        ...

    def build(self, prediction: Prediction) -> alt.Chart:
        """Create the chart.

        Args:
            prediction: Per-segment probabilities of the track.

        Returns:
            The Altair chart.
        """
        ...


@runtime_checkable
class TimeResolved(PredictionChart, Protocol):
    """A chart whose time step the viewer may change.

    Being a ``runtime_checkable`` protocol, it can be tested with ``isinstance``: the page offers
    a time-step selector to every chart that has these members, and only to those.
    """

    @property
    def step_seconds(self) -> float:
        """Width of one time step in seconds."""
        ...

    def with_step(self, seconds: float) -> Self:
        """Copy the chart with another time step.

        Args:
            seconds: Width of one time step.

        Returns:
            The chart using that step.
        """
        ...


@dataclass(frozen=True, slots=True)
class SharesChart:
    """Bars showing the share of segments each genre wins.

    Attributes:
        title: Heading shown above the chart.
        caption: One-line hint shown under the heading.
    """

    title: str = "Share of segments per genre"
    caption: str = ""

    def build(self, prediction: Prediction) -> alt.Chart:  # ruff: ignore[no-self-use]
        """Create the bar chart, largest share first.

        Args:
            prediction: Per-segment probabilities of the track.

        Returns:
            The bar chart.
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


@dataclass(frozen=True, slots=True)
class TimelineChart:
    """Lines showing each genre's probability over time.

    Attributes:
        title: Heading shown above the chart.
        caption: One-line hint shown under the heading.
    """

    title: str = "Probability over time"
    caption: str = "Click a genre in the legend to highlight it."

    def build(self, prediction: Prediction) -> alt.Chart:  # ruff: ignore[no-self-use]
        """Create the line chart.

        The legend is bound to a selection parameter, so highlighting a genre runs in the
        browser and does not call Python again.

        Args:
            prediction: Per-segment probabilities of the track.

        Returns:
            The line chart.
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
                opacity=alt
                .when(selection)
                .then(alt.value(1.0))
                .otherwise(alt.value(INACTIVE_OPACITY)),
                tooltip=[GENRE_COLUMN, TIME_COLUMN, PROBABILITY_COLUMN],
            )
            .add_params(selection)
            .properties(height=CHART_HEIGHT)
        )
        return chart


@dataclass(frozen=True, slots=True)
class HeatmapChart:
    """Heatmap with genres as rows and time steps as columns.

    Attributes:
        title: Heading shown above the chart.
        caption: One-line hint shown under the heading.
        step_seconds: Width of one column; probabilities are averaged within it.
    """

    title: str = "Genre activation over time"
    caption: str = "Probability of each genre, averaged over the chosen time step."
    step_seconds: float = SEGMENT_SECONDS

    def with_step(self, seconds: float) -> Self:
        """Copy the chart with another time step.

        Args:
            seconds: Width of one column in seconds.

        Returns:
            The chart using that step.
        """
        return replace(self, step_seconds=seconds)

    def build(self, prediction: Prediction) -> alt.Chart:
        """Create the heatmap.

        Args:
            prediction: Per-segment probabilities of the track.

        Returns:
            The heatmap coloured by probability.
        """
        chart: alt.Chart = (
            alt
            .Chart(pd.DataFrame(_long_rows(prediction.binned(self.step_seconds))))
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


CHARTS: tuple[PredictionChart, ...] = (SharesChart(), TimelineChart(), HeatmapChart())


def _long_rows(prediction: Prediction) -> list[dict[str, object]]:
    """Flatten the probabilities to one row per time step and genre.

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
