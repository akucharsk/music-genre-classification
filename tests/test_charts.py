# Copyright (C) 2026 Bartłomiej Kózka
"""Tests for the prediction charts."""

from __future__ import annotations

import numpy as np
import pytest

from music_genre_classification.charts import (
    CHARTS,
    HeatmapChart,
    PredictionChart,
    SharesChart,
    TimelineChart,
    TimeResolved,
)
from music_genre_classification.prediction import Prediction

GENRES = ("a", "b", "c")


def _prediction(segments: int = 8) -> Prediction:
    rng = np.random.default_rng(0)
    raw = rng.random((segments, len(GENRES)))
    return Prediction(GENRES, raw / raw.sum(axis=1, keepdims=True))


@pytest.mark.parametrize("chart", CHARTS, ids=lambda chart: type(chart).__name__)
def test_every_chart_builds_a_chart_with_a_title(chart: PredictionChart) -> None:
    """All registered charts satisfy the protocol and produce a renderable chart."""
    spec = chart.build(_prediction()).to_dict()

    assert chart.title
    assert "mark" in spec


def test_only_the_heatmap_has_a_time_step() -> None:
    """The runtime protocol selects exactly the charts that support a time step."""
    resolved = [type(chart).__name__ for chart in CHARTS if isinstance(chart, TimeResolved)]

    assert resolved == ["HeatmapChart"]
    assert not isinstance(SharesChart(), TimeResolved)
    assert not isinstance(TimelineChart(), TimeResolved)


def test_heatmap_with_step_is_a_modified_copy() -> None:
    """Changing the step does not mutate the original chart."""
    original = HeatmapChart()
    wider = original.with_step(30.0)

    assert wider.step_seconds == pytest.approx(30.0)
    assert original.step_seconds == pytest.approx(3.0)


def test_heatmap_uses_fewer_cells_with_a_longer_step() -> None:
    """A longer time step averages segments into fewer heatmap columns."""
    prediction = _prediction(segments=20)

    fine = HeatmapChart().build(prediction).to_dict()
    coarse = HeatmapChart().with_step(30.0).build(prediction).to_dict()

    def columns(spec: dict[str, object]) -> int:
        data = spec["datasets"]
        assert isinstance(data, dict)
        rows = next(iter(data.values()))
        return len({row["Time (s)"] for row in rows})

    assert columns(fine) == 20
    assert columns(coarse) == 2
