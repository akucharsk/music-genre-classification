# Copyright (C) 2026 Antoni Kucharski
"""Smoke test for the Streamlit app."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parents[1] / "src" / "music_genre_classification" / "app.py"


def test_app_runs_without_errors() -> None:
    """The app script executes without raising."""
    at = AppTest.from_file(str(APP_PATH)).run()
    assert not at.exception
