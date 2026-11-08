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
    assert at.title[0].value == "Music Genre Classification"
    assert len(at.file_uploader) == 1


def test_app_accepts_an_audio_file() -> None:
    """Uploading an audio file shows the file name."""
    at = AppTest.from_file(str(APP_PATH)).run()
    at.file_uploader[0].set_value(("clip.wav", b"RIFF", "audio/wav"))
    at.run()

    assert not at.exception
    assert at.caption[0].value == "clip.wav"
