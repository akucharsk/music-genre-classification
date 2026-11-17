# Copyright (C) 2026 Antoni Kucharski
"""Smoke test for the Streamlit app."""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, Self

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from music_genre_classification.classifiers import registry
from music_genre_classification.jamendo import Track, TrackPage, TrackQuery

if TYPE_CHECKING:
    from collections.abc import Iterable

APP_PATH = Path(__file__).parents[1] / "src" / "music_genre_classification" / "app.py"
AUDIO_URL = "https://prod-1.storage.jamendo.com/?trackid=1&format=mp32"
DOWNLOAD_URL = "https://prod-1.storage.jamendo.com/download/track/1/mp32/"


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


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register a trained model that always predicts jazz."""
    monkeypatch.setitem(registry.CLASSIFIERS, "fake", _FakeClassifier)
    monkeypatch.setitem(registry.MODEL_SUFFIXES, "fake", ".bin")
    monkeypatch.setattr(registry, "available_models", lambda *_args: ["fake"])


@pytest.mark.usefixtures("fake_model")
def test_analyze_classifies_an_uploaded_file() -> None:
    """Analyzing a valid upload shows the predicted genre and the segment count."""
    at = AppTest.from_file(str(APP_PATH)).run()
    at.file_uploader[0].set_value(("clip.wav", _wav_bytes(seconds=7), "audio/wav"))
    at.run()
    _clickable(at.button, "Analyze").click().run()

    assert not at.exception
    assert not at.error
    assert at.metric[0].label == "Predicted genre"
    assert at.metric[0].value == "Jazz"
    assert any("2 segments" in item.value for item in at.caption)


@pytest.mark.usefixtures("fake_model")
def test_analyze_reports_undecodable_uploads() -> None:
    """Analyzing a broken upload shows an error instead of crashing."""
    at = AppTest.from_file(str(APP_PATH)).run()
    at.file_uploader[0].set_value(("clip.wav", b"RIFF", "audio/wav"))
    at.run()
    _clickable(at.button, "Analyze").click().run()

    assert not at.exception
    assert at.error[0].value == "Could not read this audio file."


def test_analyze_warns_when_no_model_is_trained(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without trained models the app explains how to get one."""
    monkeypatch.setattr(registry, "available_models", lambda *_args: [])
    at = AppTest.from_file(str(APP_PATH)).run()
    at.file_uploader[0].set_value(("clip.wav", _wav_bytes(seconds=7), "audio/wav"))
    at.run()
    _clickable(at.button, "Analyze").click().run()

    assert not at.exception
    assert any("make train-svm" in item.value for item in at.sidebar.warning)
    assert any("make train-svm" in item.value for item in at.warning)


def test_jamendo_explains_a_missing_client_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a client id, the Jamendo tab explains how to set one."""
    monkeypatch.delenv("JAMENDO_CLIENT_ID", raising=False)
    at = AppTest.from_file(str(APP_PATH)).run()
    assert not at.exception
    assert any("JAMENDO_CLIENT_ID" in item.value for item in at.info)


def test_jamendo_search_plays_and_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    """A search shows the track, a download, and the next page."""
    _FakeJamendo.queries = []
    monkeypatch.setenv("JAMENDO_CLIENT_ID", "test-client")
    monkeypatch.setattr("music_genre_classification.jamendo.JamendoClient", _FakeJamendo)

    at = AppTest.from_file(str(APP_PATH)).run()
    _labeled(at.text_input, "Search").set_value("jazz")
    _clickable(at.button, "Search").click().run()

    assert not at.exception
    assert _FakeJamendo.queries[0].text.text == "jazz"
    assert any(item.value == "Page one" for item in at.text)
    assert len(at.download_button) == 1
    assert at.download_button[0].label == "Download"

    _clickable(at.button, "Next page").click().run()
    assert not at.exception
    assert _FakeJamendo.queries[-1].page.offset == _FakeJamendo.queries[0].page.limit
    assert any(item.value == "Page two" for item in at.text)


def _wav_bytes(*, seconds: int) -> bytes:
    rate = 22050
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(b"\x10\x20" * rate * seconds)
    return buffer.getvalue()


@pytest.mark.usefixtures("fake_model")
def test_jamendo_track_can_be_analyzed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Analyzing a Jamendo track shows the prediction next to the track's genre tags."""
    _FakeJamendo.queries = []
    monkeypatch.setenv("JAMENDO_CLIENT_ID", "test-client")
    monkeypatch.setattr("music_genre_classification.jamendo.JamendoClient", _FakeJamendo)

    at = AppTest.from_file(str(APP_PATH)).run()
    _labeled(at.text_input, "Search").set_value("jazz")
    _clickable(at.button, "Search").click().run()
    _clickable(at.button, "Analyze").click().run()

    assert not at.exception
    assert not at.error
    assert at.metric[0].value == "Jazz"
    assert any("Jamendo tags: jazz" in item.value for item in at.success)


class _FakeClassifier:
    """Classifier that predicts the second genre for every segment."""

    name: ClassVar[str] = "fake"
    genres: ClassVar[tuple[str, ...]] = ("blues", "jazz")

    @classmethod
    def load(cls, _path: Path) -> Self:
        """Return a new fake classifier.

        Returns:
            The classifier.
        """
        return cls()

    def predict_proba(  # ruff: ignore[no-self-use]
        self, features: np.ndarray
    ) -> np.ndarray:
        """Predict jazz with high probability for every segment.

        Returns:
            One row of probabilities per segment.
        """
        return np.tile([0.2, 0.8], (len(features), 1))


class _FakeJamendo:
    """Stand-in client that records queries and returns two pages."""

    queries: ClassVar[list[TrackQuery]] = []

    def __init__(self, _client_id: str) -> None:
        """Accept the client id the app passes in."""

    def __enter__(self) -> Self:
        """Return this fake client."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the fake client."""

    def download(self, _url: str) -> bytes:  # ruff: ignore[no-self-use]
        """Return a seven second recording.

        Returns:
            The bytes of a wav file.
        """
        return _wav_bytes(seconds=7)

    def search_tracks(self, query: TrackQuery) -> TrackPage:
        """Record the query and return a single fake track.

        Returns:
            One page containing the fake track.
        """
        type(self).queries.append(query)
        name = "Page one" if query.page.offset == 0 else "Page two"
        return TrackPage(
            tracks=(_sample_track(name),),
            total=30,
            offset=query.page.offset,
            limit=query.page.limit,
        )


def _sample_track(name: str) -> Track:
    return Track(
        id="1",
        name=name,
        artist_name="Artist",
        album_name="Album",
        duration=125,
        audio_url=AUDIO_URL,
        download_url=DOWNLOAD_URL,
        download_allowed=True,
        image_url="https://usercontent.jamendo.com?type=album&id=1",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        genres=("jazz",),
    )


class _Field(Protocol):
    """Text input fields the search test fills in."""

    label: str

    def set_value(self, value: str) -> object:
        """Set the field value."""


class _Runnable(Protocol):
    """A control that can rerun the Streamlit script."""

    def run(self, *, timeout: float | None = None) -> object:
        """Execute the script."""


class _Clickable(Protocol):
    """Buttons the search test presses."""

    label: str

    def click(self) -> _Runnable:
        """Press the button."""


def _labeled(items: Iterable[_Field], label: str) -> _Field:
    return next(item for item in items if item.label == label)


def _clickable(items: Iterable[_Clickable], label: str) -> _Clickable:
    return next(item for item in items if item.label == label)
