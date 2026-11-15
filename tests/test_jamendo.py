# Copyright (C) 2026 Antoni Kucharski
"""Tests for the Jamendo client."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self, cast

import httpx
import pytest

from music_genre_classification.jamendo import (
    JamendoClient,
    JamendoError,
    PageRequest,
    RangeFilters,
    SoundFilters,
    Speed,
    TextFilters,
    Track,
    TrackPage,
    TrackQuery,
    audio_mime,
    configured_client_id,
    download_audio,
    download_filename,
    is_jamendo_media_url,
)

AUDIO_URL = "https://prod-1.storage.jamendo.com/?trackid=1848357&format=mp32"
DOWNLOAD_URL = "https://prod-1.storage.jamendo.com/download/track/1848357/mp32/"
IMAGE_URL = "https://usercontent.jamendo.com?type=album&id=368084"
Handler = Callable[[httpx.Request], httpx.Response]


def test_search_sends_filters_and_parses_tracks() -> None:
    """Filtered searches encode Jamendo parameters and keep usable tracks."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_payload([_track_json(), "skip", {"id": "2"}], total=41))

    page = _search(_filtered_query(), handler)

    params = dict(requests[0].url.params)
    assert params["search"] == "sunrise"
    assert params["artist_name"] == "fankel"
    assert params["lang"] == "en"
    assert params["tags"] == "rock+jazz"
    assert "fuzzytags" not in params
    assert params["vocalinstrumental"] == "instrumental"
    assert params["acousticelectric"] == "acoustic"
    assert params["gender"] == "female"
    assert params["speed"] == "high+veryhigh"
    assert params["featured"] == "1"
    assert params["durationbetween"] == "30_180"
    assert params["order"] == "popularity_total"
    assert params["audioformat"] == "ogg"
    assert params["audiodlformat"] == "ogg"
    assert params["limit"] == "20"
    assert params["offset"] == "20"
    assert params["fullcount"] == "true"
    assert params["type"] == "single albumtrack"
    assert str(requests[0].url).startswith("https://api.jamendo.com/v3.0/tracks/")
    assert page.total == 41
    assert page.offset == 20
    assert len(page.tracks) == 1
    track = page.tracks[0]
    assert track.name == "Manana"
    assert track.artist_name == "fankel"
    assert track.duration == 272
    assert track.genres == ("rock",)
    assert track.can_download
    assert track.audio_url == AUDIO_URL


def test_search_omits_empty_filters() -> None:
    """An empty query only sends the paging and format parameters."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return httpx.Response(200, json=_payload([], total=0))

    page = _search(TrackQuery(), handler)

    assert page.tracks == ()
    assert page.total == 0
    assert set(seen) == {
        "client_id",
        "format",
        "limit",
        "offset",
        "fullcount",
        "audioformat",
        "audiodlformat",
        "imagesize",
        "include",
        "order",
        "type",
    }


def test_match_any_uses_fuzzy_tags_and_open_duration() -> None:
    """Matching any genre uses fuzzy tags, and a minimum duration supplies the end."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return httpx.Response(200, json=_payload([], total="15"))

    query = TrackQuery(
        text=TextFilters(tags=("pop", "rock"), tag_mode="any"),
        span=RangeFilters(duration_min=30),
    )
    page = _search(query, handler)

    assert seen["fuzzytags"] == "pop+rock"
    assert "tags" not in seen
    assert seen["durationbetween"] == "30_7200"
    assert page.total == 15


def test_failed_status_uses_the_api_message() -> None:
    """A failed Jamendo status surfaces the API error message."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"headers": {"status": "failed", "error_message": "bad client"}, "results": []},
        )

    with pytest.raises(JamendoError, match="bad client"):
        _search(TrackQuery(), handler)


def test_http_and_json_errors_are_reported() -> None:
    """Transport failures and invalid documents become Jamendo errors."""

    def unavailable(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    def invalid(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    with pytest.raises(JamendoError, match="Could not reach Jamendo"):
        _search(TrackQuery(), unavailable)
    with pytest.raises(JamendoError, match="invalid JSON"):
        _search(TrackQuery(), invalid)


def test_download_returns_audio_bytes() -> None:
    """Allowed Jamendo URLs are fetched as bytes."""
    content = b"ID3fake-audio"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content)

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as http,
        JamendoClient("client-id", http=http) as client,
    ):
        assert client.download(DOWNLOAD_URL) == content


def test_download_rejects_untrusted_host() -> None:
    """Downloads never leave Jamendo hosts."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"nope")

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as http,
        JamendoClient("client-id", http=http) as client,
        pytest.raises(JamendoError, match="not on Jamendo"),
    ):
        client.download("https://evil.test/track.mp3")


def test_owned_client_can_close() -> None:
    """A client that opens its own HTTP connection can close it."""
    client = JamendoClient("client-id")
    client.close()


def test_download_rejects_a_redirect_off_jamendo() -> None:
    """A Jamendo URL that redirects elsewhere is not returned."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "evil.test":
            return httpx.Response(200, content=b"stolen")
        return httpx.Response(302, headers={"location": "https://evil.test/track"})

    with (
        httpx.Client(
            transport=httpx.MockTransport(handler),
            follow_redirects=True,
        ) as http,
        JamendoClient("client-id", http=http) as client,
        pytest.raises(JamendoError, match="not on Jamendo"),
    ):
        client.download(DOWNLOAD_URL)


def test_missing_client_id_is_rejected() -> None:
    """A blank client id cannot open a client."""
    with pytest.raises(JamendoError, match="client id"):
        JamendoClient("  ")


@pytest.mark.parametrize(
    "query",
    [
        TrackQuery(page=PageRequest(limit=0)),
        TrackQuery(page=PageRequest(limit=201)),
        TrackQuery(page=PageRequest(offset=-1)),
        TrackQuery(page=PageRequest(order="nope")),
        TrackQuery(text=TextFilters(language="english")),
        TrackQuery(text=TextFilters(tags=("rock+pop",))),
        TrackQuery(span=RangeFilters(duration_min=20, duration_max=10)),
        TrackQuery(sound=SoundFilters(speeds=(cast("Speed", "nope"),))),
    ],
)
def test_invalid_queries_are_rejected(query: TrackQuery) -> None:
    """Out-of-range filters fail before a request is sent."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_payload([], total=0))

    with pytest.raises(JamendoError, match=r"invalid|Unsupported|language|Duration|filter"):
        _search(query, handler)


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        (AUDIO_URL, True),
        (DOWNLOAD_URL, True),
        (IMAGE_URL, True),
        ("https://api.jamendo.com/v3.0/tracks/", True),
        ("https://jamendo.com/track/1", True),
        ("http://prod-1.storage.jamendo.com/file", False),
        ("https://jamendo.com.evil.test/file", False),
        ("https://notjamendo.com/file", False),
        ("", False),
    ],
)
def test_is_jamendo_media_url(url: str, *, allowed: bool) -> None:
    """Only https URLs on Jamendo hosts are accepted."""
    assert is_jamendo_media_url(url) is allowed


def test_download_filename_and_mime() -> None:
    """File names drop unsafe characters and use the format extension."""
    track = _track(artist_name="AC/DC", name="Thunderstruck")
    assert download_filename(track, "mp32") == "AC_DC - Thunderstruck.mp3"
    assert download_filename(track, "flac") == "AC_DC - Thunderstruck.flac"
    assert audio_mime("ogg") == "audio/ogg"
    with pytest.raises(JamendoError, match="audio format"):
        audio_mime("wav")


def test_download_is_unavailable_without_permission() -> None:
    """A track cannot be downloaded when the artist or the URL disallows it."""
    denied = _track(download_allowed=False, download_url="")
    missing_url = _track(download_allowed=True, download_url="")
    assert not denied.can_download
    assert not missing_url.can_download


def test_configured_client_id_prefers_the_first_non_empty_string() -> None:
    """Environment and secrets values are trimmed, and blanks are skipped."""
    assert configured_client_id("  ", " secret ") == "secret"
    assert configured_client_id(None, 12, "") is None


def test_download_audio_uses_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """The download helper opens a client and fetches the given URL."""
    seen: dict[str, str] = {}

    class _Client:
        def __init__(self, client_id: str) -> None:
            """Remember the client id passed by the helper."""
            self.client_id = client_id
            seen["client_id"] = client_id

        def __enter__(self) -> Self:
            """Return this fake client."""
            return self

        def __exit__(self, *_exc: object) -> None:
            """Close the fake client."""

        def download(self, url: str) -> bytes:
            """Record the URL and return placeholder audio.

            Returns:
                Placeholder audio bytes.
            """
            seen["url"] = url
            return b"audio" if self.client_id else b""

    monkeypatch.setattr("music_genre_classification.jamendo.JamendoClient", _Client)
    assert download_audio("client-id", DOWNLOAD_URL) == b"audio"
    assert seen == {"client_id": "client-id", "url": DOWNLOAD_URL}


def _search(query: TrackQuery, handler: Handler) -> TrackPage:
    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as http,
        JamendoClient("client-id", http=http) as client,
    ):
        return client.search_tracks(query)


def _filtered_query() -> TrackQuery:
    return TrackQuery(
        text=TextFilters(
            text=" sunrise ",
            artist=" fankel ",
            language=" EN ",
            tags=("Rock", "Jazz"),
            tag_mode="all",
        ),
        sound=SoundFilters(
            vocal_instrumental="instrumental",
            acoustic_electric="acoustic",
            gender="female",
            speeds=("high", "veryhigh"),
        ),
        span=RangeFilters(featured=True, duration_min=30, duration_max=180),
        page=PageRequest(order="popularity_total", audio_format="ogg", limit=20, offset=20),
    )


def _payload(tracks: list[object], *, total: int | str) -> dict[str, object]:
    return {
        "headers": {
            "status": "success",
            "code": 0,
            "error_message": "",
            "results_count": len(tracks),
            "results_fullcount": total,
        },
        "results": tracks,
    }


def _track_json() -> dict[str, object]:
    return {
        "id": "1848357",
        "name": "Manana",
        "duration": 272,
        "artist_name": "fankel",
        "album_name": "Manana",
        "audio": AUDIO_URL,
        "audiodownload": DOWNLOAD_URL,
        "audiodownload_allowed": True,
        "image": IMAGE_URL,
        "license_ccurl": "https://creativecommons.org/licenses/by-nc-nd/3.0/",
        "musicinfo": {"tags": {"genres": ["rock", "  ", 3]}},
    }


def _track(
    *,
    artist_name: str = "Artist",
    name: str = "Song",
    download_allowed: bool = True,
    download_url: str = DOWNLOAD_URL,
) -> Track:
    return Track(
        id="1848357",
        name=name,
        artist_name=artist_name,
        album_name="Album",
        duration=90,
        audio_url=AUDIO_URL,
        download_url=download_url,
        download_allowed=download_allowed,
        image_url=IMAGE_URL,
        license_url="https://creativecommons.org/licenses/by/4.0/",
        genres=("rock",),
    )
