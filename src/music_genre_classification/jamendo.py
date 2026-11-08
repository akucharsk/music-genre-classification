# Copyright (C) 2026 Antoni Kucharski
"""Jamendo API client for searching, playing, and downloading tracks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, Self, cast
from urllib.parse import urlsplit

import httpx

if TYPE_CHECKING:
    from collections.abc import Mapping

API_ROOT = "https://api.jamendo.com/v3.0"
JAMENDO_HOST = "jamendo.com"
JAMENDO_HOST_SUFFIX = ".jamendo.com"

DEFAULT_PAGE_SIZE = 10
MIN_PAGE_SIZE = 1
MAX_PAGE_SIZE = 200
MIN_OFFSET = 0
MIN_DURATION_SECONDS = 0
MAX_DURATION_SECONDS = 7200
COVER_IMAGE_SIZE = 200
LANGUAGE_CODE_LENGTH = 2
REQUEST_TIMEOUT_SECONDS = 60.0
MAX_FILENAME_STEM = 80

SUCCESS_STATUS = "success"
RESPONSE_FORMAT = "json"
TRACK_TYPES = "single albumtrack"
FULL_COUNT = "true"
FEATURED_FLAG = "1"
INCLUDE_MUSICINFO = "musicinfo"
TAG_MODE_ALL = "all"
TRUE_TEXT = "true"
FILENAME_FALLBACK = "track"

MISSING_CLIENT_ID = "Jamendo client id is missing."
UNSUPPORTED_FORMAT = "Unsupported audio format."
UNSUPPORTED_ORDER = "Unsupported sort order."
UNSUPPORTED_FILTER = "Unsupported search filter."
INVALID_PAGE = "Page size or offset is invalid."
INVALID_DURATION = "Duration range is invalid."
INVALID_LANGUAGE = "Lyrics language must be a two-letter code."
REQUEST_FAILED = "Jamendo request failed."
INVALID_JSON = "Jamendo returned invalid JSON."
INVALID_RESPONSE = "Jamendo returned an unexpected response."
UNREACHABLE = "Could not reach Jamendo."
UNEXPECTED_HOST = "Refusing a URL that is not on Jamendo."

TAG_MODE_VALUES = frozenset({TAG_MODE_ALL, "any"})
VOCAL_VALUES = frozenset({"vocal", "instrumental"})
ACOUSTIC_VALUES = frozenset({"acoustic", "electric"})
GENDER_VALUES = frozenset({"male", "female"})
SPEED_VALUES = frozenset({"verylow", "low", "medium", "high", "veryhigh"})
ORDERS = frozenset(
    {
        "relevance",
        "popularity_total",
        "listens_total",
        "downloads_total",
        "releasedate_desc",
        "duration_desc",
        "name",
    },
)
TAG_SEPARATORS = frozenset("+ \t")
UNSAFE_FILENAME_CHARACTERS = frozenset('<>:"/\\|?*')
AUDIO_FORMAT_DETAILS: dict[str, tuple[str, str]] = {
    "mp31": ("mp3", "audio/mpeg"),
    "mp32": ("mp3", "audio/mpeg"),
    "ogg": ("ogg", "audio/ogg"),
    "flac": ("flac", "audio/flac"),
}

type TagMode = Literal["all", "any"]
type VocalInstrumental = Literal["vocal", "instrumental"]
type AcousticElectric = Literal["acoustic", "electric"]
type Gender = Literal["male", "female"]
type Speed = Literal["verylow", "low", "medium", "high", "veryhigh"]
type AudioFormat = Literal["mp31", "mp32", "ogg", "flac"]


class JamendoError(Exception):
    """A Jamendo request could not be completed."""


@dataclass(frozen=True, slots=True)
class TextFilters:
    """Free-text, artist, language, and genre filters."""

    text: str = ""
    artist: str = ""
    language: str | None = None
    tags: tuple[str, ...] = ()
    tag_mode: TagMode = "all"


@dataclass(frozen=True, slots=True)
class SoundFilters:
    """Timbre and performance filters."""

    vocal_instrumental: VocalInstrumental | None = None
    acoustic_electric: AcousticElectric | None = None
    gender: Gender | None = None
    speeds: tuple[Speed, ...] = ()


@dataclass(frozen=True, slots=True)
class RangeFilters:
    """Featured flag and duration window, in seconds."""

    featured: bool = False
    duration_min: int | None = None
    duration_max: int | None = None


@dataclass(frozen=True, slots=True)
class PageRequest:
    """Sort order, audio format, and result window."""

    order: str = "relevance"
    audio_format: AudioFormat = "mp32"
    limit: int = DEFAULT_PAGE_SIZE
    offset: int = MIN_OFFSET


@dataclass(frozen=True, slots=True)
class TrackQuery:
    """A paginated Jamendo track search."""

    text: TextFilters = field(default_factory=TextFilters)
    sound: SoundFilters = field(default_factory=SoundFilters)
    span: RangeFilters = field(default_factory=RangeFilters)
    page: PageRequest = field(default_factory=PageRequest)

    def page_at(self, offset: int) -> TrackQuery:
        """Return a copy of this query starting at offset."""
        return replace(self, page=replace(self.page, offset=offset))

    def to_params(self, client_id: str) -> dict[str, str]:
        """Build the query parameters for the tracks endpoint.

        Returns:
            Query parameters for the tracks endpoint.
        """
        params = _page_params(client_id, self.page)
        _apply_text(params, self.text)
        _apply_sound(params, self.sound)
        _apply_span(params, self.span)
        return params


@dataclass(frozen=True, slots=True)
class Track:
    """A Jamendo track that can be played or downloaded."""

    id: str
    name: str
    artist_name: str
    album_name: str
    duration: int
    audio_url: str
    download_url: str
    download_allowed: bool
    image_url: str
    license_url: str
    genres: tuple[str, ...] = ()

    @property
    def can_download(self) -> bool:
        """Whether the artist allows this track to be downloaded."""
        return self.download_allowed and bool(self.download_url)


@dataclass(frozen=True, slots=True)
class TrackPage:
    """One page of Jamendo search results."""

    tracks: tuple[Track, ...]
    total: int
    offset: int
    limit: int


class JamendoClient:
    """HTTP client for the Jamendo tracks API."""

    def __init__(
        self,
        client_id: str,
        *,
        http: httpx.Client | None = None,
        base_url: str = API_ROOT,
    ) -> None:
        """Open a client for the given Jamendo client id.

        Raises:
            JamendoError: If the client id is blank.
        """
        stripped = client_id.strip()
        if not stripped:
            raise JamendoError(MISSING_CLIENT_ID)
        self._client_id = stripped
        self._http = _new_http() if http is None else http
        self._owns_http = http is None
        self._tracks_url = f"{base_url.rstrip('/')}/tracks/"

    def close(self) -> None:
        """Close the HTTP client when this object created it."""
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> Self:
        """Return this client for use in a with-statement."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the owned HTTP client."""
        self.close()

    def search_tracks(self, query: TrackQuery) -> TrackPage:
        """Return one page of tracks matching query."""
        _validate(query)
        body = self._read(self._tracks_url, query.to_params(self._client_id))
        return _parse_page(_decode(body), offset=query.page.offset, limit=query.page.limit)

    def download(self, url: str) -> bytes:
        """Download audio bytes from a Jamendo URL.

        Returns:
            The audio file contents.
        """
        return self._read(url, None)

    def _read(self, url: str, params: Mapping[str, str] | None) -> bytes:
        if not is_jamendo_media_url(url):
            raise JamendoError(UNEXPECTED_HOST)
        try:
            response = self._http.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise JamendoError(UNREACHABLE) from exc
        if not is_jamendo_media_url(str(response.url)):
            raise JamendoError(UNEXPECTED_HOST)
        return response.content


def configured_client_id(*candidates: object) -> str | None:
    """Return the first non-empty string among client id candidates."""
    for candidate in candidates:
        if isinstance(candidate, str):
            stripped = candidate.strip()
            if stripped:
                return stripped
    return None


def download_audio(client_id: str, url: str) -> bytes:
    """Download a Jamendo track and return its bytes.

    Returns:
        The audio file contents.
    """
    with JamendoClient(client_id) as client:
        return client.download(url)


def download_filename(track: Track, audio_format: str) -> str:
    """Return a safe file name for a downloaded track."""
    extension, _mime = _audio_details(audio_format)
    return f"{_filename_stem(track)}.{extension}"


def audio_mime(audio_format: str) -> str:
    """Return the MIME type for a Jamendo audio format code."""
    _extension, mime = _audio_details(audio_format)
    return mime


def is_jamendo_media_url(url: str) -> bool:
    """Return whether url is an https resource on a Jamendo host."""
    parsed = urlsplit(url)
    host = parsed.hostname
    if parsed.scheme != "https" or host is None:
        return False
    normalized = host.lower().rstrip(".")
    return normalized == JAMENDO_HOST or normalized.endswith(JAMENDO_HOST_SUFFIX)


def _new_http() -> httpx.Client:
    return httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True)


def _page_params(client_id: str, page: PageRequest) -> dict[str, str]:
    return {
        "client_id": client_id,
        "format": RESPONSE_FORMAT,
        "limit": str(page.limit),
        "offset": str(page.offset),
        "fullcount": FULL_COUNT,
        "audioformat": page.audio_format,
        "audiodlformat": page.audio_format,
        "imagesize": str(COVER_IMAGE_SIZE),
        "include": INCLUDE_MUSICINFO,
        "order": page.order,
        "type": TRACK_TYPES,
    }


def _apply_text(params: dict[str, str], filters: TextFilters) -> None:
    _assign(params, "search", filters.text)
    _assign(params, "artist_name", filters.artist)
    _assign(params, "lang", _language(filters.language))
    key = "tags" if filters.tag_mode == TAG_MODE_ALL else "fuzzytags"
    _assign_joined(params, key, filters.tags)


def _apply_sound(params: dict[str, str], filters: SoundFilters) -> None:
    _assign(params, "vocalinstrumental", filters.vocal_instrumental)
    _assign(params, "acousticelectric", filters.acoustic_electric)
    _assign(params, "gender", filters.gender)
    _assign_joined(params, "speed", filters.speeds)


def _apply_span(params: dict[str, str], filters: RangeFilters) -> None:
    if filters.featured:
        params["featured"] = FEATURED_FLAG
    duration = _duration_between(filters.duration_min, filters.duration_max)
    if duration is not None:
        params["durationbetween"] = duration


def _assign(params: dict[str, str], key: str, value: str | None) -> None:
    if value is None:
        return
    stripped = value.strip()
    if stripped:
        params[key] = stripped


def _assign_joined(params: dict[str, str], key: str, values: tuple[str, ...]) -> None:
    cleaned = tuple(item.strip().lower() for item in values if item.strip())
    if cleaned:
        params[key] = "+".join(cleaned)


def _duration_between(minimum: int | None, maximum: int | None) -> str | None:
    if minimum is None and maximum is None:
        return None
    start = MIN_DURATION_SECONDS if minimum is None else minimum
    end = MAX_DURATION_SECONDS if maximum is None else maximum
    return f"{start}_{end}"


def _language(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip().lower()
    if not stripped:
        return None
    return stripped


def _validate(query: TrackQuery) -> None:
    message = _validation_error(query)
    if message is not None:
        raise JamendoError(message)


def _validation_error(query: TrackQuery) -> str | None:
    page = query.page
    language = _language(query.text.language)
    checks = (
        (_page_ok(page), INVALID_PAGE),
        (page.audio_format in AUDIO_FORMAT_DETAILS, UNSUPPORTED_FORMAT),
        (page.order in ORDERS, UNSUPPORTED_ORDER),
        (query.text.tag_mode in TAG_MODE_VALUES, UNSUPPORTED_FILTER),
        (_known(query.sound.vocal_instrumental, VOCAL_VALUES), UNSUPPORTED_FILTER),
        (_known(query.sound.acoustic_electric, ACOUSTIC_VALUES), UNSUPPORTED_FILTER),
        (_known(query.sound.gender, GENDER_VALUES), UNSUPPORTED_FILTER),
        (_all_known(query.sound.speeds, SPEED_VALUES), UNSUPPORTED_FILTER),
        (_tags_ok(query.text.tags), UNSUPPORTED_FILTER),
        (_language_ok(language), INVALID_LANGUAGE),
        (_duration_ok(query.span.duration_min, query.span.duration_max), INVALID_DURATION),
    )
    for passed, message in checks:
        if not passed:
            return message
    return None


def _page_ok(page: PageRequest) -> bool:
    limit_ok = MIN_PAGE_SIZE <= page.limit <= MAX_PAGE_SIZE
    return limit_ok and page.offset >= MIN_OFFSET


def _known(value: str | None, allowed: frozenset[str]) -> bool:
    return value is None or value in allowed


def _all_known(values: tuple[str, ...], allowed: frozenset[str]) -> bool:
    return all(item in allowed for item in values)


def _tags_ok(tags: tuple[str, ...]) -> bool:
    return all(_tag_ok(tag) for tag in tags)


def _tag_ok(tag: str) -> bool:
    stripped = tag.strip()
    if not stripped:
        return False
    return all(character not in TAG_SEPARATORS for character in stripped)


def _language_ok(value: str | None) -> bool:
    if value is None:
        return True
    return len(value) == LANGUAGE_CODE_LENGTH and value.isalpha() and value.isascii()


def _duration_ok(minimum: int | None, maximum: int | None) -> bool:
    if minimum is not None and minimum < MIN_DURATION_SECONDS:
        return False
    if maximum is not None and (maximum < MIN_DURATION_SECONDS or maximum > MAX_DURATION_SECONDS):
        return False
    return minimum is None or maximum is None or minimum <= maximum


def _decode(body: bytes) -> dict[str, object]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise JamendoError(INVALID_JSON) from exc
    return _as_dict(payload)


def _parse_page(payload: dict[str, object], *, offset: int, limit: int) -> TrackPage:
    headers = _as_dict(payload.get("headers"))
    _raise_for_status(headers)
    tracks = _parse_tracks(_as_list(payload.get("results")))
    return TrackPage(
        tracks=tracks,
        total=_total(headers, len(tracks)),
        offset=offset,
        limit=limit,
    )


def _raise_for_status(headers: Mapping[str, object]) -> None:
    if headers.get("status") == SUCCESS_STATUS:
        return
    raise JamendoError(_error_message(headers))


def _error_message(headers: Mapping[str, object]) -> str:
    message = _string(headers.get("error_message"))
    return message or REQUEST_FAILED


def _total(headers: Mapping[str, object], fallback: int) -> int:
    parsed = _whole_number(headers.get("results_fullcount"))
    if parsed is None or parsed < MIN_OFFSET:
        return fallback
    return parsed


def _parse_tracks(results: list[object]) -> tuple[Track, ...]:
    tracks: list[Track] = []
    for item in results:
        track = _try_parse_track(item)
        if track is not None:
            tracks.append(track)
    return tuple(tracks)


def _try_parse_track(item: object) -> Track | None:
    raw = _dict_or_none(item)
    if raw is None:
        return None
    identity = _identity(raw)
    audio_url = _jamendo_url(raw.get("audio"))
    if identity is None or audio_url is None:
        return None
    track_id, name, artist_name = identity
    return _track_from(raw, track_id, name, artist_name, audio_url)


def _track_from(
    raw: Mapping[str, object],
    track_id: str,
    name: str,
    artist_name: str,
    audio_url: str,
) -> Track:
    duration = _whole_number(raw.get("duration"))
    return Track(
        id=track_id,
        name=name,
        artist_name=artist_name,
        album_name=_optional_text(raw.get("album_name")),
        duration=MIN_DURATION_SECONDS if duration is None else duration,
        audio_url=audio_url,
        download_url=_jamendo_url(raw.get("audiodownload")) or "",
        download_allowed=_flag(raw.get("audiodownload_allowed")),
        image_url=_jamendo_url(raw.get("image")) or "",
        license_url=_optional_text(raw.get("license_ccurl")),
        genres=_genres(raw),
    )


def _identity(raw: Mapping[str, object]) -> tuple[str, str, str] | None:
    track_id = _string(raw.get("id"))
    name = _string(raw.get("name"))
    artist_name = _string(raw.get("artist_name"))
    if track_id is None or name is None or artist_name is None:
        return None
    return track_id, name, artist_name


def _genres(raw: Mapping[str, object]) -> tuple[str, ...]:
    info = _dict_or_none(raw.get("musicinfo"))
    tags = None if info is None else _dict_or_none(info.get("tags"))
    genres = None if tags is None else tags.get("genres")
    if not isinstance(genres, list):
        return ()
    return tuple(item.strip() for item in genres if isinstance(item, str) and item.strip())


def _jamendo_url(value: object) -> str | None:
    url = _string(value)
    if url is None or not is_jamendo_media_url(url):
        return None
    return url


def _optional_text(value: object) -> str:
    return _string(value) or ""


def _flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == TRUE_TEXT
    return False


def _string(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def _whole_number(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _as_dict(value: object) -> dict[str, object]:
    parsed = _dict_or_none(value)
    if parsed is None:
        raise JamendoError(INVALID_RESPONSE)
    return parsed


def _dict_or_none(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    parsed: dict[str, object] = {}
    for key, item in value.items():
        if isinstance(key, str):
            parsed[key] = cast("object", item)
    return parsed


def _as_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise JamendoError(INVALID_RESPONSE)
    return [cast("object", item) for item in value]


def _audio_details(audio_format: str) -> tuple[str, str]:
    try:
        return AUDIO_FORMAT_DETAILS[audio_format]
    except KeyError as exc:
        raise JamendoError(UNSUPPORTED_FORMAT) from exc


def _filename_stem(track: Track) -> str:
    parts = [part for part in (track.artist_name, track.name) if part.strip()]
    stem = _sanitize_filename(" - ".join(parts))
    if stem:
        return stem
    fallback = _sanitize_filename(track.id)
    return fallback or FILENAME_FALLBACK


def _sanitize_filename(value: str) -> str:
    cleaned = "".join(
        character if _filename_character_ok(character) else "_" for character in value
    )
    return " ".join(cleaned.split()).strip(" .")[:MAX_FILENAME_STEM].strip(" .")


def _filename_character_ok(character: str) -> bool:
    return character.isprintable() and character not in UNSAFE_FILENAME_CHARACTERS
