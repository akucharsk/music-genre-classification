# Copyright (C) 2026 Antoni Kucharski
"""Streamlit entry point."""

from __future__ import annotations

import os

import numpy as np
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from music_genre_classification.data_preprocessing import (
    AudioError,
    MfccConfig,
    extract_audio_mfcc,
)
from music_genre_classification.jamendo import (
    MAX_DURATION_SECONDS,
    AcousticElectric,
    AudioFormat,
    Gender,
    JamendoClient,
    JamendoError,
    PageRequest,
    RangeFilters,
    SoundFilters,
    Speed,
    TagMode,
    TextFilters,
    Track,
    TrackPage,
    TrackQuery,
    VocalInstrumental,
    audio_mime,
    configured_client_id,
    download_audio,
    download_filename,
    is_jamendo_media_url,
)

AUDIO_EXTENSIONS = ("wav", "mp3", "ogg", "flac", "m4a", "aac")
PAGE_SIZES = (10, 20, 50)
SECONDS_PER_MINUTE = 60
MFCC_CONFIG = MfccConfig()
MFCC_ROW_HEIGHT = 12
ANALYZE_LABEL = "Analyze"
DECODE_ERROR = "Could not read this audio file."
TOO_SHORT = "The audio is shorter than one segment, nothing to analyze."
QUERY_KEY = "jamendo_query"
SEARCH_FORM_KEY = "jamendo-search"
CLIENT_ID_SETTING = "JAMENDO_CLIENT_ID"
CLIENT_ID_HELP = (
    "Add a Jamendo client id to search tracks. Set `JAMENDO_CLIENT_ID` in the "
    "environment or in `.streamlit/secrets.toml`. "
    "Create one at [devportal.jamendo.com](https://devportal.jamendo.com/)."
)
JAMENDO_INTRO = (
    "Search Jamendo tracks, play them in the browser, and download them when the artist allows it."
)
GENRES = (
    "ambient",
    "blues",
    "classical",
    "country",
    "disco",
    "electronic",
    "folk",
    "funk",
    "hiphop",
    "jazz",
    "latin",
    "lounge",
    "metal",
    "pop",
    "punk",
    "reggae",
    "relaxation",
    "rock",
    "soundtrack",
    "world",
)
TAG_MODES: dict[str, TagMode] = {"Match all": "all", "Match any": "any"}
VOCAL_OPTIONS: dict[str, VocalInstrumental | None] = {
    "Any vocals": None,
    "Vocal": "vocal",
    "Instrumental": "instrumental",
}
ACOUSTIC_OPTIONS: dict[str, AcousticElectric | None] = {
    "Any acoustics": None,
    "Acoustic": "acoustic",
    "Electric": "electric",
}
GENDER_OPTIONS: dict[str, Gender | None] = {
    "Any singer": None,
    "Female": "female",
    "Male": "male",
}
SPEED_OPTIONS: dict[str, Speed] = {
    "Very slow": "verylow",
    "Slow": "low",
    "Medium": "medium",
    "Fast": "high",
    "Very fast": "veryhigh",
}
ORDER_OPTIONS: dict[str, str] = {
    "Relevance": "relevance",
    "Popularity": "popularity_total",
    "Listens": "listens_total",
    "Downloads": "downloads_total",
    "Newest": "releasedate_desc",
    "Longest": "duration_desc",
    "Track name": "name",
}
FORMAT_OPTIONS: dict[str, AudioFormat] = {
    "MP3": "mp32",
    "MP3 (smaller)": "mp31",
    "OGG": "ogg",
    "FLAC": "flac",
}


def _render_upload() -> None:
    """Show the local audio file picker."""
    st.write("Upload an audio file to classify its genre.")
    audio_file = st.file_uploader(
        "Audio file",
        type=AUDIO_EXTENSIONS,
        help="Accepted formats: WAV, MP3, OGG, FLAC, M4A, and AAC.",
    )
    if audio_file is not None:
        st.audio(audio_file, format=audio_file.type)
        st.caption(audio_file.name)
        if st.button(ANALYZE_LABEL, key="analyze-upload"):
            _render_analysis(audio_file.getvalue())


def _render_analysis(audio: bytes) -> None:
    """Extract MFCC features from audio and show them as an image."""
    with st.spinner("Extracting features..."):
        try:
            segments = extract_audio_mfcc(audio, MFCC_CONFIG)
        except AudioError:
            st.error(DECODE_ERROR)
            return
    if not segments:
        st.warning(TOO_SHORT)
        return
    st.caption(
        f"{len(segments)} segments x {len(segments[0])} frames x {MFCC_CONFIG.num_mfcc} MFCCs"
    )
    st.image(_mfcc_image(segments), width="stretch", caption="MFCC over time")


def _mfcc_image(segments: list[list[list[float]]]) -> np.ndarray:
    """Join segment MFCCs along time and scale them to a grayscale image in [0, 1].

    Returns:
        Array with one block of rows per coefficient and one column per frame.
    """
    matrix = np.concatenate([np.asarray(segment) for segment in segments]).T
    span = np.ptp(matrix)
    scaled = (matrix - matrix.min()) / span if span else np.zeros_like(matrix)
    return np.repeat(scaled, MFCC_ROW_HEIGHT, axis=0)


def _render_jamendo() -> None:
    """Show Jamendo search, playback, and download."""
    st.write(JAMENDO_INTRO)
    client_id = _client_id()
    if client_id is None:
        st.info(CLIENT_ID_HELP)
        return
    _run_search(client_id)


def _client_id() -> str | None:
    return configured_client_id(os.environ.get(CLIENT_ID_SETTING), _secret_client_id())


def _secret_client_id() -> object:
    try:
        return st.secrets[CLIENT_ID_SETTING]
    except (KeyError, StreamlitSecretNotFoundError):
        return None


def _run_search(client_id: str) -> None:
    submitted = _search_form()
    if submitted is not None:
        st.session_state[QUERY_KEY] = submitted
    stored = st.session_state.get(QUERY_KEY)
    if isinstance(stored, TrackQuery):
        _show_results(client_id, stored)


def _search_form() -> TrackQuery | None:
    """Return a query when the user submits the search form."""
    with st.form(SEARCH_FORM_KEY):
        search_text = st.text_input("Search", placeholder="Track, artist, album, or tag")
        with st.expander("Filters"):
            artist, tags, tag_mode, language = _text_filter_inputs()
            sound = _sound_filter_inputs()
            span = _span_filter_inputs()
            page = _page_inputs()
        submitted = st.form_submit_button("Search")
    if not submitted:
        return None
    text = TextFilters(
        text=search_text,
        artist=artist,
        language=language,
        tags=tags,
        tag_mode=tag_mode,
    )
    return TrackQuery(text=text, sound=sound, span=span, page=page)


def _text_filter_inputs() -> tuple[str, tuple[str, ...], TagMode, str]:
    artist = st.text_input("Artist")
    selected = st.multiselect("Genres", GENRES)
    tags = tuple(str(tag) for tag in selected)
    match_label = str(
        st.radio(
            "Genre match",
            list(TAG_MODES),
            horizontal=True,
            help="Match all selected genres, or any of them.",
        ),
    )
    language = st.text_input("Lyrics language", placeholder="en")
    return artist, tags, TAG_MODES[match_label], language


def _sound_filter_inputs() -> SoundFilters:
    vocal_label = str(st.selectbox("Vocals", list(VOCAL_OPTIONS)))
    acoustic_label = str(st.selectbox("Acoustics", list(ACOUSTIC_OPTIONS)))
    gender_label = str(st.selectbox("Singer", list(GENDER_OPTIONS)))
    selected = st.multiselect("Speed", list(SPEED_OPTIONS))
    speeds = tuple(SPEED_OPTIONS[str(label)] for label in selected)
    return SoundFilters(
        vocal_instrumental=VOCAL_OPTIONS[vocal_label],
        acoustic_electric=ACOUSTIC_OPTIONS[acoustic_label],
        gender=GENDER_OPTIONS[gender_label],
        speeds=speeds,
    )


def _span_filter_inputs() -> RangeFilters:
    featured = st.checkbox("Featured tracks only")
    duration_min = st.number_input(
        "Minimum duration (seconds)",
        min_value=0,
        max_value=MAX_DURATION_SECONDS,
        value=None,
        step=1,
        placeholder="Any",
    )
    duration_max = st.number_input(
        "Maximum duration (seconds)",
        min_value=0,
        max_value=MAX_DURATION_SECONDS,
        value=None,
        step=1,
        placeholder="Any",
    )
    return RangeFilters(
        featured=featured,
        duration_min=duration_min,
        duration_max=duration_max,
    )


def _page_inputs() -> PageRequest:
    order_label = str(st.selectbox("Sort", list(ORDER_OPTIONS)))
    format_label = str(st.selectbox("Audio format", list(FORMAT_OPTIONS)))
    limit = int(st.selectbox("Tracks per page", PAGE_SIZES))
    return PageRequest(
        order=ORDER_OPTIONS[order_label],
        audio_format=FORMAT_OPTIONS[format_label],
        limit=limit,
    )


def _show_results(client_id: str, query: TrackQuery) -> None:
    try:
        page = _fetch_page(client_id, query)
    except JamendoError as exc:
        st.error(str(exc))
        return
    if page.total <= 0:
        st.info("No tracks matched those filters.")
        return
    _turn_page(query, page)
    if not page.tracks:
        st.info("No tracks on this page.")
        return
    _render_tracks(client_id, page, query.page.audio_format)


def _fetch_page(client_id: str, query: TrackQuery) -> TrackPage:
    with st.spinner("Searching Jamendo"), JamendoClient(client_id) as client:
        return client.search_tracks(query)


def _turn_page(query: TrackQuery, page: TrackPage) -> None:
    st.caption(_page_label(page))
    updated = _page_controls(query, page)
    if updated is not None:
        st.session_state[QUERY_KEY] = updated
        st.rerun()


def _page_label(page: TrackPage) -> str:
    if not page.tracks:
        return f"0 of {page.total}"
    start = page.offset + 1
    end = page.offset + len(page.tracks)
    return f"{start}-{end} of {page.total}"


def _page_controls(query: TrackQuery, page: TrackPage) -> TrackQuery | None:
    previous_column, next_column = st.columns(2)
    with previous_column:
        previous = st.button("Previous page", disabled=page.offset <= 0)
    with next_column:
        nxt = st.button("Next page", disabled=page.offset + page.limit >= page.total)
    if previous:
        return query.page_at(max(page.offset - page.limit, 0))
    if nxt:
        return query.page_at(page.offset + page.limit)
    return None


def _render_tracks(client_id: str, page: TrackPage, audio_format: str) -> None:
    for index, track in enumerate(page.tracks):
        _render_track(client_id, track, audio_format, f"{page.offset}-{index}")


def _render_track(client_id: str, track: Track, audio_format: str, key: str) -> None:
    with st.container(border=True):
        if is_jamendo_media_url(track.image_url):
            artwork, details = st.columns((1, 4))
            with artwork:
                st.image(track.image_url, width="stretch", alt=track.name)
            with details:
                _render_track_details(client_id, track, audio_format, key)
            return
        _render_track_details(client_id, track, audio_format, key)


def _render_track_details(client_id: str, track: Track, audio_format: str, key: str) -> None:
    st.text(track.name)
    st.caption(_summary(track))
    _render_player(track, audio_format)
    _render_download(client_id, track, audio_format, key)
    _render_track_analysis(client_id, track, key)
    if track.license_url:
        st.caption(f"License: {track.license_url}")


def _summary(track: Track) -> str:
    parts = [track.artist_name]
    if track.album_name:
        parts.append(track.album_name)
    parts.append(_clock(track.duration))
    if track.genres:
        parts.append(", ".join(track.genres))
    return " | ".join(parts)


def _clock(duration: int) -> str:
    minutes, seconds = divmod(max(duration, 0), SECONDS_PER_MINUTE)
    return f"{minutes}:{seconds:02d}"


def _render_player(track: Track, audio_format: str) -> None:
    if is_jamendo_media_url(track.audio_url):
        st.audio(track.audio_url, format=audio_mime(audio_format))


def _render_track_analysis(client_id: str, track: Track, key: str) -> None:
    """Offer feature extraction for the streamable version of a track."""
    if not is_jamendo_media_url(track.audio_url):
        return
    if st.button(ANALYZE_LABEL, key=f"analyze-{key}"):
        try:
            audio = download_audio(client_id, track.audio_url)
        except JamendoError as error:
            st.error(str(error))
            return
        _render_analysis(audio)


def _render_download(client_id: str, track: Track, audio_format: str, key: str) -> None:
    """Offer a file download when the artist allows it."""
    if not track.can_download or not is_jamendo_media_url(track.download_url):
        st.caption("Download disabled by the artist")
        return
    url = track.download_url

    def _fetch() -> bytes:
        return download_audio(client_id, url)

    st.download_button(
        "Download",
        data=_fetch,
        file_name=download_filename(track, audio_format),
        mime=audio_mime(audio_format),
        key=f"download-{key}",
        help="Save this track from Jamendo.",
    )


st.set_page_config(page_title="Music Genre Classification")
st.title("Music Genre Classification")

upload_tab, jamendo_tab = st.tabs(["Upload", "Jamendo"])
with upload_tab:
    _render_upload()
with jamendo_tab:
    _render_jamendo()
