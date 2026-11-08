# Copyright (C) 2026 Antoni Kucharski
"""Streamlit entry point."""

from __future__ import annotations

import streamlit as st

AUDIO_EXTENSIONS = ("wav", "mp3", "ogg", "flac", "m4a", "aac")

st.set_page_config(page_title="Music Genre Classification")

st.title("Music Genre Classification")
st.write("Upload an audio file to classify its genre.")

audio_file = st.file_uploader(
    "Audio file",
    type=AUDIO_EXTENSIONS,
    help="Accepted formats: WAV, MP3, OGG, FLAC, M4A, and AAC.",
)

if audio_file is not None:
    st.audio(audio_file, format=audio_file.type)
    st.caption(audio_file.name)
