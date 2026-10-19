# Copyright (C) 2026 Antoni Kucharski
"""Command-line entry point."""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as stcli

APP_PATH = Path(__file__).with_name("app.py")


def main() -> None:
    """Launch the Streamlit app."""
    sys.argv = ["streamlit", "run", str(APP_PATH), *sys.argv[1:]]
    sys.exit(stcli.main())
