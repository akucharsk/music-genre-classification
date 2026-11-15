# Copyright (C) 2026 Bartłomiej Kózka
"""Tests for the interactive menu."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from music_genre_classification import menu
from music_genre_classification.menu import ACTIONS, Action, render_menu, run_menu


def _console() -> tuple[Console, StringIO]:
    buffer = StringIO()
    return Console(file=buffer, width=80), buffer


def test_render_menu_numbers_every_action() -> None:
    """Each action is listed with its number, and 0 quits."""
    console, buffer = _console()

    render_menu(console, (Action("run", "Run the app"), Action("test", "Run the tests")))

    output = buffer.getvalue()
    assert "1" in output
    assert "Run the app" in output
    assert "Run the tests" in output
    assert "Quit" in output


def test_run_menu_runs_chosen_targets_until_quit() -> None:
    """Chosen options run their make targets in order and 0 ends the loop."""
    console, buffer = _console()
    answers = iter([2, 1, 0])
    executed: list[str] = []

    def run(target: str) -> int:
        executed.append(target)
        return 0

    run_menu(console, ask=lambda _count: next(answers), run=run)

    assert executed == [ACTIONS[1].target, ACTIONS[0].target]
    assert "exit code 0" in buffer.getvalue()


def test_run_target_reports_missing_make(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clear error is raised when make is not installed."""
    monkeypatch.setattr("shutil.which", lambda _name: None)

    with pytest.raises(FileNotFoundError, match="make"):
        menu.run_target("test")


def test_run_target_returns_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exit code of make is passed on, and Ctrl+C maps to 130."""
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/make")

    class _Done:
        returncode = 3

    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: _Done())
    assert menu.run_target("test") == 3

    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("subprocess.run", interrupt)
    assert menu.run_target("test") == 130
