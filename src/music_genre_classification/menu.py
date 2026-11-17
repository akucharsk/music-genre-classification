# Copyright (C) 2026 Bartłomiej Kózka
"""Interactive terminal menu that runs the project's make targets."""

from __future__ import annotations

import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Callable

QUIT_CHOICE = 0
MISSING_MAKE = "`make` was not found on PATH."


@dataclass(frozen=True, slots=True)
class Action:
    """One menu entry.

    Attributes:
        target: Name of the make target to run.
        label: Text shown to the user.
    """

    target: str
    label: str


ACTIONS = (
    Action("run", "Run the app"),
    Action("test", "Run the tests"),
    Action("lint", "Lint (ruff)"),
    Action("format", "Format the code (ruff)"),
    Action("typecheck", "Type-check (mypy)"),
    Action("docstrings", "Check docstrings"),
    Action("check", "Run all checks (as CI does)"),
    Action("preprocess", "Extract MFCC features from GTZAN"),
    Action("train-svm", "Train the SVM classifier"),
    Action("train-cnn", "Train the CNN classifier"),
    Action("train-transformer", "Train the Transformer classifier"),
    Action("import-weights", "Import weights of the original project"),
    Action("install", "Install dependencies and git hooks"),
)


def render_menu(console: Console, actions: tuple[Action, ...] = ACTIONS) -> None:
    """Print the numbered list of actions.

    Args:
        console: Console to print to.
        actions: Entries to list; they are numbered from 1.
    """
    table = Table(title="Music Genre Classification", show_header=False, box=None)
    table.add_column(justify="right", style="bold cyan")
    table.add_column()
    for number, action in enumerate(actions, start=1):
        table.add_row(str(number), action.label)
    table.add_row(str(QUIT_CHOICE), "Quit")
    console.print(table)


def ask_choice(count: int) -> int:
    """Ask the user for a menu number.

    Args:
        count: Number of actions; valid answers are 0 to ``count``.

    Returns:
        The chosen number, where 0 means quit.
    """
    choices = [str(number) for number in range(count + 1)]
    return IntPrompt.ask("Choose an option", choices=choices, show_choices=False)


def run_target(target: str) -> int:
    """Run one make target, returning to the caller when it ends or is interrupted.

    Args:
        target: Name of the make target.

    Returns:
        The exit code of make; 130 when the user interrupted the target.

    Raises:
        FileNotFoundError: If make is not installed.
    """
    make = shutil.which("make")
    if make is None:
        raise FileNotFoundError(MISSING_MAKE)
    try:
        return subprocess.run([make, target], check=False).returncode  # ruff: ignore[subprocess-without-shell-equals-true]
    except KeyboardInterrupt:
        return 130


def run_menu(
    console: Console,
    ask: Callable[[int], int] = ask_choice,
    run: Callable[[str], int] = run_target,
    actions: tuple[Action, ...] = ACTIONS,
) -> None:
    """Show the menu repeatedly until the user quits.

    Args:
        console: Console to print to.
        ask: Function returning the user's choice; injected for testing.
        run: Function running a make target and returning its exit code; injected for testing.
        actions: Entries to offer.
    """
    while True:
        render_menu(console, actions)
        choice = ask(len(actions))
        if choice == QUIT_CHOICE:
            return
        action = actions[choice - 1]
        code = run(action.target)
        console.print(f"[bold]{action.label}[/bold] finished with exit code {code}\n")


def main() -> None:
    """Open the interactive menu."""
    try:
        run_menu(Console())
    except (KeyboardInterrupt, EOFError):
        return


if __name__ == "__main__":
    main()
