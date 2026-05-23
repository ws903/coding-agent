# src/agent/cli_input.py
"""REPL input handling: prompt_toolkit session + ESC-during-task watcher.

The two concerns are different but share terminal state:
- PromptSession reads a line of input with proper editing (arrow keys,
  history, ESC clears).
- _esc_aborts spawns a background thread during orch.run that watches stdin
  for ESC and calls orch.abort() through call_soon_threadsafe.
"""

from __future__ import annotations

import contextlib
import select
import sys
import threading
import time
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings

from agent.cli import console as _con  # See cli_ui.py for why we use attribute access

if TYPE_CHECKING:
    from agent.core.orchestrator import Orchestrator

# ESC watcher tuning. Values picked to match prompt_toolkit's defaults
# where applicable; documented inline so future maintainers don't have to
# guess what these numbers mean.
_WATCHER_POLL_INTERVAL = 0.1  # seconds between Unix select() polls
_WATCHER_POLL_INTERVAL_WIN = 0.05  # seconds between Windows kbhit() polls
_WATCHER_STOP_TIMEOUT = 1.0  # how long _esc_aborts waits for the thread to join
_ESC_SEQUENCE_DISAMBIG_TIMEOUT = 0.05  # peek-window after \x1b to detect escape
# sequences (arrow keys etc) before treating it as a lone ESC press


def _build_repl_keybindings() -> KeyBindings:
    """Key bindings for the main REPL prompt.

    ESC clears the input line (matches Claude Code behavior at the prompt).
    Ctrl+C raises KeyboardInterrupt so the outer loop can exit cleanly.
    """
    kb = KeyBindings()

    @kb.add("escape", eager=True)
    def _esc_clear(event):
        event.app.current_buffer.reset()

    @kb.add("c-c")
    def _ctrl_c(event):
        event.app.exit(exception=KeyboardInterrupt)

    return kb


def _make_prompt_session() -> PromptSession:
    """Build the REPL PromptSession.

    Replaces Rich's Prompt.ask (which used input() underneath and produced
    garbled escape sequences for arrow keys, ESC, etc.). prompt_toolkit gives
    us cross-platform line editing, history within a session, and bindable
    keys -- including ESC to clear the input line.
    """
    return PromptSession(
        message=HTML("<ansicyan><b>></b></ansicyan> "),
        key_bindings=_build_repl_keybindings(),
    )


_PROMPT_SESSION: PromptSession | None = None


def _get_session() -> PromptSession:
    """Lazy singleton. Avoids constructing a PromptSession at import or loop
    entry, which triggers Windows-console probing that fails in headless CI.
    Tests mock `_get_user_input` so this never runs in unit tests."""
    global _PROMPT_SESSION
    if _PROMPT_SESSION is None:
        _PROMPT_SESSION = _make_prompt_session()
    return _PROMPT_SESSION


async def _get_user_input() -> str:
    """One-line shim that tests patch instead of mocking PromptSession itself."""
    return await _get_session().prompt_async()


def _is_lone_escape_unix(
    stream, timeout: float = _ESC_SEQUENCE_DISAMBIG_TIMEOUT
) -> bool:
    """After reading 0x1b, return True if it was a standalone ESC press
    (nothing follows within `timeout`) and False if it was the prefix of
    an escape sequence (arrow keys, function keys, Alt+X, etc.).

    Escape sequences are transmitted as a burst (`\\x1b[A` for up-arrow,
    etc.) that arrives within microseconds. A bare ESC keypress is followed
    by nothing. prompt_toolkit uses the same timeout-based disambiguation
    via its `Application.ttimeoutlen` setting.

    Any follow-up bytes are drained so they don't leak into the next
    read or trigger a second match.
    """
    follow_ready, _, _ = select.select([stream], [], [], timeout)
    if not follow_ready:
        return True
    # Drain the rest of the sequence (non-blocking; typically 2-3 more bytes).
    while True:
        more, _, _ = select.select([stream], [], [], 0)
        if not more:
            return False
        stream.read(1)


def _watch_for_esc_unix(
    orch: Orchestrator, stop_event: threading.Event
) -> None:  # pragma: no cover -- requires real TTY + termios
    """Unix watcher: puts stdin in cbreak mode and polls for ESC.

    On a real ESC keypress (0x1b followed by no other bytes within the
    disambiguation window), calls orch.abort() and exits. Multi-byte
    escape sequences (arrow keys, Alt-combos) are silently drained.
    Terminal state is always restored in the finally clause.
    """
    import termios
    import tty

    fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
    except termios.error:
        # Not a real TTY (CI, headless) -- nothing to watch.
        return
    try:
        tty.setcbreak(fd)
        while not stop_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], _WATCHER_POLL_INTERVAL)
            if not ready:
                continue
            ch = sys.stdin.read(1)
            if ch == "\x1b" and _is_lone_escape_unix(sys.stdin):
                orch.abort()
                _con.console.print("\n[yellow]ESC pressed -- aborting task...[/yellow]")
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _watch_for_esc_win(
    orch: Orchestrator, stop_event: threading.Event
) -> None:  # pragma: no cover -- requires real Windows console
    """Windows watcher: polls msvcrt.kbhit() for ESC."""
    import msvcrt

    while not stop_event.is_set():
        if not msvcrt.kbhit():
            time.sleep(_WATCHER_POLL_INTERVAL_WIN)
            continue
        ch = msvcrt.getch()
        if ch == b"\x1b":
            orch.abort()
            _con.console.print("\n[yellow]ESC pressed -- aborting task...[/yellow]")
            return


@contextlib.contextmanager
def _esc_aborts(orch: Orchestrator):
    """Context manager: while active, ESC keypress calls orch.abort().

    No-op when stdin isn't a TTY (CI, piped input). Uses a background daemon
    thread that polls stdin with a short timeout so we can stop it cleanly on
    exit. Terminal state is restored by the watcher's own finally clause.
    """
    if not sys.stdin.isatty():
        yield
        return

    stop_event = threading.Event()
    target = _watch_for_esc_win if sys.platform == "win32" else _watch_for_esc_unix
    thread = threading.Thread(target=target, args=(orch, stop_event), daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=_WATCHER_STOP_TIMEOUT)
