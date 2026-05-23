# src/agent/console.py
"""Shared Rich Console instance.

A single module-level console is imported by every UI module so that they
share output buffering, theme, and TTY detection. Putting it in its own
module avoids circular imports between `cli.py` and the extracted UI/input
modules that all need to print.
"""

from rich.console import Console

console = Console()
