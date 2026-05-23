"""CLI subpackage.

Re-exports everything from `agent.cli.main` so existing
``from agent.cli import _foo`` imports keep working. The `main` module
already re-exports the surface from its sibling modules (ui, input,
intent, console) via its own ``__all__``.

The `console` submodule deliberately is NOT re-exported as a name --
submodules in this package do ``from agent.cli import console as _con``
to get the *module* (so tests can ``@patch("agent.cli.console.console")``).
Re-exporting the Console *instance* would mask the module and break that
pattern.
"""

from agent.cli.main import *  # noqa: F401, F403
