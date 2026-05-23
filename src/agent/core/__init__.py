"""Core domain: plan-execute-verify state machine + models.

Import classes from their submodules directly, e.g.::

    from agent.core.orchestrator import Orchestrator
    from agent.core.models import Plan, Step

Eager re-exports here would create circular imports with `agent.tools`
(which depends on core.models) and `agent.cli` (which depends on the
whole core stack).
"""
