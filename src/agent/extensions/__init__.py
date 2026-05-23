"""User-extensible filesystem-discovered managers.

All three follow the same pattern (ADR 0004): read markdown/JSON from a
known location at startup, expose a `catalog_section()` summary, load full
bodies on demand.

Import managers directly::

    from agent.extensions.skills import SkillsManager
    from agent.extensions.agents import AgentsManager
    from agent.extensions.mcp import MCPManager, load_mcp_config
"""
