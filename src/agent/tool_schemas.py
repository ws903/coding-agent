# src/agent/tool_schemas.py
"""OpenAI-format tool schemas for the executor.

Read tools (read_file, list_files, search_text) execute live against
the filesystem during the executor loop. Edit tools (create_file,
edit_file, replace_file) record FileEdit intents that the orchestrator
applies + lint-gates after the executor returns. run_command queues
shell commands for the orchestrator to run via the sandbox.
"""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file. Returns the file's text "
                "with line numbers prepended."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the project root.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional 1-indexed start line.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional 1-indexed end line (inclusive).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory (recursive).",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path relative to project root. Defaults to '.'",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Optional glob pattern (e.g. '*.py') to filter by filename.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": (
                "Search for a literal substring across all files in the project. "
                "Returns matching file path, line number, and line content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Literal text to search for.",
                    },
                    "path_filter": {
                        "type": "string",
                        "description": "Optional filename glob (e.g. '*.py').",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": (
                "Record an edit to create a new file. The edit is applied + lint-gated "
                "by the orchestrator after the executor returns; do not expect to read "
                "the file back during this turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the new file, relative to project root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file contents.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Record a search/replace edit on an existing file. The search string "
                "must match a unique substring (whitespace is normalized as a fallback). "
                "The edit is applied + lint-gated by the orchestrator after this turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "search": {
                        "type": "string",
                        "description": "Exact text to find. Include enough context to be unique.",
                    },
                    "replace": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["path", "search", "replace"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_file",
            "description": (
                "Record a whole-file rewrite. Prefer edit_file for targeted changes; "
                "use replace_file only for small files (<300 lines) or wholesale rewrites."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Queue a shell command to be run by the orchestrator after edits are applied. "
                "Subject to the command allowlist (no rm -rf, sudo, force push, etc.)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill",
            "description": (
                "Read the full instructions for a named skill from "
                ".agent/skills/. Use this only when a skill from the catalog "
                "applies to the current step. Returns the skill body as text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name as listed in the 'Available skills' section.",
                    },
                },
                "required": ["name"],
            },
        },
    },
]
