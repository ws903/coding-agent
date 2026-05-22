import re
import shlex

SAFE_COMMANDS = {
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "file",
    "stat",
    "du",
    "df",
    "find",
    "tree",
    "realpath",
    "dirname",
    "basename",
    "grep",
    "rg",
    "ag",
    "ack",
    "fgrep",
    "egrep",
    "sort",
    "uniq",
    "cut",
    "tr",
    "diff",
    "jq",
    "yq",
    "echo",
    "printf",
    "date",
    "whoami",
    "pwd",
    "env",
    "which",
    "type",
    "true",
    "false",
    "test",
    "python",
    "python3",
    "node",
    "pytest",
    "pip",
    "pip3",
    "npm",
    "npx",
    "cargo",
    "make",
    "git",
    "ruff",
    "sed",
    "awk",
    "tee",
    "mkdir",
    "touch",
    "cp",
    "mv",
}

BLOCKED_PATTERNS = [
    re.compile(r"\brm\s+(-[^\s]*)?-r", re.IGNORECASE),
    re.compile(r"\brm\s+(-[^\s]*)?\s*/", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bdd\b.*\bof=/dev/", re.IGNORECASE),
    re.compile(r"\bchmod\s+777\b"),
    re.compile(r"\bchown\s+-R\b"),
    re.compile(r"\b(shutdown|reboot|halt|poweroff)\b", re.IGNORECASE),
    re.compile(r"\bgit\s+push\s+.*--force\b"),
    re.compile(r"\bgit\s+reset\s+--hard\b"),
    re.compile(r"\bgit\s+clean\s+-[fdx]"),
    re.compile(
        r"\b(DROP\s+(TABLE|DATABASE)|TRUNCATE\s|DELETE\s+FROM)\b", re.IGNORECASE
    ),
    re.compile(r"\b(curl|wget)\b.*\|\s*(bash|sh|zsh|python|python3|perl|ruby|node)\b"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bapt-get\b"),
    re.compile(r"\byum\b"),
    re.compile(r"\bbrew\s+install\b"),
]

DANGEROUS_PIPE_TARGETS = {
    "bash",
    "sh",
    "zsh",
    "python",
    "python3",
    "perl",
    "ruby",
    "node",
}

_PIPE_SPLIT = re.compile(r"\|{1,2}|&&|;")


class CommandBlocked(Exception):
    def __init__(self, command: str, reason: str):
        self.command = command
        self.reason = reason
        super().__init__(f"Blocked: {reason}")


def check_command(command: str) -> tuple[str, str]:
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(command):
            return "block", f"matches blocked pattern: {pattern.pattern}"

    segments = _PIPE_SPLIT.split(command)
    for i, segment in enumerate(segments):
        segment = segment.strip()
        if not segment:
            continue
        for pattern in BLOCKED_PATTERNS:
            if pattern.search(segment):
                return "block", f"segment '{segment}' matches blocked pattern"
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        if not tokens:
            continue
        first_token = tokens[0]
        if i > 0 and first_token in DANGEROUS_PIPE_TARGETS:
            return "block", f"dangerous pipe target: {first_token}"

    first_segment = segments[0].strip()
    if first_segment:
        try:
            tokens = shlex.split(first_segment)
        except ValueError:
            tokens = first_segment.split()
        if tokens and tokens[0] in SAFE_COMMANDS:
            return "allow", "safe command"

    return "ask", "unknown command"
