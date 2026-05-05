import re

from agent.models import Plan, Step, FileEdit, Answer


def parse_planner_response(text: str) -> Plan | Answer:
    plan_match = re.search(r"##\s+Plan:", text)
    if plan_match:
        return parse_plan(text)

    answer_match = re.search(r"##\s+Answer\s*\n(.+?)(?=\n##\s|\Z)", text, re.DOTALL)
    if answer_match:
        return Answer(text=answer_match.group(1).strip())

    return parse_plan(text)


def parse_plan(text: str) -> Plan:
    goal_match = re.search(r"##\s+Plan:\s*(.+)", text)
    goal = goal_match.group(1).strip() if goal_match else ""

    step_pattern = re.compile(
        r"###\s+Step\s+(\d+):\s*(.+?)(?=\n###\s+Step|\Z)",
        re.DOTALL,
    )

    steps = []
    for match in step_pattern.finditer(text):
        step_id = int(match.group(1))
        step_text = match.group(2).strip()
        action = step_text.split("\n")[0].strip()

        files_match = re.search(r"-\s*Files needed:\s*(.+)", step_text)
        files_needed = []
        if files_match:
            files_needed = [f.strip() for f in files_match.group(1).split(",")]

        verify_match = re.search(r"-\s*Verify:\s*(.+)", step_text)
        verify_command = verify_match.group(1).strip() if verify_match else None

        steps.append(
            Step(
                id=step_id,
                action=action,
                files_needed=files_needed,
                verify_command=verify_command,
            )
        )

    return Plan(goal=goal, steps=steps)


def parse_edits(
    text: str, extract_commands: bool = False
) -> list[FileEdit] | tuple[list[FileEdit], list[str]]:
    edits: list[FileEdit] = []

    sr_pattern = re.compile(
        r"^(\S[^\n]*?)\n<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
        re.MULTILINE | re.DOTALL,
    )
    for match in sr_pattern.finditer(text):
        path = match.group(1).strip()
        edits.append(
            FileEdit(
                path=path,
                action="search_replace",
                search=match.group(2),
                replace=match.group(3),
            )
        )

    create_pattern = re.compile(
        r"^CREATE\s+(\S+)\n```\w*\n(.*?)\n```",
        re.MULTILINE | re.DOTALL,
    )
    for match in create_pattern.finditer(text):
        edits.append(
            FileEdit(
                path=match.group(1).strip(),
                action="create",
                content=match.group(2),
            )
        )

    rewrite_pattern = re.compile(
        r"^REWRITE\s+(\S+)\n```\w*\n(.*?)\n```",
        re.MULTILINE | re.DOTALL,
    )
    for match in rewrite_pattern.finditer(text):
        edits.append(
            FileEdit(
                path=match.group(1).strip(),
                action="rewrite",
                content=match.group(2),
            )
        )

    if extract_commands:
        commands = re.findall(r"^RUN:\s*(.+)$", text, re.MULTILINE)
        return edits, commands

    return edits
