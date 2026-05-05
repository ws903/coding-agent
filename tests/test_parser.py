from agent.models import Answer, Plan
from agent.parser import parse_plan, parse_edits, parse_planner_response


class TestParsePlan:
    def test_basic_plan(self):
        text = """## Plan: Add authentication

### Step 1: Create User model
- Files needed: src/models.py, src/config.py
- Verify: pytest tests/test_models.py

### Step 2: Add login route
- Files needed: src/routes.py
- Verify: pytest tests/test_auth.py
"""
        plan = parse_plan(text)
        assert plan.goal == "Add authentication"
        assert len(plan.steps) == 2
        assert plan.steps[0].id == 1
        assert plan.steps[0].action == "Create User model"
        assert plan.steps[0].files_needed == ["src/models.py", "src/config.py"]
        assert plan.steps[0].verify_command == "pytest tests/test_models.py"
        assert plan.steps[1].id == 2
        assert plan.steps[1].action == "Add login route"

    def test_plan_no_verify(self):
        text = """## Plan: Simple change

### Step 1: Update readme
- Files needed: README.md
"""
        plan = parse_plan(text)
        assert plan.steps[0].verify_command is None

    def test_plan_with_extra_content(self):
        text = """Some preamble text the model might add.

## Plan: Fix the bug

Here is my thinking about this...

### Step 1: Fix the handler
- Files needed: src/handler.py
- Verify: pytest

More explanation text here.

### Step 2: Update tests
- Files needed: tests/test_handler.py
- Verify: pytest tests/test_handler.py
"""
        plan = parse_plan(text)
        assert plan.goal == "Fix the bug"
        assert len(plan.steps) == 2

    def test_empty_returns_empty_plan(self):
        plan = parse_plan("no plan here")
        assert plan.goal == ""
        assert len(plan.steps) == 0


class TestParseEdits:
    def test_search_replace_block(self):
        text = """I'll make the following changes:

src/main.py
<<<<<<< SEARCH
def hello():
    return "hi"
=======
def hello():
    return "hello world"
>>>>>>> REPLACE
"""
        edits = parse_edits(text)
        assert len(edits) == 1
        assert edits[0].path == "src/main.py"
        assert edits[0].action == "search_replace"
        assert edits[0].search == 'def hello():\n    return "hi"'
        assert edits[0].replace == 'def hello():\n    return "hello world"'

    def test_multiple_search_replace(self):
        text = """
src/a.py
<<<<<<< SEARCH
old_a
=======
new_a
>>>>>>> REPLACE

src/b.py
<<<<<<< SEARCH
old_b
=======
new_b
>>>>>>> REPLACE
"""
        edits = parse_edits(text)
        assert len(edits) == 2
        assert edits[0].path == "src/a.py"
        assert edits[1].path == "src/b.py"

    def test_create_file_block(self):
        text = """
CREATE src/new_file.py
```
def new_function():
    pass
```
"""
        edits = parse_edits(text)
        assert len(edits) == 1
        assert edits[0].path == "src/new_file.py"
        assert edits[0].action == "create"
        assert "def new_function" in edits[0].content

    def test_rewrite_file_block(self):
        text = """
REWRITE src/small.py
```
def updated():
    return True
```
"""
        edits = parse_edits(text)
        assert len(edits) == 1
        assert edits[0].path == "src/small.py"
        assert edits[0].action == "rewrite"
        assert "def updated" in edits[0].content

    def test_commands_extracted(self):
        text = """
RUN: pip install flask
RUN: pytest tests/
"""
        _, commands = parse_edits(text, extract_commands=True)
        assert len(commands) == 2
        assert commands[0] == "pip install flask"
        assert commands[1] == "pytest tests/"

    def test_no_edits_returns_empty(self):
        edits = parse_edits("just some explanation text")
        assert len(edits) == 0


class TestParsePlannerResponse:
    def test_answer_only_returns_answer(self):
        text = """## Answer

This app is a local coding agent. It uses Ollama for inference.
"""
        result = parse_planner_response(text)
        assert isinstance(result, Answer)
        assert "local coding agent" in result.text

    def test_plan_only_returns_plan(self):
        text = """## Plan: Add health check

### Step 1: Create endpoint
- Files needed: src/app.py
"""
        result = parse_planner_response(text)
        assert isinstance(result, Plan)
        assert result.goal == "Add health check"
        assert len(result.steps) == 1

    def test_both_prefers_plan(self):
        text = """## Answer
some text

## Plan: Do the thing

### Step 1: Do it
- Files needed: x.py
"""
        result = parse_planner_response(text)
        assert isinstance(result, Plan)
        assert result.goal == "Do the thing"

    def test_neither_falls_back_to_empty_plan(self):
        result = parse_planner_response("just rambling, no headers")
        assert isinstance(result, Plan)
        assert result.goal == ""
        assert len(result.steps) == 0

    def test_answer_strips_whitespace(self):
        text = "## Answer\n\n   Hello world.   \n\n"
        result = parse_planner_response(text)
        assert isinstance(result, Answer)
        assert result.text == "Hello world."
