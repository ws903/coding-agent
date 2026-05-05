from pathlib import Path

from agent.db import AgentDB


def make_db(tmp_path: Path) -> AgentDB:
    return AgentDB(tmp_path / ".agent" / "agent.db")


def test_init_creates_tables(tmp_path):
    db = make_db(tmp_path)
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {row[0] for row in tables}
    assert "config" in table_names
    assert "conversations" in table_names
    assert "messages" in table_names
    assert "plans" in table_names
    assert "edits" in table_names


def test_config_get_set(tmp_path):
    db = make_db(tmp_path)
    db.set_config("planner_model", "qwen3:14b")
    assert db.get_config("planner_model") == "qwen3:14b"


def test_config_get_default(tmp_path):
    db = make_db(tmp_path)
    assert db.get_config("nonexistent", "fallback") == "fallback"


def test_config_upsert(tmp_path):
    db = make_db(tmp_path)
    db.set_config("model", "a")
    db.set_config("model", "b")
    assert db.get_config("model") == "b"


def test_create_conversation(tmp_path):
    db = make_db(tmp_path)
    conv_id = db.create_conversation("interactive", "Fix the bug")
    conv = db.get_conversation(conv_id)
    assert conv["mode"] == "interactive"
    assert conv["task"] == "Fix the bug"
    assert conv["status"] == "active"


def test_add_message(tmp_path):
    db = make_db(tmp_path)
    conv_id = db.create_conversation("interactive", "task")
    db.add_message(conv_id, "user", "hello")
    db.add_message(conv_id, "planner", "plan output")
    messages = db.get_messages(conv_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "planner"


def test_save_plan(tmp_path):
    db = make_db(tmp_path)
    conv_id = db.create_conversation("autonomous", "task")
    db.save_plan(conv_id, 1, "## Plan\n### Step 1: do thing")
    db.save_plan(conv_id, 2, "## Revised Plan\n### Step 1: different")
    plans = db.get_plans(conv_id)
    assert len(plans) == 2
    assert plans[0]["version"] == 1
    assert plans[1]["version"] == 2


def test_save_edit(tmp_path):
    db = make_db(tmp_path)
    conv_id = db.create_conversation("autonomous", "task")
    db.save_edit(conv_id, 1, "src/main.py", "create", before=None, after="print('hi')")
    edits = db.get_edits(conv_id)
    assert len(edits) == 1
    assert edits[0]["file_path"] == "src/main.py"
    assert edits[0]["edit_type"] == "create"


def test_update_conversation_status(tmp_path):
    db = make_db(tmp_path)
    conv_id = db.create_conversation("interactive", "task")
    db.update_conversation_status(conv_id, "completed")
    conv = db.get_conversation(conv_id)
    assert conv["status"] == "completed"
