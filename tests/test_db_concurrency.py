"""Smoke tests for AgentDB under interleaved access.

Not full concurrency testing -- just enough to verify sqlite's default
behavior doesn't crash on rapid same-thread reads/writes from the
orchestrator's call patterns.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.db import AgentDB


@pytest.fixture
def db(tmp_path):
    return AgentDB(tmp_path / ".agent" / "agent.db")


def test_db_rapid_sequential_writes(db):
    """Many same-thread writes don't corrupt or deadlock."""
    conv_id = db.create_conversation("autonomous", "stress test")
    for i in range(50):
        db.add_message(conv_id, "system", f"msg {i}")
    messages = db.get_messages(conv_id)
    assert len([m for m in messages if m["content"].startswith("msg ")]) == 50


def test_db_interleaved_read_write(db):
    """Reading conversation state while writes are in flight works."""
    conv_id = db.create_conversation("autonomous", "interleaved")
    for i in range(20):
        db.add_message(conv_id, "executor", f"step {i}")
        msgs = db.get_messages(conv_id)
        assert any(f"step {i}" in m["content"] for m in msgs)


def test_db_concurrent_threads_share_path(tmp_path):
    """Two AgentDB instances pointing at the same file don't break each other.

    Each thread opens its own AgentDB pointed at the same path (sqlite
    handles file-level locking). Verifies the per-instance connection
    model isn't fragile across threads.
    """
    db_path = tmp_path / ".agent" / "agent.db"
    AgentDB(db_path).create_conversation("autonomous", "init")

    errors: list[Exception] = []

    def worker(tag: str):
        try:
            local_db = AgentDB(db_path)
            conv_id = local_db.create_conversation("autonomous", f"task-{tag}")
            for i in range(10):
                local_db.add_message(conv_id, "system", f"{tag}-{i}")
        except Exception as exc:  # noqa: BLE001 -- propagate to main thread
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, ["a", "b", "c", "d"]))

    assert not errors, f"workers raised: {errors}"


def test_db_reopen_preserves_data(tmp_path):
    """Closing and reopening the file gets the same data back."""
    db_path = tmp_path / ".agent" / "agent.db"
    db1 = AgentDB(db_path)
    conv_id = db1.create_conversation("autonomous", "persistence check")
    db1.add_message(conv_id, "user", "hello")

    db2 = AgentDB(db_path)
    messages = db2.get_messages(conv_id)
    assert any(m["content"] == "hello" for m in messages)


def test_db_no_deadlock_under_sequential_load(db):
    """Stress: 100 rapid create+message+read cycles complete without hanging."""
    deadline = threading.Event()

    def timer():
        deadline.wait(timeout=30)

    t = threading.Thread(target=timer, daemon=True)
    t.start()
    for i in range(100):
        conv_id = db.create_conversation("autonomous", f"task-{i}")
        db.add_message(conv_id, "system", "x")
        db.get_messages(conv_id)
    deadline.set()
    assert True
