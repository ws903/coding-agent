import sqlite3
import uuid
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS config (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    started_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mode        TEXT CHECK(mode IN ('interactive', 'autonomous')),
    task        TEXT,
    status      TEXT CHECK(status IN ('active', 'completed', 'failed', 'aborted'))
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id     TEXT REFERENCES conversations(id),
    role        TEXT CHECK(role IN ('user', 'planner', 'executor', 'verifier', 'system')),
    content     TEXT,
    timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS plans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id     TEXT REFERENCES conversations(id),
    version     INTEGER DEFAULT 1,
    content     TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edits (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id     TEXT REFERENCES conversations(id),
    step_id     INTEGER,
    file_path   TEXT,
    edit_type   TEXT CHECK(edit_type IN ('create', 'rewrite', 'search_replace')),
    before      TEXT,
    after       TEXT,
    timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class AgentDB:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def set_config(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO config (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_config(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM config WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def create_conversation(self, mode: str, task: str) -> str:
        conv_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO conversations (id, mode, task, status) VALUES (?, ?, ?, 'active')",
            (conv_id, mode, task),
        )
        self.conn.commit()
        return conv_id

    def get_conversation(self, conv_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_conversation_status(self, conv_id: str, status: str) -> None:
        self.conn.execute(
            "UPDATE conversations SET status = ? WHERE id = ?", (status, conv_id)
        )
        self.conn.commit()

    def add_message(self, conv_id: str, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO messages (conv_id, role, content) VALUES (?, ?, ?)",
            (conv_id, role, content),
        )
        self.conn.commit()

    def get_messages(self, conv_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE conv_id = ? ORDER BY id", (conv_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def save_plan(self, conv_id: str, version: int, content: str) -> None:
        self.conn.execute(
            "INSERT INTO plans (conv_id, version, content) VALUES (?, ?, ?)",
            (conv_id, version, content),
        )
        self.conn.commit()

    def get_plans(self, conv_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM plans WHERE conv_id = ? ORDER BY version", (conv_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def save_edit(
        self, conv_id: str, step_id: int, file_path: str, edit_type: str,
        before: str | None, after: str | None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO edits (conv_id, step_id, file_path, edit_type, before, after) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (conv_id, step_id, file_path, edit_type, before, after),
        )
        self.conn.commit()

    def get_edits(self, conv_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM edits WHERE conv_id = ? ORDER BY id", (conv_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.conn.close()
