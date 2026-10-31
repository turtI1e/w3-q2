"""Storage layer (SQLite). No HTTP and no AI in this file.

Why SQLite: it is a single file, needs no server, and gives us real transactions.
The "UPDATE ... WHERE state = INPUT_REQUIRED" trick is what makes the
cancel-vs-result race safe: only ONE writer can flip the state.
"""
import hashlib
import json
import os
import sqlite3
import threading
from contextlib import contextmanager

DB_PATH = os.getenv("DB_PATH", "agent.db")
ACTIVE_STATE = "TASK_STATE_INPUT_REQUIRED"


class StateConflict(Exception):
    """The task was not in the state we expected (someone else got there first)."""


SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks(
  seq        INTEGER PRIMARY KEY AUTOINCREMENT,
  id         TEXT UNIQUE NOT NULL,
  principal  TEXT NOT NULL,
  context_id TEXT NOT NULL,
  batch_id   TEXT NOT NULL,
  state      TEXT NOT NULL,
  task_json  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS tasks_by_user ON tasks(principal, seq);

CREATE TABLE IF NOT EXISTS messages(
  principal  TEXT NOT NULL,
  message_id TEXT NOT NULL,
  msg_hash   TEXT NOT NULL,
  task_id    TEXT NOT NULL,
  PRIMARY KEY(principal, message_id)
);

-- the "ledger": one row per executed action. The primary key means the same
-- action can never be recorded (i.e. "paid") twice for a task.
CREATE TABLE IF NOT EXISTS executions(
  task_id       TEXT NOT NULL,
  action_id     TEXT NOT NULL,
  package_id    TEXT NOT NULL,
  receipt_nonce TEXT NOT NULL,
  PRIMARY KEY(task_id, action_id)
);

CREATE TABLE IF NOT EXISTS decisions(
  key           TEXT PRIMARY KEY,
  decision_json TEXT NOT NULL
);
"""

_lock = threading.RLock()
_db = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
_db.executescript(SCHEMA)
_db.execute("PRAGMA journal_mode=WAL")
_db.execute("PRAGMA synchronous=NORMAL")


def canon(obj) -> str:
    """Recursively key-sorted, compact JSON."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(obj) -> str:
    return hashlib.sha256(canon(obj).encode("utf-8")).hexdigest()


@contextmanager
def tx():
    with _lock:
        _db.execute("BEGIN IMMEDIATE")
        try:
            yield _db
            _db.execute("COMMIT")
        except BaseException:
            _db.execute("ROLLBACK")
            raise


# ---------- messages (idempotency) ----------
def get_message(principal: str, message_id: str):
    with _lock:
        r = _db.execute(
            "SELECT msg_hash, task_id FROM messages WHERE principal=? AND message_id=?",
            (principal, message_id),
        ).fetchone()
    return (r[0], r[1]) if r else None


# ---------- tasks ----------
def insert_task(principal, task, batch_id, message_id, msg_hash):
    """Store a brand-new task AND its message record in one transaction."""
    with tx() as c:
        c.execute(
            "INSERT INTO tasks(id, principal, context_id, batch_id, state, task_json) VALUES(?,?,?,?,?,?)",
            (task["id"], principal, task["contextId"], batch_id,
             task["status"]["state"], json.dumps(task, ensure_ascii=False)),
        )
        c.execute(
            "INSERT INTO messages(principal, message_id, msg_hash, task_id) VALUES(?,?,?,?)",
            (principal, message_id, msg_hash, task["id"]),
        )


def get_task(principal: str, task_id: str):
    """Only ever returns a task that belongs to `principal` (user isolation)."""
    with _lock:
        r = _db.execute(
            "SELECT task_json FROM tasks WHERE id=? AND principal=?", (task_id, principal)
        ).fetchone()
    return json.loads(r[0]) if r else None


def list_tasks(principal: str, limit: int, offset: int):
    with _lock:
        rows = _db.execute(
            "SELECT task_json FROM tasks WHERE principal=? ORDER BY seq DESC LIMIT ? OFFSET ?",
            (principal, limit, offset),
        ).fetchall()
        total = _db.execute("SELECT COUNT(*) FROM tasks WHERE principal=?", (principal,)).fetchone()[0]
    return [json.loads(r[0]) for r in rows], total


def transition(principal, new_task, executions=(), message_id=None, msg_hash=None):
    """Move an INPUT_REQUIRED task to a terminal state. Atomic.

    Raises StateConflict if the task is no longer INPUT_REQUIRED, so when a
    cancel and a result arrive together exactly one of them wins.
    """
    try:
        with tx() as c:
            cur = c.execute(
                "UPDATE tasks SET state=?, task_json=? WHERE id=? AND principal=? AND state=?",
                (new_task["status"]["state"], json.dumps(new_task, ensure_ascii=False),
                 new_task["id"], principal, ACTIVE_STATE),
            )
            if cur.rowcount != 1:
                raise StateConflict()
            for e in executions:
                c.execute(
                    "INSERT INTO executions(task_id, action_id, package_id, receipt_nonce) VALUES(?,?,?,?)",
                    (new_task["id"], e["actionId"], e["packageId"], e["receiptNonce"]),
                )
            if message_id:
                c.execute(
                    "INSERT INTO messages(principal, message_id, msg_hash, task_id) VALUES(?,?,?,?)",
                    (principal, message_id, msg_hash, new_task["id"]),
                )
    except sqlite3.IntegrityError as e:  # duplicate action / message -> treat as "already done"
        raise StateConflict() from e


# ---------- AI decision cache ----------
def cache_get(key: str):
    with _lock:
        r = _db.execute("SELECT decision_json FROM decisions WHERE key=?", (key,)).fetchone()
    return json.loads(r[0]) if r else None


def cache_put(key: str, decision: dict):
    with _lock:
        _db.execute(
            "INSERT OR REPLACE INTO decisions(key, decision_json) VALUES(?,?)",
            (key, json.dumps(decision, ensure_ascii=False)),
        )
