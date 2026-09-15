"""对话会话存储：sessions 与 messages 两张表，与文献库账本同库（library.db）。

设计要点：
- 会话是「轻资产」：标题取首条消息截断，元数据只有时间戳；消息全文入库，
  重启/换端都能恢复完整上下文；
- 与 Library 相同的连接模式：每次请求自己开、用完即关（SQLite 文件库无并发写竞争）；
- check_same_thread=False：连接的创建与关闭可能落在不同线程池线程（同 Library）。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    role          TEXT NOT NULL,
    content       TEXT NOT NULL,
    intent        TEXT,
    understanding TEXT,
    notes         TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id);
"""


class ChatStore:
    """chat_sessions / chat_messages 的读写入口。"""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # ---- 会话 ----

    def create_session(self, title: str = "新对话") -> str:
        sid = uuid.uuid4().hex[:12]
        now = datetime.now(UTC).isoformat(timespec="seconds")
        self.conn.execute(
            "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (sid, title[:30] or "新对话", now, now),
        )
        self.conn.commit()
        return sid

    def list_sessions(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT s.id, s.title, s.created_at, s.updated_at,"
            " (SELECT COUNT(*) FROM chat_messages m WHERE m.session_id = s.id) AS message_count"
            " FROM chat_sessions s ORDER BY s.updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def touch_session(self, sid: str, *, title: str | None = None) -> None:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        if title:
            self.conn.execute(
                "UPDATE chat_sessions SET updated_at = ?, title = ? WHERE id = ?",
                (now, title[:30], sid),
            )
        else:
            self.conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (now, sid)
            )
        self.conn.commit()

    def delete_session(self, sid: str) -> bool:
        cur = self.conn.execute("DELETE FROM chat_sessions WHERE id = ?", (sid,))
        self.conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (sid,))
        self.conn.commit()
        return cur.rowcount > 0

    # ---- 消息 ----

    def append_message(
        self,
        sid: str,
        *,
        role: str,
        content: str,
        intent: str = "",
        understanding: str = "",
        notes: list[str] | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        self.conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, intent, understanding, notes, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sid, role, content, intent, understanding, json.dumps(notes or [], ensure_ascii=False), now),
        )
        self.conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (now, sid)
        )
        self.conn.commit()

    def get_messages(self, sid: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, role, content, intent, understanding, notes, created_at"
            " FROM chat_messages WHERE session_id = ? ORDER BY id",
            (sid,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["notes"] = json.loads(d["notes"] or "[]")
            out.append(d)
        return out
