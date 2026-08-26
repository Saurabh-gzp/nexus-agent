"""Long-term memory: sessions, messages, task summaries, user preferences, facts.

Retrieval = recent-window + semantic search (no raw conversation dumps).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


class MemoryStore:
    def __init__(self, db_path: Path, llm=None, rag=None):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self.rag = rag
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init()
        self.session_id = ""

    def _init(self) -> None:
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, title TEXT, created REAL, updated REAL,
            goal TEXT, status TEXT DEFAULT 'active', meta TEXT DEFAULT '{}');
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT,
            agent TEXT, content TEXT, created REAL, tokens INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY, session_id TEXT, title TEXT, agent TEXT,
            status TEXT, result TEXT, score REAL, created REAL, finished REAL,
            meta TEXT DEFAULT '{}');
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, key TEXT,
            value TEXT, importance REAL DEFAULT 0.5, created REAL, hits INTEGER DEFAULT 0);
        CREATE INDEX IF NOT EXISTS idx_msg_sess ON messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_task_sess ON tasks(session_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_fact_key ON facts(kind, key);
        """)
        self.conn.commit()

    # ---------------- sessions ----------------
    def start_session(self, goal: str = "", title: str = "") -> str:
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            self.conn.execute(
                "INSERT INTO sessions(id,title,created,updated,goal) VALUES (?,?,?,?,?)",
                (sid, title or (goal[:60] if goal else "session"), now, now, goal))
            self.conn.commit()
        self.session_id = sid
        return sid

    def resume_session(self, sid: str) -> bool:
        row = self._exec("SELECT id FROM sessions WHERE id=?", (sid,)).fetchone()
        if row:
            self.session_id = sid
            return True
        return False

    def resolve_session(self, ref: str) -> Optional[str]:
        """Resolve a session by NUMBER (as shown in /sessions), by id, or by
        id-prefix. Returns the session id, or None if not found.

        Fix (v1.5): session ids are UUID hex — a prefix can be ALL DIGITS
        (e.g. '123456'), which `isdigit()` used to misread as a session number
        and return None. Exact id is checked FIRST, then the number path,
        and an out-of-range number falls through to prefix matching.
        """
        ref = (ref or "").strip()
        if not ref:
            return None
        # 1) exact id first (a 12-char hex id may be all digits)
        if self._exec("SELECT id FROM sessions WHERE id=?",
                             (ref,)).fetchone():
            return ref
        # 2) number (as shown by /sessions, newest first)
        if ref.isdigit():
            n = int(ref)
            if n < 1:
                return None
            rows = self.list_sessions(n)          # same order as /sessions
            if n <= len(rows):
                return rows[n - 1]["id"]
            # out-of-range number → fall through to prefix matching below
        # 3) id prefix (e.g. "/resume debf")
        row = self._exec(
            "SELECT id FROM sessions WHERE id LIKE ? ORDER BY updated DESC LIMIT 1",
            (ref + "%",)).fetchone()
        return row["id"] if row else None

    def latest_session(self) -> Optional[str]:
        row = self._exec("SELECT id FROM sessions ORDER BY updated DESC LIMIT 1").fetchone()
        return row["id"] if row else None

    def _exec(self, sql: str, args: tuple = ()):
        with self._lock:
            return self.conn.execute(sql, args)

    def list_sessions(self, limit: int = 20) -> List[dict]:
        rows = self._exec(
            "SELECT s.id, s.title, s.goal, s.created, s.status, "
            "(SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) AS msgs "
            "FROM sessions s ORDER BY s.updated DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ---------------- messages ----------------
    def add_message(self, role: str, content: str, agent: str = "", tokens: int = 0) -> None:
        if not self.session_id:
            self.start_session()
        with self._lock:
            self.conn.execute(
                "INSERT INTO messages(session_id,role,agent,content,created,tokens) VALUES (?,?,?,?,?,?)",
                (self.session_id, role, agent, content, time.time(), tokens))
            self.conn.execute("UPDATE sessions SET updated=? WHERE id=?",
                              (time.time(), self.session_id))
            self.conn.commit()

    def recent_messages(self, limit: int = 12, session_id: Optional[str] = None) -> List[dict]:
        sid = session_id or self.session_id
        if not sid:
            return []
        rows = self._exec(
            "SELECT role, content, agent, created FROM messages WHERE session_id=? "
            "ORDER BY id DESC LIMIT ?", (sid, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ---------------- tasks ----------------
    def save_task(self, task_id: str, title: str, agent: str, status: str,
                  result: str = "", score: float = 0.0, meta: Optional[dict] = None) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO tasks(id,session_id,title,agent,status,result,score,"
                "created,finished,meta) VALUES (?,?,?,?,?,?,?,"
                "COALESCE((SELECT created FROM tasks WHERE id=?),?),?,?)",
                (task_id, self.session_id, title, agent, status, result[:8000], score,
                 task_id, time.time(), time.time(), json.dumps(meta or {})))
            self.conn.commit()
        # semantic memory
        if self.rag and status == "done" and result:
            try:
                self.rag.index_text(f"TASK: {title}\nAGENT: {agent}\nRESULT:\n{result[:4000]}",
                                    source=f"memory://task/{task_id}",
                                    meta={"kind": "task_summary"}, collection="memory")
            except Exception:
                pass

    def past_tasks(self, limit: int = 10) -> List[dict]:
        rows = self._exec(
            "SELECT id,title,agent,status,score,created FROM tasks ORDER BY created DESC LIMIT ?",
            (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ---------------- facts / preferences ----------------
    def remember(self, kind: str, key: str, value: str, importance: float = 0.5) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO facts(kind,key,value,importance,created) VALUES (?,?,?,?,?) "
                "ON CONFLICT(kind,key) DO UPDATE SET value=excluded.value, "
                "importance=excluded.importance, created=excluded.created",
                (kind, key, value, importance, time.time()))
            self.conn.commit()
        if self.rag:
            try:
                self.rag.index_text(f"{kind}: {key} = {value}",
                                    source=f"memory://fact/{kind}/{key}",
                                    meta={"kind": "fact"}, collection="memory")
            except Exception:
                pass

    def recall(self, kind: Optional[str] = None, limit: int = 30) -> List[dict]:
        q = "SELECT kind,key,value,importance FROM facts"
        args: tuple = ()
        if kind:
            q += " WHERE kind=?"
            args = (kind,)
        q += " ORDER BY importance DESC, created DESC LIMIT ?"
        rows = self._exec(q, args + (limit,)).fetchall()
        return [dict(r) for r in rows]

    def forget(self, kind: str, key: str) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM facts WHERE kind=? AND key=?", (kind, key))
            self.conn.commit()
            return cur.rowcount > 0

    # ---------------- context building ----------------
    def build_context(self, query: str, recent: int = 8, semantic_k: int = 4) -> str:
        parts: List[str] = []
        prefs = self.recall("preference", 8)
        if prefs:
            parts.append("### User preferences\n" +
                         "\n".join(f"- {p['key']}: {p['value']}" for p in prefs))
        msgs = self.recent_messages(recent)
        if msgs:
            convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in msgs[-recent:])
            parts.append(f"### Recent conversation\n{convo}")
        if self.rag and query:
            try:
                docs = self.rag.retrieve(query, semantic_k, collection="memory")
                if docs:
                    parts.append("### Relevant past work\n" +
                                 "\n".join(f"- {d.text[:280]}" for d in docs))
            except Exception:
                pass
        return "\n\n".join(parts)

    def stats(self) -> dict:
        def c(t: str) -> int:
            return int(self._exec(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
        return {"sessions": c("sessions"), "messages": c("messages"),
                "tasks": c("tasks"), "facts": c("facts"), "current": self.session_id}

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
