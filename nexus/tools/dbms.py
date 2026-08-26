"""SQLite DBMS tool — schema + query inside the workspace only."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List

from .base import Risk, ToolRegistry, ToolResult
from .paths import in_workspace

_WRITE = ("insert", "update", "delete", "drop", "alter", "create", "replace", "pragma")


class DbmsTools:
    def __init__(self, workspace: Path) -> None:
        self.root = Path(workspace).resolve()

    def _db(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        p = p.resolve()
        if not in_workspace(p, self.root):
            raise ValueError("db path escapes workspace")
        if p.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
            raise ValueError("only .db / .sqlite / .sqlite3 allowed")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def sqlite_exec(self, db_path: str, sql: str) -> ToolResult:
        try:
            db = self._db(db_path)
            sql = (sql or "").strip()
            if not sql:
                return ToolResult(False, error="empty SQL")
            low = sql.lower().lstrip()
            con = sqlite3.connect(str(db))
            con.row_factory = sqlite3.Row
            try:
                cur = con.execute(sql)
                if low.startswith("select") or low.startswith("pragma") or low.startswith("with"):
                    rows = cur.fetchmany(200)
                    cols = [d[0] for d in cur.description] if cur.description else []
                    lines = [" | ".join(cols)] if cols else []
                    for r in rows:
                        lines.append(" | ".join("" if v is None else str(v) for v in r))
                    body = "\n".join(lines) or "(0 rows)"
                    return ToolResult(True, output=f"{db.relative_to(self.root)} → {len(rows)} row(s)\n{body}")
                con.commit()
                return ToolResult(True, output=f"OK rowcount={cur.rowcount} db={db.relative_to(self.root)}")
            finally:
                con.close()
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, error=f"sqlite_exec: {e}")

    def sqlite_schema(self, db_path: str) -> ToolResult:
        try:
            db = self._db(db_path)
            if not db.exists():
                return ToolResult(False, error=f"no such db: {db_path}")
            con = sqlite3.connect(str(db))
            try:
                tabs = con.execute(
                    "SELECT name, sql FROM sqlite_master WHERE type IN ('table','index','view') "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY type, name"
                ).fetchall()
                lines = [f"{n}:\n{s or ''}" for n, s in tabs]
                return ToolResult(True, output="\n\n".join(lines) or "(empty schema)")
            finally:
                con.close()
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, error=f"sqlite_schema: {e}")

    def register(self, reg: ToolRegistry) -> None:
        S = {"type": "string"}
        who = ["coder", "worker", "supervisor", "researcher", "solo"]
        reg.add("sqlite_exec",
                "Run ONE SQL statement on a workspace SQLite file (.db/.sqlite). "
                "CREATE/INSERT/UPDATE/SELECT. Path must stay inside the workspace.",
                {"type": "object", "properties": {"db_path": S, "sql": S},
                 "required": ["db_path", "sql"]},
                self.sqlite_exec, Risk.WRITE, agents=who)
        reg.add("sqlite_schema",
                "Show CREATE statements for tables/indexes in a workspace SQLite file.",
                {"type": "object", "properties": {"db_path": S}, "required": ["db_path"]},
                self.sqlite_schema, Risk.READ_ONLY, agents=who)
