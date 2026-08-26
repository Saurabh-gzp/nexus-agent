"""Lightweight vector store — SQLite + numpy (Termux friendly, no pgvector/qdrant needed).

Hybrid retrieval = cosine similarity (dense) + BM25-ish keyword score (sparse).
If Qdrant/Chroma is ever needed, just implement the VectorStore interface.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


@dataclass
class Document:
    id: str
    text: str
    source: str
    meta: dict
    score: float = 0.0

    def cite(self) -> str:
        loc = self.meta.get("chunk_index")
        return f"{self.source}" + (f"#chunk{loc}" if loc is not None else "")


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]{2,}", text.lower())


class VectorStore:
    """SQLite-backed store with in-memory numpy matrix for fast search."""

    def __init__(self, db_path: Path, dim: int = 1024):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()
        self._matrix = None
        self._ids: List[str] = []
        self._dirty = True

    def _init_db(self) -> None:
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            collection TEXT NOT NULL DEFAULT 'default',
            source TEXT,
            text TEXT NOT NULL,
            meta TEXT DEFAULT '{}',
            embedding BLOB,
            created REAL
        );
        CREATE INDEX IF NOT EXISTS idx_coll ON chunks(collection);
        CREATE INDEX IF NOT EXISTS idx_src ON chunks(source);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    @staticmethod
    def make_id(source: str, text: str, idx: int = 0) -> str:
        return hashlib.sha1(f"{source}:{idx}:{text[:200]}".encode()).hexdigest()[:20]

    def add(self, texts: List[str], embeddings: List[List[float]], sources: List[str],
            metas: Optional[List[dict]] = None, collection: str = "default") -> int:
        metas = metas or [{} for _ in texts]
        rows = []
        for t, e, s, m in zip(texts, embeddings, sources, metas):
            cid = self.make_id(s, t, m.get("chunk_index", 0))
            blob = (np.asarray(e, dtype=np.float32).tobytes() if np is not None
                    else json.dumps(e).encode())
            rows.append((cid, collection, s, t, json.dumps(m), blob, time.time()))
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO chunks(id,collection,source,text,meta,embedding,created) "
                "VALUES (?,?,?,?,?,?,?)", rows)
            self._conn.commit()
            self._dirty = True
        return len(rows)

    def delete_source(self, source: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM chunks WHERE source=?", (source,))
            self._conn.commit()
            self._dirty = True
            return cur.rowcount

    def clear(self, collection: Optional[str] = None) -> int:
        with self._lock:
            if collection:
                cur = self._conn.execute("DELETE FROM chunks WHERE collection=?", (collection,))
            else:
                cur = self._conn.execute("DELETE FROM chunks")
            self._conn.commit()
            self._dirty = True
            return cur.rowcount

    def count(self, collection: Optional[str] = None) -> int:
        q = "SELECT COUNT(*) FROM chunks" + (" WHERE collection=?" if collection else "")
        with self._lock:
            cur = self._conn.execute(q, (collection,) if collection else ())
            return int(cur.fetchone()[0])

    def sources(self) -> List[Tuple[str, int]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY 2 DESC")
            return list(cur.fetchall())

    def has_source(self, source: str, mtime: Optional[float] = None) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "SELECT meta FROM chunks WHERE source=? LIMIT 1", (source,))
            row = cur.fetchone()
        if not row:
            return False
        if mtime is None:
            return True
        try:
            return abs(float(json.loads(row[0]).get("mtime", 0)) - mtime) < 1.0
        except Exception:
            return False

    # ------------------------------------------------------------------
    def _load_matrix(self, collection: Optional[str] = None):
        if np is None:
            return None, [], []
        q = "SELECT id, source, text, meta, embedding FROM chunks"
        args: tuple = ()
        if collection:
            q += " WHERE collection=?"
            args = (collection,)
        rows = self._conn.execute(q, args).fetchall()
        if not rows:
            return None, [], []
        vecs, records = [], []
        for cid, src, text, meta, blob in rows:
            if not blob:
                continue
            v = np.frombuffer(blob, dtype=np.float32)
            if v.size == 0:
                continue
            vecs.append(v)
            records.append((cid, src, text, meta))
        if not vecs:
            return None, [], []
        dim = max(v.size for v in vecs)
        mat = np.zeros((len(vecs), dim), dtype=np.float32)
        for i, v in enumerate(vecs):
            mat[i, :v.size] = v
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms, records, norms

    def search(self, query_embedding: List[float], top_k: int = 6,
               collection: Optional[str] = None, query_text: str = "",
               min_score: float = 0.0) -> List[Document]:
        with self._lock:
            mat, records, _ = self._load_matrix(collection)
        if mat is None or not records:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        if q.size < mat.shape[1]:
            q = np.pad(q, (0, mat.shape[1] - q.size))
        q = q[:mat.shape[1]]
        qn = np.linalg.norm(q) or 1.0
        dense = mat @ (q / qn)

        # sparse keyword boost (hybrid)
        if query_text:
            qt = set(_tokens(query_text))
            if qt:
                boost = np.array([
                    len(qt & set(_tokens(r[2][:1200]))) / (len(qt) or 1) for r in records
                ], dtype=np.float32)
                dense = 0.82 * dense + 0.18 * boost

        k = min(top_k, len(records))
        idx = np.argpartition(-dense, k - 1)[:k]
        idx = idx[np.argsort(-dense[idx])]
        out: List[Document] = []
        for i in idx:
            sc = float(dense[i])
            if sc < min_score:
                continue
            cid, src, text, meta = records[i]
            try:
                m = json.loads(meta)
            except Exception:
                m = {}
            out.append(Document(id=cid, text=text, source=src, meta=m, score=round(sc, 4)))
        return out

    def keyword_search(self, query: str, top_k: int = 6) -> List[Document]:
        """Fallback jab embeddings available na ho."""
        toks = _tokens(query)
        if not toks:
            return []
        like = " OR ".join(["text LIKE ?"] * len(toks))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, source, text, meta FROM chunks WHERE {like} LIMIT 200",
                tuple(f"%{t}%" for t in toks)).fetchall()
        scored = []
        for cid, src, text, meta in rows:
            tl = text.lower()
            score = sum(tl.count(t) for t in toks) / (len(text) ** 0.5 + 1)
            try:
                m = json.loads(meta)
            except Exception:
                m = {}
            scored.append(Document(cid, text, src, m, round(score, 4)))
        scored.sort(key=lambda d: -d.score)
        return scored[:top_k]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
