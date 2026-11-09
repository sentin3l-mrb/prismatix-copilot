"""SQLite-backed vector store. Vectors are stored as REAL BLOBs and exact
NN search runs in numpy — fine for thousands of chunks, trivial to teach."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, db_path: str | Path = ".cache/rag.sqlite3"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        # FastAPI serves requests from threadpool workers, so the shared
        # connection must allow cross-thread use; the lock serializes it.
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                   id INTEGER PRIMARY KEY,
                   source TEXT,
                   text TEXT,
                   dim INTEGER,
                   vec BLOB)"""
        )

    def upsert(self, source: str, text: str, vec: list[float]) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO chunks (source, text, dim, vec) VALUES (?,?,?,?)",
                (source, text, len(vec), np.asarray(vec, dtype=np.float32).tobytes()),
            )

    def count(self) -> int:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def clear(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM chunks")
            self.conn.commit()

    def commit(self) -> None:
        with self._lock:
            self.conn.commit()

    def search(self, query_vec: list[float], k: int = 4) -> list[dict]:
        """Exact cosine similarity search (unnormalized vecs are rescaled)."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, source, text, vec FROM chunks").fetchall()
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        if not rows:
            return []
        mat = np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        sims = mat @ q
        top = np.argsort(-sims)[:k]
        return [
            {"id": rows[i][0], "source": rows[i][1], "text": rows[i][2],
             "score": float(sims[i])}
            for i in top
        ]
