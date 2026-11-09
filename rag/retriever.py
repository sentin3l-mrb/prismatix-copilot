"""Retriever wires ingest -> embed -> store and exposes two commands used by
the app: `index()` (rebuild) and `retrieve()` (top-k chunks for a query)."""

from __future__ import annotations

from pathlib import Path

from . import embed
from .ingest import chunk_file
from .store import VectorStore

KB_DIR = Path(__file__).parent / "knowledge_base"
DB_PATH = Path(__file__).resolve().parent.parent / ".cache" / "rag.sqlite3"


class Retriever:
    def __init__(self, db_path=DB_PATH):
        self.store = VectorStore(db_path)

    # -- indexing -------------------------------------------------------------
    def index(self) -> int:
        """(Re)build the index over all files in the knowledge base dir."""
        self.store.clear()
        n = 0
        for path in sorted(KB_DIR.iterdir()):
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            for chunk in chunk_file(path):
                vec = embed.embed([chunk.text])[0]
                self.store.upsert(chunk.source, chunk.text, vec)
                n += 1
        self.store.commit()
        return n

    # -- retrieval -------------------------------------------------------------
    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        qvec = embed.embed([query])[0]
        return self.store.search(qvec, k=k)

    def retrieve_for_prompt(self, query: str, k: int = 4) -> str:
        """Render retrieved chunks as a context block for the chat prompt."""
        hits = self.retrieve(query, k=k)
        if not hits:
            return ""
        parts = []
        for h in hits:
            parts.append(f"[source: {h['source']} | score {h['score']:.2f}]\n{h['text']}")
        return "\n\n---\n\n".join(parts)
