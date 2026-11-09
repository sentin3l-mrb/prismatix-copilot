"""Chunking: split markdown-ish files on headings, then pack paragraphs into
~600 character chunks and prefix each with its section label so the embedding
carries topic information, not just raw text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

HEADING = re.compile(r"^#{1,6}\s+")
CHUNK_TARGET = 600


@dataclass
class Chunk:
    source: str      # file name, shown to the model for provenance
    text: str


def chunk_file(path: Path) -> list[Chunk]:
    raw = path.read_text().splitlines()
    chunks: list[Chunk] = []
    buf: list[str] = []
    cur_heading = ""

    def flush() -> None:
        if not buf:
            return
        text = f"[{path.name} / {cur_heading or 'intro'}]\n" + "\n".join(buf).strip()
        chunks.append(Chunk(source=path.name, text=text[: 3 * CHUNK_TARGET]))
        buf.clear()

    for line in raw:
        if HEADING.match(line):          # new section -> finish the old one
            flush()
            cur_heading = line.lstrip("#").strip()
            continue
        if not line.strip():
            if sum(map(len, buf)) > CHUNK_TARGET:   # soft target reached
                flush()
            continue
        buf.append(line)
    flush()
    return chunks


def chunk_dir(dirpath: Path) -> list[Chunk]:
    out: list[Chunk] = []
    for p in sorted(Path(dirpath).iterdir()):
        if p.suffix.lower() in {".md", ".txt"}:
            out.extend(chunk_file(p))
    return out
