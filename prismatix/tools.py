"""Safe, read-only tools for /do agent mode.

Sandbox rules:
1. EVERY path is resolved against the repo root; absolute paths and any
   traversal that escapes the root are rejected.
2. Only four verbs exist: look at files, search text, read a file, run the
   project's own pytest suite. Nothing writes, nothing deletes, nothing
   touches the network; the only subprocess is pytest over tests/.
3. Every tool returns a STRING (what the model's next turn will see),
   truncated to a size budget so one tool call cannot flood the context.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_TOOL_OUTPUT = 4000


def _safe_path(p: str) -> Path:
    """Resolve `p` inside REPO_ROOT or raise. This is the entire sandbox."""
    cand = (REPO_ROOT / p).resolve()
    root = REPO_ROOT.resolve()
    if cand != root and root not in cand.parents:
        raise ValueError(f"path outside project: {p!r}")
    return cand


def _clip(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated at {limit} chars)"


# -- the four tools -----------------------------------------------------------
def list_files(dir: str = ".", max_entries: int = 60) -> str:
    """List files under the project (name + size), one level."""
    d = _safe_path(dir)
    entries = sorted(p for p in d.iterdir() if not p.name.startswith("."))
    out = [f"{p.relative_to(REPO_ROOT)} ({p.stat().st_size} B)"
           for p in entries[: max(1, int(max_entries))]]
    return _clip("\n".join(out) or "(empty directory)")


def read_file(path: str, max_chars: int = 4000) -> str:
    """Read a text file inside the project."""
    p = _safe_path(path)
    if not p.is_file():
        return f"error: {path!r} is not a file"
    return _clip(p.read_text(errors="replace")[: max(200, int(max_chars))])


def search_text(pattern: str, max_hits: int = 20) -> str:
    """Grep-like regex search over *.py, *.md, *.toml, *.jsonl, *.yaml files."""
    rx = re.compile(pattern)
    hits: list[str] = []
    for path in sorted(REPO_ROOT.rglob("*")):
        if path.is_dir() or path.name.startswith("."):
            continue
        if path.suffix not in {".py", ".md", ".toml", ".jsonl", ".yaml", ".yml"}:
            continue
        try:
            for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if rx.search(line):
                    rel = path.relative_to(REPO_ROOT)
                    hits.append(f"{rel}:{i}: {line.strip()[:120]}")
                    if len(hits) >= max_hits:
                        return _clip("\n".join(hits))
        except OSError:
            continue
    return _clip("\n".join(hits) or "(no matches)")


def run_tests() -> str:
    """Run the project's own offline pytest suite (tests/ only).

    Recursion guard: tests themselves call this tool; if pytest is
    already running above us, report that instead of spawning pytest.
"""
    import os
    if os.environ.get("PRISMATIX_IN_PYTEST") or os.environ.get("PYTEST_CURRENT_TEST"):
        return "skipped: pytest already active above this call (recursion guard)"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    tail = "\n".join(proc.stdout.splitlines()[-12:])
    return _clip(f"exit={proc.returncode}\n{tail}")


registry: dict[str, dict] = {
    "list_files": {
        "fn": list_files,
        "desc": "List files under the project, one level. Args: dir='.', max_entries=60.",
        "params": {"dir": {"type": "string"}, "max_entries": {"type": "integer"}},
    },
    "read_file": {
        "fn": read_file,
        "desc": "Read a text file inside the project. Args: path (required), max_chars=4000.",
        "params": {"path": {"type": "string"}, "max_chars": {"type": "integer"}},
    },
    "search_text": {
        "fn": search_text,
        "desc": "Regex-search project files. Args: pattern (required), max_hits=20.",
        "params": {"pattern": {"type": "string"}, "max_hits": {"type": "integer"}},
    },
    "run_tests": {
        "fn": run_tests,
        "desc": "Run the offline pytest suite, no args.",
        "params": {},
    },
}


def openai_tool_list() -> list[dict]:
    """Tool descriptors in the OpenAI-ish schema Qwen templates expect."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": spec["desc"],
                "parameters": {
                    "type": "object",
                    "properties": spec["params"],
                    "required": [
                        k for k, v in spec["params"].items() if v.get("x-required")
                    ],
                },
            },
        }
        for name, spec in registry.items()
    ]


def execute(name: str, args: dict) -> str:
    """Run one tool call defensively; every failure becomes a text reply."""
    spec = registry.get(name)
    if spec is None:
        return f"error: unknown tool {name!r}; available: {', '.join(registry)}"
    try:
        return _clip(str(spec["fn"](**args)))
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


# -- parsing the model's tool-call protocol ------------------------------------
# the model's protocol: a fenced JSON blob naming a tool + arguments.
# Built from chr(96) so the fence is NEVER accidentally mangled by editors.
_FENCE = chr(96) * 3
TOOL_CALL_RX = re.compile(rf"{_FENCE}(?:json)?\s*\n(.*?)\n?{_FENCE}", re.S)


def parse_tool_call(response: str):
    """Extract the first FENCED block that parses as {"name", "arguments"}.

    Tolerates fences anywhere (same line or separate line), markdown code
    fences elsewhere, and leading prose. Returns (name, args) or None when
    the model produced no tool call (it went for its final answer).
    """
    fence = _FENCE
    for m in re.finditer(rf"{fence}(?:json)?[ \t]*\n?(.*?){fence}", response, re.S):
        blob = m.group(1).strip()
        try:
            obj = json.loads(blob)
            name = obj.get("name") or (obj.get("function") or {}).get("name")
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            if name and isinstance(args, dict):
                return str(name), args
        except json.JSONDecodeError:
            continue                    # some other code fence; keep looking
    # last resort: a bare JSON object at the very start (unfenced)
    m2 = re.match(r"\s*(\{.*\})", response, re.S)
    if m2 and '"name"' in m2.group(1):
        try:
            obj = json.loads(m2.group(1))
            return obj.get("name"), obj.get("arguments", {})
        except json.JSONDecodeError:
            pass
    return None
