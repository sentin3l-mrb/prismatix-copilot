"""Prismatix Desktop: a local web UI for the copilot.

architecture:
    browser window (static/index.html, no CDN)
       |  fetch + SSE (localhost only)
    FastAPI (this file)  ->  the SAME engine/router/agent as the TUI
       ->  mlx (4-bit model on Apple GPU)

No HTML/CSS/JS build tooling: one hand-written index.html keeps the
"completely offline app" promise intact.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from prismatix.agent import run_agent
from prismatix.commands import HELP, route
from prismatix.engine import LocalEngine

STATIC = Path(__file__).parent / "static"
PORT = 7600


def resolve_adapter(cli_adapter: Optional[str]) -> Optional[str]:
    """--adapter wins; otherwise the project's own LoRA when present."""
    if cli_adapter:
        return cli_adapter
    default = Path("adapters/prismatix-style-v2/adapters.safetensors")
    return "adapters/prismatix-style-v2" if default.exists() else None


def launch_window(url: str) -> bool:
    """Open a native macOS window (pywebview/WKWebView).

    Returns False when pywebview is unavailable (pure-server fallback).
    Blocks until the user closes the window.
    """
    try:
        import webview
    except ImportError:
        return False
    webview.create_window(
        "prismatix — local copilot · by sentin3l-mrb", url,
        width=1120, height=760, min_size=(760, 560),
    )
    webview.start()          # native event loop; blocks until window closes
    return True


def wait_for_server(port: int, timeout_s: float = 15.0) -> bool:
    """Block until the local port accepts TCP connections (or give up)."""
    import socket
    import time
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.4):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def sse(event: str, data: dict) -> str:
    """one Server-Sent-Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def build_app(engine: Optional[LocalEngine] = None, retriever=None,
              model_id: Optional[str] = None, adapter: Optional[str] = None) -> FastAPI:
    """app factory so tests can inject a fake engine."""
    app = FastAPI(title="prismatix desktop")
    eng = engine or LocalEngine(model_id=model_id, adapter_path=adapter)
    app.state.retriever = retriever

    @app.get("/")
    def index():
        from fastapi.responses import FileResponse
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @app.get("/api/health")
    def health():
        return {
            "model": eng.model_id,
            "chunks": retriever.store.count() if retriever else 0,
        }

    @app.get("/api/chat")
    def chat(q: str, mode: str = "chat"):
        turn = route(f"/{mode} {q}" if mode != "chat" else q)
        if turn is None or turn.max_tokens == 0 or turn.mode == "unknown":
            return JSONResponse({"error": "unknown mode", "help": HELP})

        rag_ctx = ""
        if mode == "rag" and retriever is not None:
            rag_ctx = retriever.retrieve_for_prompt(q, k=4)
            turn = route(f"/rag {q}", rag_context=rag_ctx)

        def gen() -> Iterator[str]:
            yield sse("meta", {"mode": turn.mode})
            parts, t0 = [], time.perf_counter()
            try:
                for piece in eng.stream(turn.messages, system=turn.system,
                                        max_tokens=turn.max_tokens):
                    parts.append(piece)
                    yield sse("token", {"text": piece})
            except Exception as e:              # show failures in the UI
                yield sse("error", {"message": repr(e)})
                return
            answer = "".join(parts)
            secs = time.perf_counter() - t0
            yield sse("stats", {"words": len(answer.split()),
                                "secs": round(secs, 1)})

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/agent")
    def agent(q: str):
        def gen() -> Iterator[str]:
            yield sse("meta", {"mode": "agent", "max_steps": 6})

            steps: "queue.Queue[dict]" = queue.Queue()
            done: "queue.Queue[tuple]" = queue.Queue()

            def cb(tool, args, output):
                steps.put({"name": tool, "args": args,
                           "output": " ".join(str(output).split())[:200]})

            def worker():
                try:
                    done.put(("answer", run_agent(eng, q, on_step=cb)))
                except Exception as e:
                    done.put(("error", {"text": repr(e)}))

            threading.Thread(target=worker, daemon=True).start()
            while True:
                try:
                    yield sse("tool", steps.get(timeout=0.05))
                except queue.Empty:
                    pass
                try:
                    kind, payload = done.get_nowait()
                    yield sse(kind, {"text": payload} if kind == "answer"
                              else payload)
                    break
                except queue.Empty:
                    continue

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
