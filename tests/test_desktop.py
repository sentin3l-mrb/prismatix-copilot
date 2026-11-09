"""Desktop UI API tests with a fake engine (no model, no GPU)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from prismatix.desktop import build_app


class FakeEngine:
    model_id = "fake-4B"
    chunks = 0

    def stream(self, messages, system=None, max_tokens=500):
        yield "Hello "
        yield "world"


def client():
    return TestClient(build_app(engine=FakeEngine()))


def test_health_and_index():
    c = client()
    assert c.get("/api/health").json() == {"model": "fake-4B", "chunks": 0}
    assert "prismatix" in c.get("/").text


def test_chat_sse_stream():
    with client().stream("GET", "/api/chat", params={"q": "hi"}) as r:
        body = "".join(r.iter_text())
    assert body.count("event: ") == 4          # meta, 2 tokens, stats
    assert '"text": "Hello "' in body
    assert '"words": 2' in body


def test_unknown_mode_returns_error_json():
    r = client().get("/api/chat", params={"q": "hi", "mode": "nope"})
    assert r.json()["error"] == "unknown mode"


def test_resolve_adapter_prefers_cli_then_own_adapter():
    from prismatix.desktop import resolve_adapter

    assert resolve_adapter("adapters/prismatix-style-v3") == \
        "adapters/prismatix-style-v3"
    own = resolve_adapter(None)
    assert own in ("adapters/prismatix-style-v2", None)


def test_launch_window_falls_back_without_webview(monkeypatch):
    from prismatix import desktop

    monkeypatch.setitem(__import__("sys").modules, "webview", None)  # import -> ImportError
    assert desktop.launch_window("http://127.0.0.1:1") is False


def test_native_launch_pipeline_binds_port_and_serves():
    """The exact pipeline --web uses: uvicorn in a thread + wait_for_server."""
    import threading
    import httpx
    import uvicorn

    app = build_app(engine=FakeEngine())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=7618,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()

    from prismatix.desktop import wait_for_server
    assert wait_for_server(7618, timeout_s=10)
    assert httpx.get("http://127.0.0.1:7618/api/health").json() == \
        {"model": "fake-4B", "chunks": 0}
    server.should_exit = True
