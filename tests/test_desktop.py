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
