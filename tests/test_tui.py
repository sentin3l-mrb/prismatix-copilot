"""TUI smoke test with a mock engine: UI composes, routing renders, no model.

    .venv/bin/python -m pytest tests/test_tui.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prismatix.engine import GenResult
from prismatix.tui import PrismatixApp


class FakeEngine:
    """Stands in for LocalEngine so tests never touch model weights."""

    model_id = "fake-model"

    def __init__(self, *a, **k):
        pass

    def stream(self, messages, system=None, max_tokens=200):
        yield from ["hello ", "world"]
        self.last_call = messages


def test_tui_renders_and_answers(monkeypatch):
    app = PrismatixApp()
    monkeypatch.setattr(app, "engine", FakeEngine())
    # scrap retrieval: no index needed in this test
    app.rag = False
    app.retriever = None

    async def run_once():
        async with app.run_test(size=(80, 24)) as pilot:
            inp = app.query_one("Input")
            inp.value = "write me something"
            await pilot.press("enter")
            await pilot.pause(0.2)
            # conversation memory must contain the mock answer
            assert ["hello world"] == [
                m["content"] for m in app.convo.history if m["role"] == "assistant"
            ], app.convo.history

    import asyncio
    asyncio.run(run_once())
