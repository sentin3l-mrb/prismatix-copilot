"""Command-router tests (no model needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prismatix.commands import Conversation, route


def test_sql_route():
    t = route("/sql monthly revenue")
    assert t.mode == "sql" and "Business question" in t.messages[-1]["content"]


def test_brief_route():
    t = route("/brief revenue up 10%")
    assert t.mode == "brief" and "revenue up 10%" in t.messages[-1]["content"]


def test_plain_chat_is_code_mode():
    t = route("write a function to dedupe a list")
    assert t.mode == "chat" and t.max_tokens == 700


def test_rag_route_without_index():
    t = route("/rag what is the refund window?")
    assert t.mode == "rag"


def test_quit_and_unknown():
    assert route("/quit") is None
    assert route("/wibble x").mode == "unknown"


def test_conversation_history_capped():
    c = Conversation([], max_turns=2)
    c.extend("a", "b")
    c.extend("c", "d")
    c.extend("e", "f")
    assert c.context() == [{"role": "user", "content": "c"},
                           {"role": "assistant", "content": "d"},
                           {"role": "user", "content": "e"},
                           {"role": "assistant", "content": "f"}]
    c.reset()
    assert c.context() == []
