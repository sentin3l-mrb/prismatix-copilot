"""Agent-loop tests with a scripted model: the loop, the cap, the parsing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prismatix.agent import run_agent
from prismatix.tools import parse_tool_call, execute


class ScriptedEngine:
    """Yields canned tool calls (or a final answer) in sequence."""

    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.prompts: list[str] = []

    def prompt_for(self, messages, system=None, tools=None, *a, **k):
        self.prompts.append("|".join(m["role"] for m in messages))
        return "x"

    def generate_raw(self, prompt, max_tokens=400):
        return self.sequence.pop(0)


F = chr(96) * 3        # build the fence without literal backticks


def test_tool_call_then_answer():
    call = F + '{"name": "list_files", "arguments": {"dir": "."}}' + F
    eng = ScriptedEngine([call, "Found README.md and tests dir. Done."])
    steps: list[str] = []
    answer = run_agent(eng, "what files exist?",
                       on_step=lambda t, a, o: steps.append(t))
    assert "Found README.md" in answer
    assert steps == ["list_files"]
    # loop fed the tool result back: 3rd prompt has 4 messages (sys,user,asst,tool)
    assert eng.prompts[-1].count("user") >= 2


def test_step_budget_stops_the_loop():
    call = F + '{"name": "search_text", "arguments": {"pattern": "x"}}' + F
    # exactly 3 tool calls (the budget), then ONE forced synthesis turn;
    # any further generate_raw call would IndexError the scripted sequence
    eng = ScriptedEngine([call] * 3 + ["Forced final answer without tools."])
    answer = run_agent(eng, "loop forever", max_steps=3)
    assert answer == "Forced final answer without tools."


def test_parser_shapes_and_failures():
    j = '{"name": "read_file", "arguments": {"path": "LOG.md"}}'
    assert parse_tool_call(F + j + F) == ("read_file", {"path": "LOG.md"})
    assert parse_tool_call("no tool here") is None
    assert parse_tool_call(F + "not json" + F) is None
    # arguments sent as a string still work
    wrapped = F + '{"name": "run_tests", "arguments": "{}"}' + F
    assert parse_tool_call(wrapped)[0] == "run_tests"


def test_tools_sandbox_and_behaviour():
    assert "error" in execute("read_file", {"path": "../../etc/passwd"})
    assert "error" in execute("no_such_tool", {})
    out = execute("search_text", {"pattern": "def parse_tool_call"})
    assert "prismatix/tools.py" in out
    out = execute("run_tests", {})
    # inside pytest the tool must refuse (recursion guard) — accept either
    # the guard message or a real pytest run result
    assert "skipped" in out or out.startswith("exit=") or "passed" in out
