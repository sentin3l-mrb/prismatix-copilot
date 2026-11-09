"""Agent mode (/do): the OpenCode-style loop on a small local model.

model -> tool call -> real result -> feed back -> repeat (max_steps cap).
Tools are read-only (prismatix/tools.py sandbox); after max_steps the loop
stops and reports what happened, it never guesses past its budget.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from prismatix.engine import LocalEngine
from prismatix.tools import openai_tool_list, parse_tool_call, registry, execute

AGENT_SYSTEM = (
    "You are Prismatix in AGENT mode. You answer questions about THIS project "
    "by using tools step by step. Rules: 1) When you need a fact, issue ONE "
    "tool call in a fenced block as {\"name\": ..., \"arguments\": {...}}. "
    "2) Wait for the result before the next step; never invent file contents. "
    "3) After at most ~4 tool calls, stop and give a final answer in 2-5 "
    "sentences, citing the files you read. Available tools: "
    + ", ".join(registry) + ". "
)

MAX_STEPS = 6


def run_agent(
    engine: LocalEngine,
    goal: str,
    on_step: Optional[Callable[[str, str, str], None]] = None,
    max_steps: int = MAX_STEPS,
) -> str:
    """Run the loop; returns the final answer text.

    on_step(tool_name, tool_args, tool_result) is a display callback
    (the TUI prints lines, tests collect output headlessly).
    """
    messages: List[dict] = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": goal},
    ]

    for step in range(max_steps):
        prompt = engine.prompt_for(messages, system=None, tools=openai_tool_list())
        reply = engine.generate_raw(prompt, max_tokens=400)
        call = parse_tool_call(reply)

        if call is None:                       # the model's final answer
            messages.append({"role": "assistant", "content": reply})
            return reply.strip()

        name, args = call
        result = execute(name, args)
        if on_step:
            on_step(name, args, result)

        # loop memory: assistant's tool call + the real tool output.
        # Every injection carries a step counter: 4B models do not stop by
        # themselves, so we keep reminding them the budget is finite.
        left = max_steps - step - 1
        messages.append({"role": "assistant", "content": reply})
        messages.append({
            "role": "user",
            "content": (
                f"Tool result for step {step + 1} of {max_steps}:\n\n{result}\n\n"
                + (
                    f"You have {left} steps left."
                    if left else "NO STEPS LEFT."
                )
                + " As soon as you have enough information, STOP calling "
                "tools and write the final answer as plain text now."
            ),
        })

    # budget exhausted: one forced synthesis turn, tools withheld
    prompt = engine.prompt_for(messages, system=None)
    reply = engine.generate_raw(prompt, max_tokens=400)
    return reply.strip()
