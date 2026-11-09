"""Slash-command routing shared by the TUI and the headless CLI.

Each command builds an OpenAI-style message list; the engine turns that into a
chat-template prompt. Keeping routing separate from UI makes the app testable
headless: `python -m prismatix --headless "/sql revenue by region"`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from prismatix.engine import SYSTEM_PROMPT

HELP = """\
commands:
  /help              show this help
  /code <task>       write code (plan -> code block -> edge-case note)
  /sql <question>    turn a business question into SQL for the Northwind DB
  /brief <notes>     3-sentence manager summary, numbers first
  /email <task>      draft a short customer/team email
  /rag <question>    answer using the local knowledge base (retrieval)
  /index             rebuild the retrieval index
  /stats             show engine stats
  /reset             clear conversation history
  /quit              exit
"""

SQL_CONTEXT = (
    "The database is Northwind Cloud's production Postgres.\n"
    "Tables: customers(id, name, region, plan, signed_up); "
    "orders(id, customer_id, amount, order_date); products(id, name, price, active)."
)


@dataclass
class Turn:
    """One routed user input ready for the engine."""
    mode: str
    messages: List[dict]
    system: str
    max_tokens: int


@dataclass
class Conversation:
    """Short rolling memory of the current session (last N turns)."""
    history: List[dict]
    max_turns: int = 6

    def context(self) -> List[dict]:
        return self.history[-self.max_turns * 2:]  # max_turns = user+assistant pairs

    def extend(self, user: str, assistant: str) -> None:
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": assistant})

    def reset(self) -> None:
        self.history.clear()


def route(text: str, history: Optional[List[dict]] = None,
          rag_context: str = "") -> Optional[Turn]:
    """Return None for /quit and /index style control commands (not model turns)."""
    text = text.strip()
    if not text:
        return None
    if text == "/quit":
        return None

    if text == "/reset":
        if history is not None:
            history.clear()
        return None
    if text == "/help":
        return Turn(mode="help", messages=[], system=SYSTEM_PROMPT, max_tokens=0)

    words = text.split(maxsplit=1)
    cmd = words[0].lower() if words and words[0].startswith("/") else ""
    rest = text[len(cmd):].strip() if cmd else text

    if cmd in ("/sql",):
        user = (
            f"{SQL_CONTEXT}\n\nBusiness question: {rest}\n"
            "Respond with a fenced ```sql block, then one sentence explaining it."
        )
        return Turn(mode="sql", messages=[{"role": "user", "content": user}],
                    system=SYSTEM_PROMPT, max_tokens=350)

    if cmd == "/brief":
        user = (
            "Write a 3-sentence manager brief from these notes. "
            "Numbers first, no greetings, no filler.\nNotes:\n" + rest
        )
        return Turn(mode="brief", messages=[{"role": "user", "content": user}],
                    system=SYSTEM_PROMPT, max_tokens=220)

    if cmd == "/email":
        return Turn(mode="email", messages=[{"role": "user", "content": rest}],
                    system=SYSTEM_PROMPT, max_tokens=350)

    if cmd == "/rag":
        if not rag_context:
            return Turn(mode="rag", messages=[{"role": "user",
                        "content": "No indexed documents found. Run /index first."}],
                        system="You answer shortly.", max_tokens=80)
        user = (
            f"Answer using ONLY the sources below. Cite the source file you used.\n\n"
            f"SOURCES:\n{rag_context}\n\nQUESTION: {rest}"
        )
        return Turn(mode="rag", messages=[{"role": "user", "content": user}],
                    system="You answer from provided sources, concisely, in 2-4 sentences.",
                    max_tokens=350)

    if cmd == "/code" or not cmd:
        user = rest
        msgs: List[dict] = []
        if history:
            msgs.extend(history)
        msgs.append({"role": "user", "content": user})
        return Turn(mode="chat", messages=msgs, system=SYSTEM_PROMPT, max_tokens=700)

    if cmd == "/index" or cmd == "/stats":
        return None

    if cmd:
        return Turn(mode="unknown",
                    messages=[{"role": "user",
                               "content": f"Unknown command {cmd}. From now on answer minimally."}],
                    system="One line: unknown command, suggest /help.", max_tokens=40)

    return None
