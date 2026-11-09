"""The agent benchmark: 6 questions whose answers are checkable facts hidden
in the project itself. Run with the real engine:

    python -m eval.run_agent_eval --out artifacts/eval_runs/agent_base.jsonl

Each task grades the agent's FINAL ANSWER deterministically:
  must_include: all these strings must appear (file names, tool names, facts)
  must_not_include: filler/hallucination markers
  min_tools: the agent had to use at least this many tool calls
  max_steps_budget is enforced inside run_agent (forced synthesis at 0 left)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prismatix.agent import run_agent
from prismatix.engine import LocalEngine

AGENT_TASKS: list[dict] = [
    {
        "id": "agent_where_graders",
        "goal": "Which file defines the CodeGrader and SQLGrader classes?",
        "must_include": ["eval/graders.py"],
        "min_tools": 1,
        "max_tokens": 260,
    },
    {
        "id": "agent_where_plans",
        "goal": "In what file would you find the plan prices ($ per seat) of Northwind?",
        "must_include": ["company_handbook.md"],
        "min_tools": 1,
        "max_tokens": 260,
    },
    {
        "id": "agent_tool_read",
        "goal": "Which tool would you use to read a file's content, and which file lists all tools?",
        "must_include": ["read_file", "prismatix/tools.py"],
        "min_tools": 1,
        "max_tokens": 300,
    },
    {
        "id": "agent_dataset_size",
        "goal": "How many SFT examples does training/build_dataset.py generate?",
        "must_include": ["400"],
        "min_tools": 1,
        "max_tokens": 260,
    },
    {
        "id": "agent_rag_chunks",
        "goal": "Which files live in rag/knowledge_base/ (name at least two)?",
        "must_include": ["company_handbook.md", "api_style_guide.md"],
        "min_tools": 1,
        "max_tokens": 300,
    },
    {
        "id": "agent_command_index",
        "goal": "Which slash command rebuilds the retrieval index in this app?",
        "must_include": ["/index"],
        "min_tools": 1,
        "max_tokens": 260,
    },
]


def grade(task: dict, answer: str, n_tool_uses: int) -> dict:
    low = answer.lower()
    missing = [s for s in task["must_include"] if s.lower() not in low]
    filler = [s for s in ("i don't know", "cannot access", "as an ai")
              if s in low]
    return {
        "pass": (not missing) and (not filler) and n_tool_uses >= task["min_tools"],
        "missing": missing,
        "banned": filler,
        "tools_used": n_tool_uses,
    }


def run(out_path: Path, model_id: str | None = None,
        adapter: str | None = None, max_tokens_scale: int = 1) -> dict:
    engine = LocalEngine(model_id=model_id, adapter_path=adapter)
    rows, passed = [], 0
    for task in AGENT_TASKS:
        uses: list[str] = []

        def step(tool, args, result):
            uses.append(tool)

        answer = run_agent(engine, task["goal"], on_step=step,
                           max_tokens=max_tokens_scale * task["max_tokens"])
        g = grade(task, answer, len(uses))
        rows.append({"task_id": task["id"], "goal": task["goal"], "answer": answer,
                     "grade": g})
        symbol = "PASS" if g["pass"] else "FAIL"
        detail = "missing: " + ", ".join(g["missing"]) if g["missing"] else "ok"
        print(f"[{symbol}] {task['id']:<26} {detail}")
        passed += bool(g["pass"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(json.dumps(r) for r in rows))
    summary = {"passed": passed, "total": len(AGENT_TASKS),
               "pass_rate": passed / len(AGENT_TASKS)}
    print(f"\nagent benchmark: {passed}/{len(AGENT_TASKS)}")
    Path(str(out_path).replace(".jsonl", ".summary.json")).write_text(
        json.dumps(summary, indent=2))
    return summary
