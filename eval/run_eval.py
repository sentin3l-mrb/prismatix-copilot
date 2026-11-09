"""Run the Prismatix benchmark against a local model (optionally with a LoRA
adapter) and save an honest, replayable JSONL log of every task + score.

Usage:
    python -m eval.run_eval --out artifacts/eval_runs/base.jsonl
    python -m eval.run_eval --adapter adapters/prismatix-style --out artifacts/eval_runs/lora.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from eval.tasks import ALL_TASKS
from prismatix.engine import LocalEngine


def run_eval(model_id: str, adapter: str | None, out_path: Path,
             max_tokens: int, seed: int, limit: int | None) -> dict:
    engine = LocalEngine(model_id=model_id, adapter_path=adapter)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.perf_counter()
    with out_path.open("w") as fh:
        for task in ALL_TASKS if limit is None else ALL_TASKS[:limit]:
            messages = [{"role": "user", "content": task["prompt"]}]
            result = engine.chat(messages, max_tokens=max_tokens, seed=seed)
            graded = task["grader"](result.text)
            row = {
                "task_id": task["id"],
                "kind": task["kind"],
                "model": engine.model_id,
                "adapter": adapter,
                "grader": graded,
                "stats": {
                    "prompt_tokens": result.prompt_tokens,
                    "output_tokens": result.output_tokens,
                    "seconds": round(result.seconds, 2),
                    "tps": round(result.tps, 1),
                },
                "response": result.text,
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            symbol = "PASS" if graded.get("pass") else "FAIL"
            print(f"[{symbol}] {task['id']:<22} {graded.get('detail','')[:80]}")
            rows.append(row)

    wall = time.perf_counter() - t0
    summary = summarize(rows, wall)
    with out_path.with_suffix(".summary.json").open("w") as fh:
        json.dump({"summary": summary, "model": engine.model_id,
                   "adapter": adapter, "wall_seconds": round(wall, 1)}, fh, indent=2)
    print("\n=== summary ===")
    for kind, k in summary["by_kind"].items():
        print(f"{kind:<8} pass {k['pass']}/{k['total']}")
    print(f"total    pass {summary['passed']}/{summary['total']}"
          f"  ({summary['pass_rate']*100:.0f}%)  wall {wall:.0f}s")
    print(f"avg gen  {summary['avg_tps']:.1f} tok/s")
    return summary


def summarize(rows: list) -> dict:
    by_kind = defaultdict(lambda: {"pass": 0, "total": 0})
    tps, passed = [], 0
    for r in rows:
        by_kind[r["kind"]]["total"] += 1
        passed += bool(r["grader"].get("pass"))
        by_kind[r["kind"]]["pass"] += bool(r["grader"].get("pass"))
        tps.append(r["stats"]["tps"])
    return {
        "passed": passed,
        "total": len(rows),
        "pass_rate": passed / max(1, len(rows)),
        "by_kind": dict(by_kind),
        "avg_tps": sum(tps) / max(1, len(tps)),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run_eval(args.model, args.adapter, Path(args.out), args.max_tokens, args.seed, args.limit)
