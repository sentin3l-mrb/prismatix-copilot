"""Compare two eval run summaries (base vs adapter) into a markdown table.

    python -m eval.compare artifacts/eval_runs/base.jsonl.artifacts ... 
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _summary(path: Path) -> dict:
    p = path.with_suffix(".summary.json")
    if not p.exists():
        raise SystemExit(f"missing summary: run `python -m eval.run_eval ... --out {path}` first")
    return json.loads(p.read_text())["summary"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs=2, help="two .jsonl eval outputs: base then candidate")
    args = ap.parse_args()

    base = _summary(Path(args.runs[0]))
    cand = _summary(Path(args.runs[1]))

    labels = ["base", "candidate"]
    out = ["| category | base pass | candidate pass |", "|---|---|---|"]
    for kind in sorted(set(base["by_kind"]) | set(cand["by_kind"])):
        b = base["by_kind"].get(kind, {"pass": 0, "total": 0})
        c = cand["by_kind"].get(kind, {"pass": 0, "total": 0})
        out.append(f"| {kind} | {b['pass']}/{b['total']} | {c['pass']}/{c['total']} |")
    out.append(f"| **total** | {base['passed']}/{base['total']} "
               f"({base['pass_rate']*100:.0f}%) | {cand['passed']}/{cand['total']} "
               f"({cand['pass_rate']*100:.0f}%) |")
    out.insert(0, f"throughput: base {base['avg_tps']:.1f} tok/s -> "
                  f"candidate {cand['avg_tps']:.1f} tok/s")
    text = "\n".join(out)
    print(text)
    md = Path("artifacts") / "compare.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(text + "\n")


if __name__ == "__main__":
    main()
