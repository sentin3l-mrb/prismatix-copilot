"""Headless mode for CI/tests/demos: answer one prompt and print.

    python -m prismatix --headless "/sql revenue by region"
"""

from __future__ import annotations

import argparse
import sys

from prismatix.commands import HELP, route
from prismatix.engine import LocalEngine
from rag.retriever import Retriever


def headless_main(argv=None) -> None:
    ap = argparse.ArgumentParser("prismatix headless")
    ap.add_argument("prompt")
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter")
    ap.add_argument("--max-tokens", type=int, default=500)
    args = ap.parse_args(argv)

    turn = route(args.prompt)
    if turn is None or turn.max_tokens == 0:
        print(HELP)
        return

    rag_ctx = ""
    if turn.mode == "rag":
        rag_ctx = Retriever().retrieve_for_prompt(args.prompt.split(maxsplit=1)[-1])
        turn = route(args.prompt, rag_context=rag_ctx)

    engine = LocalEngine(model_id=args.model, adapter_path=args.adapter)
    out = engine.chat(turn.messages, max_tokens=turn.max_tokens, system=turn.system)
    print(out.text)
    print(f"\n[out {out.output_tokens} tok · {out.tps:.1f} tok/s]", file=sys.stderr)


def build_parser():
    ap = argparse.ArgumentParser("prismatix", description="local copilot TUI")
    ap.add_argument("--headless", nargs="?", const="", default=None,
                    metavar="PROMPT", help="run one prompt headlessly and print")
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--no-rag", action="store_true")
    return ap


def main():
    args = build_parser().parse_args()

    if args.headless is not None:
        if not args.headless:
            raise SystemExit("usage: prismatix --headless 'PROMPT'")
        argv = [args.headless, "--max-tokens", str(args.max_tokens)]
        if args.model:
            argv += ["--model", args.model]
        if args.adapter:
            argv += ["--adapter", args.adapter]
        headless_main(argv)
        return

    from prismatix.tui import PrismatixApp
    PrismatixApp(model_id=args.model, adapter=args.adapter,
                 rag=not args.no_rag).run()


if __name__ == "__main__":
    main()
