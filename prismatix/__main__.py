"""Entry point: `python -m prismatix` (TUI), `--headless` (CI/demo) or
`--web` (desktop-style browser UI on http://127.0.0.1:7600)."""

from __future__ import annotations

from prismatix.headless import build_parser


def main():
    args = build_parser().parse_args()

    if getattr(args, "web", False):
        import uvicorn
        from prismatix.desktop import build_app
        from rag.retriever import Retriever

        app = build_app(retriever=Retriever())
        print("prismatix desktop on http://127.0.0.1:7600  (ctrl+c to stop)")
        uvicorn.run(app, host="127.0.0.1", port=7600, log_level="warning")
        return

    if args.headless is not None:
        from prismatix.headless import main as headless_main
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
