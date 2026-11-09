"""Entry point: `python -m prismatix` (TUI), `--headless` (CI/demo) or
`--web` (desktop-style browser UI on http://127.0.0.1:7600).

Desktop and TUI default to the project's own LoRA adapter
(adapters/prismatix-style-v2) when present — the base model runs otherwise."""

from __future__ import annotations

from pathlib import Path

from prismatix.headless import build_parser


def main():
    args = build_parser().parse_args()

    if getattr(args, "web", False):
        import threading

        import uvicorn
        from prismatix.desktop import build_app, launch_window, wait_for_server
        from rag.retriever import Retriever

        adapter = resolve_adapter(args.adapter)
        app = build_app(model_id=args.model, adapter=adapter,
                        retriever=Retriever())

        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=7600,
                                               log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        if not wait_for_server(7600):
            print("warning: server did not bind in 15s; opening anyway")

        url = "http://127.0.0.1:7600"
        if launch_window(url):
            print("native window closed — stopping server")
            return
        # no pywebview: fall back to a blocking server + browser tab
        print(f"prismatix desktop on {url}  (ctrl+c to stop)")
        import webbrowser
        webbrowser.open(url)
        server.run()
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

    from prismatix.desktop import resolve_adapter

    from prismatix.tui import PrismatixApp
    PrismatixApp(model_id=args.model, adapter=resolve_adapter(args.adapter),
                 rag=not args.no_rag).run()


if __name__ == "__main__":
    main()
