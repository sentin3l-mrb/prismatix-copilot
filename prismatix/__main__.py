"""Entry point: `python -m prismatix` (TUI) or `python -m prismatix --headless`. """

from __future__ import annotations

from prismatix.headless import build_parser


def main():
    args = build_parser().parse_args()

    if getattr(args, "web", False):
        import uvicorn
        from prismatix.desktop import build_app

        app = build_app()
        uvicorn.run(app, host="127.0.0.1", port=7600, log_level="warning")
        return

    if args.headless is not None:


if __name__ == "__main__":
    main()
