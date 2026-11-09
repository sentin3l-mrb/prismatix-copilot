"""Entry point: `python -m prismatix` (TUI) or `python -m prismatix --headless`. """

from __future__ import annotations

from prismatix.headless import build_parser, main as _real_main


def main():
    _real_main()


if __name__ == "__main__":
    main()
