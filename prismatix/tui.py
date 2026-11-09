"""Prismatix TUI — an OpenCode-style local coding/business copilot.

Built with Textual; the model streams into a scrollable log. All inference is
local (MLX on Apple Silicon); RAG answers cite local source files.
"""

from __future__ import annotations

import time

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Input, RichLog

from prismatix.commands import HELP, Conversation, route
from prismatix.engine import LocalEngine
from rag.retriever import Retriever


class PrismatixApp(App):
    TITLE = "prismatix"
    SUB_TITLE = "local copilot on Qwen3.5-4B (mlx)"
    BINDINGS = [("ctrl+c", "quit", "Quit"), ("ctrl+l", "clear", "Clear")]
    AUTO_FOCUS = "Input"      # keyboard focus starts in the chat input
    CSS = """
    RichLog { height: 1fr; padding: 0 1; }
    Input { dock: bottom; margin: 0 1 1 1; }
    """

    def __init__(self, model_id=None, adapter=None, rag=True):
        super().__init__()
        self.engine = LocalEngine(model_id=model_id, adapter_path=adapter)
        self.adapter = adapter
        self.rag = rag
        self.retriever = Retriever() if rag else None
        self.convo = Conversation(history=[])
        self.streaming = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield RichLog(markup=True, wrap=True, id="log")
        yield Input(placeholder="/help for commands · /quit to exit")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Input).focus()
        log = self.query_one("#log", RichLog)
        log.write(f"[bold]prismatix[/] — {self.engine.model_id}"
                  + (f" + adapter {self.adapter}" if self.adapter else "")
                  + "  ·  by sentin3l-mrb")
        if self.rag:
            n = self.retriever.store.count()
            log.write(f"[dim]retrieval index: {n} chunks (run /index to rebuild)[/]")
        log.write("[dim]" + HELP.replace("\n", "  |  ") + "[/]")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self.streaming:                      # one generation at a time
            event.input.value = event.value
            return
        event.input.value = ""
        self.run_worker(self._handle(event.value), exclusive=False)

    # -- async generation worker ----------------------------------------------
    async def _handle(self, text: str) -> None:
        log = self.query_one("#log", RichLog)
        first = text.split(maxsplit=1)[0].lower()
        if first == "/quit":
            self.exit()
            return
        if first == "/index":
            log.write("[yellow]rebuilding index...[/]")
            t0 = time.perf_counter()
            n = self.retriever.index()
            log.write(f"indexed {n} chunks in {time.perf_counter()-t0:.1f}s")
            return
        if first == "/clear":
            self.query_one("#log", RichLog).clear()
            return
        if first == "/stats":
            n = self.retriever.store.count() if self.retriever else 0
            log.write(f"model={self.engine.model_id} adapter={self.adapter or 'none'}"
                      f" chunks={n}")
            return

        rag_ctx = ""
        if self.rag and first == "/rag":
            rag_ctx = self.retriever.retrieve_for_prompt(
                text[len(first):].strip(), k=4)
        turn = route(text, history=self.convo.history, rag_context=rag_ctx)
        if turn is None:
            return
        if turn.mode == "help" or turn.mode == "unknown":
            log.write(HELP)
            return
        if turn.mode == "do":
            goal = text[len("/do"):].strip()
            log.write(f"[bold cyan]agent[/] {goal}")

            def on_step(tool, args, output):
                one = " ".join(str(output).split())[:150]
                log.write(f"[magenta]tool[/] {escape(tool + ' ' + str(args))} -> {escape(one)}")

            import prismatix.agent as _agent
            self.streaming = True
            try:
                answer = _agent.run_agent(self.engine, goal, on_step=on_step)
                log.write(f"[bold green]agent answer[/]\n{escape(answer)}")
            except Exception as e:
                log.write(f"[red]agent error:[/] {e!r}")
            finally:
                self.streaming = False
            return

        log.write(f"\n[bold cyan]you ({turn.mode})[/] {escape(text)}")
        self.streaming = True
        try:
            # Stream tokens into a line buffer: RichLog renders every write()
            # as its own block, so writing per-token would stack single tokens
            # vertically. We flush per LINE instead (natural chat output).
            chunks = []
            buf = ""
            t0 = time.perf_counter()
            for piece in self.engine.stream(turn.messages, system=turn.system,
                                            max_tokens=turn.max_tokens):
                chunks.append(piece)
                buf += piece
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    log.write(escape(line))
            if buf:
                log.write(escape(buf))
            answer = "".join(chunks)
            secs = time.perf_counter() - t0
            log.write(f"\n[dim]{len(answer.split())} words · {secs:.1f}s[/]")
            self.convo.extend(text, answer)
        except Exception as e:                  # surface failures, keep UI alive
            log.write(f"[red]generation error:[/] {e!r}")
        finally:
            self.streaming = False
