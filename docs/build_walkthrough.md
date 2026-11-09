# Prismatix — the complete build walkthrough

*How a small open-weight model on a MacBook became: a measured benchmark, a
retrieval system, three LoRA fine-tuning recipes, an agentic tool loop, a
terminal app, a desktop app, and an installed macOS application.*

This document is deliberately meticulous. Every claim is backed by a number
that exists in `LOG.md`, an artifact in `artifacts/`, or a test in `tests/`.
Nothing here is aspirational: if a feature was attempted and failed, the
failure is documented with its diagnosis (see §11).

Repository: https://github.com/sentin3l-mrb/prismatix-copilot
Author: Denis Polkanov (@sentin3l-mrb)

---

## 1. What the product is, in one paragraph

**Prismatix** is a privacy-first, fully-local AI assistant for coding and
small-business work. It runs `Qwen/Qwen3.5-4B` (Apache-2.0) quantized to
4 bits via Apple MLX on a 16 GB MacBook Pro M1 Pro. Its behaviour is adapted
by a **LoRA adapter trained by the author** (16.2M parameters, the
"house style" variant, `adapters/prismatix-style-v2`). It adds **retrieval**
over a local document corpus, a **checkable 16-task benchmark**, an
**agentic tool loop** (`/do`) with read-only sandboxed tools, and ships as a
**terminal TUI** and an **installed native macOS application**
(`/Applications/Prismatix.app`).

Numbers you can quote under examination:

| measurement | value | source |
|---|---|---|
| base model benchmark | **13/16 pass (81%)** | `artifacts/eval_runs/base.summary.json` |
| v2 LoRA benchmark | 12/16 (75%), briefs 4/4 | `lora_v2.summary.json` |
| v3 LoRA benchmark | 12/16 (75%), emails 3/3, val loss 0.089 | `lora_v3.summary.json` |
| generation throughput | 41.2 tok/s base · 28.2 tok/s w/ adapter | eval summaries |
| model load time | 4.0–4.7 s cold | `LOG.md` #9 |
| retrieval hits (5 labelled queries) | 5/5 correct source in top-3 | `tests/test_rag.py` |
| retrieval latency | 4–6 ms per query (indexed) | `LOG.md` #7 |
| training run | 300 iters ≈ 27 min on M1 Pro (16 GB) | `LOG.md` #13 |
| trainable parameters | 16.2M of 4.205B = 0.386% (rank 16) · 8.1M = 0.193% (rank 8) | training stdout |
| offline pytest suite | 27 tests, all passing, no model needed | `tests/` |

---

## 2. Timeline: how it was actually built

The honest sequence, with the failures left in:

1. **Environment.** System Python 3.9.6 was too old for MLX → installed
   Python 3.12.16 via `uv` (no sudo needed). `uv venv` + `uv pip install`.
2. **Model selection.** Started with `Qwen2.5-Coder-3B-Instruct`. A license
   audit via the HF API showed `license: other` — the actual Qwen *Research
   License* (non-commercial only), *not* Apache-2.0. Switched to
   **`Qwen/Qwen3.5-4B`** (Apache-2.0, newer, mlx 4-bit conversion verified).
   Lesson: model cards can lie by omission; always fetch the license *file*.
3. **First inference.** Local smoke test revealed two things at once: (a) 42
   tok/s generation was fine; (b) Qwen3.5 is a *thinking hybrid* that leaked
   raw "Thinking Process:" text into answers. Fixed by passing
   `enable_thinking=False` in the chat template.
4. **Benchmark before fine-tuning.** Built 16 checkable tasks and — crucially —
   unit-tested the *graders* first. This caught three grader bugs that would
   otherwise have faked a perfect eval (§6).
5. **RAG.** Chunking → `bge-small-en-v1.5` embeddings via fastembed (ONNX, no
   PyTorch) → SQLite exact cosine search → verified 5/5 retrieval hits.
6. **Fine-tuning attempts.** v1 (lr 1e-4 × LoRA scale 20) produced a
   textbook loss curve and **0/16 degenerate output** — diagnosed and retried:
   v2 (lr 1e-5 × scale 10) achieved the style goal; v3 (rank 8 × scale 2)
   tested whether a weaker imprint preserves base skills.
7. **App.** Terminal TUI (Textual) → headless mode for CI → FastAPI + SSE web
   UI → native `pywebview` window → packaged `.app` in `/Applications`.
8. **Agent mode.** `/do` — the OpenCode-style tool loop: read-only sandboxed
   tools, tool-call dialect parsing (two dialects found live!), step budget,
   forced synthesis turn.

All of it lives in 38 commits with the author's own identity in the history.

---

## 3. Architecture: what happens when you press enter

```
UI surfaces (three of them, one shared brain)
  ├─ TUI        prismatix/tui.py        (Textual, terminal)
  ├─ Desktop    prismatix/desktop.py    (FastAPI + SSE → browser/native window)
  └─ Headless   prismatix/headless.py   (one prompt, for CI and demos)
        │
        ▼  same shared router
   prismatix/commands.py  route() → Turn{mode, messages, system, max_tokens}
        │
        ▼
   prismatix/engine.py    LocalEngine
        ├─ load()          mlx_lm.load(repo, adapter_path=...) — lazy, once
        ├─ prompt_for()    apply_chat_template(..., enable_thinking=False, tools=…)
        ├─ chat()          blocking completion (eval + headless)
        ├─ stream()        token iterator (TUI + SSE), make_sampler(temp)
        └─ generate_raw()  raw prompt completion (agent loop)
        │
        ├─ /rag pre-step:  rag/retriever.py  → top-k chunks → into prompt
        └─ /do loop:       prismatix/agent.py → tools.py sandbox → feed back
```

Design forces that shaped this shape:

* **One brain, many bodies.** All three UIs call the same `route()` so a
  behaviour fix applies everywhere. Early on there was one router duplicate
  per UI — it drifted apart at the first UI change and was unified.
* **Lazy everything.** Model weights load at first generation (4 s), the
  embedding model is a `functools.lru_cache` singleton (110 MB), the RAG
  index persists in `.cache/rag.sqlite3`.
* **Failures surface, never vanish.** TUI and desktop print red error lines;
  nothing is swallowed silently.

---

## 4. The engine (`prismatix/engine.py`) — tokens, quantization, thinking mode

**Tokens.** Text is split into sub-word token IDs from a fixed vocabulary
(≈150k). `smoke_test.py` prints them: `"SELECT region, SUM(revenue) …"` →
`["SELECT", " region", ",", " SUM", "(re", "venue", …]`. Generation is
iterative: one token per forward pass, ~41 tok/s on the M1 Pro's GPU.

**Quantization (why a 4B model fits in ~2.4 GB).** 4.2B float parameters at
FP16 = ~8 GB; at 4 bits each weight sits on a 16-level grid with per-group
scale `w ≈ s·(q − z)` → ~2.4 GB. On Apple Silicon memory *bandwidth* is the
bottleneck, so the smaller representation is also *faster*. Adapters train
on top of the quantized base (QLoRA-style).

**Thinking mode.** Qwen3.5's chat template accepts `enable_thinking`. With it
enabled the model writes long reasoning before answering — great for math,
bad for a copilot (a smoke test leaked *"Thinking Process: …"* into output).
Prismatix always passes `enable_thinking=False`; thinking stays opt-in in code.

**Sampling.** In mlx-lm 0.32.x, `stream_generate()` no longer accepts a
`temperature=` kwarg; sampling moved into `make_sampler(temp=…)` objects.
Passing the old kwarg crashed the TUI only (the headless path never passed
it — a nice lesson in divergent code paths, LOG #19).

---

## 5. RAG (`rag/`) — retrieval in five files

1. **`ingest.py`** — chunks markdown on headings, packs paragraphs to ~600
   chars, and prefixes each chunk with `"[file / section]"` so the *vector*
   carries topic information.
2. **`embed.py`** — `BAAI/bge-small-en-v1.5` (MIT) via `fastembed`: ONNX
   Runtime, CPU, 384-dim vectors, ~110 MB model — deliberately *no PyTorch*.
3. **`store.py`** — SQLite: each chunk row holds its vector as a BLOB.
   Search is **exact** cosine similarity computed in numpy over all rows
   (fine for thousands of chunks, trivial to verify). Cross-thread access
   required `check_same_thread=False` + a `threading.Lock` because FastAPI
   serves from worker threads (LOG #23 — a bug the browser UI exposed).
4. **`retriever.py`** — `index()` rebuilds (5 files → 18 chunks); `retrieve()`
   returns top-k with scores; `retrieve_for_prompt()` renders the citation
   block: `[source: file.md | score 0.85] … chunk text`.
5. **Injection & grounding.** The routed prompt says *"Answer using ONLY the
   sources below. Cite the source file you used."* — open-book exam style.

Verified: 5 labelled queries hit the correct source every time
(`tests/test_rag.py`), latency 4–6 ms (an index lookup, not an LLM call).

---

## 6. The benchmark (`eval/`) — why each task is checkable

Three families, six grader mechanics:

1. **Coding (6 tasks, `CodeGrader`)** — extracts the first ```python block
   from the reply and executes hidden unit tests against the model's own
   code. Pass = the code runs and does the right thing. Base run: 5/6
   (the fail: a dict-comprehension SyntaxError — "looks right, doesn't run").
2. **SQL (3 tasks, `SQLGrader`)** — the model's query and a reference query
   both execute against the same **seeded in-memory SQLite database**
   (`BUSINESS_SCHEMA` + 3 tables, 16 rows); result rows are compared.
   Order-insensitive by design; UPDATEs/DELETEs are rejected by extraction
   (read-only). Base: 3/3 after harness fixes.
3. **Business writing (4 briefs + 3 emails, `RubricGrader`)** — deterministic
   rubrics: required figures verbatim (`has_numbers(["48,200", "2.4"])`),
   ≤3 sentences (`max_sentences`), ≤150 words, subject/greeting/CTA/sign-off
   present, filler-phrases banned.

**Harness honesty (LOG #11).** The first live run scored 10/16 with *buggy
graders* (a bare-SQL extraction bug and a prompt that didn't ask for the
columns the grader wanted). Regrading the *same* responses after fixes gave
12/16, and a fresh run gave the official 13/16. Both numbers are kept in the
log. The benchmark was *scored before it was trusted*: `tests/` runs the
graders against gold + poisoned answers with no model at all.

---

## 7. Fine-tuning (`training/`) — three recipes and the failure that taught us

**Dataset (`build_dataset.py`).** 400 chat-format examples, fully
self-authored from hand-written templates with seeded randomization of
surface details: 150 Python (plan → code block → edge-case note), 90 SQL
(query + one-sentence explanation), 80 briefs (3 sentences, numbers first),
80 emails (subject/greeting/CTA/sign-off). No scraping, no third-party APIs,
no license risk. The eval set uses *different* instances — the benchmark
measures format generalization, not memorized items. Split: 360 train / 40
valid (`split.py`).

**LoRA math, plainly.** Instead of updating 4.2B frozen weights, thin
rank-r matrices `A·B` are inserted next to some projections; the effective
update is `W + B·A`. Rank 16 → 16.2M trainable (0.386%), a 62 MB
`adapters.safetensors`. The *effective step size* is roughly
`learning_rate × scale` — the product of the two knobs, not either alone.

**The three runs (the heart of the project):**

| recipe | trainable | loss story | generation | verdict |
|---|---|---|---|---|
| v1: lr 1e-4, scale 20, 300 iters | 16.2M | train 0.09 — "too good" | **0/16, degenerate loops** | catastrophic; diagnosed |
| v2: lr 1e-5, scale 10, 200 iters | 16.2M | val 0.001–0.009 (saturated fast) | coherent; **briefs 2/4→4/4** | style goal achieved, 12/16 |
| v3: lr 1e-5, scale 2, 200 iters | 8.1M | val 0.089 (healthy curve) | coherent, emails 3/3 kept | 12/16, balanced |

**v1's diagnosis (LOG #14–15).** Validation loss looked *great* while every
generation collapsed into repeated loops (probed even at iter 100). The
lever was the product lr × scale, not the iteration count. After v2 we also
learned that near-zero val loss ≈ memorization of templated data within
<100 steps. **The report-grade lesson: never trust cross-entropy alone —
evaluate generations.**

**v3's crash (LOG #17).** macOS killed the GPU command buffer mid-run
(`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`) when a competing
process (a TUI instance) grabbed GPU/memory. Restarted with
`max_seq_length 1024`; completed cleanly.

**The scientific conclusion (README, restated):** base 13/16 > v2 = v3
12/16. RAG is the correctness lever; LoRA is the *style* lever; the base
model is the generalist. Knowing which lever does what is the result.

---

## 7b. The model-swap experiment (Nov 10): 4B vs 9B, same everything else

With the benchmark harness, agent loop and eval scripts all
`--model`-parameterized, swapping brains is one flag:
`--model mlx-community/Qwen3.5-9B-4bit`. The 9B weights (~5.4 GB on disk)
were downloaded, the smoke test loaded in 5.8 s and generated at 27 tok/s
(only ~35% slower than 4B despite 2.2× parameters).

**16-task results: 13/16 (81%) — identical total, different error profile.**
Code went from 5/6 to **6/6** (the unparenthesized-`:=` class of failure
disappeared — deeper arithmetic competency). SQL fell 3/3 → 2/3: on the
"Same schema…" prompt the 9B invented a `plans` table instead of noticing
which tables exist (the 4B got it right for probably less sophisticated
reasons). One email lost its required CTA markers. All within one seed —
single-instance evidence, not a leaderboard claim.

**Agent results: 2/6 on the 9B vs 3/6 on the 4B — the surprising finding.**
Failure quotes from the 9B (LOG #29): it reported *"the search tool
encountered a persistent error"* and gave up; where the 4B had *guessed*
answers from prompt-adjacent memory (README.md "lists all tools"), the 9B
answered with more nuance but *still didn't open* `prismatix/tools.py` —
and its richer search expressions kept tripping our strict regex tool
(on that task the 4B's simpler queries accidentally succeeded).
Prediction logged BEFORE the run: 14–15/16 on the suite and 4–5/6 on the
agent if reasoning helps; the agent part of the prediction was wrong and
kept, which is the point of writing predictions down.

Practical takeaways for the report:
1. **More parameters bought code competency, not tool navigation.** The
   capability gap for agent work is in *calibration and tool ergonomics*
   (strict regex tool + no feedback loop), not raw intelligence.
2. **Cost side:** 2.3× slower wall-clock on the full suite (95 s vs 53 s),
   ~6 GB working set against 16 GB total, and the LoRA adapters **do not
   transfer** between bases (they are trained against 4B layer shapes) —
   a fresh recipe would be needed for the 9B.
3. The single biggest practical improvement this experiment points to is
   better *tool design* (lenient matching, fuzzy fallbacks, clearer error
   feedback), not a bigger model. That's rare in student projects and
   entirely data-backed here.

---

## 8. Agent mode (`agent.py`, `tools.py`) — `/do`

The loop (identical in shape to OpenCode/Claude Code):

```
goal → model → tool call → execute for real → paste result back → repeat
        (capped at max_steps=6) → forced synthesis turn WITHOUT tools
```

* **Sandbox (`tools.py`).** Four read-only verbs: `list_files`,
  `search_text`, `read_file`, `run_tests`. Every path resolves against the
  repo root — traversal attempts (`../../etc/passwd`) raise an error that
  becomes text for the model. `run_tests` skips pytest only when `tests/`
  is the target, and carries a **recursion guard** (`PYTEST_CURRENT_TEST` +
  a `conftest.py` marker): tests call the tool; without the guard, pytest
  spawns pytest forever (found that out the hard way — a 5-minute hang).
* **Tool-call dialects.** The live model emitted Qwen3.5's markup dialect
  (`<function=…><parameter=…>`), not the fenced JSON of older Qwen models.
  `parse_tool_call()` now accepts **both**, plus plain prose (no call =
  final answer).
* **Stopping instinct.** 4B models don't know when to stop: without
  intervention, the loop drained all 6 steps on tool calls. Fix: every tool
  result is injected with a step counter ("5 steps left… NO STEPS LEFT"),
  and on exhaustion **one forced synthesis turn** runs with tools withheld.
  With that, the agent answered the demo question correctly in 3 steps
  (`eval/graders.py` located, CodeGrader flow explained accurately).
* Output clipping: every tool reply is truncated to ~4000 chars so one call
  cannot flood the context.

---

## 9. The surfaces

**TUI (`textual`).** Two lessons encoded in code: (a) `RichLog` renders every
`write()` as its own block — writing per token *stacked single tokens
vertically*, so the stream buffers and flushes per **line**; (b) model text
is **escaped** before styled writes — square brackets in answers (e.g.
`[0-9]`) otherwise get eaten by rich's markup parser.

**Desktop (`desktop.py` + `static/index.html`).** FastAPI app factory
(engine injectable → testable without a GPU). Endpoints: `/api/health`,
`/api/chat?…` (SSE: `meta` → `token`× → `stats`), `/api/agent?…`
(SSE: `tool` events from a worker thread + final `answer`). The frontend is
**one hand-written offline HTML file** — no CDN, no fonts, no build step;
fonts are system, highlight rendering is a 15-line markdown-lite
(escapes HTML, renders fences, bold, inline code). The *same* router and
engine as the TUI — the desktop app is a view over one shared brain.

**Native window.** `launch_window()` builds a `pywebview` `WKWebView` window
(title, width/height, min-size) and runs the native event loop. Startup
order matters: `uvicorn` runs in a **daemon thread**, `wait_for_server()`
polls `socket.create_connection(("127.0.0.1", 7600))` until the port accepts,
*then* the window opens. `daemon=True` gives desktop-app semantics: window
close ⇒ process exit ⇒ server gone.

---

## 10. Packaging (`scripts/install_app.sh`) — how `/Applications/Prismatix.app` exists

Hand-built bundle (no PyInstaller, honestly local):

```
Prismatix.app
└── Contents
    ├── Info.plist          (id io.github.sentin3l-mrb.prismatix, version 0.1.0)
    ├── MacOS/prismatix     (bash launcher, see below)
    └── Resources/prismatix.icns
```

* **Icon.** Generated without imaging libraries: `make_icon.py` writes a
  hand-crafted 20-line PDF (dark rounded square + blue "P" in Helvetica),
  `sips` converts to PNGs (16…512 px + @2x), `iconutil` packs `.icns`.
* **Launcher.** `exec venv-python -m prismatix --web` with `cd "$REPO"` first.
  Two packaging bugs discovered live: (1) Finder starts apps with `/` as
  cwd, so without the `cd` the package wasn't importable; (2) bash heredoc
  escaping corrupted the baked-in repo path (which contains *spaces*), fixed
  by generating the launcher with Python `repr()`-quoted paths.
* **Single engine, many windows.** On launch, `wait_for_server(7600, 0.7)`
  detects an already-running engine; a second launch **attaches its window**
  instead of starting a duplicate model (no "address already in use").
* Install: `bash scripts/install_app.sh --install` → copies the bundle to
  `/Applications`, linting the plist with `plutil`.
* The compiled bundle is gitignored; the *scripts* to rebuild it are the
  product. This keeps the repo reviewable while `build/` stays regenerable.

---

## 11. Failures ledger (the pedagogical core of this document)

| # | what broke | root cause | fix | log |
|---|---|---|---|---|
| 1 | license "other" meant research-only | card metadata ≠ actual license file | model switch to Apache-2.0 | #5 |
| 2 | "Thinking Process:" leaked into answers | Qwen3.5 thinking-hybrid default | `enable_thinking=False` | #9 |
| 3 | first eval scored 10/16 with a lying harness | grader bugs | graders unit-tested first; regrades recorded | #11 |
| 4 | **LoRA v1: 0/16 degenerate output** | effective step = lr×scale too big | 10× lr cut; recipe v2 | #14–15 |
| 5 | early checkpoint also 0/6 | over-strong from the start | confirmed the failure isn't iters | #15 |
| 6 | Metal crash at iter 51 | GPU oversubscription by other processes | `max_seq_length 1024`, quiet machine | #17 |
| 7 | TUI printed stacked single tokens | RichLog renders per-`write()` | line-buffered flush | — |
| 8 | TUI-only generation crash | mlx-lm 0.32 moved temperature into a sampler object | `make_sampler(temp=…)` | #19 |
| 9 | browser 502 / can't start | `__main__` dispatch IndentationError | rewrite | #23 |
| 10 | `/rag` dead in desktop | launcher passed `retriever=None` | wire `Retriever()` | #23 |
| 11 | sqlite cross-thread crash | FastAPI worker threads | `check_same_thread=False` + lock | #23 |
| 12 | pytest recursion hung 5 min | agent's `run_tests` in tests | recursion guard + conftest marker | — |
| 13 | model never stopped calling tools | 4B models lack stopping instinct | step counters + forced synthesis turn | #20 |
| 14 | `.app` launched with no module | Finder cwd = `/` | `cd "$REPO"` in launcher | #24 |
| 15 | heredoc baked broken paths | spaces in repo path + escaping | python-generated launcher | #24 |

Reproducing any of these: `git log` + `LOG.md` + the artifacts.

---

## 12. How to run everything (reviewer checklist)

```bash
cd prismatix-copilot
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e '.[dev]'

.venv/bin/python -m pytest tests/ -q        # 27 tests, no model, <3 s
python scripts/smoke_test.py                 # downloads model (~2.4 GB), 42 tok/s
python -m eval.run_eval --out artifacts/eval_runs/base.jsonl   # 13/16 reproducible
python -m training.build_dataset && python -m training.split
bash training/train.sh                       # ~17–27 min, writes an adapter
python -m prismatix                           # terminal TUI
python -m prismatix --web                    # native window + local server
bash scripts/install_app.sh --install        # /Applications/Prismatix.app + icon
python -m prismatix --headless '/sql revenue by region'
```

Requirements: an Apple-silicon Mac (MLX is Apple-only), 16 GB unified memory
(11 GB free is enough — close the IDE), no network *after* the model and
embedding model are cached once.

---

## 13. What we do NOT claim (the integrity section)

* **Not trained from scratch.** 0.386% of a 4B model was adapted, with a LoRA,
  in minutes; the base model is Qwen's under Apache-2.0.
* **The LoRA did not make the model "smarter".** It changed output format and
  style; correctness overall was 13/16 base vs 12/16 with adapters. RAG is
  the correctness lever; that's the measured finding, and it is more
  interesting than an overclaim.
* **16 tasks, one seed.** Evidence, not a leaderboard claim.
* **No confidential data was ever sent anywhere.** No third-party APIs, no
  scraping: all documents and training data are self-authored; stored in
  the repo and inspected in the report.
* **AI-assisted development is disclosed.** The author built this with
  AI-pair-programming support, and every artifact is reproducible from this
  repo. The failures are retained on purpose: the log is as valuable as the
  code.

---

## 14. Glossary (so the numbers speak)

* **Token** — unit of text the model processes (sub-word); IDs from a fixed vocabulary.
* **Quantization** — representing weights with fewer bits (16→4) with per-group scales.
* **KV cache** — the model's working memory for attention during generation.
* **Embedding** — a vector placing texts close if they mean similar things.
* **Cosine similarity** — angle between vectors; 1 = same direction.
* **RAG** — retrieval-augmented generation: an open-book-thin (-but real) knowledge injection.
* **LoRA** — low-rank adaptation: learning thin `A·B` matrices instead of updating `W`.
* **QLoRA (effective here) — training a LoRA adapter on top of a quantized base model.
* **SSE** — server-sent events: one-way HTTP stream (our token channel).
* **WKWebView** — Apple's system browser engine; the window around our UI.
* **`plutil -lint`** — the gate that says your Info.plist is valid.

*End of the walkthrough. For the day-by-day receipts in chronological order read `LOG.md`.*
