"""Thin wrapper around mlx-lm for local inference on Apple Silicon.

Design notes
------------
* Lazily loads a small open-weight model (default: Qwen3.5-4B, Apache-2.0,
  4-bit quantized). An optional LoRA adapter path can be passed;
  mlx-lm merges it at load time (adapter weights only, ~tens of MB).
* Every generation returns a GenResult with real throughput/memory stats so we
  can log honest numbers in LOG.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

# 2026-11-09 license audit, see LOG.md #3: switched from Qwen2.5-Coder-3B-Instruct
# (Qwen Research License, non-commercial-only) to Qwen3.5-4B — Apache-2.0, newer,
# and better quality-per-GB on Apple Silicon. Verify with:
#   curl -s https://huggingface.co/api/models/Qwen/Qwen3.5-4B | python3 -m json.tool | grep -i license
DEFAULT_MODEL_ID = "mlx-community/Qwen3.5-4B-4bit"

# The shared persona for the copilot. "Business mode" output shapes are
# trained into the model with our LoRA adapter (see training/).
SYSTEM_PROMPT = (
    "You are Prismatix, a concise local assistant for developers and small businesses."
    " For coding tasks: give a short plan, then a complete, runnable code block, then"
    " one or two notes on edge cases. For SQL: give only the query, then one sentence"
    " explaining it. For business summaries: write 3 short sentences, facts first,"
    " numbers included, no filler."
)


@dataclass
class GenResult:
    """One full completion plus the honest numbers we keep in the log."""

    text: str
    prompt_tokens: int
    output_tokens: int
    seconds: float
    tps: float = 0.0
    peak_memory_gb: float = 0.0
    model_id: str = ""
    adapter: Optional[str] = None

    def as_row(self, task_id: str = "", score: str = "") -> str:
        return (
            f"{task_id:<12} model={self.model_id} adapter={self.adapter or 'none'} "
            f"out_tok={self.output_tokens} sec={self.seconds:.1f} "
            f"tps={self.tps:.1f} osc_mem={self.peak_memory_gb:.2f}GB {score}"
        )


@dataclass
class LocalEngine:
    model_id: Optional[str] = None   # None -> DEFAULT_MODEL_ID (see __post_init__)
    adapter_path: Optional[str] = None
    max_kv_size: int = 4096

    _model: object = field(default=None, init=False, repr=False)
    _tokenizer: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.model_id is None:
            self.model_id = DEFAULT_MODEL_ID

    # -- lifecycle -----------------------------------------------------------
    def load(self) -> None:
        """Load model + tokenizer once, then reuse for many generations."""
        if self._model is not None:
            return
        from mlx_lm import load

        self._model, self._tokenizer = load(
            self.model_id,
            adapter_path=self.adapter_path,
        )

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None

    # -- prompting -----------------------------------------------------------
    def prompt_for(
        self,
        messages: List[dict],
        system: Optional[str] = None,
        add_generation_prompt: bool = True,
        thinking: bool = False,
    ) -> str:
        msgs = []
        sys = system or SYSTEM_PROMPT
        if sys:
            msgs.append({"role": "system", "content": sys})
        msgs.extend(messages)
        # Qwen3.5 is a thinking hybrid; its chat template supports
        # `enable_thinking`. Copilots want direct answers, so default False.
        # (Templates that don't know this kwarg simply ignore it.)
        return self._tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=thinking,
        )

    def count_tokens(self, messages: List[dict], system: Optional[str] = None) -> int:
        return len(self._tokenizer.encode(self.prompt_for(messages, system=system)))

    # -- one-shot generation --------------------------------------------------
    def chat(
        self,
        messages: List[dict],
        max_tokens: int = 400,
        temperature: float = 0.2,
        seed: Optional[int] = None,
        system: Optional[str] = None,
    ) -> GenResult:
        """Non-streaming completion with stats. Good for eval runs (fast + simple)."""
        import mlx.core as mx
        from mlx_lm import generate as lm_generate

        self.load()
        if seed is not None:
            mx.random.seed(seed)

        prompt = self.prompt_for(messages, system=system)
        prompt_tokens = len(self._tokenizer.encode(prompt))

        t0 = time.perf_counter()
        text = lm_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        seconds = time.perf_counter() - t0
        out_tokens = len(self._tokenizer.encode(text))
        return GenResult(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=out_tokens,
            seconds=seconds,
            tps=out_tokens / max(seconds, 1e-6),
            model_id=self.model_id,
            adapter=self.adapter_path,
        )

    # -- streaming (what the TUI uses) ---------------------------------------
    def stream(
        self,
        messages: List[dict],
        max_tokens: int = 600,
        temperature: float = 0.2,
        system: Optional[str] = None,
    ) -> Iterator[str]:
        """Yield token-by-token text; richer UX for the live TUI."""
        from mlx_lm import stream_generate

        self.load()
        prompt = self.prompt_for(messages, system=system)
        for chunk in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            # mlx-lm versions differ: older yield str, newer yield objects with .text
            text = getattr(chunk, "text", None)
            if text is not None:
                yield text
            elif isinstance(chunk, str):
                yield chunk
