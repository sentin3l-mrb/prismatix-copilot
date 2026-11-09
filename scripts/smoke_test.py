"""Milestone 1 smoke test: run the base model locally and measure it.

Run:  python scripts/smoke_test.py
"""

import json
import platform
import sys
import time
from pathlib import Path

# allow running this file directly: `python scripts/smoke_test.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prismatix.engine import LocalEngine


def main():
    print(f"python {platform.python_version()} on {platform.machine()} (Apple Silicon: arm64)")
    engine = LocalEngine()

    t0 = time.perf_counter()
    engine.load()
    print(f"loaded {engine.model_id} in {time.perf_counter() - t0:.1f}s")

    # -- tokenization sanity check ------------------------------------------
    ids = engine._tokenizer.encode("SELECT region, SUM(revenue) FROM sales GROUP BY region;")
    pieces = [engine._tokenizer.decode([i]) for i in ids]
    print(f"tokens ({len(ids)}): {json.dumps(pieces)}\n")

    # -- a realistic hybrid task ---------------------------------------------
    t0 = time.perf_counter()
    result = engine.chat([{
        "role": "user",
        "content": (
            "Write a Python function `summarize_orders(orders)` that takes a list of dicts with "
            "keys 'amount' (float) and 'region' (str) and returns total revenue per region as a "
            "dict. Then write a 2-sentence update for a product manager about what it tracks."
        ),
    }], max_tokens=400, seed=42)
    print(result.text)
    print("\n--- honest numbers ---")
    print(f"prompt tokens: {result.prompt_tokens}  output tokens: {result.output_tokens}")
    print(f"generation: {result.tps:.1f} tok/s in {result.seconds:.1f}s")


if __name__ == "__main__":
    main()
