"""Embedding backend: BAAI/bge-small-en-v1.5 via fastembed (ONNX Runtime).

Why ONNX Runtime instead of PyTorch: it runs bge-small on CPU in ~90 ms per
short chunk while shipping no 2 GB PyTorch wheel — the model itself is ~110 MB.
"""

from __future__ import annotations

from functools import lru_cache

# MIT-licensed model, well supported by fastembed.
MODEL_NAME = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def model():
    """Singleton so the TUI loads weights exactly once (~110 MB RAM)."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=MODEL_NAME)
    # downloads to ~/.cache/fastembed on first use


def embed(texts: list[str]) -> list[list[float]]:
    """list[str] -> list of 384-dim float vectors (normalized)."""
    if not texts:
        return []
    return [v.tolist() for v in model().embed(texts)]
