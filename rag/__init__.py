"""Local RAG: chunk knowledge-base files, embed with fastembed (ONNX, CPU,
no PyTorch), store vectors in SQLite, retrieve by cosine similarity."""

from rag.retriever import Retriever  # noqa: F401
