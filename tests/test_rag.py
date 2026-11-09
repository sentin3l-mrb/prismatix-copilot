"""Retrieval quality: each query should surface the right source chunk.

    .venv/bin/python -m pytest tests/test_rag.py -q
    Uses the cached fastembed model (~110 MB) and runs in seconds.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CASES = [
    ("How long do I have to cancel for a full refund?", "company_handbook.md"),
    ("What are the first reply and resolution time targets?", "support_playbook.md"),
    ("What does the Enterprise plan cost per seat?", "company_handbook.md"),
    ("Which tables and columns are in the northwind database?", "api_style_guide.md"),
    ("How is churn rate defined and reported?", "business_brief_style.md"),
]


@pytest.mark.parametrize("query,expected_source", CASES)
def test_retrieval_hits_right_source(query, expected_source):
    from rag.retriever import Retriever

    hits = Retriever().retrieve(query, k=3)
    assert {h["source"] for h in hits}, "no index — run /index first"
    assert expected_source in {h["source"] for h in hits}, \
        f"{query!r} -> {[h['source'] for h in hits]}"


def test_retriever_prompt_block_format():
    from rag.retriever import Retriever

    block = Retriever().retrieve_for_prompt("refund window")
    assert "source:" in block and "---" in block
