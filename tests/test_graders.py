"""Offline checks for the graders (no model required).

    .venv/bin/python -m pytest tests/ -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.graders import (
    CodeGrader,
    RubricGrader,
    SQLGrader,
    has_email_parts,
    has_numbers,
    max_sentences,
    max_words,
    no_filler,
)


# --- SQLGrader: the reference queries must be valid and sane -----------------
# Note: by design we compare result ROWS, not row ORDER — an answer with the
# right rows but a different ORDER BY still counts as correct.
def test_sql_grader():
    g = SQLGrader(reference_sql=(
        "SELECT c.region, SUM(o.amount) AS revenue "
        "FROM orders o JOIN customers c ON c.id = o.customer_id "
        "GROUP BY c.region ORDER BY revenue DESC"))
    assert g("SELECT ROUND(1)")["pass"] is False      # wrong query fails
    assert g("DELETE FROM orders")["pass"] is False   # not a SELECT
    equivalent = ("```sql\nSELECT c.region, SUM(o.amount) FROM orders o "
                  "JOIN customers c ON c.id=o.customer_id GROUP BY c.region;\n```")
    assert g(equivalent)["pass"] is True              # same rows (any order)


# --- CodeGrader: gold solutions pass; wrong ones fail ------------------------
def test_code_grader():
    g = CodeGrader(tests={
        "twice": lambda ns: ns["twice"](21) == 42,
        "zero": lambda ns: ns["twice"](0) == 0,
    })
    gold = "```python\ndef twice(x):\n    return 2 * x\n```"
    assert g(gold)["pass"] is True
    assert g("```python\ndef twice(x):\n    return 3 * x\n```")["passed"] == 1
    assert g("no code here")["pass"] is False


# --- Rubric: gold brief passes all conditions --------------------------------
def test_rubric_brief_gold():
    gold = ("Revenue was $6,800, up 3%. Churn fell from 3.2% to 2.4%. "
            "Two enterprise deals slipping to Q3 is the main risk.")
    g = RubricGrader(conditions=[
        ("nums", has_numbers(["6,800", "2.4"]), 1.0),
        ("len", max_sentences(3), 1.0),
        ("fill", no_filler, 1.0),
    ], threshold=1.0)
    assert g(gold)["pass"] is True

    bad = "Great question! I hope this helps: revenue was about $6.8k."
    r = g(bad)
    assert r["pass"] is False


def test_rubric_email_parts():
    g = RubricGrader(conditions=[
        ("parts", has_email_parts(cta="any time|this week"), 1.0),
        ("short", max_words(120), 1.0),
    ], threshold=1.0)
    gold = ("Subject: Meeting\nHi Tom,\nAny time this week works for me.\n"
            "Thanks,\nPrismatix")
    out = g(gold)
    assert out["pass"], out
