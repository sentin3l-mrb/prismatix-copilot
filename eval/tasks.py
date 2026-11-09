"""The Prismix benchmark: 16 checkable tasks.

- 6 coding tasks   -> hidden unit tests (CodeGrader)
- 3 SQL tasks      -> executed against a seeded SQLite DB, compared with a
                      reference query's rows (SQLGrader)
- 4 business briefs / 3 emails -> deterministic rubrics (RubricGrader)

Score types are hard (executed correctness) rather than vibes (human taste),
which keeps base-vs-LoRA comparisons honest for the write-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

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


# ==========================================================================
# 1. CODING TASKS  (hidden unit tests exercise the model's own code)
# ==========================================================================
def _raises(fn):
    def run(ns):
        try:
            fn(ns)
        except ValueError:
            return
        raise AssertionError("expected ValueError")
    return run


CODING_TASKS: List[dict] = [
    {
        "id": "code_word_count",
        "kind": "code",
        "prompt": (
            "Write a Python function `word_frequencies(text: str) -> dict` that returns "
            "a dict mapping lowercase words to counts, ignoring punctuation and case. "
            "Example: 'Dogs chase dogs.' -> {'dogs': 2, 'chase': 1}."
        ),
        "grader": CodeGrader(tests={
            "basic": lambda ns: ns["word_frequencies"]("Dogs chase dogs.") == {"dogs": 2, "chase": 1},
            "punct": lambda ns: ns["word_frequencies"]("Hello, HELLO") == {"hello": 2},
            "empty": lambda ns: ns["word_frequencies"]("") == {},
        }),
    },
    {
        "id": "code_moving_avg",
        "kind": "code",
        "prompt": (
            "Write a Python function `moving_avg(nums: list, k: int) -> list` that returns "
            "the k-window moving averages, one float per window, for every window that is "
            "fully inside the list. For  [1,2,3,4] with k=2 -> [1.5, 2.5, 3.5]. Raise ValueError if k < 1."
        ),
        "grader": CodeGrader(tests={
            "windows": lambda ns: ns["moving_avg"]([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5],
            "k_equals_len": lambda ns: ns["moving_avg"]([10], 1) == [10.0],
            "raises": _raises(lambda ns: ns["moving_avg"]([1, 2], 0)),
        }),
    },
    {
        "id": "code_churn_rate",
        "kind": "code",
        "prompt": (
            "Write a Python function `churn_by_plan(customers: list) -> dict` where each customer "
            "is a dict with keys 'plan' (str) and 'churned' (bool). Return a dict mapping each plan "
            "to its churn rate as a float in [0, 1]."
        ),
        "grader": CodeGrader(tests={
            "rates": lambda ns: abs(ns["churn_by_plan"]([
                {"plan": "pro", "churned": True},
                {"plan": "pro", "churned": False},
                {"plan": "starter", "churned": False},
            ])["pro"] - 0.5) < 1e-9,
            "starter": lambda ns: ns["churn_by_plan"]([
                {"plan": "starter", "churned": False},
            ])["starter"] == 0.0,
            "keys": lambda ns: set(ns["churn_by_plan"]([
                {"plan": "x", "churned": False}])) == {"x"},
        }),
    },
    {
        "id": "code_unique_words",
        "kind": "code",
        "prompt": (
            "Write a Python function `first_unique_words(text: str, n: int) -> list` that returns "
            "the first n words (lowercased, punctuation removed) with no repeats, in order of "
            "first appearance."
        ),
        "grader": CodeGrader(tests={
            "order": lambda ns: ns["first_unique_words"]("apple banana apple cherry", 2) == ["apple", "banana"],
            "n_larger": lambda ns: ns["first_unique_words"]("a a b", 5) == ["a", "b"],
            "strip_punct": lambda ns: ns["first_unique_words"]("Go, go, stop!", 3) == ["go", "stop"],
        }),
    },
    {
        "id": "code_overlap_days",
        "kind": "code",
        "prompt": (
            "Write a Python function `overlap_days(start1: str, end1: str, start2: str, end2: str) -> int` "
            "taking ISO date strings, returning the number of days the two [start, end) ranges overlap. "
            "Return 0 if they don't overlap."
        ),
        "grader": CodeGrader(tests={
            "partial": lambda ns: ns["overlap_days"]("2024-01-01", "2024-01-10", "2024-01-05", "2024-01-20") == 5,
            "disjoint": lambda ns: ns["overlap_days"]("2024-01-01", "2024-01-05", "2024-02-01", "2024-02-10") == 0,
        }),
    },
    {
        "id": "code_csv_scores",
        "kind": "code",
        "prompt": (
            "Write a Python function `top_scorers(csv: str) -> list` that parses CSV text with header "
            "'name,score', returning the top 3 (name, score:int) tuples sorted by score descending "
            "(stable order). Return fewer tuples when there are fewer than 3 data rows."
        ),
        "grader": CodeGrader(tests={
            "top3": lambda ns: ns["top_scorers"]("name,score\nann,5\nbob,9\ncal,7\ndee,9") == [("bob", 9), ("dee", 9), ("cal", 7)],
            "shorter": lambda ns: ns["top_scorers"]("name,score\nann,4") == [("ann", 4)],
        }),
    },
]


# ==========================================================================
# 2. BUSINESS -> SQL  (verified against a seeded SQLite DB)
# ==========================================================================
SQL_TASKS: List[dict] = [
    {
        "id": "sql_region_revenue",
        "kind": "sql",
        "prompt": (
            "Database schema: customers(id, name, region, plan, signed_up); "
            "orders(id, customer_id, amount, order_date). "
            "Write a SQL query that returns each customer region and its total revenue, "
            "ordered by revenue descending."
        ),
        "grader": SQLGrader(reference_sql=(
            "SELECT c.region, SUM(o.amount) AS revenue "
            "FROM orders o JOIN customers c ON c.id = o.customer_id "
            "GROUP BY c.region ORDER BY revenue DESC"
        )),
    },
    {
        "id": "sql_top_customers",
        "kind": "sql",
        "prompt": (
            "Same schema (customers: id, name, region, plan, signed_up; orders: id, customer_id, "
            "amount, order_date). Return the name AND total order revenue of the 3 customers with "
            "the highest total revenue, highest first. Ties broken by name."
        ),
        "grader": SQLGrader(reference_sql=(
            "SELECT c.name, SUM(o.amount) AS rev FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "GROUP BY c.id ORDER BY rev DESC, c.name ASC LIMIT 3"
        )),
    },
    {
        "id": "sql_plan_counts",
        "kind": "sql",
        "prompt": (
            "Same schema. How many customers are on each plan? Return plan and customer "
            "count, ordered by count descending."
        ),
        "grader": SQLGrader(reference_sql=(
            "SELECT plan, COUNT(*) FROM customers GROUP BY plan ORDER BY COUNT(*) DESC"
        )),
    },
]


# ==========================================================================
# 3. BUSINESS WRITING  (rubric grader)
# ==========================================================================
BRIEF_STYLE = (
    "Write a short status/update summary for a busy manager. Style: facts first, "
    "numbers included, 3 sentences max, no greetings, no filler."
)

BRIEF_TASKS: List[dict] = [
    {
        "id": "brief_q2_update",
        "kind": "brief",
        "prompt": (
            BRIEF_STYLE + " Input: 'Q2 revenue $48,200, up 12% from Q1. Churn rate fell from 3.1% "
            "to 2.4%. Biggest risk: 2 enterprise deals slipped to Q3.'"
        ),
        "grader": RubricGrader(conditions=[
            ("numbers", has_numbers(["48,200", "2.4"]), 1.0),
            ("concise", max_sentences(3), 1.0),
            ("no_fill", no_filler, 1.0),
        ], threshold=1.0),
    },
    {
        "id": "brief_ship_delay",
        "kind": "brief",
        "prompt": (
            BRIEF_STYLE + " Input: 'The reporting feature ships 5 days late (originally March 14, now "
            "March 19). QA found 3 minor bugs, all fixed. No license or billing impact.'"
        ),
        "grader": RubricGrader(conditions=[
            ("numbers", has_numbers(["5", "3"]), 1.0),
            ("concise", max_sentences(3), 1.0),
            ("no_fill", no_filler, 1.0),
        ], threshold=1.0),
    },
    {
        "id": "brief_support",
        "kind": "brief",
        "prompt": (
            BRIEF_STYLE + " Input: 'Support tickets this week: 41 (down from 63). Median first-reply time "
            "1 hour 20 minutes. We hired one new support engineer starting Monday.'"
        ),
        "grader": RubricGrader(conditions=[
            ("numbers", has_numbers(["41", "63"]), 1.0),
            ("concise", max_sentences(3), 1.0),
            ("no_fill", no_filler, 1.0),
        ], threshold=1.0),
    },
    {
        "id": "brief_plans",
        "kind": "brief",
        "prompt": (
            BRIEF_STYLE + " Input: 'March: 54 new customers (target 60). Starter plan 40% of signups. "
            "Pro revenue share up to 33%. Upgrade offer email went to 900 customers Tuesday.'"
        ),
        "grader": RubricGrader(conditions=[
            ("numbers", has_numbers(["54", "60"]), 1.0),
            ("concise", max_sentences(3), 1.0),
            ("no_fill", no_filler, 1.0),
        ], threshold=1.0),
    },
]


EMAIL_TASKS: List[dict] = [
    {
        "id": "email_followup",
        "kind": "email",
        "prompt": (
            "Draft a short, polite follow-up email to a prospective customer, Dana Whitfield, who "
            "asked for pricing two weeks ago and hasn't replied. Include a subject line, greeting, "
            "reasonable call to action (e.g. 'any time this week'), and sign-off. Under 120 words."
        ),
        "grader": RubricGrader(conditions=[
            ("parts", has_email_parts(cta="any time|this week"), 1.0),
            ("short", max_words(120), 1.0),
            ("no_fill", no_filler, 1.0),
        ], threshold=1.0),
    },
    {
        "id": "email_apology",
        "kind": "email",
        "prompt": (
            "Draft a brief apology email to a customer, Marcus Lee, whose Friday report failed to "
            "generate due to a bug we introduced (fixed now). Subject line, sincere apology, what "
            "changed, no overpromises. Under 150 words."
        ),
        "grader": RubricGrader(conditions=[
            ("parts", has_email_parts(subject=True, cta="fixed|apolog|sorry"), 1.0),
            ("short", max_words(150), 1.0),
            ("no_fill", no_filler, 1.0),
        ], threshold=1.0),
    },
    {
        "id": "email_meeting",
        "kind": "email",
        "prompt": (
            "Draft a concise internal email inviting the team to a 30-minute pricing review on "
            "Thursday at 3pm. Ask people to bring their top 2 proposals. Subject line, greeting, "
            "agenda, sign-off, under 100 words."
        ),
        "grader": RubricGrader(conditions=[
            ("parts", has_email_parts(cta="bring"), 1.0),
            ("short", max_words(100), 1.0),
            ("no_fill", no_filler, 1.0),
        ], threshold=1.0),
    },
]


ALL_TASKS: List[dict] = CODING_TASKS + SQL_TASKS + BRIEF_TASKS + EMAIL_TASKS
