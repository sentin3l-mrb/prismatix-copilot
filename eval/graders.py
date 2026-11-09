"""Deterministic graders for the Prismatix benchmark.

Three families, chosen so every score is reproducible and honest:

1. CodeGrader   - extracts a Python code block, executes it, and runs hidden
                  asserts (like a real test suite).
2. SQLGrader    - runs the model's SQL and a reference query against the same
                  seeded SQLite database and compares the result rows.
3. RubricGrader - deterministic text checks for business writing: required
                  facts present, word budget respected, no filler phrases.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def extract_code_block(text: str, lang: str = "python") -> Optional[str]:
    """First fenced ```python block; regex handles imperfect formatting."""
    pattern = rf"```(?:{lang})?\s*\n(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return matches[0].strip() if matches else None


def extract_sql(text: str) -> Optional[str]:
    block = extract_code_block(text, "sql")
    if block:
        # strip things like ';' duplicates that break nothing but keep it tidy
        return block.strip().rstrip(";") + ";"
    # fall back: use the whole response if it looks like bare SQL
    t = text.strip().rstrip(";") + ";"
    return t if re.match(r"(?is)\s*(select|with)\b", t) else None


# --------------------------------------------------------------------------
# 1. code grader
# --------------------------------------------------------------------------
@dataclass
class CodeGrader:
    """`tests` maps name -> function(solution_ns). Each test raises
    AssertionError on failure. We report the pass rate across tests."""
    tests: Dict[str, Callable[[dict], None]]

    def __call__(self, response: str) -> Dict:
        code = extract_code_block(response)
        if code is None:
            return {"pass": False, "passed": 0, "total": len(self.tests),
                    "detail": "no python code block found"}
        ns: dict = {}
        try:
            exec(compile(code, "<solution>", "exec"), ns)  # scoped, not shell
        except Exception as e:  # syntax/runtime error in the model's code
            return {"pass": False, "passed": 0, "total": len(self.tests),
                    "detail": f"exec error: {type(e).__name__}: {e}"}

        passed = 0
        first_fail = ""
        for name, test in self.tests.items():
            try:
                outcome = test(ns)
                # tests may either return a truthy bool or use plain asserts
                ok = outcome is not False
            except AssertionError as e:
                ok, outcome = False, None
                outcome_msg = f"{name}: {e}"
            except Exception as e:           # test crashed -> fail loudly
                ok = False
                outcome_msg = f"{name} raised {type(e).__name__}: {e}"
            else:
                outcome_msg = outcome if outcome is False else ""
            if ok:
                passed += 1
            else:
                first_fail = first_fail or (locals().get("outcome_msg") or f"{name}: returned False")
        return {"pass": passed == len(self.tests), "passed": passed,
                "total": len(self.tests), "detail": first_fail or "all passed"}


# --------------------------------------------------------------------------
# 2. SQL grader
# --------------------------------------------------------------------------
BUSINESS_SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY, name TEXT, region TEXT,
    plan TEXT, signed_up DATE
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY, customer_id INTEGER,
    amount REAL, order_date DATE
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY, name TEXT, price REAL, active INTEGER
);
"""

SEED_ROWS = {
    "customers": [
        (1, "Aurora Labs", "north", "pro", "2024-01-12"),
        (2, "Bergstrom & Co", "south", "starter", "2024-02-03"),
        (3, "Creo Digital", "north", "pro", "2024-02-20"),
        (4, "Delta Freight", "east", "enterprise", "2024-03-05"),
        (5, "Ember Studio", "south", "starter", "2024-04-15"),
        (6, "Fjord Analytics", "east", "pro", "2024-05-02"),
    ],
    "orders": [
        (1, 1, 250.0, "2024-01-20"),
        (2, 1, 450.0, "2024-02-11"),
        (3, 2, 120.0, "2024-03-01"),
        (4, 3, 640.0, "2024-04-09"),
        (5, 3, 310.0, "2024-05-19"),
        (6, 4, 990.0, "2024-03-30"),
        (7, 4, 120.0, "2024-05-28"),
        (8, 5, 80.0, "2024-06-02"),
        (9, 6, 470.0, "2024-06-14"),
        (10, 6, 560.0, "2024-07-21"),
    ],
    "products": [
        (1, "Starter seat", 80.0, 1),
        (2, "Pro seat", 320.0, 1),
        (3, "Enterprise seat", 990.0, 1),
        (4, "Legacy addon", 25.0, 0),
    ],
}


def seeded_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(BUSINESS_SCHEMA)
    for table, rows in SEED_ROWS.items():
        cols = ", ".join("?" * len(rows[0]))
        conn.executemany(f"INSERT INTO {table} VALUES ({cols})", rows)
    return conn


@dataclass
class SQLGrader:
    reference_sql: str

    def __call__(self, response: str) -> Dict:
        candidate = extract_sql(response)
        if candidate is None:
            return {"pass": False, "detail": "no SQL found"}
        try:
            with seeded_db() as c1, seeded_db() as c2:
                got = c1.execute(candidate).fetchall()
                want = c2.execute(self.reference_sql).fetchall()
        except Exception as e:
            return {"pass": False, "detail": f"sqlite error: {e}"}
        ok = sorted(map(repr, got)) == sorted(map(repr, want))
        return {"pass": ok, "detail": "rows match" if ok else f"got={got!r} want={want!r}"}


# --------------------------------------------------------------------------
# 3. rubric grader (business writing)
# --------------------------------------------------------------------------
FILLER = re.compile(
    r"\b(i hope this helps|i 'm sorry|as an ai|great question|certainly!|of course!)\b",
    re.I,
)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class RubricGrader:
    """score = weighted check of deterministic conditions.

    conditions: list of (name, fn(response)->bool, weight).
    pass if score fraction >= threshold.
    """
    conditions: List
    threshold: float = 0.8

    def __call__(self, response: str) -> Dict:
        passed = 0.0
        total = sum(w for _, _, w in self.conditions)
        failed = []
        for name, fn, w in self.conditions:
            try:
                ok = fn(response)
            except Exception as e:
                ok = False
                name = f"{name} (err {e})"
            if ok:
                passed += w
            else:
                failed.append(name)
        frac = passed / total
        return {
            "pass": frac >= self.threshold,
            "score": round(frac, 3),
            "failed": failed,
            "detail": "ok" if not failed else "failed: " + "; ".join(failed),
        }


# --- reusable rubric building blocks ---------------------------------------
def has_numbers(required: List[str]) -> Callable:
    """All given figures (percentages, counts, money) appear in the reply."""
    def check(resp: str) -> bool:
        return all(num in resp for num in required)
    return check


def max_sentences(n: int) -> Callable:
    def check(resp: str) -> bool:
        # ignore headings/markdown junk; split prose only
        prose = "\n".join(
            ln for ln in resp.splitlines() if ln.strip() and not ln.strip().startswith(("#", "-"))
        )
        sents = [s for s in SENT_SPLIT.split(prose) if s.strip()]
        return 0 < len(sents) <= n
    return check


def no_filler(resp: str) -> bool:
    return FILLER.search(resp) is None


def max_words(n: int) -> Callable:
    def check(resp: str) -> bool:
        return 0 < len(resp.split()) <= n
    return check


def has_email_parts(subject: bool = True, greeting: bool = True,
                    signoff: bool = True, cta: str | None = None) -> Callable:
    def check(resp: str) -> bool:
        r = resp.lower()
        if CTA := cta:
            # any gentler variant of the call to action word set
            variants = CTA.lower().split("|")
            if not any(v in r for v in variants):
                return False
        if subject and not re.search(r"(?im)^\s*(subject\s*:|\*\*subject\b)", resp):
            return False
        if greeting and not re.search(r"(hi|hello|hey|dear)\s+\w", r):
            return False
        if signoff and not re.search(r"\b(thanks|sincerely|best|kind regards|cheers)\b", r):
            return False
        return True
    return check
