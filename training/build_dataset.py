"""Build the Prismatix SFT dataset.

Design principles (kept honest for the report):
1. SELF-AUTHORED: every template hand-writes the ideal answer, then randomizes
   surface details (names, numbers, phrasing) with a fixed seed. No scraped
   data, no third-party API calls, no license risk.
2. It teaches STYLE + FORMAT (the house answer shape), not new knowledge —
   so a 3B model can learn it in a few hundred steps on a laptop.
3. Benchmark tasks (eval/tasks.py) are DIFFERENT instances: the eval tests
   generalization of the format, not memorized items. Function names,
   signatures and numbers all differ from the eval set.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from prismatix.engine import SYSTEM_PROMPT

SEED = 20260101

COMPANIES = ["Bright Harbor", "Ledger Lane", "Courier & Sons", "Windmill Foods",
             "Vector Freight", "Alpine Goods", "Bluecrest Labs"]
PEOPLE = ["Dana W.", "Marcus L.", "Priya S.", "Tom R.", "Elena K.", "Sam O."]


# ---------------------------------------------------------------------------
# brief task: notes -> 3-sentence manager brief
# ---------------------------------------------------------------------------
def brief_pair(rng) -> tuple[str, str]:
    revenue = rng.randrange(8, 90) * 100 + 200
    prev = max(6, int(revenue * rng.uniform(0.84, 1.18)))
    delta = revenue - prev
    pct = abs(delta) / prev * 100
    churn_new = rng.randrange(18, 41) / 10
    churn_old = round(churn_new + rng.uniform(0.2, 1.4), 1)
    risk = rng.choice([
        "two enterprise deals may slip to next quarter",
        "one support engineer is out this week",
        "API latency rose on the east cluster",
    ])
    notes = (f"Revenue ${revenue:,} ({'up' if delta >= 0 else 'down'} {pct:.0f}%). "
             f"Churn {churn_old}% -> {churn_new}%. Risk: {risk}.")
    head = (f"Revenue was ${revenue:,}, up {pct:.0f}%." if delta >= 0
            else f"Revenue fell to ${revenue:,}, down {pct:.0f}%.") if abs(delta) > 0 else \
           f"Revenue held at ${revenue:,}."
    ideal = f"{head} Churn improved from {churn_old}% to {churn_new}%. " \
            f"{risk.capitalize()} is the main risk."
    user = ("Write a 3-sentence manager brief from these notes. Numbers first, no greetings, "
            "no filler.\nNotes: " + notes)
    return user, ideal


# ---------------------------------------------------------------------------
# code tasks (names/signatures differ from the eval)
# ---------------------------------------------------------------------------
def code_revenue_by_region(rng) -> tuple[str, str]:
    func = rng.choice(["revenue_by_region", "total_per_region"])
    user = (
        f"Write a Python function `{func}(rows: list) -> dict` where each row is a dict with "
        "keys 'amount' (float) and 'region' (str). Return total revenue per region."
    )
    ans = f'''Plan: fold the rows into a dict keyed by region.

```python
def {func}(rows: list) -> dict:
    """Return total revenue per region, e.g. {{"north": 250.0}}."""
    totals = {{}}
    for row in rows:
        totals[row["region"]] = totals.get(row["region"], 0.0) + row["amount"]
    return totals
```

Note: assumes amounts are non-negative; verify data before trusting totals.'''
    return user, ans


def code_top_customers(rng) -> tuple[str, str]:
    n = rng.choice([2, 3])
    func = rng.choice(["top_spenders", "best_customers"])
    user = (
        f"Write a Python function `{func}(orders: list, n: int = {n}) -> list` where each order is "
        "a dict with keys 'customer' (str) and 'amount' (float). Return the top n customers by "
        "total order amount, highest first; ties broken alphabetically."
    )
    ans = f'''Plan: aggregate spend per customer, sort with a stable tie-break, then slice.

```python
def {func}(orders: list, n: int = {n}) -> list:
    """Return the {n} customers with the highest total spend."""
    totals = {{}}
    for o in orders:
        totals[o["customer"]] = totals.get(o["customer"], 0.0) + o["amount"]
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:n]
```

Note: returns fewer than n entries when there are fewer customers than n.'''
    return user, ans


def code_churn(rng) -> tuple[str, str]:
    func = rng.choice(["churn_rate", "churn_fraction"])
    key = rng.choice(["plan", "tier"])
    user = (
        f"Write a Python function `{func}(customers: list) -> dict` mapping each '{key}' "
        f"to its churn rate in [0, 1]. Each customer dict has keys '{key}' (str) "
        "and 'cancelled' (bool)."
    )
    ans = f'''Plan: count total and cancelled per {key}, then divide.

```python
def {func}(customers: list) -> dict:
    """Return churn rate per {key}, e.g. {{"pro": 0.25}}."""
    total, cancelled = {{}}, {{}}
    for c in customers:
        k = c["{key}"]
        total[k] = total.get(k, 0) + 1
        cancelled[k] = cancelled.get(k, 0) + int(c["cancelled"])
    return {{k: cancelled[k] / total[k] for k in total}}
```

Note: keys with zero cancellations return 0.0, which matches the churn definition.'''
    return user, ans


def code_window(rng) -> tuple[str, str]:
    func = rng.choice(["window_mean", "rolling_mean"])
    user = (
        f"Write a Python function `{func}(vals: list, win: int) -> list` returning the mean of "
        "each fully-included window of length win as floats. Raise ValueError when "
        "win < 1 or win > len(vals)."
    )
    ans = f'''Plan: slide a window across the list and average each one.

```python
def {func}(vals: list, win: int) -> list:
    """Return window means, e.g. {func}([1, 2, 3], 2) == [1.5, 2.5]."""
    if win < 1 or win > len(vals):
        raise ValueError("win must satisfy 1 <= win <= len(vals)")
    return [sum(vals[i:i + win]) / win for i in range(len(vals) - win + 1)]
```

Note: output length is len(vals) - win + 1, never empty for valid input.'''
    return user, ans


CODE_TEMPLATES = [code_revenue_by_region, code_top_customers, code_churn, code_window]


# ---------------------------------------------------------------------------
# business -> SQL (same schema, different questions than the eval)
# ---------------------------------------------------------------------------
def sql_pair(rng) -> tuple[str, str]:
    schema = ("Database schema: customers(id, name, region, plan, signed_up); "
              "orders(id, customer_id, amount, order_date).")
    explainers = [
        "This joins orders to customers, filters the period, then groups and sorts as asked.",
        "The count is computed over orders joined with customers using the filtered columns.",
        "It filters active products below the price cap and sorts cheapest first.",
    ]
    kind = rng.randrange(3)
    if kind == 0:
        month = f"{rng.randrange(1, 13):02d}"
        day = f"{rng.randrange(1, 28):02d}"
        user = (schema + "\nWrite SQL: total revenue per plan for orders after "
                f"2024-{month}-{day}, sorted alphabetically by plan.")
        sql = ("SELECT c.plan, SUM(o.amount) FROM orders o JOIN customers c "
               f"ON c.id = o.customer_id AND o.order_date > '2024-{month}-{day}' "
               "GROUP BY c.plan ORDER BY c.plan;")
    elif kind == 1:
        min_amt = rng.randrange(1, 6) * 100
        reg = rng.choice(["north", "south", "east"])
        user = (schema + "\nWrite SQL: how many orders over $" + str(min_amt) +
                f" came from customers in the '{reg}' region?")
        sql = ("SELECT COUNT(*) FROM orders o JOIN customers c "
               f"ON c.id = o.customer_id WHERE o.amount > {min_amt} "
               f"AND c.region = '{reg}';")
    else:
        cap = rng.randrange(2, 10) * 100
        user = (schema + "\nWrite SQL: active products priced under " + str(cap) +
                ", name and price, cheapest first.")
        sql = (f"SELECT name, price FROM products WHERE active = 1 AND price < {cap} "
               "ORDER BY price;")
    ans = f"```sql\n{sql}\n```\n" + explainers[kind]
    return user, ans


# ---------------------------------------------------------------------------
# email tasks
# ---------------------------------------------------------------------------
def email_pair(rng) -> tuple[str, str]:
    who = rng.choice(PEOPLE)
    company = rng.choice(COMPANIES)
    first = who.split()[0]
    kind = rng.randrange(3)
    if kind == 0:
        user = (f"Draft a short follow-up email to {who} at {company} about an invoice "
                "30 days past due. Subject line, greeting, one concrete next step, "
                "sign-off, under 120 words.")
        ans = (f"**Subject:** Invoice reminder — {company}\n\n"
               f"Hi {first},\n\n"
               f"Our invoice for {company} is 30 days past due. Could you confirm when it "
               "will be paid, or tell me if anything is blocking the process?\n\n"
               "Thanks,\nPrismatix Support")
    elif kind == 1:
        user = (f"Draft a brief email scheduling a 20-minute renewal call with {who} at "
                f"{company}. Subject, greeting, propose two concrete times, sign-off, "
                "under 100 words.")
        ans = (f"**Subject:** Renewal call — {company}\n\n"
               f"Hi {first},\n\n"
               "The call covers renewal options and what changed since your last term. "
               "I can do Tuesday at 2pm or Thursday at 4pm.\n\n"
               "Thanks,\nPrismatix Support")
    else:
        user = (f"Draft a short good-news email to {who} saying their feature request "
                "(CSV export) shipped today. Subject line, greeting, where to find it, "
                "sign-off, under 100 words.")
        ans = (f"**Subject:** CSV export is live\n\n"
               f"Hi {first},\n\n"
               "The feature you requested shipped today. Find it under Reports -> Export; "
               "older reports export there too.\n\n"
               "Thanks,\nPrismatix Support")
    return user, ans


# ---------------------------------------------------------------------------
# main builder: each row is chat-format JSONL {"messages": [...]}
# ---------------------------------------------------------------------------
def build(out_path: Path, n_code=150, n_sql=90, n_brief=80, n_email=80) -> None:
    rng = random.Random(SEED)
    rows: list[dict] = []

    for _ in range(n_code):
        u, a = rng.choice(CODE_TEMPLATES)(rng)
        rows.append(_pair(u, a))
    for _ in range(n_sql):
        u, a = sql_pair(rng)
        rows.append(_pair(u, a))
    for _ in range(n_brief):
        u, a = brief_pair(rng)
        rows.append(_pair(u, a))
    for _ in range(n_email):
        u, a = email_pair(rng)
        rows.append(_pair(u, a))

    rng.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} examples -> {out_path}")


def _pair(user: str, assistant: str) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training/dataset.jsonl")
    args = ap.parse_args()
    build(Path(args.out))
