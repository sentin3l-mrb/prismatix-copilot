# Coding task conventions (synthetic, for /code mode)

For code answers inside Prismatix:
1. First line: one-sentence plan.
2. Then a complete fenced ```python code block. No "..." placeholders, no TODOs.
3. Last line: one note about an edge case.

Accepted data shapes are the same as Northwind's API style guide:
dicts/lists in and out, float money, ISO-8601 date strings, int counts.
Raise ValueError for invalid input; never return None silently.

Example shape of a good answer:
    Plan: fold the list into a dict keyed by region.
    ```python
    def revenue_by_region(orders):
        """Return total revenue per region, e.g. {'north': 250.0}."""
        ...
    ```
    Note: assumes amounts are non-negative.
