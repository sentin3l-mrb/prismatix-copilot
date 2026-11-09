# Northwind Cloud API Style Guide (synthetic internal doc)

## Python code style
- Functions take plain data (dicts, lists), return plain data. No global state.
- Every public function has a one-line docstring with an example.
- Money amounts are `float` dollars; times are ISO-8601 strings; counts are `int`.
- Errors: raise `ValueError` for bad input, never return -1 or None-on-error.

## Database
- The `northwind` production database has three tables:
  - `customers(id, name, region, plan, signed_up)`
  - `orders(id, customer_id, amount, order_date)`
  - `products(id, name, price, active)`
- Regions are: north, south, east, west. Plans are: starter, pro, enterprise.
- `orders.amount` is net revenue after discounts.

## Reporting conventions
- "Revenue" always means SUM(orders.amount) in the reporting period.
- Churned customer = a customer whose plan was set to the value 'cancelled'.
- Weekly reports use Monday as the first day of the week.
