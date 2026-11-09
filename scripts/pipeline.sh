#!/usr/bin/env bash
# Runs after the model is cached: smoke test -> base eval -> LoRA training.
# Usage: bash scripts/pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_XET=1        # xet CDN stalls on this network (LOG #8)

echo "== smoke test =="
.venv/bin/python scripts/smoke_test.py

echo "== baseline eval (16 tasks) =="
.venv/bin/python -m eval.run_eval --out artifacts/eval_runs/base.jsonl

echo "== LoRA training (~300 iters) =="
bash training/train.sh
