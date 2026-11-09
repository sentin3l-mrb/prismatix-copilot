#!/usr/bin/env bash
# LoRA fine-tune of the house style on Apple Silicon (mlx-lm).
# 4-bit base + rank-16 LoRA on attention/MLP projections; loss computed on the
# assistant completion only (--mask-prompt). ~300 steps / ~20 min on M1 Pro.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_ID="${MODEL_ID:-mlx-community/Qwen3.5-4B-4bit}"

.venv/bin/python -m mlx_lm lora \
    --model "$MODEL_ID" \
    --train \
    --data training/split \
    --fine-tune-type lora \
    --rank 16 \
    --batch-size 1 \
    --iters "${ITERS:-300}" \
    --learning-rate 1e-4 \
    --mask-prompt \
    --grad-accumulation-steps 4 \
    --steps-per-report 20 \
    --steps-per-eval 100 \
    --save-every 100 \
    --adapter-path adapters/prismatix-style \
    --seed 42
