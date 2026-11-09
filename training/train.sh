#!/usr/bin/env bash
# LoRA fine-tune of the house style on Apple Silicon (mlx-lm).
# 4-bit base + rank-16 LoRA; loss on assistant completion only (--mask-prompt).
# ~300 steps; expect ~20-40 min on an M1 Pro with the browser closed.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_XET=1

.venv/bin/python -m mlx_lm lora --config training/lora_config.yaml
