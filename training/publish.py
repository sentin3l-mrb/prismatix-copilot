"""Publish the Prismatix fine-tune to the author's Hugging Face account.

Run on the author's machine with a WRITE token:

    export HF_TOKEN=hf_xxx
    .venv/bin/python -m training.publish --hf-id <your-name>/prismatix-style-v2

Uploads three artifacts:
  1. the sorted-LoRA adapter (adapters.safetensors + adapter_config.json)
  2. a model card (README.md) with YAML header: license, base model,
     dataset, benchmark snippet, usage
  3. the companion dataset repo (400 self-authored SFT examples)
Everything is Apache-2.0/MIT consistent with the repo. Nothing is uploaded
without the token; the token is never written into any repo file.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ADAPTER_DIR = Path("adapters/prismatix-style-v2")
DATASET_FILE = Path("training/dataset.jsonl")

BASE_MODEL_EVAL = (
    "Benchmark: 16 tasks with deterministic checkable graders. Base model "
    "13/16 (81%); this adapter 12/16 (75%) with the 'manager brief' category "
    "improved from 2/4 to 4/4. The adapter changes output **style/format**; "
    "it does not add knowledge."
)

MODEL_CARD = """---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
tags:
- mlx
- lora
- code
- business-writing
- local-llm
- apple-silicon
datasets:
- {dataset_id}
- Qwen/Qwen3.5-4B
---

# Prismatix style adapter — a LoRA fine-tune of Qwen3.5-4B

A rank-16 LoRA adapter (16.2M trainable parameters, 0.386% of the base)
trained on **400 self-authored SFT examples** to give a small local model a
consistent "developer & business copilot" answer format:

* code: one-line plan → complete `python` block → edge-case note
* SQL: query first (fenced), then one explanatory sentence
* manager briefs: ≤3 sentences, numbers stated verbatim, no filler
* emails: subject line, greeting, concrete call to action, short sign-off

Experiments, failures (including a first recipe that produced degenerate
generations and was diagnosed in the log), the benchmark harness, and the
terminal/desktop app live at
https://github.com/sentin3l-mrb/prismatix-copilot — everything runs **fully
locally** on Apple Silicon via MLX.

{eval_block}

## Usage (Apple Silicon)

```bash
pip install mlx-lm
python -m mlx_lm.chat --model mlx-community/Qwen3.5-4B-4bit \\
    --adapter-path {hf_id}

# or, in Python:
from mlx_lm import load, generate
model, tokenizer = load("mlx-community/Qwen3.5-4B-4bit", adapter_path="{hf_id}")
```

Prompt-style defaults for the brief persona:

```
Write a 3-sentence manager brief from these notes. Numbers first, no
greetings, no filler.
Notes: <your notes here>
```

## Honest scope

Style format only — the adapter does not add world knowledge or improve
correctness beyond the base model (verified against a 16-task executed
benchmark, see the project README). Intended for local, non-production use;
SQLite/agents/write-tools are out of scope for this card.
"""

DATASET_CARD = """---
license: apache-2.0
---

# prismatix-sft

400 chat-format SFT examples, deterministic, **self-authored from templates**
(no scraping, no third-party API generation):
150 code answers · 90 business→SQL · 80 executive briefs · 80 emails.
Used to train the {model_id} LoRA adapter.

Format: JSONL, one example per line: `{"messages": [system, user, assistant]}`.
"""


def publish(hf_id: str, dataset_id: str) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN first (token is never stored/committed)")
    api = HfApi(token=token)

    api.create_repo(repo_id=hf_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_type="model", repo_id=hf_id, folder_path=ADAPTER_DIR,
        commit_message="prismatix style LoRA adapter (rank-16, 400 SFT examples)",
    )
    card = MODEL_CARD.format(dataset_id=dataset_id, hf_id=hf_id,
                             eval_block=BASE_MODEL_EVAL,
                             model_id=hf_id)
    api.upload_file(repo_type="model", repo_id=hf_id, path_or_pathobj="README.md",
                    path_in_repo="README.md",
                    path_or_fileobj=card.encode(),
                    commit_message="model card: usage + honest scope")

    api.create_repo(repo_id=dataset_id, repo_type="dataset", exist_ok=True)
    api.upload_file(repo_type="dataset", repo_id=dataset_id,
                    path_or_pathobj="dataset.jsonl", path_in_repo="dataset.jsonl",
                    path_or_fileobj=DATASET_FILE.read_text().encode(),
                    commit_message="400 self-authored SFT examples")
    dcard = DATASET_CARD.format(model_id=hf_id)
    api.upload_file(repo_type="dataset", repo_id=dataset_id,
                    path_or_pathobj="README.md", path_in_repo="README.md",
                    path_or_fileobj=dcard.encode(),
                    commit_message="dataset card")

    n = sum(1 for _ in DATASET_FILE.open())
    who = api.whoami()["name"]
    print(f"published as {who}:")
    print(f"  model  https://huggingface.co/{hf_id}")
    print(f"  data   https://huggingface.co/datasets/{dataset_id}  ({n} examples)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", required=True, help="e.g. yourname/prismatix-style-v2")
    ap.add_argument("--dataset-id", default=None,
                    help="default: same owner as --hf-id with /prismatix-sft")
    a = ap.parse_args()
    ds = a.dataset_id or (a.hf_id.split("/")[0] + "/prismatix-sft")
    publish(a.hf_id, ds)
