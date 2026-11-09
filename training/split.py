"""Split the SFT dataset into train/valid (fair, seeded, no leakage)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main(dataset: Path, out_dir: Path, n_valid: int = 40) -> None:
    rows = [json.loads(line) for line in dataset.open()]
    rng = random.Random(1234)
    rng.shuffle(rows)
    valid, train = rows[:n_valid], rows[n_valid:]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.jsonl").write_text("\n".join(map(json.dumps, train)) + "\n")
    (out_dir / "valid.jsonl").write_text("\n".join(map(json.dumps, valid)) + "\n")
    print(f"train {len(train)}  valid {len(valid)}  -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="training/dataset.jsonl")
    ap.add_argument("--out", default="training/split")
    ap.add_argument("--n-valid", type=int, default=40)
    a = ap.parse_args()
    main(Path(a.dataset), Path(a.out), a.n_valid)
