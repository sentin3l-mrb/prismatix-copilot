# Experiment Log

Honest, append-only record of what we tried, what failed, and the numbers.

| # | date | milestone | what happened | result / numbers | notes |
|---|------|-----------|---------------|------------------|-------|
| 1 | 2026-11-09 | env | Installed Python 3.12.16 via `uv` (system python 3.9.6 too old for mlx). uv 0.13.7 | OK in 25s | `uv` made no-sudo installs trivial |
| 2 | 2026-11-09 | env | `uv venv --python 3.12` + `uv pip install -e '.[dev]'` (mlx-lm, fastembed, textual, numpy, pytest) | installed OK | mlx-lm pulls transformers; no torch needed |
| 3 | 2026-11-09 | M1 | `python scripts/smoke_test.py` direct run | ModuleNotFoundError: prismatix | lesson: `python scripts/x.py` puts `scripts/` on sys.path, not repo root; added path shim |
| 4 | 2026-11-09 | M2 | unit tests for graders offline | 3 fails -> fixed: (a) has_numbers args reversed, (b) CodeGrader ignored boolean-False returns (only caught raised asserts!), (c) subject-line regex missing re.I | exactly why we test graders before the model: silent grader bugs fake a perfect eval |
| 5 | 2026-11-09 | M1 | **License audit via HF API + model card.** Original pick `Qwen/Qwen2.5-Coder-3B-Instruct` advertises `license: other` -> actual file = **Qwen Research License** (reproduction, modification, distribution allowed **non-commercial ONLY**, §2a) | not our claimed Apache-2.0 | **Switched to `Qwen/Qwen3.5-4B`**: Apache-2.0 (report: `license: apache-2.0`), newer (2026), 5.3M downloads, mlx-community 4-bit build verified to exist. University (non-commercial) use would have been fine under either license, but Apache-2.0 has zero terms to track |
| … | | | | | |
