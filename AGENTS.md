# AGENTS.md — TrainPilot

Instructions for AI coding agents working in this repo. Read this before making changes.

## What this project is
TrainPilot is an open-source Python library for PyTorch. On a short run (~30 steps after warmup) it measures
whether the GPU is computing or waiting for data, tries mixed precision (and optionally `torch.compile`),
checks that training did not break, and writes a one-page report plus a code snippet the user pastes in themselves.

Goal of v0.1: honest diagnosis and an honest measurement. Not "speed up any model".

## Docs to follow
- `ROADMAP.md` — stages and checklists. Work stage by stage; do not start the next stage until the current one passes its check.
- `ARCHITECTURE.md` — module layout and API.
- `TESTING.md` — expected results on the 3 reference examples.

## Repo layout
- `trainpilot/measure.py` — step timing, data-wait time, memory
- `trainpilot/diagnose.py` — "waiting for data" vs "GPU busy"
- `trainpilot/experiments.py` — baseline vs AMP comparison, loss/NaN check
- `trainpilot/report.py` — report + code snippet
- `trainpilot/cli.py` — `trainpilot run train.py`
- `examples/` — 3 reference training scripts (do not "optimize" them; they are slow on purpose)

## Measurement rules (never break these)
- Always call `torch.cuda.synchronize()` before reading a timer on GPU.
- Exclude warmup steps (default 10) from all stats.
- Use the median step time, not the mean.
- Baseline and experiment use the same seed, same batches, same conditions.
- Measure one-time setup cost (e.g. `torch.compile` compilation) separately from per-step time.
- Every number in the report is labeled as a short-run result; full-run time is an estimate.

## Hardware rules
- Target: single NVIDIA GPU, mainly free T4 on Colab / Kaggle.
- On T4 use FP16 + `GradScaler`. Never BF16 on T4.
- Do not raise DataLoader `num_workers` inside notebooks if it breaks the run.
- On CPU-only machines: do not crash, report "GPU not found".

## Safety rules
- Never modify the user's original model, optimizer or script. Experiments run on copies.
- Never apply a change silently. If unsure a change is safe, skip it and write "skipped because …".
- "Training did not break" means: loss did not diverge and there are no NaN. Do not claim the model's quality is unchanged.

## Out of scope for v0.1 (do not implement)
- Multi-GPU / distributed training
- Automatically patching the user's script
- Batch size tuning, memory optimization
- Frameworks other than plain PyTorch loops
- Promising a specific speedup percentage

## How to verify a change
1. `pip install -e .`
2. Run `pytest` (must pass, including on CPU).
3. On a GPU, run TrainPilot on all 3 examples and compare with `TESTING.md`:
   - 01 CIFAR → "waiting for data"
   - 02 TinyStories → AMP gives a clear speedup, loss OK
   - 03 ResNet-50 → "GPU busy". On T4, FP16 still gave a clear speedup and the loss stayed OK, so recommend mixed precision. Say "little to gain" only when that short test itself shows no clear speedup.
4. Sanity-check one timing by hand with `time.time()` + `synchronize()` (within ~5%).

## Style
- Python 3.10+, type hints, small functions.
- Report text in plain language, no profiler jargon on the first screen.
- Keep dependencies minimal: `torch` only for the core library.
