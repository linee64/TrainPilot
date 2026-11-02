# TrainPilot — Roadmap (MVP)

Goal: an open-source library that, on a short run, says whether the GPU is computing or waiting, tries mixed precision, and writes a report plus a snippet to paste.

Rule: **do not start the next stage until the current checklist passes.** Days are order, not a calendar. Faster is fine.

---

## Stage 0. Setup
- [x] Create the `trainpilot` GitHub repo (MIT license)
- [x] Folder layout (see `docs/ARCHITECTURE.md`)
- [x] `pyproject.toml`, `pip install -e .` works
- [x] Put 3 reference scripts in `examples/`
- [x] Colab and Kaggle accounts; confirm a T4 is available

**Done when:** `import trainpilot` works in Colab.

## Stage 1. Step timing
- [x] `measure.py`: time each step with `torch.cuda.synchronize()`
- [x] Drop 10 warmup steps; measure the next 30
- [x] Examples per second, peak memory (`torch.cuda.max_memory_allocated`)
- [x] Median, not the mean (outliers)

**Check:** run on ResNet-50, compare the step time with a hand timer using `time.time()`. Difference < 5%.

## Stage 2. Diagnosis
- [x] Time data wait (until the batch arrives) separately from compute
- [x] Wait share = wait / whole step
- [x] Rule: wait > 30% → waiting for data, otherwise GPU busy
- [x] Reason in plain language

**Check:** CIFAR → waiting for data, ResNet → busy. Both correct.

## Stage 3. Mixed-precision experiment
- [x] Copy of the model and optimizer; the original training is not touched
- [x] Same seed, same batches
- [x] `torch.autocast` FP16 + `GradScaler` (FP16 only on a T4, never BF16)
- [x] Compare median step time: baseline vs AMP
- [x] "Training did not break": loss did not diverge, no NaN
- [x] (Optional) `torch.compile`: measure compile time separately

**Check:** TinyStories — AMP is clearly faster, loss is healthy.

## Stage 4. Report and CLI
- [x] `report.py`: one-page report (markdown + console)
- [x] "Code to paste": a few lines the user inserts themselves
- [x] Label: numbers are from a short test; a full run is an estimate
- [x] "Skipped because …" for anything that was not checked
- [x] `cli.py`: `trainpilot run train.py`

**Check:** the report is readable without profiler jargon.

## Stage 5. Colab and Kaggle
- [x] Notebook: one-cell run
- [x] Run all 3 examples on a T4
- [x] Record real numbers in the table (see `docs/TESTING.md`)

**Check:** passed. 01 waiting for data, fix the loader. 02 and 03 GPU busy; on a T4, FP16 made the step faster and the loss stayed OK. Say "little is left" only when the short test itself showed no clear speedup.

## Stage 6. README
- [x] First line: what it does and how it differs from a profiler
- [x] Chart/table with real before/after numbers
- [x] Quick start
- [x] v0.1 limits

**Check:** passed. README opens with the idea and the difference from a profiler. The table uses T4 numbers from `docs/TESTING.md`.

## Stage 7. Release
- [x] Fix bugs
- [x] Tag `v0.1.0`, GitHub release
- [x] (Optional) publish to PyPI
- [x] Post on LinkedIn / X

**Check:** passed. Version 0.1.0.

---

## Not in v0.1
- Multi-GPU / distributed training
- Automatically patching someone else's script
- Batch-size search, memory tuning
- Promising a specific speedup percentage
