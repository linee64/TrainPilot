# TrainPilot

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

TrainPilot finds out why your PyTorch training is slow on free Colab/Kaggle GPUs and gives you a ready-to-paste fix.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/linee64/TrainPilot/blob/main/notebooks/trainpilot_quickstart.ipynb)

## Why

On a free T4, a training step is often slow for one of two reasons: the GPU is waiting for the next batch, or it is busy doing heavy math. Students usually cannot tell which, so they turn on mixed precision or `torch.compile` and hope. TrainPilot measures a short run, names the cause, and only suggests a change that made *this* short test faster without blowing up the loss.

## What it does

- Splits each step into data wait vs compute, then says **waiting for data** or **GPU busy**.
- Tries mixed precision (FP16 + `GradScaler`; never BF16 on a T4).
- Optionally tries `torch.compile` on top of AMP, with prepare time counted separately and payback in steps.
- Checks that loss did not diverge and there are no NaN. That is not a claim that model quality is unchanged.
- Writes a one-page report and a short snippet. **It never edits your script.**

Warmup steps (default 10) are dropped. Stats use the median of the next 30 steps. Baseline and experiment share the same seed and the same batches. Without a GPU the report says `GPU not found` and does not crash.

## Quick start

```bash
pip install git+https://github.com/linee64/TrainPilot.git
```

Your script must call `measure_batches` and `compare_mixed_precision`; `trainpilot run` runs it and saves the report to `trainpilot-report.md`.

```python
# train.py
import torch
from torch import nn
from trainpilot import compare_mixed_precision, diagnose, measure_batches, render_report

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

def make_loader():
    data = torch.utils.data.TensorDataset(
        torch.randn(1280, 3, 32, 32),
        torch.randint(0, 10, (1280,)),
    )
    return torch.utils.data.DataLoader(data, batch_size=32, num_workers=0)

model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10),
).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
ce = nn.CrossEntropyLoss()

def step(batch):
    x, y = batch[0].to(device), batch[1].to(device)
    optimizer.zero_grad(set_to_none=True)
    loss = ce(model(x), y)
    loss.backward()
    optimizer.step()
    return int(x.shape[0])

def loss_fn(candidate, batch):
    x, y = batch[0].to(device), batch[1].to(device)
    return ce(candidate(x), y)

diagnosis = diagnose(measure_batches(make_loader(), step))
experiment = compare_mixed_precision(model, optimizer, make_loader(), loss_fn, seed=0)
print(render_report(diagnosis, experiment))
```

```bash
trainpilot run train.py
trainpilot run train.py --compile
```

`step(batch)` runs one training step and returns the example count. `loss_fn(model, batch)` must use the model it is given and return a scalar loss. `--compile` is passed through to `try_compile`. If the script already prints `render_report`, the CLI does not print it again.

## Example report

```
The GPU is busy computing. It is not sitting idle.

On this short run, 0.2% of a typical step is spent waiting for the next
batch. The GPU is busy computing the rest.

## What we tried

Mixed precision (FP16) made a typical step faster on this short test:
310.4 ms down to 96.3 ms. That is about 3.22 times faster.
A busy GPU can still be busy with slow math. This short test found a
faster step, so that is the change to paste.
Loss did not diverge and there are no NaN.

## Code to paste

scaler = torch.amp.GradScaler("cuda")
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = loss_fn(model(inputs), targets)
scaler.scale(loss).backward()
scaler.step(optimizer); scaler.update()
```

These numbers are from a short test. A full training run is only an estimate.

## Real results on a T4 (Colab)

![FP32 vs AMP median step time on a Tesla T4](docs/t4_results.png)

| Workload | Verdict | Data wait | FP32 step | AMP step | Speedup |
|---|---|---|---|---|---|
| CIFAR-10 ResNet-18 | GPU busy | 0.3% | 116.8 ms | 46.8 ms | 2.5x |
| DistilBERT IMDB | GPU busy | 0.2% | 310.4 ms | 96.3 ms | 3.22x |
| nanoGPT Shakespeare | GPU busy | 0.5% | 182.0 ms | 55.2 ms | 3.3x |
| nanoGPT + torch.compile | GPU busy | 0.5% | 55.2 ms (AMP) | 47.8 ms | 1.15x, prepare 27.2 s, pays off after ~3674 steps |
| Heavy augmentation, workers=2 | Waiting for data | 75.8% | 64.1 ms | 35.0 ms | loader is the bottleneck |
| Script already using AMP | GPU busy | 0.0% | 24.5 ms | 25.5 ms | no change suggested |

Short runs (10 warmup + 30 measured steps), numbers vary run to run.

## What it does not do (v0.1)

- One NVIDIA GPU. No multi-GPU or distributed training.
- It does not patch your script, tune batch size, or touch memory settings.
- "Training did not break" only means the short test did not blow up (no diverged loss, no NaN).
- It does not promise a speedup percentage. The table above is one Colab T4 run.

On notebooks it never suggests more than two DataLoader workers. If `num_workers=2` and `pin_memory` are already set, it says so and tells you to make loading cheaper instead.

## How it differs from torch.profiler / TensorBoard

- A profiler shows a trace. TrainPilot answers one question: is the GPU waiting or working, and did one safe change help on this short run?
- Before and after use the same seed and the same batches, so the step-time comparison is fair.
- You paste the snippet yourself. TrainPilot does not rewrite the training loop.

## Roadmap

v0.1 is diagnosis, mixed precision, and optional `torch.compile`. Not in this version: auto-patching, batch-size search, multi-GPU.

## Contributing

```bash
pip install -e ".[dev]"
pytest
```

GPU tests skip when CUDA is not available. Keep the public API (`measure_batches`, `diagnose`, `compare_mixed_precision`, `render_report`) unless a change is required. Do not edit the reference scripts in `examples/`.

## License

MIT
