# TrainPilot — Architecture

```
trainpilot/
├── trainpilot/
│   ├── __init__.py       # public API; no Pilot class
│   ├── measure.py        # step timing, memory, data wait
│   ├── diagnose.py       # waiting for data vs GPU busy
│   ├── experiments.py    # baseline vs AMP (optional torch.compile)
│   ├── report.py         # one-page report + snippet
│   └── cli.py            # trainpilot run train.py [--compile]
├── examples/
│   ├── 01_cifar_data_bound.py
│   ├── 02_tinystories_fp32.py
│   └── 03_resnet50_compute_bound.py
├── notebooks/
│   ├── trainpilot_quickstart.ipynb
│   └── trainpilot_colab.ipynb
├── docs/
│   ├── ARCHITECTURE.md
│   ├── ROADMAP.md
│   ├── TESTING.md
│   ├── make_chart.py
│   └── t4_results.png
├── tests/
├── AGENTS.md
├── pyproject.toml
├── README.md
└── LICENSE
```

## How to hook it up

```python
from trainpilot import compare_mixed_precision, diagnose, measure_batches, render_report

diagnosis = diagnose(measure_batches(loader, step))
experiment = compare_mixed_precision(model, optimizer, loader, loss_fn, seed=0)
print(render_report(diagnosis, experiment))
```

There is no `Pilot` class in this version. `trainpilot run train.py` runs the script, writes `trainpilot-report.md`, and does not edit the file. The script must call `measure_batches` and `compare_mixed_precision`.

## Data flow

1. `measure` → raw numbers (step time, wait, memory)
2. `diagnose` → verdict
3. `experiments` → baseline vs AMP, loss/NaN check
4. `report` → markdown + paste-ready code

## Measurement rules

- Always `torch.cuda.synchronize()` before reading a timer
- Warmup steps are excluded
- Median step time, not the mean
- Same seed and same batches for the comparison
