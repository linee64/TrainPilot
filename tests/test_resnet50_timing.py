"""Stage 1 check: ResNet-50 step time versus a hand timer.

Skipped when this machine has no CUDA device. On a GPU, the median step
time from ``measure_steps`` must stay within 5% of a separate loop that
uses ``time.time()`` and ``torch.cuda.synchronize()``.
"""

import statistics
import time

import pytest
import torch

from trainpilot.measure import measure_steps


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not found")
def test_resnet50_median_matches_manual_timer() -> None:
    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False

    device = "cuda"
    model = torchvision.models.resnet50(num_classes=100).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    inputs = torch.randn(32, 3, 224, 224, device=device)
    targets = torch.randint(0, 100, (32,), device=device)

    def step() -> int:
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
        return int(inputs.shape[0])

    measured = measure_steps(step, warmup=10, steps=30)
    manual = _manual_median_seconds(step, warmup=10, steps=30)

    assert measured.status == "ok"
    assert measured.peak_memory_bytes is not None
    assert measured.peak_memory_bytes > 0
    assert manual > 0
    assert abs(measured.median_step_seconds - manual) / manual < 0.05


def _manual_median_seconds(step_fn, warmup: int, steps: int) -> float:
    samples: list[float] = []
    for index in range(warmup + steps):
        torch.cuda.synchronize()
        started = time.time()
        step_fn()
        torch.cuda.synchronize()
        elapsed = time.time() - started
        if index >= warmup:
            samples.append(elapsed)
    return float(statistics.median(samples))
