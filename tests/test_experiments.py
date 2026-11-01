"""CPU tests for mixed-precision copies. The GPU check lives in test_amp_tinystories."""

import torch
from torch import nn

from trainpilot.experiments import (
    LOSS_OK_NOTE,
    SKIPPED_COMPILE_NOT_REQUESTED,
    SKIPPED_NO_GPU,
    assess_losses,
    compare_mixed_precision,
    select_amp_dtype,
)
from trainpilot.experiments import _clone_model_and_optimizer, _collect_batches, _execute
from trainpilot.measure import GPU_NOT_FOUND


def test_cpu_skips_and_leaves_the_original_alone() -> None:
    torch.manual_seed(0)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    seen: list[int] = []

    def loss_fn(candidate: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        seen.append(id(candidate))
        return candidate(batch).sum()

    result = compare_mixed_precision(
        model,
        optimizer,
        [torch.randn(2, 3)],
        loss_fn,
        warmup=0,
        steps=1,
    )

    assert result.status == GPU_NOT_FOUND
    assert result.skipped_reason == SKIPPED_NO_GPU
    assert result.loss_ok is None
    assert result.label == "short-run result"
    assert result.compile_note == SKIPPED_NO_GPU
    assert torch.equal(model.weight, before)
    assert seen == []


def test_copy_trains_without_changing_the_original() -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    batch = torch.randn(2, 4)
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    weight = model.weight.detach().clone()
    momentum = optimizer.state[model.weight]["momentum_buffer"].detach().clone()

    model_copy, optimizer_copy = _clone_model_and_optimizer(model, optimizer)

    def loss_fn(candidate: nn.Module, data: torch.Tensor) -> torch.Tensor:
        return candidate(data).sum()

    _execute(model_copy, optimizer_copy, [batch, batch], loss_fn, seed=0, warmup=0, steps=2, amp=False)

    assert torch.equal(model.weight, weight)
    assert torch.equal(optimizer.state[model.weight]["momentum_buffer"], momentum)
    assert not torch.equal(model_copy.weight, weight)
    assert id(model_copy) != id(model)

    adam_model = nn.Linear(3, 1)
    adam = torch.optim.AdamW(adam_model.parameters(), lr=1e-3)
    adam_batch = torch.randn(2, 3)
    adam_model(adam_batch).sum().backward()
    adam.step()
    adam_weight = adam_model.weight.detach().clone()
    adam_copy, _adam_optimizer = _clone_model_and_optimizer(adam_model, adam)
    _execute(
        adam_copy,
        _adam_optimizer,
        [adam_batch],
        lambda candidate, data: candidate(data).sum(),
        seed=0,
        warmup=0,
        steps=1,
        amp=False,
    )
    assert torch.equal(adam_model.weight, adam_weight)


def test_same_seed_replays_the_same_batches() -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            return self.scale * batch + torch.rand(())

    seen: list[float] = []

    def loss_fn(candidate: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        output = candidate(batch)
        seen.append(float(output.detach()))
        return output.sum() * 0 + candidate.scale

    batches = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
    source = Net()
    source_optimizer = torch.optim.SGD(source.parameters(), lr=0.01)
    first_model, first_optimizer = _clone_model_and_optimizer(source, source_optimizer)
    second_model, second_optimizer = _clone_model_and_optimizer(source, source_optimizer)

    _execute(first_model, first_optimizer, batches, loss_fn, seed=0, warmup=1, steps=2, amp=False)
    first_seen = list(seen)
    seen.clear()
    _execute(second_model, second_optimizer, batches, loss_fn, seed=0, warmup=1, steps=2, amp=False)

    assert seen == first_seen
    assert _collect_batches([1, 2, 3, 4], 3) == [1, 2, 3]


def test_loss_check_rejects_nan_and_divergence_only() -> None:
    assert assess_losses([1.0, 1.0, 1.0], [1.1, 1.0, 0.9]) == (True, LOSS_OK_NOTE)
    assert assess_losses([1.0, 1.0, 1.0], [9.0, 9.0, 9.0])[0] is True
    ok, note = assess_losses([1.0, 1.0, 1.0], [12.0, 12.0, 12.0])
    assert ok is False
    assert note == "Mixed-precision loss diverged."
    assert assess_losses([1.0, float("nan"), 1.0], [1.0, 1.0, 1.0])[1] == "NaN in the baseline loss."
    assert assess_losses([1.0, 1.0, 1.0], [1.0, float("inf"), 1.0])[1] == "NaN in the mixed-precision loss."


def test_amp_dtype_is_fp16_even_on_t4() -> None:
    assert select_amp_dtype("Tesla T4") is torch.float16
    assert select_amp_dtype("NVIDIA A100-SXM4-40GB") is torch.float16
    assert select_amp_dtype("Tesla T4") is not torch.bfloat16
    assert select_amp_dtype("NVIDIA A100-SXM4-40GB") is not torch.bfloat16


def test_compile_is_skipped_unless_requested() -> None:
    assert SKIPPED_COMPILE_NOT_REQUESTED.startswith("Skipped because")


def test_short_batch_list_is_rejected() -> None:
    try:
        _collect_batches([1], 2)
    except RuntimeError as exc:
        assert "Not enough batches" in str(exc)
    else:
        raise AssertionError("expected a short loader to fail")
