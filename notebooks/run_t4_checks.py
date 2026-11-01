"""Short check of the three reference setups on one GPU.

The example scripts are not imported or edited. Without a GPU the script
prints ``GPU not found`` and exits. Loader workers stay at 0 so a notebook
does not spawn processes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from trainpilot.diagnose import diagnose
from trainpilot.experiments import compare_mixed_precision
from trainpilot.measure import measure_batches

WINDOW = 40
EXPECTED = {
    "01 CIFAR": "waiting for data",
    "02 TinyStories": "gpu busy",
    "03 ResNet-50": "gpu busy",
}


def main(only: str | None = None) -> int:
    if not torch.cuda.is_available():
        print("GPU not found")
        return 0

    torch.backends.cudnn.benchmark = False
    device_name = torch.cuda.get_device_name(0)
    print(f"device: {device_name}", flush=True)
    print(
        "example\tstep_before_ms\tstep_after_ms\tspeedup\tverdict\t"
        "diagnosis_ok\tloss_ok\trecommendation",
        flush=True,
    )
    failures: list[str] = []
    runners = (
        ("01 CIFAR", run_cifar),
        ("02 TinyStories", run_tinystories),
        ("03 ResNet-50", run_resnet50),
    )
    if only is not None:
        runners = tuple(item for item in runners if item[0] == only)
        if not runners:
            print(f"Unknown example: {only}")
            return 1
    for name, runner in runners:
        print(f"\n== {name} ==", flush=True)
        try:
            diagnosis, experiment = runner()
            _report(name, diagnosis, experiment)
            if not _verdict_ok(name, diagnosis.verdict):
                failures.append(name)
        except Exception as exc:
            failures.append(name)
            print(f"FAILED {name}: {exc}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return 1 if failures else 0


def run_cifar() -> tuple[object, object]:
    import torchvision

    torch.manual_seed(0)
    device = "cuda"
    transforms = torchvision.transforms
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.49, 0.48, 0.45), (0.25, 0.24, 0.26)),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(_data_root(), train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=False
    )
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 10),
    ).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    def loss_from_batch(candidate: nn.Module, batch) -> torch.Tensor:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        return loss_fn(candidate(inputs), targets)

    # Time the loader itself. Prefetched batches hide the wait and look GPU-busy.
    diagnosis = diagnose(
        measure_batches(loader, lambda batch: _train_step(model, optimizer, batch, loss_from_batch))
    )
    experiment = compare_mixed_precision(
        model, optimizer, loader, loss_from_batch, seed=0, try_compile=False
    )
    return diagnosis, experiment


def run_tinystories() -> tuple[object, object]:
    from datasets import load_dataset

    torch.manual_seed(0)
    device = "cuda"
    raw = load_dataset("roneneldan/TinyStories", split="train[:1000]")
    text = "\n".join(raw["text"])
    chars = sorted(set(text))
    index = {char: pos for pos, char in enumerate(chars)}
    data = torch.tensor([index[char] for char in text], dtype=torch.long)
    vocab, context, batch_size = len(chars), 256, 64

    class Block(nn.Module):
        def __init__(self, width: int, heads: int) -> None:
            super().__init__()
            self.norm_1 = nn.LayerNorm(width)
            self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
            self.norm_2 = nn.LayerNorm(width)
            self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

        def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            normed = self.norm_1(hidden)
            attended = self.attention(normed, normed, normed, attn_mask=mask, need_weights=False)[0]
            hidden = hidden + attended
            return hidden + self.mlp(self.norm_2(hidden))

    class GPT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tokens = nn.Embedding(vocab, 384)
            self.positions = nn.Embedding(context, 384)
            self.blocks = nn.ModuleList([Block(384, 6) for _ in range(6)])
            self.norm = nn.LayerNorm(384)
            self.head = nn.Linear(384, vocab)

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            length = token_ids.shape[1]
            hidden = self.tokens(token_ids) + self.positions(torch.arange(length, device=token_ids.device))
            mask = torch.triu(torch.ones(length, length, device=token_ids.device, dtype=torch.bool), 1)
            for block in self.blocks:
                hidden = block(hidden, mask)
            return self.head(self.norm(hidden))

    def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(len(data) - context - 1, (batch_size,))
        inputs = torch.stack([data[start : start + context] for start in starts])
        targets = torch.stack([data[start + 1 : start + context + 1] for start in starts])
        return inputs.to(device), targets.to(device)

    batches = [make_batch() for _ in range(WINDOW)]
    model = GPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def loss_from_batch(candidate: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        inputs, targets = batch
        logits = candidate(inputs)
        return nn.functional.cross_entropy(logits.view(-1, vocab), targets.view(-1))

    diagnosis = diagnose(measure_batches(batches, lambda batch: _train_step(model, optimizer, batch, loss_from_batch)))
    experiment = compare_mixed_precision(model, optimizer, batches, loss_from_batch, seed=0, try_compile=False)
    return diagnosis, experiment


def run_resnet50() -> tuple[object, object]:
    import torchvision

    torch.manual_seed(0)
    device = "cuda"
    images = torch.randn(WINDOW * 32, 3, 224, 224)
    targets = torch.randint(0, 100, (WINDOW * 32,))
    batches = [
        (images[start : start + 32], targets[start : start + 32])
        for start in range(0, WINDOW * 32, 32)
    ]
    model = torchvision.models.resnet50(num_classes=100).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    return _check(model, optimizer, batches, device, loss_fn)


def _check(model, optimizer, batches, device: str, loss_fn: nn.Module):
    def loss_from_batch(candidate: nn.Module, batch) -> torch.Tensor:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        return loss_fn(candidate(inputs), targets)

    diagnosis = diagnose(
        measure_batches(batches, lambda batch: _train_step(model, optimizer, batch, loss_from_batch))
    )
    experiment = compare_mixed_precision(model, optimizer, batches, loss_from_batch, seed=0, try_compile=False)
    return diagnosis, experiment


def _train_step(model, optimizer, batch, loss_from_batch) -> int:
    count = int(batch[0].shape[0])
    optimizer.zero_grad()
    loss = loss_from_batch(model, batch)
    loss.backward()
    optimizer.step()
    return count


def _report(name: str, diagnosis, experiment) -> None:
    verdict = diagnosis.verdict
    before = _ms(experiment.baseline_median_step_seconds)
    after = _ms(experiment.amp_median_step_seconds)
    speedup = "" if experiment.median_step_speedup is None else f"{experiment.median_step_speedup:.2f}"
    loss_ok = "" if experiment.loss_ok is None else str(experiment.loss_ok)
    match = "yes" if verdict == EXPECTED[name] else "no"
    print(
        f"{name}\t{before}\t{after}\t{speedup}\t{verdict}\t{match}\t{loss_ok}\t{_recommendation(verdict, experiment)}",
        flush=True,
    )
    if experiment.loss_note:
        print(experiment.loss_note, flush=True)
    if diagnosis.wait_fraction is not None:
        print(f"wait_percent: {diagnosis.wait_fraction * 100:.1f}", flush=True)
    if verdict == "waiting for data":
        print("step times above are the math only; most of the step is data wait", flush=True)
    reason = getattr(diagnosis, "reason", "")
    if reason:
        print(reason, flush=True)


def _verdict_ok(name: str, verdict: str) -> bool:
    return verdict == EXPECTED[name]


def _recommendation(verdict: str, experiment) -> str:
    speedup = experiment.median_step_speedup
    if verdict == "waiting for data":
        return "num_workers and pin_memory"
    if experiment.loss_ok and speedup is not None and speedup >= 1.2:
        return "mixed precision (FP16)"
    return "little left to speed up"


def _ms(seconds: float | None) -> str:
    if seconds is None:
        return ""
    return f"{seconds * 1000:.1f}"


def _data_root() -> Path:
    colab = Path("/content/data")
    if colab.exists():
        return colab
    return Path("data")


if __name__ == "__main__":
    sys.exit(main())
