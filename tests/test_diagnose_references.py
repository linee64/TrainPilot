"""Stage 2 check: CIFAR waits for data, ResNet-50 keeps the GPU busy.

Skipped when this machine has no CUDA device. The loaders follow the
reference scripts without changing those files. ``num_workers`` stays 0 so
the test does not spawn workers.
"""

import pytest
import torch

from trainpilot.diagnose import GPU_BUSY, WAITING_FOR_DATA, diagnose
from trainpilot.measure import measure_batches


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not found")
def test_cifar_is_waiting_for_data(tmp_path) -> None:
    torchvision = pytest.importorskip("torchvision")
    transforms = torchvision.transforms
    torch.manual_seed(0)

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
    dataset = torchvision.datasets.CIFAR10(
        tmp_path, train=True, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=False
    )
    device = "cuda"
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(64, 128, 3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(128, 256, 3, padding=1),
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(256, 10),
    ).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    def step(batch) -> int:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
        return int(inputs.shape[0])

    diagnosis = diagnose(measure_batches(loader, step))
    assert diagnosis.verdict == WAITING_FOR_DATA


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not found")
def test_resnet50_gpu_is_busy() -> None:
    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False

    device = "cuda"
    images = torch.randn(40 * 32, 3, 224, 224)
    targets = torch.randint(0, 100, (40 * 32,))
    dataset = torch.utils.data.TensorDataset(images, targets)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
    )
    model = torchvision.models.resnet50(num_classes=100).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    def step(batch) -> int:
        inputs, labels = batch
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), labels)
        loss.backward()
        optimizer.step()
        return int(inputs.shape[0])

    diagnosis = diagnose(measure_batches(loader, step))
    assert diagnosis.verdict == GPU_BUSY
