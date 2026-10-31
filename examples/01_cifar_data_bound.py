"""Reference 1: GPU waits for data (data-bound).
Small CNN on CIFAR-10 with deliberately slow loading: heavy CPU augmentations, num_workers=0, no pin_memory.
Expected TrainPilot finding: GPU mostly idle, waiting for data; DataLoader settings help most."""
import time, torch, torch.nn as nn, torchvision, torchvision.transforms as T

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

tf = T.Compose([T.RandomResizedCrop(32, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                T.RandomRotation(15), T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                T.ToTensor(), T.Normalize((0.49, 0.48, 0.45), (0.25, 0.24, 0.26))])
ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tf)
loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=False)

model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
    nn.Flatten(), nn.Linear(256, 10)).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

EPOCHS = 2
for epoch in range(EPOCHS):
    model.train(); t0 = time.time()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
    if device == "cuda": torch.cuda.synchronize()
    print(f"epoch {epoch} time {time.time()-t0:.1f}s")
