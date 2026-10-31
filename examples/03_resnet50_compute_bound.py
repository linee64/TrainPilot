"""Reference 3: GPU already busy (compute-bound).
ResNet-50 at 224x224 on synthetic in-memory data, so data loading costs nothing.
Expected TrainPilot finding: GPU is already near 100%; little to gain except maybe mixed precision.
Checks that TrainPilot stays honest when there is nothing to speed up."""
import time, torch, torch.nn as nn, torchvision

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

N, B = 2048, 32
X = torch.randn(N, 3, 224, 224); Y = torch.randint(0, 100, (N,))
ds = torch.utils.data.TensorDataset(X, Y)
loader = torch.utils.data.DataLoader(ds, batch_size=B, shuffle=True, num_workers=2, pin_memory=True)

model = torchvision.models.resnet50(num_classes=100).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

EPOCHS = 2
for epoch in range(EPOCHS):
    model.train(); t0 = time.time()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
    if device == "cuda": torch.cuda.synchronize()
    print(f"epoch {epoch} time {time.time()-t0:.1f}s")
