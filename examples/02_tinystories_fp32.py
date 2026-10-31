"""Reference 2: compute in FP32 where mixed precision helps.
Small GPT-style transformer (char-level) on a slice of TinyStories, trained in plain FP32.
Expected TrainPilot finding: GPU busy but in FP32; mixed precision (FP16 on T4) gives a clear speedup."""
import time, math, torch, torch.nn as nn
from datasets import load_dataset  # pip install datasets

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

raw = load_dataset("roneneldan/TinyStories", split="train[:20000]")
text = "\n".join(raw["text"])
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
V, CTX, B = len(chars), 256, 64

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.ln1 = nn.LayerNorm(d); s.att = nn.MultiheadAttention(d, h, batch_first=True)
        s.ln2 = nn.LayerNorm(d); s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x, mask):
        h = s.ln1(x); x = x + s.att(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + s.mlp(s.ln2(x))

class GPT(nn.Module):
    def __init__(s, d=384, h=6, L=6):
        super().__init__(); s.tok = nn.Embedding(V, d); s.pos = nn.Embedding(CTX, d)
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(L)]); s.ln = nn.LayerNorm(d); s.head = nn.Linear(d, V)
    def forward(s, idx):
        T_ = idx.shape[1]; x = s.tok(idx) + s.pos(torch.arange(T_, device=idx.device))
        mask = torch.triu(torch.ones(T_, T_, device=idx.device, dtype=torch.bool), 1)
        for b in s.blocks: x = b(x, mask)
        return s.head(s.ln(x))

def batch():
    ix = torch.randint(len(data) - CTX - 1, (B,))
    x = torch.stack([data[i:i+CTX] for i in ix]); y = torch.stack([data[i+1:i+CTX+1] for i in ix])
    return x.to(device), y.to(device)

model = GPT().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
STEPS = 1000; t0 = time.time()
for step in range(STEPS):
    x, y = batch()
    loss = nn.functional.cross_entropy(model(x).view(-1, V), y.view(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.4f}")
if device == "cuda": torch.cuda.synchronize()
print(f"total time {time.time()-t0:.1f}s")
