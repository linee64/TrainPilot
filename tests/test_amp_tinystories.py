"""Stage 3 check: TinyStories mixed precision is faster and the loss stays finite.

Skipped without a CUDA device. The model matches examples/02_tinystories_fp32.py.
The text slice is shorter than that script so the check stays a short run.
``datasets`` must be installed. The original example file is not imported,
because importing it starts a full training loop.
"""

import pytest
import torch
from torch import nn

from trainpilot.experiments import LOSS_OK_NOTE, compare_mixed_precision


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not found")
def test_tinystories_amp_is_faster_and_loss_is_ok() -> None:
    datasets = pytest.importorskip("datasets")
    torch.manual_seed(0)
    device = "cuda"

    raw = datasets.load_dataset("roneneldan/TinyStories", split="train[:1000]")
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
        def __init__(self, width: int = 384, heads: int = 6, layers: int = 6) -> None:
            super().__init__()
            self.tokens = nn.Embedding(vocab, width)
            self.positions = nn.Embedding(context, width)
            self.blocks = nn.ModuleList([Block(width, heads) for _ in range(layers)])
            self.norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, vocab)

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

    model = GPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    window = [make_batch() for _ in range(40)]

    def loss_fn(candidate: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        inputs, targets = batch
        logits = candidate(inputs)
        return nn.functional.cross_entropy(logits.view(-1, vocab), targets.view(-1))

    result = compare_mixed_precision(model, optimizer, window, loss_fn, seed=0, try_compile=False)

    assert result.status == "ok"
    assert result.amp_dtype == "float16"
    assert result.loss_ok is True
    assert result.loss_note == LOSS_OK_NOTE
    assert result.median_step_speedup is not None
    assert result.median_step_speedup >= 1.2
    assert result.compile_note == "Skipped because torch.compile was not requested."
