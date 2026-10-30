"""Optional PyTorch smoke test for the final JSONL shards.

This is deliberately a tiny character/token GRU, not a production trainer. Its
purpose is to prove that packs can be read, windowed, masked, and optimized
without shape errors. Production training must use the fields documented in
README.md and the training framework's own document-reset attention path.

Usage:
    python examples/train_smoke.py --dataset data/shards --vocab-size 32000 --steps 100
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List, Tuple

from ebook_pipeline.dataset import PackedDataset

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("Install PyTorch to run this optional smoke training script") from exc


class TinyGRULM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        hidden = self.embedding(input_ids)
        output, _state = self.gru(hidden)
        return self.head(output)


def _window(pack: Dict[str, object], offset: int, width: int) -> Tuple[List[int], List[int], List[int]]:
    input_ids = [int(value) for value in pack["input_ids"]]  # type: ignore[index]
    loss_mask = [int(value) for value in pack["loss_mask"]]  # type: ignore[index]
    end = min(len(input_ids) - 1, offset + width)
    start = max(0, end - width)
    x = input_ids[start:end]
    y = input_ids[start + 1 : end + 1]
    mask = loss_mask[start + 1 : end + 1]
    return x, y, mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    with PackedDataset(args.dataset) as dataset:
        if not dataset.rows:
            raise SystemExit("dataset has no packs")
        model = TinyGRULM(args.vocab_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses: List[float] = []
        for step in range(1, args.steps + 1):
            row = dataset.rows[(step * 7919) % len(dataset.rows)]
            pack = dataset[str(row["pack_id"])]
            max_offset = max(0, len(pack["input_ids"]) - 2)
            offset = random.randint(0, max_offset)
            x_list, y_list, mask_list = _window(pack, offset, args.window)
            if not x_list:
                continue
            x = torch.tensor([x_list], dtype=torch.long, device=device)
            y = torch.tensor([y_list], dtype=torch.long, device=device)
            mask = torch.tensor([mask_list], dtype=torch.float32, device=device)
            logits = model(x)
            token_loss = nn.functional.cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1), reduction="none")
            loss = (token_loss * mask.reshape(-1)).sum() / mask.sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(f"step={step} loss={losses[-1]:.6f} window={len(x_list)}")
        print(f"initial_loss={losses[0]:.6f} final_loss={losses[-1]:.6f} steps={len(losses)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())