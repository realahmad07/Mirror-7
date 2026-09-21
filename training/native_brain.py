from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from training.dataset import load_jsonl
from training.model import require_torch

PAD = 256
VOCAB_SIZE = 257


@dataclass(frozen=True)
class NativeBrainConfig:
    embedding_dim: int = 96
    hidden_dim: int = 256
    layers: int = 2
    state_dim: int = 128
    dropout: float = 0.10
    max_bytes: int = 512

    def to_dict(self):
        return asdict(self)


def _encode(text: str, max_bytes: int):
    return list(text.encode("utf-8")[:max_bytes])


def _batch(rows, max_bytes):
    torch, _ = require_torch()
    max_len = max(1, min(max_bytes, max(len(x) for x in rows)))
    values, mask = [], []
    for row in rows:
        row = row[:max_len]
        pad = max_len - len(row)
        values.append(row + [PAD] * pad)
        mask.append([1.0] * len(row) + [0.0] * pad)
    return torch.tensor(values, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)


def build_native_brain(config: NativeBrainConfig):
    torch, nn = require_torch()
    import torch.nn.functional as F

    class NativeBrain(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.embedding = nn.Embedding(
                VOCAB_SIZE, config.embedding_dim, padding_idx=PAD
            )
            self.gru = nn.GRU(
                config.embedding_dim,
                config.hidden_dim,
                num_layers=config.layers,
                batch_first=True,
                dropout=config.dropout if config.layers > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(config.hidden_dim)
            self.state = nn.Linear(config.hidden_dim, config.state_dim)
            self.transition = nn.Sequential(
                nn.Linear(config.state_dim, config.state_dim),
                nn.GELU(),
                nn.LayerNorm(config.state_dim),
                nn.Linear(config.state_dim, config.state_dim),
            )

        def encode(self, ids, mask):
            x, _ = self.gru(self.embedding(ids))
            x = self.norm(x)
            weights = mask.unsqueeze(-1)
            pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            return F.normalize(self.state(pooled), dim=-1)

        def forward(self, ids, mask):
            state = self.encode(ids, mask)
            predicted = F.normalize(self.transition(state), dim=-1)
            return state, predicted

    return NativeBrain()


def _loss(state_a, state_b, predicted_b, temperature):
    torch, _ = require_torch()
    import torch.nn.functional as F

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = state_a @ state_b.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    contrastive = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )
    transition = 1.0 - (predicted_b * state_b.detach()).sum(dim=-1).mean()
    return contrastive + transition, contrastive.detach(), transition.detach()


def train(args):
    torch, _ = require_torch()
    examples = load_jsonl(args.dataset)
    train_items = [x for x in examples if x.split == "train"]
    val_items = [x for x in examples if x.split == "validation"]
    if not train_items or not val_items:
        raise ValueError("dataset must contain train and validation splits")

    config = NativeBrainConfig(
        args.embedding_dim,
        args.hidden_dim,
        args.layers,
        args.state_dim,
        args.dropout,
        args.max_bytes,
    )
    model = build_native_brain(config)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    rng = random.Random(args.seed)
    history = []

    def run_epoch(items, training):
        model.train(training)
        order = list(range(len(items)))
        if training:
            rng.shuffle(order)
        totals = [0.0, 0.0, 0.0]
        batches = 0
        for start in range(0, len(order), args.batch_size):
            batch = [items[i] for i in order[start : start + args.batch_size]]
            ui, um = _batch(
                [_encode(x.user_text, config.max_bytes) for x in batch],
                config.max_bytes,
            )
            ti, tm = _batch(
                [_encode(x.target_text, config.max_bytes) for x in batch],
                config.max_bytes,
            )
            ui, um, ti, tm = (
                ui.to(device),
                um.to(device),
                ti.to(device),
                tm.to(device),
            )
            with torch.set_grad_enabled(training):
                us, pred = model(ui, um)
                ts = model.encode(ti, tm)
                loss, c_loss, t_loss = _loss(us, ts, pred, args.temperature)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
            totals[0] += float(loss.detach().cpu())
            totals[1] += float(c_loss.cpu())
            totals[2] += float(t_loss.cpu())
            batches += 1
        return {
            "loss": totals[0] / max(1, batches),
            "contrastive": totals[1] / max(1, batches),
            "transition": totals[2] / max(1, batches),
        }

    for epoch in range(1, args.epochs + 1):
        rec = {
            "epoch": epoch,
            "train": run_epoch(train_items, True),
            "validation": run_epoch(val_items, False),
        }
        history.append(rec)
        print(json.dumps(rec))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "mirror7-native-brain-v1",
            "config": config.to_dict(),
            "history": history,
            "device": str(device),
            "state_dict": model.state_dict(),
        },
        output,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        )
    )


def main():
    p = argparse.ArgumentParser(
        description="Train Mirror 7 native semantic/state brain."
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default="artifacts/mirror7_native_brain_v1.pt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--embedding-dim", type=int, default=96)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--state-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--max-bytes", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    train(p.parse_args())


if __name__ == "__main__":
    main()
