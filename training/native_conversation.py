from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from training.dataset import load_jsonl
from training.model import require_torch
from training.native_brain import PAD, VOCAB_SIZE, _batch, _encode


@dataclass(frozen=True)
class NativeConversationConfig:
    embedding_dim: int = 96
    hidden_dim: int = 256
    state_dim: int = 128
    layers: int = 1
    dropout: float = 0.10
    max_bytes: int = 512

    def to_dict(self):
        return asdict(self)


def build_native_conversation(config: NativeConversationConfig):
    torch, nn = require_torch()
    import torch.nn.functional as F

    class NativeConversation(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.embedding = nn.Embedding(VOCAB_SIZE, config.embedding_dim, padding_idx=PAD)
            self.encoder = nn.GRU(
                config.embedding_dim, config.hidden_dim,
                num_layers=config.layers, batch_first=True
            )
            self.encoder_norm = nn.LayerNorm(config.hidden_dim)
            self.state = nn.Linear(config.hidden_dim, config.state_dim)

            self.decoder_embedding = nn.Embedding(VOCAB_SIZE, config.embedding_dim)
            self.decoder = nn.GRU(
                config.embedding_dim + config.state_dim,
                config.hidden_dim,
                num_layers=config.layers, batch_first=True
            )
            self.decoder_norm = nn.LayerNorm(config.hidden_dim)
            self.head = nn.Linear(config.hidden_dim, VOCAB_SIZE)

        def encode(self, ids, mask):
            x, _ = self.encoder(self.embedding(ids))
            x = self.encoder_norm(x)
            weights = mask.unsqueeze(-1)
            pooled = (x * weights).sum(1) / weights.sum(1).clamp_min(1.0)
            return torch.tanh(self.state(pooled))

        def forward(self, input_ids, input_mask, target_ids):
            state = self.encode(input_ids, input_mask)
            emb = self.decoder_embedding(target_ids)
            state_seq = state.unsqueeze(1).expand(-1, target_ids.shape[1], -1)
            x, _ = self.decoder(torch.cat([emb, state_seq], dim=-1))
            return self.head(self.decoder_norm(x)), state

    return NativeConversation()


def train(args):
    torch, _ = require_torch()
    examples = load_jsonl(args.dataset)
    train_items = [x for x in examples if x.split == "train"]
    if not train_items:
        raise ValueError("dataset must contain a train split")

    config = NativeConversationConfig(
        args.embedding_dim, args.hidden_dim, args.state_dim,
        args.layers, args.dropout, args.max_bytes
    )
    model = build_native_conversation(config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for start in range(0, len(train_items), args.batch_size):
            batch = train_items[start:start + args.batch_size]
            ui, um = _batch([_encode(x.user_text, config.max_bytes) for x in batch], config.max_bytes)
            targets = [list(x.target_text.encode("utf-8")[:config.max_bytes]) for x in batch]
            ti, _ = _batch(targets, config.max_bytes)
            ui, um, ti = ui.to(device), um.to(device), ti.to(device)
            logits, _ = model(ui, um, ti[:, :-1])
            labels = ti[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                ignore_index=PAD
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        print({"epoch": epoch, "train_loss": total / max(1, batches)})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "mirror7-native-conversation-v1",
        "config": config.to_dict(),
        "state_dict": model.state_dict(),
        "device": str(device),
    }, output)
    print({"output": str(output), "parameters": sum(p.numel() for p in model.parameters())})


def main():
    p = argparse.ArgumentParser(description="Train Mirror 7 native conversation brain from scratch.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default="artifacts/mirror7_native_conversation_v1.pt")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--embedding-dim", type=int, default=96)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--state-dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--max-bytes", type=int, default=512)
    p.add_argument("--device", default=None)
    train(p.parse_args())


if __name__ == "__main__":
    main()
