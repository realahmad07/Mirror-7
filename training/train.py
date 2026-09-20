from __future__ import annotations

import argparse
from pathlib import Path
import random

from .bytes import PAD, encode
from .dataset import load_jsonl, summarize, validate_split_isolation
from .model import ModelConfig, build_model, require_torch, save_checkpoint


def _collate(encoded, pad=PAD):
    torch, _ = require_torch()
    max_len = max(len(x.input_ids) for x in encoded)
    inputs = []
    labels = []
    weights = []
    for item in encoded:
        ids = list(item.input_ids)
        input_row = ids[:-1]
        label_row = ids[1:]
        mask = [0.0] * len(label_row)
        for i in range(max(0, item.loss_start - 1), min(len(mask), item.loss_end - 1)):
            mask[i] = item.quality
        pad_count = max_len - len(ids)
        inputs.append(input_row + [pad] * pad_count)
        labels.append(label_row + [-100] * pad_count)
        weights.append(mask + [0.0] * pad_count)
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(weights, dtype=torch.float32),
    )


def _masked_loss(logits, labels, weights):
    torch, _ = require_torch()
    import torch.nn.functional as F

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    losses = F.cross_entropy(
        flat_logits, flat_labels, ignore_index=-100, reduction="none"
    )
    flat_weights = weights.reshape(-1)
    active = flat_labels.ne(-100)
    weighted = losses * flat_weights
    denom = flat_weights[active].sum().clamp_min(1.0)
    return weighted.sum() / denom


def train(args) -> dict[str, object]:
    torch, _ = require_torch()
    examples = load_jsonl(args.dataset)
    validate_split_isolation(examples)
    train_items = [x for x in examples if x.split == "train"]
    val_items = [x for x in examples if x.split == "validation"]
    if not train_items:
        raise ValueError("dataset has no train split")
    if not val_items:
        raise ValueError("dataset has no validation split")

    config = ModelConfig(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        max_sequence_bytes=args.max_sequence_bytes,
    )
    model = build_model(config)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_encoded = [encode(x, args.max_sequence_bytes) for x in train_items]
    val_encoded = [encode(x, args.max_sequence_bytes) for x in val_items]

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        order = list(range(len(train_encoded)))
        random.Random(args.seed + epoch).shuffle(order)
        for start in range(0, len(order), args.batch_size):
            batch = [train_encoded[i] for i in order[start : start + args.batch_size]]
            inputs, labels, weights = _collate(batch)
            inputs = inputs.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            loss = _masked_loss(logits, labels, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += float(loss.detach().cpu())
            batches += 1

        model.eval()
        with torch.no_grad():
            val_total = 0.0
            val_batches = 0
            for start in range(0, len(val_encoded), args.batch_size):
                batch = val_encoded[start : start + args.batch_size]
                inputs, labels, weights = _collate(batch)
                inputs = inputs.to(device)
                labels = labels.to(device)
                weights = weights.to(device)
                logits, _ = model(inputs)
                val_total += float(_masked_loss(logits, labels, weights).detach().cpu())
                val_batches += 1

        record = {
            "epoch": epoch,
            "train_loss": running / max(1, batches),
            "validation_loss": val_total / max(1, val_batches),
        }
        history.append(record)
        print(record)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(examples)
    save_checkpoint(
        output,
        model,
        config,
        {
            "dataset_fingerprint": summary["fingerprint"],
            "dataset_summary": summary,
            "history": history,
            "device": str(device),
        },
    )
    return {"output": str(output), "history": history, "device": str(device)}


def main():
    parser = argparse.ArgumentParser(
        description="Train Mirror 7's byte-level response realization layer."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="artifacts/mirror7_response.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--max-sequence-bytes", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
