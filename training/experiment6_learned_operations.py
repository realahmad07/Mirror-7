"""Experiment 6: learned operation semantics for Mirror's native state brain.

This is a controlled pilot. Operation identifiers are opaque (OP_A..OP_F); their
meanings are not encoded in the identifier names. The model must learn an
operation embedding and compose variable-length transformations. The benchmark
holds out operation combinations and concept identities, and includes order and
shuffled-operation controls.

This experiment is intentionally separate from the Exp-5 checkpoint format so
the result cannot be mistaken for a continuation of the previous synthetic
operation vocabulary.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    state_dim: int = 64
    op_dim: int = 32
    hidden_dim: int = 128
    concepts: int = 24
    operations: int = 6
    concept_dim: int = 8
    max_chain: int = 5
    epochs: int = 60
    batch_size: int = 128
    lr: float = 5e-4
    margin: float = 0.20


class LearnedOperationBrain(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.concept = nn.Embedding(cfg.concepts, cfg.state_dim)
        self.operation = nn.Embedding(cfg.operations, cfg.op_dim)
        self.step = nn.Sequential(
            nn.Linear(cfg.state_dim + cfg.op_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.state_dim),
        )
        self.norm = nn.LayerNorm(cfg.state_dim)

    def encode_concept(self, concept_ids):
        return F.normalize(self.concept(concept_ids), dim=-1)

    def apply(self, state, op_ids):
        op = self.operation(op_ids)
        delta = self.step(torch.cat([state, op], dim=-1))
        return F.normalize(self.norm(state + delta), dim=-1)

    def compose(self, concept_ids, op_sequences):
        state = self.encode_concept(concept_ids)
        for i in range(op_sequences.shape[1]):
            state = self.apply(state, op_sequences[:, i])
        return state


@dataclass(frozen=True)
class Example:
    concept: int
    ops: tuple[int, ...]
    target: torch.Tensor
    held_out: bool


def make_world(cfg: Config, seed: int):
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    base = F.normalize(torch.randn(cfg.concepts, cfg.concept_dim, generator=g), dim=-1)

    # Opaque operations have learned latent effects. Names/IDs carry no semantics.
    matrices = []
    for op in range(cfg.operations):
        m = torch.randn(cfg.concept_dim, cfg.concept_dim, generator=g) * 0.35
        if op % 2 == 0:
            m += torch.eye(cfg.concept_dim) * (0.15 + 0.03 * op)
        matrices.append(m)

    def target_for(c, ops):
        x = base[c]
        for op in ops:
            x = torch.tanh(x @ matrices[op].T)
            x = F.normalize(x, dim=-1)
        return x

    examples = []
    for c in range(cfg.concepts):
        for length in range(1, cfg.max_chain + 1):
            # Keep enough coverage for every length while holding out structured
            # combinations and concept identities.
            for _ in range(16):
                ops = tuple(rng.randrange(cfg.operations) for _ in range(length))
                held = (
                    (c in {cfg.concepts - 2, cfg.concepts - 1} and length >= 3)
                    or (length >= 4 and ops[0] == 0 and ops[1] == 1)
                    or (length >= 5 and ops[-2:] == (4, 5))
                )
                examples.append(Example(c, ops, target_for(c, ops), held))
    return examples


def loss_fn(pred, target, negatives, margin):
    pos = (pred * target).sum(-1)
    neg = (pred * negatives).sum(-1)
    return (1 - pos).mean() + F.relu(neg - pos + margin).mean()


def _batch(examples, device):
    c = torch.tensor([e.concept for e in examples], dtype=torch.long, device=device)
    n = max(len(e.ops) for e in examples)
    ops = torch.zeros((len(examples), n), dtype=torch.long, device=device)
    for i, e in enumerate(examples):
        ops[i, :len(e.ops)] = torch.tensor(e.ops, device=device)
    target = F.normalize(torch.stack([e.target for e in examples]).to(device), dim=-1)
    return c, ops, target


def evaluate(model, examples, cfg, device):
    model.eval()
    rows = [e for e in examples if e.held_out]
    if not rows:
        raise AssertionError("held-out set is empty")
    with torch.no_grad():
        margins, correct = [], []
        order_margins = []
        identity = []
        for e in rows:
            c, ops, target = _batch([e], device)
            pred = model.compose(c, ops)
            pos = float((pred * target).sum())
            wrong = []
            for j in range(min(8, len(examples))):
                w = examples[(j * 37) % len(examples)]
                wrong.append(float((pred * w.target.to(device)).sum()))
            neg = max(wrong)
            margins.append(pos - neg)
            correct.append(pos > neg)
            if len(e.ops) >= 2:
                shuffled = list(e.ops)
                shuffled[-1], shuffled[-2] = shuffled[-2], shuffled[-1]
                so = torch.tensor([shuffled], device=device)
                sp = model.compose(c, so)
                order_margins.append(pos - float((sp * target).sum()))
            if len(e.ops) >= 2 and e.ops[-2:] == (0, 0):
                identity.append(float((pred * model.encode_concept(c)).sum()))
    return {
        "count": len(rows),
        "accuracy": sum(correct) / len(correct),
        "margin": sum(margins) / len(margins),
        "order_margin": sum(order_margins) / max(1, len(order_margins)),
        "identity_similarity": sum(identity) / max(1, len(identity)),
    }


def train(args):
    cfg = Config(epochs=args.epochs, lr=args.lr)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    examples = make_world(cfg, args.seed)
    train_rows = [e for e in examples if not e.held_out]
    val_rows = [e for e in examples if e.held_out]
    model = LearnedOperationBrain(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)
    rng = random.Random(args.seed)
    for epoch in range(1, cfg.epochs + 1):
        rng.shuffle(train_rows)
        total = 0.0
        for start in range(0, len(train_rows), cfg.batch_size):
            batch = train_rows[start:start + cfg.batch_size]
            c, ops, target = _batch(batch, device)
            pred = model.compose(c, ops)
            shuffled = target[torch.randperm(target.shape[0], device=device)]
            loss = loss_fn(pred, target, shuffled, cfg.margin)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            metrics = evaluate(model, examples, cfg, device)
            print(json.dumps({"epoch": epoch, "loss": total / max(1, (len(train_rows)+cfg.batch_size-1)//cfg.batch_size), **metrics}))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "mirror7-native-brain-exp6-pilot-v1",
                "config": cfg.__dict__, "seed": args.seed,
                "state_dict": model.state_dict()}, output)
    print(json.dumps({"output": str(output), "parameters": sum(p.numel() for p in model.parameters()),
                      "final": evaluate(model, examples, cfg, device)}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/content/drive/MyDrive/Mirror7/mirror7_exp6_pilot.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    train(p.parse_args())


if __name__ == "__main__":
    main()
