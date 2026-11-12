"""Experiment 6: learned operation semantics for Mirror's native state brain.

This pilot uses opaque operation IDs and variable-length compositions. Padding is
represented by -1 and is masked during composition; it is never interpreted as
operation zero. Seeds control model initialization, minibatch negatives, and row
shuffling. Results are bounded synthetic evidence, not open-domain reasoning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
    concept_dim: int = 64
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

    def encode_concept(self, concept_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.concept(concept_ids), dim=-1)

    def apply(self, state: torch.Tensor, op_ids: torch.Tensor) -> torch.Tensor:
        op = self.operation(op_ids)
        delta = self.step(torch.cat([state, op], dim=-1))
        return F.normalize(self.norm(state + delta), dim=-1)

    def compose(self, concept_ids: torch.Tensor, op_sequences: torch.Tensor) -> torch.Tensor:
        """Compose valid operations; -1 is a masked right-padding sentinel."""
        if op_sequences.ndim != 2:
            raise ValueError("operation sequences must be a rank-2 tensor")
        if concept_ids.ndim != 1 or concept_ids.shape[0] != op_sequences.shape[0]:
            raise ValueError("concept IDs must be rank-1 and match the batch size")
        if bool((op_sequences < -1).any()):
            raise ValueError("operation padding sentinel must be -1")
        if bool((op_sequences >= self.cfg.operations).any()):
            raise ValueError("operation ID is outside the configured vocabulary")

        valid = op_sequences >= 0
        if op_sequences.shape[1] > 1 and bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError("operation padding must be a contiguous right suffix")

        state = self.encode_concept(concept_ids)
        for column in range(op_sequences.shape[1]):
            active = valid[:, column].unsqueeze(-1)
            safe_ids = op_sequences[:, column].clamp_min(0)
            proposed = self.apply(state, safe_ids)
            state = torch.where(active, proposed, state)
        return state


@dataclass(frozen=True)
class Example:
    concept: int
    ops: tuple[int, ...]
    target: torch.Tensor
    held_out: bool


def make_world(cfg: Config, seed: int) -> list[Example]:
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    base = F.normalize(torch.randn(cfg.concepts, cfg.concept_dim, generator=generator), dim=-1)

    matrices = []
    for op in range(cfg.operations):
        matrix = torch.randn(cfg.concept_dim, cfg.concept_dim, generator=generator) * 0.35
        if op % 2 == 0:
            matrix += torch.eye(cfg.concept_dim) * (0.15 + 0.03 * op)
        matrices.append(matrix)

    def target_for(concept: int, operations: tuple[int, ...]) -> torch.Tensor:
        state = base[concept]
        for operation in operations:
            state = torch.tanh(state @ matrices[operation].T)
            state = F.normalize(state, dim=-1)
        return state

    examples = []
    for concept in range(cfg.concepts):
        for length in range(1, cfg.max_chain + 1):
            for _ in range(16):
                operations = tuple(rng.randrange(cfg.operations) for _ in range(length))
                held_out = (
                    (concept in {cfg.concepts - 2, cfg.concepts - 1} and length >= 3)
                    or (length >= 4 and operations[0] == 0 and operations[1] == 1)
                    or (length >= 5 and operations[-2:] == (4, 5))
                )
                examples.append(Example(
                    concept=concept,
                    ops=operations,
                    target=target_for(concept, operations),
                    held_out=held_out,
                ))
    return examples


def world_fingerprint(cfg: Config, seed: int, examples: list[Example]) -> str:
    """Hash the generated benchmark, including targets and train/held-out labels."""
    payload = {
        "format": "mirror7-exp6-semantic-world-v2",
        "seed": int(seed),
        "world_config": {
            "concepts": cfg.concepts,
            "operations": cfg.operations,
            "concept_dim": cfg.concept_dim,
            "max_chain": cfg.max_chain,
            "examples_per_concept_length": 16,
        },
        "examples": [
            {
                "concept": row.concept,
                "operations": list(row.ops),
                "held_out": row.held_out,
                "target": [round(float(value), 7) for value in row.target.tolist()],
            }
            for row in examples
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def loss_fn(
    prediction: torch.Tensor,
    target: torch.Tensor,
    negatives: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    positive_score = (prediction * target).sum(-1)
    negative_score = (prediction * negatives).sum(-1)
    return (1 - positive_score).mean() + F.relu(negative_score - positive_score + margin).mean()


def _batch(examples: list[Example], device: str | torch.device):
    if not examples:
        raise ValueError("cannot create a batch from zero examples")
    concept_ids = torch.tensor([row.concept for row in examples], dtype=torch.long, device=device)
    max_length = max(len(row.ops) for row in examples)
    operations = torch.full(
        (len(examples), max_length), -1, dtype=torch.long, device=device,
    )
    for index, row in enumerate(examples):
        if row.ops:
            operations[index, :len(row.ops)] = torch.tensor(row.ops, dtype=torch.long, device=device)
    targets = F.normalize(torch.stack([row.target for row in examples]).to(device), dim=-1)
    return concept_ids, operations, targets


def evaluate(
    model: LearnedOperationBrain,
    examples: list[Example],
    cfg: Config,
    device: str | torch.device,
    *,
    split: str = "held_out",
) -> dict[str, Any]:
    """Evaluate a deduplicated train or held-out candidate set and order controls."""
    if split not in {"train", "held_out"}:
        raise ValueError("split must be 'train' or 'held_out'")
    model.eval()
    # Random generation samples operation chains with replacement. Collapse
    # repeated (concept, operation-sequence) tasks before constructing the
    # candidate matrix so identical targets cannot count as false negatives.
    unique_rows = {}
    for row in examples:
        if row.held_out == (split == "held_out"):
            unique_rows.setdefault((row.concept, row.ops), row)
    rows = list(unique_rows.values())
    if len(rows) < 2:
        raise ValueError("held-out evaluation requires at least two unique tasks")

    predictions = []
    positive_scores = []
    order_margins = []
    with torch.no_grad():
        targets = F.normalize(torch.stack([row.target for row in rows]).to(device), dim=-1)
        for row in rows:
            concept_ids, operations, _ = _batch([row], device)
            prediction = model.compose(concept_ids, operations)[0]
            predictions.append(prediction)
            positive_scores.append(float((prediction * row.target.to(device)).sum()))

            if len(row.ops) >= 2 and row.ops[-1] != row.ops[-2]:
                swapped = row.ops[:-2] + (row.ops[-1], row.ops[-2])
                swapped_tensor = torch.tensor([swapped], dtype=torch.long, device=device)
                swapped_prediction = model.compose(concept_ids, swapped_tensor)[0]
                swapped_score = float((swapped_prediction * row.target.to(device)).sum())
                order_margins.append(positive_scores[-1] - swapped_score)

        prediction_matrix = F.normalize(torch.stack(predictions), dim=-1)
        similarities = prediction_matrix @ targets.T
        candidate_margins = []
        correct = []
        for index in range(len(rows)):
            positive = similarities[index, index]
            negative_scores = similarities[index].clone()
            negative_scores[index] = -torch.inf
            hardest_negative = negative_scores.max()
            candidate_margins.append(float(positive - hardest_negative))
            correct.append(bool(positive > hardest_negative))

        collapse_similarity = None
        if len(rows) > 1:
            pairwise = prediction_matrix @ prediction_matrix.T
            off_diagonal = ~torch.eye(len(rows), dtype=torch.bool, device=device)
            collapse_similarity = float(pairwise[off_diagonal].mean())

    order_control_count = len(order_margins)
    order_margin = statistics.fmean(order_margins) if order_margins else None
    return {
        "split": split,
        "count": len(rows),
        "candidate_count": len(rows),
        "chance_accuracy": 1.0 / len(rows),
        "accuracy": sum(correct) / len(correct),
        "margin": statistics.fmean(candidate_margins),
        "mean_positive_similarity": statistics.fmean(positive_scores),
        "order_control_count": order_control_count,
        "order_margin": order_margin,
        "order_violation_rate": (
            sum(value <= 0 for value in order_margins) / order_control_count
            if order_control_count else None
        ),
        "collapse_similarity": collapse_similarity,
        "held_out_concept_count": sum(
            row.concept >= cfg.concepts - 2 for row in rows
        ),
        "structured_composition_count": sum(
            (len(row.ops) >= 4 and row.ops[:2] == (0, 1))
            or (len(row.ops) >= 5 and row.ops[-2:] == (4, 5))
            for row in rows
        ),
    }


def seed_everything(seed: int) -> None:
    """Seed model initialization and all RNGs used by the training loop."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(
    cfg: Config,
    seed: int,
    device: str | torch.device,
    *,
    log_every: int = 10,
) -> tuple[LearnedOperationBrain, list[Example], list[dict[str, Any]]]:
    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    seed_everything(seed)
    examples = make_world(cfg, seed)
    train_rows = [row for row in examples if not row.held_out]
    held_out_rows = [row for row in examples if row.held_out]
    if not train_rows or len(held_out_rows) < 2:
        raise ValueError("benchmark must contain train rows and at least two held-out rows")

    model = LearnedOperationBrain(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)
    shuffle_rng = random.Random(seed)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        shuffle_rng.shuffle(train_rows)
        total_loss = 0.0
        batch_count = 0
        for start in range(0, len(train_rows), cfg.batch_size):
            batch = train_rows[start:start + cfg.batch_size]
            concept_ids, operations, targets = _batch(batch, device)
            prediction = model.compose(concept_ids, operations)
            negatives = targets[torch.randperm(targets.shape[0], device=device)]
            loss = loss_fn(prediction, targets, negatives, cfg.margin)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batch_count += 1

        if epoch == 1 or epoch % max(1, log_every) == 0 or epoch == cfg.epochs:
            history.append({
                "epoch": epoch,
                "loss": total_loss / max(1, batch_count),
                **evaluate(model, examples, cfg, device),
            })
    return model, examples, history


def train(args: argparse.Namespace) -> None:
    cfg = Config(epochs=args.epochs, lr=args.lr)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, examples, history = train_model(cfg, args.seed, device)
    final = evaluate(model, examples, cfg, device)
    data_sha256 = world_fingerprint(cfg, args.seed, examples)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "mirror7-native-brain-exp6-pilot-v2",
        "config": asdict(cfg),
        "seed": args.seed,
        "dataset_sha256": data_sha256,
        "state_dict": model.state_dict(),
    }, output)
    print(json.dumps({
        "output": str(output),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dataset_sha256": data_sha256,
        "history": history,
        "final": final,
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/content/drive/MyDrive/Mirror7/mirror7_exp6_pilot.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
