"""Three-seed screening runner for native semantic Experiment 6.

Passing these checks is not sufficient for promotion: results still need a
frozen baseline comparison and a richer natural-semantic benchmark.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch

from .experiment6_learned_operations import Config, evaluate, train_model, world_fingerprint


def run_multiseed_benchmark(
    seeds: Iterable[int] = (17, 29, 43),
    *,
    config: Config | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    seed_list = [int(seed) for seed in seeds]
    if len(seed_list) < 3:
        raise ValueError("the screening benchmark requires at least three seeds")
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("benchmark seeds must be unique")

    cfg = config or Config()
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for seed in seed_list:
        model, examples, history = train_model(cfg, seed, resolved_device)
        train_metrics = evaluate(model, examples, cfg, resolved_device, split="train")
        metrics = evaluate(model, examples, cfg, resolved_device, split="held_out")
        checks = {
            "positive_heldout_margin": metrics["margin"] > 0,
            "above_candidate_chance": metrics["accuracy"] > metrics["chance_accuracy"],
            "positive_order_margin": (
                metrics["order_margin"] is not None and metrics["order_margin"] > 0
            ),
            "no_representation_collapse": (
                metrics["collapse_similarity"] is not None
                and metrics["collapse_similarity"] < 0.98
            ),
        }
        results.append({
            "seed": seed,
            "dataset_sha256": world_fingerprint(cfg, seed, examples),
            "train_metrics": train_metrics,
            "held_out_metrics": metrics,
            "training_history": history,
            "metrics": metrics,
            "screening_checks": checks,
            "screening_passed": all(checks.values()),
        })

    metric_names = (
        "accuracy", "margin", "mean_positive_similarity",
        "order_margin", "order_violation_rate", "collapse_similarity",
    )
    aggregate = {}
    for name in metric_names:
        values = [row["metrics"][name] for row in results if row["metrics"][name] is not None]
        aggregate[name] = {
            "mean": statistics.fmean(values) if values else None,
            "population_std": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None,
        }

    return {
        "format": "mirror7-exp6-multiseed-screening-v1",
        "config": asdict(cfg),
        "seeds": seed_list,
        "device": str(resolved_device),
        "results": results,
        "aggregate": aggregate,
        "screening_passed_all_seeds": all(row["screening_passed"] for row in results),
        "promotion_eligible": False,
        "promotion_blockers": [
            "No frozen Experiment 5/7 baseline comparison is included.",
            "The benchmark uses synthetic latent transformations, not natural semantic relations.",
            "The result must be reviewed with independent held-out and adversarial evaluation.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--contrastive-weight", type=float, default=0.10)
    parser.add_argument("--transition-supervision-weight", type=float, default=1.0)
    parser.add_argument("--rollout-supervision-weight", type=float, default=0.0)
    parser.add_argument(
        "--transition-architecture",
        choices=("mlp", "linear_operator"),
        default="mlp",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="artifacts/exp6_multiseed_screening.json")
    args = parser.parse_args(argv)

    try:
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    except ValueError as exc:
        parser.error(f"invalid seed list: {exc}")
    report = run_multiseed_benchmark(
        seeds,
        config=Config(
            epochs=args.epochs, lr=args.lr,
            contrastive_weight=args.contrastive_weight,
            transition_supervision_weight=args.transition_supervision_weight,
            rollout_supervision_weight=args.rollout_supervision_weight,
            transition_architecture=args.transition_architecture,
        ),
        device=args.device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "screening_passed_all_seeds": report["screening_passed_all_seeds"],
        "promotion_eligible": report["promotion_eligible"],
        "aggregate": report["aggregate"],
        "per_seed": report["results"],
    }, indent=2))
    return 0 if report["screening_passed_all_seeds"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
