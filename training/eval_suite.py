from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from training.dataset import load_jsonl, validate_split_isolation
from training.model import require_torch
from training.native_brain import _batch, _encode
from training.native_conversation import PAD, _prepare_targets, load_checkpoint


def repetition_ratio(text: str) -> float:
    """Calculate ratio of repeated word 3-grams (0.0 = completely unique, 1.0 = loop)."""
    words = text.split()
    if len(words) < 4:
        return 0.0
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    return 1.0 - (len(set(trigrams)) / len(trigrams))


def vocabulary_diversity(text: str) -> float:
    """Distinct words / total words."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def ngram_f1(pred: str, target: str, n: int = 4) -> float:
    """Character/byte n-gram F1 overlap between prediction and reference target."""
    if len(pred) < n or len(target) < n:
        return 0.0
    p_ngrams = set(pred[i : i + n] for i in range(len(pred) - n + 1))
    t_ngrams = set(target[i : i + n] for i in range(len(target) - n + 1))
    intersection = p_ngrams.intersection(t_ngrams)
    if not intersection:
        return 0.0
    prec = len(intersection) / len(p_ngrams)
    rec = len(intersection) / len(t_ngrams)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def run_eval_suite(
    checkpoint_path: str,
    dataset_path: str = "training/data/mirror_oasst2_curated_v1.jsonl",
    num_examples: int = 100,
    max_new_bytes: int = 256,
    min_new_bytes: int = 10,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 5,
    device: str = "cpu",
    output_path: str | None = None,
) -> dict:
    torch, _ = require_torch()
    examples = load_jsonl(dataset_path)
    validate_split_isolation(examples)
    test_items = [x for x in examples if x.split == "test"]
    if not test_items:
        raise ValueError("test split is empty")

    # Select deterministic, uniformly spaced 100 examples from the test split
    stride = max(1, len(test_items) // num_examples)
    benchmark_items = [test_items[i * stride] for i in range(min(num_examples, len(test_items)))]

    model, metadata = load_checkpoint(checkpoint_path, device=device)
    config = model.config
    model.eval()

    # 1. Compute loss on benchmark items
    total_loss = 0.0
    loss_batches = 0
    batch_size = 16
    with torch.no_grad():
        for start in range(0, len(benchmark_items), batch_size):
            batch = benchmark_items[start : start + batch_size]
            ui, um = _batch(
                [_encode(x.user_text, config.max_bytes) for x in batch],
                config.max_bytes,
            )
            target_rows = _prepare_targets(
                [x.target_text for x in batch],
                config.max_bytes,
            )
            ti, _ = _batch(target_rows, config.max_bytes)
            ui, um, ti = ui.to(device), um.to(device), ti.to(device)

            logits, _, _, _ = model(ui, um, ti[:, :-1])
            labels = ti[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=PAD,
            )
            total_loss += float(loss.cpu())
            loss_batches += 1

    benchmark_loss = total_loss / max(1, loss_batches)

    # 2. Free-running generation and quality metrics
    outputs = []
    nonempty_count = 0
    exact_matches = 0
    exact_eligible = 0

    rep_ratios = []
    vocab_diversities = []
    ngram_f1_scores = []
    pred_byte_lengths = []

    for i, ex in enumerate(benchmark_items):
        pred_text = model.generate_text(
            ex.user_text,
            max_new_tokens=max_new_bytes,
            min_new_tokens=min_new_bytes,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        ).strip()

        target_text = ex.target_text.strip()
        target_bytes = len(target_text.encode("utf-8"))
        pred_bytes = len(pred_text.encode("utf-8"))

        is_nonempty = pred_bytes > 0
        if is_nonempty:
            nonempty_count += 1

        eligible = target_bytes <= max_new_bytes
        is_exact = eligible and (pred_text == target_text)
        if eligible:
            exact_eligible += 1
            if is_exact:
                exact_matches += 1

        rep = repetition_ratio(pred_text)
        vd = vocabulary_diversity(pred_text)
        f1 = ngram_f1(pred_text, target_text, n=4)

        rep_ratios.append(rep)
        vocab_diversities.append(vd)
        ngram_f1_scores.append(f1)
        pred_byte_lengths.append(pred_bytes)

        outputs.append(
            {
                "index": i + 1,
                "example_id": ex.example_id,
                "prompt": ex.user_text,
                "target": target_text,
                "prediction": pred_text,
                "target_bytes": target_bytes,
                "prediction_bytes": pred_bytes,
                "repetition_ratio": rep,
                "vocabulary_diversity": vd,
                "char_4gram_f1": f1,
                "exact_match": is_exact,
            }
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "total_benchmark_examples": len(benchmark_items),
        "benchmark_loss": benchmark_loss,
        "nonempty_rate": nonempty_count / max(1, len(benchmark_items)),
        "exact_match_rate": exact_matches / max(1, exact_eligible),
        "avg_repetition_ratio": sum(rep_ratios) / max(1, len(rep_ratios)),
        "avg_vocabulary_diversity": sum(vocab_diversities) / max(1, len(vocab_diversities)),
        "avg_char_4gram_f1": sum(ngram_f1_scores) / max(1, len(ngram_f1_scores)),
        "mean_prediction_bytes": sum(pred_byte_lengths) / max(1, len(pred_byte_lengths)),
        "min_prediction_bytes": min(pred_byte_lengths) if pred_byte_lengths else 0,
        "max_prediction_bytes": max(pred_byte_lengths) if pred_byte_lengths else 0,
        "decoding_params": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "max_new_bytes": max_new_bytes,
            "min_new_bytes": min_new_bytes,
        },
    }

    report = {"summary": summary, "outputs": outputs}

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main():
    p = argparse.ArgumentParser(description="Mirror 7 Comprehensive Evaluation Suite")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="training/data/mirror_oasst2_curated_v1.jsonl")
    p.add_argument("--num-examples", type=int, default=100)
    p.add_argument("--max-new-bytes", type=int, default=256)
    p.add_argument("--min-new-bytes", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.2)
    p.add_argument("--no-repeat-ngram-size", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="artifacts/benchmark_report.json")
    args = p.parse_args()

    report = run_eval_suite(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        num_examples=args.num_examples,
        max_new_bytes=args.max_new_bytes,
        min_new_bytes=args.min_new_bytes,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        device=args.device,
        output_path=args.output,
    )

    print("=" * 60)
    print("MIRROR 7 BENCHMARK SUMMARY")
    print("=" * 60)
    for k, v in report["summary"].items():
        if k != "decoding_params":
            if isinstance(v, float):
                print(f"  {k:30s}: {v:.4f}")
            else:
                print(f"  {k:30s}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
