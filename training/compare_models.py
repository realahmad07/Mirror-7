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
    words = text.split()
    if len(words) < 4:
        return 0.0
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    return 1.0 - (len(set(trigrams)) / len(trigrams))


def vocabulary_diversity(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def ngram_f1(pred: str, target: str, n: int = 4) -> float:
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


def word_overlap_jaccard(pred: str, prompt: str) -> float:
    p_words = set(w.lower() for w in pred.split() if len(w) > 2)
    prompt_words = set(w.lower() for w in prompt.split() if len(w) > 2)
    if not p_words or not prompt_words:
        return 0.0
    return len(p_words.intersection(prompt_words)) / len(p_words.union(prompt_words))


def evaluate_configuration(
    model,
    benchmark_items,
    decoding_kwargs,
    max_new_bytes=256,
):
    nonempty_count = 0
    proper_eos_count = 0
    rep_ratios = []
    vocab_diversities = []
    f1_scores = []
    prompt_jaccards = []
    lengths = []
    outputs = []

    for idx, ex in enumerate(benchmark_items):
        pred_text = model.generate_text(
            ex.user_text,
            max_new_tokens=max_new_bytes,
            **decoding_kwargs,
        ).strip()

        pred_bytes = len(pred_text.encode("utf-8"))
        is_nonempty = pred_bytes > 0
        if is_nonempty:
            nonempty_count += 1

        # Check if model emitted EOS cleanly before hitting max_new_bytes limit
        emitted_eos = 0 < pred_bytes < max_new_bytes
        if emitted_eos:
            proper_eos_count += 1

        rep = repetition_ratio(pred_text)
        vd = vocabulary_diversity(pred_text)
        f1 = ngram_f1(pred_text, ex.target_text, n=4)
        jaccard = word_overlap_jaccard(pred_text, ex.user_text)

        rep_ratios.append(rep)
        vocab_diversities.append(vd)
        f1_scores.append(f1)
        prompt_jaccards.append(jaccard)
        lengths.append(pred_bytes)

        outputs.append(
            {
                "index": idx + 1,
                "prompt": ex.user_text,
                "target": ex.target_text[:200],
                "prediction": pred_text,
                "bytes": pred_bytes,
                "emitted_eos": emitted_eos,
                "repetition_ratio": rep,
                "vocabulary_diversity": vd,
                "char_4gram_f1": f1,
                "prompt_jaccard": jaccard,
            }
        )

    summary = {
        "nonempty_rate": nonempty_count / max(1, len(benchmark_items)),
        "proper_eos_rate": proper_eos_count / max(1, len(benchmark_items)),
        "avg_repetition_ratio": sum(rep_ratios) / max(1, len(rep_ratios)),
        "avg_vocabulary_diversity": sum(vocab_diversities) / max(1, len(vocab_diversities)),
        "avg_char_4gram_f1": sum(f1_scores) / max(1, len(f1_scores)),
        "avg_prompt_jaccard": sum(prompt_jaccards) / max(1, len(prompt_jaccards)),
        "mean_bytes": sum(lengths) / max(1, len(lengths)),
        "min_bytes": min(lengths) if lengths else 0,
        "max_bytes": max(lengths) if lengths else 0,
    }
    return summary, outputs


def run_comparison(
    baseline_path="artifacts/baseline_v3_5ep.pt",
    v4_path="artifacts/mirror7_native_conversation_v3_v4.pt",
    dataset_path="training/data/mirror_oasst2_curated_v1.jsonl",
    num_examples=100,
    max_new_bytes=256,
    device="cpu",
    output_file="artifacts/model_comparison_100.json",
):
    print(f"Loading dataset: {dataset_path}")
    examples = load_jsonl(dataset_path)
    validate_split_isolation(examples)
    test_items = [x for x in examples if x.split == "test"]
    print(f"Total test examples: {len(test_items)}")

    # Deterministic stride over held-out test split
    stride = max(1, len(test_items) // num_examples)
    benchmark_items = [test_items[i * stride] for i in range(min(num_examples, len(test_items)))]
    print(f"Selected {len(benchmark_items)} fixed held-out test questions.")

    configs = {
        "Unconstrained (Raw Sampling)": {
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "min_new_tokens": 1,
        },
        "Constrained (Penalized Decoding)": {
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 5,
            "min_new_tokens": 10,
        },
    }

    results = {}

    for model_name, path in [("Baseline (Epoch 5)", baseline_path), ("V4 (Epoch 6)", v4_path)]:
        print(f"\n--- Loading {model_name} from {path} ---")
        model, payload = load_checkpoint(path, device=device)
        model.eval()
        results[model_name] = {}

        for mode_name, kwargs in configs.items():
            print(f"  Evaluating {model_name} under {mode_name}...")
            summary, outputs = evaluate_configuration(
                model, benchmark_items, kwargs, max_new_bytes=max_new_bytes
            )
            results[model_name][mode_name] = {
                "summary": summary,
                "outputs": outputs,
                "decoding_kwargs": kwargs,
            }

    # Save to JSON
    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved full comparative evaluation report to {output_file}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="artifacts/baseline_v3_5ep.pt")
    p.add_argument("--v4", default="artifacts/mirror7_native_conversation_v3_v4.pt")
    p.add_argument("--dataset", default="training/data/mirror_oasst2_curated_v1.jsonl")
    p.add_argument("--num-examples", type=int, default=100)
    p.add_argument("--max-new-bytes", type=int, default=256)
    p.add_argument("--output", default="artifacts/model_comparison_100.json")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    run_comparison(
        baseline_path=args.baseline,
        v4_path=args.v4,
        dataset_path=args.dataset,
        num_examples=args.num_examples,
        max_new_bytes=args.max_new_bytes,
        device=args.device,
        output_file=args.output,
    )
