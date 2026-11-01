from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from training.dataset import load_jsonl, validate_split_isolation
from training.model import require_torch


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


def word_jaccard(text1: str, text2: str) -> float:
    w1 = set(w.lower() for w in text1.split() if len(w) > 2)
    w2 = set(w.lower() for w in text2.split() if len(w) > 2)
    if not w1 or not w2:
        return 0.0
    return len(w1.intersection(w2)) / len(w1.union(w2))


def load_model_any(checkpoint_path: str, device: str = "cpu"):
    torch, _ = require_torch()
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fmt = raw.get("format", "")
    if fmt == "mirror7-native-conversation-v5-input-conditioned":
        from training.native_conversation_v5 import load_checkpoint as load_v5
        return load_v5(checkpoint_path, device=device)
    elif fmt == "mirror7-native-conversation-v3-cross-attention":
        from training.native_conversation import load_checkpoint as load_v3
        return load_v3(checkpoint_path, device=device)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def run_benchmark(
    checkpoint_path: str,
    dataset_path: str = "training/data/mirror_oasst2_curated_v1.jsonl",
    split_type: str = "untouched",  # "dev" or "untouched"
    num_examples: int = 100,
    max_new_bytes: int = 256,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 5,
    min_new_tokens: int = 10,
    device: str = "cpu",
    output_path: str | None = None,
):
    model, metadata = load_model_any(checkpoint_path, device=device)
    model.eval()

    examples = load_jsonl(dataset_path)
    validate_split_isolation(examples)
    test_items = [x for x in examples if x.split == "test"]

    # Partition: dev uses offset 0, untouched final uses offset 7
    offset = 0 if split_type == "dev" else 7
    stride = max(1, len(test_items) // num_examples)
    benchmark_items = [test_items[i * stride + offset] for i in range(num_examples)]

    # Compute loss on benchmark items
    torch, _ = require_torch()
    total_loss = 0.0
    batches = 0
    batch_size = 16
    with torch.no_grad():
        for start in range(0, len(benchmark_items), batch_size):
            batch = benchmark_items[start : start + batch_size]
            from training.native_brain import _batch, _encode
            from training.native_conversation import PAD, _prepare_targets
            ui, um = _batch([_encode(x.user_text, 512) for x in batch], 512)
            ti, _ = _batch(_prepare_targets([x.target_text for x in batch], 512), 512)
            ui, um, ti = ui.to(device), um.to(device), ti.to(device)
            logits, _, _, _ = model(ui, um, ti[:, :-1])
            labels = ti[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=PAD,
            )
            total_loss += float(loss.cpu())
            batches += 1

    eval_loss = total_loss / max(1, batches)

    outputs = []
    proper_eos = 0
    rep_ratios = []
    vocab_divs = []
    f1_scores = []
    prompt_jaccards = []
    lengths = []

    for idx, ex in enumerate(benchmark_items):
        pred_text = model.generate_text(
            ex.user_text,
            max_new_tokens=max_new_bytes,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        ).strip()

        pred_bytes = len(pred_text.encode("utf-8"))
        emitted_eos = 0 < pred_bytes < max_new_bytes
        if emitted_eos:
            proper_eos += 1

        rep = repetition_ratio(pred_text)
        vd = vocabulary_diversity(pred_text)
        f1 = ngram_f1(pred_text, ex.target_text, n=4)
        pj = word_jaccard(pred_text, ex.user_text)

        rep_ratios.append(rep)
        vocab_divs.append(vd)
        f1_scores.append(f1)
        prompt_jaccards.append(pj)
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
                "prompt_jaccard": pj,
            }
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "split_type": split_type,
        "format": metadata.get("format", ""),
        "total_examples": len(benchmark_items),
        "benchmark_loss": eval_loss,
        "proper_eos_rate": proper_eos / max(1, len(benchmark_items)),
        "avg_repetition_ratio": sum(rep_ratios) / max(1, len(rep_ratios)),
        "avg_vocabulary_diversity": sum(vocab_divs) / max(1, len(vocab_divs)),
        "avg_char_4gram_f1": sum(f1_scores) / max(1, len(f1_scores)),
        "avg_prompt_jaccard": sum(prompt_jaccards) / max(1, len(prompt_jaccards)),
        "mean_bytes": sum(lengths) / max(1, len(lengths)),
        "min_bytes": min(lengths) if lengths else 0,
        "max_bytes": max(lengths) if lengths else 0,
    }

    report = {"summary": summary, "outputs": outputs}
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="training/data/mirror_oasst2_curated_v1.jsonl")
    p.add_argument("--split-type", default="untouched", choices=["dev", "untouched"])
    p.add_argument("--num-examples", type=int, default=100)
    p.add_argument("--max-new-bytes", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.2)
    p.add_argument("--no-repeat-ngram-size", type=int, default=5)
    p.add_argument("--min-new-tokens", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    report = run_benchmark(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        split_type=args.split_type,
        num_examples=args.num_examples,
        max_new_bytes=args.max_new_bytes,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        min_new_tokens=args.min_new_tokens,
        device=args.device,
        output_path=args.output,
    )

    print("=" * 60)
    print(f"BENCHMARK SUMMARY ({args.split_type.upper()} SPLIT)")
    print("=" * 60)
    for k, v in report["summary"].items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
