from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .bytes import BOS, CONTEXT, GOAL, USER, RESPONSE, SPECIAL_TOKENS
from .dataset import load_jsonl, validate_split_isolation
from .model import require_torch


def _prefix(example):
    context = example.canonical_context().decode("utf-8")
    return [
        BOS,
        CONTEXT,
        *context.encode("utf-8"),
        GOAL,
        USER,
        *example.user_text.encode("utf-8"),
        RESPONSE,
    ]


def _decode(ids):
    payload = [value for value in ids if 0 <= value <= 255]
    return bytes(payload).decode("utf-8", errors="replace")


def _repetition_ratio(text: str) -> float:
    """Calculate ratio of repeated 3-grams to measure degeneration.
    Returns 0.0 for completely distinct n-grams, close to 1.0 for repetitive loops.
    """
    words = text.split()
    if len(words) < 4:
        return 0.0
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    return 1.0 - (len(set(trigrams)) / len(trigrams))


def evaluate(args):
    torch, _ = require_torch()
    examples = load_jsonl(args.dataset)
    validate_split_isolation(examples)
    items = [x for x in examples if x.split == args.split]
    if not items:
        raise ValueError(f"dataset has no {args.split} split")

    checkpoint_path = Path(args.checkpoint)
    raw_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fmt = raw_payload.get("format")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if fmt == "mirror7-native-conversation-v3-cross-attention":
        from .native_conversation import (
            PAD,
            _prepare_targets,
            build_native_conversation,
            load_checkpoint as load_v3,
        )
        from .native_brain import _batch, _encode

        model, metadata = load_v3(checkpoint_path, device=str(device))
        config = model.config

        # 1. Compute evaluation split loss (teacher-forced)
        model.eval()
        total_loss = 0.0
        batches = 0
        batch_size = 16
        with torch.no_grad():
            for start in range(0, len(items), batch_size):
                batch = items[start : start + batch_size]
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
                batches += 1

        eval_loss = total_loss / max(1, batches)

        # 2. Evaluate free-running generation
        exact = 0
        exact_eligible = 0
        nonempty = 0
        outputs = []

        for example in items[: args.max_examples]:
            text = model.generate_text(
                example.user_text,
                max_new_tokens=args.max_new_bytes,
                min_new_tokens=getattr(args, "min_new_bytes", 1),
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=getattr(args, "top_p", 1.0),
                repetition_penalty=getattr(args, "repetition_penalty", 1.0),
                no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            ).strip()

            target = example.target_text.strip()
            target_bytes = len(target.encode("utf-8"))
            prediction_bytes = len(text.encode("utf-8"))
            eligible = target_bytes <= args.max_new_bytes
            is_nonempty = bool(text)
            nonempty += is_nonempty
            is_exact = eligible and (text == target)
            if eligible:
                exact_eligible += 1
                exact += is_exact

            rep_ratio = _repetition_ratio(text)

            outputs.append(
                {
                    "example_id": example.example_id,
                    "target": example.target_text,
                    "prediction": text,
                    "target_bytes": target_bytes,
                    "prediction_bytes": prediction_bytes,
                    "exact_match_eligible": eligible,
                    "exact_match": is_exact,
                    "repetition_ratio": rep_ratio,
                }
            )

        avg_rep = sum(x["repetition_ratio"] for x in outputs) / max(1, len(outputs))

        result = {
            "checkpoint": str(args.checkpoint),
            "format": fmt,
            "split": args.split,
            "eval_loss": eval_loss,
            "examples": len(outputs),
            "nonempty": nonempty,
            "exact_match": exact,
            "nonempty_rate": nonempty / max(1, len(outputs)),
            "exact_match_eligible": exact_eligible,
            "exact_match_rate": exact / max(1, exact_eligible),
            "avg_repetition_ratio": avg_rep,
            "truncated_target_count": len(outputs) - exact_eligible,
            "checkpoint_metadata": metadata.get("metadata", {}),
            "outputs": outputs,
        }
        if getattr(args, "output", None):
            out_file = Path(args.output)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        try:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except UnicodeEncodeError:
            print(json.dumps(result, ensure_ascii=True, indent=2))
        return result

    elif fmt == "mirror7-byte-gru-v1":
        from .model import load_checkpoint as load_v1

        model, metadata = load_v1(args.checkpoint)
        model.to(device)

        exact = 0
        exact_eligible = 0
        nonempty = 0
        outputs = []
        for example in items[: args.max_examples]:
            prefix = torch.tensor(
                [_prefix(example)], dtype=torch.long, device=device
            )
            generated = model.generate(
                prefix,
                max_new_bytes=args.max_new_bytes,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            generated_ids = generated[0].tolist()
            response_start = generated_ids.index(RESPONSE) + 1
            tail = generated_ids[response_start:]
            if SPECIAL_TOKENS["EOS"] in tail:
                tail = tail[: tail.index(SPECIAL_TOKENS["EOS"])]
            text = _decode(tail).strip()
            target = example.target_text.strip()
            target_bytes = len(target.encode("utf-8"))
            prediction_bytes = len(text.encode("utf-8"))
            eligible = target_bytes <= args.max_new_bytes
            nonempty += bool(text)
            if eligible:
                exact_eligible += 1
                exact += text == target
            outputs.append(
                {
                    "example_id": example.example_id,
                    "target": example.target_text,
                    "prediction": text,
                    "target_bytes": target_bytes,
                    "prediction_bytes": prediction_bytes,
                    "exact_match_eligible": eligible,
                    "exact_match": eligible and text == target,
                }
            )

        result = {
            "checkpoint": str(args.checkpoint),
            "format": fmt,
            "split": args.split,
            "examples": len(outputs),
            "nonempty": nonempty,
            "exact_match": exact,
            "nonempty_rate": nonempty / max(1, len(outputs)),
            "exact_match_eligible": exact_eligible,
            "exact_match_rate": exact / max(1, exact_eligible),
            "truncated_target_count": len(outputs) - exact_eligible,
            "checkpoint_metadata": metadata.get("metadata", {}),
            "outputs": outputs,
        }
        if getattr(args, "output", None):
            out_file = Path(args.output)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        try:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except UnicodeEncodeError:
            print(json.dumps(result, ensure_ascii=True, indent=2))
        return result
    else:
        raise ValueError(f"unsupported Mirror 7 checkpoint format: {fmt!r}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a Mirror 7 response checkpoint."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--max-new-bytes", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--min-new-bytes", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
