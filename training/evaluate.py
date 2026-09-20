from __future__ import annotations

import argparse
import json

from .bytes import BOS, CONTEXT, GOAL, USER, RESPONSE, SPECIAL_TOKENS
from .dataset import load_jsonl, validate_split_isolation
from .model import load_checkpoint, require_torch


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


def evaluate(args):
    torch, _ = require_torch()
    examples = load_jsonl(args.dataset)
    validate_split_isolation(examples)
    items = [x for x in examples if x.split == args.split]
    if not items:
        raise ValueError(f"dataset has no {args.split} split")

    model, metadata = load_checkpoint(args.checkpoint)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)

    exact = 0
    nonempty = 0
    outputs = []
    for example in items[: args.max_examples]:
        prefix = torch.tensor([_prefix(example)], dtype=torch.long, device=device)
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
        nonempty += bool(text)
        exact += text == example.target_text.strip()
        outputs.append(
            {
                "example_id": example.example_id,
                "target": example.target_text,
                "prediction": text,
                "exact_match": text == example.target_text.strip(),
            }
        )

    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "examples": len(outputs),
        "nonempty": nonempty,
        "exact_match": exact,
        "nonempty_rate": nonempty / max(1, len(outputs)),
        "exact_match_rate": exact / max(1, len(outputs)),
        "checkpoint_metadata": metadata.get("metadata", {}),
        "outputs": outputs,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Mirror 7 response checkpoint.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--max-new-bytes", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
