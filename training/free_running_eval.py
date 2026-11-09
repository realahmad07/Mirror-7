"""Reproducible free-running evaluation for NativeConversationV5 checkpoints.

These diagnostics measure generation behavior; they do not certify factual correctness.
Use the same frozen prompt file, seeds, decoding settings, and baseline when comparing
checkpoint revisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "what", "how", "why",
    "when", "where", "who", "are", "you", "can", "does", "explain", "show",
    "step", "give", "tell", "about", "into", "using", "between", "prove",
    "state", "find", "write", "please", "could", "would",
}


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", str(value).casefold())


def _normalize(value: str) -> str:
    return " ".join(_tokens(value))


def score_generation(prompt: str, response: str, target: str | None = None) -> dict[str, Any]:
    """Return transparent lexical/format diagnostics without implying truth."""
    text = str(response or "").strip()
    words = _tokens(text)
    prompt_terms = {
        item for item in _tokens(prompt)
        if len(item) >= 3 and item not in _STOPWORDS
    }
    response_terms = set(words)
    grounding = (
        len(prompt_terms & response_terms) / len(prompt_terms)
        if prompt_terms else None
    )

    char_window = text[:1000]
    trigrams = [char_window[i:i + 3] for i in range(max(0, len(char_window) - 2))]
    char_diversity = (
        len(set(trigrams)) / len(trigrams) if trigrams else (1.0 if text else 0.0)
    )
    fivegrams = [tuple(words[i:i + 5]) for i in range(max(0, len(words) - 4))]
    word_fivegram_diversity = (
        len(set(fivegrams)) / len(fivegrams) if fivegrams else (1.0 if words else 0.0)
    )
    exact_match = None
    if target is not None:
        exact_match = _normalize(text) == _normalize(target)

    return {
        "nonempty": bool(text),
        "utf8_bytes": len(text.encode("utf-8", errors="replace")),
        "replacement_character_count": text.count("\ufffd"),
        "char_trigram_diversity": round(char_diversity, 4),
        "word_fivegram_diversity": round(word_fivegram_diversity, 4),
        "prompt_keyword_coverage": round(grounding, 4) if grounding is not None else None,
        "terminated": bool(text) and text[-1] in ".!?;:}]\"",
        "exact_match": exact_match,
        "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def load_prompt_cases(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_no}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"line {line_no} must contain a JSON object")
        prompt = item.get("prompt") or item.get("user_text") or item.get("input")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"line {line_no} has no non-empty prompt/user_text/input")
        target = item.get("target") or item.get("target_text")
        cases.append({
            "case_id": str(item.get("case_id") or f"case-{line_no}"),
            "prompt": prompt.strip(),
            "target": target if isinstance(target, str) else None,
        })
        if limit is not None and len(cases) >= limit:
            break
    if not cases:
        raise ValueError("prompt file contains no usable cases")
    return cases


def evaluate_checkpoint(
    checkpoint: str | Path,
    cases: Iterable[dict[str, Any]],
    *,
    seeds: Iterable[int] = (17, 29, 43),
    device: str | None = None,
    max_new_tokens: int = 384,
    temperature: float = 0.0,
    top_k: int = 40,
) -> dict[str, Any]:
    import torch
    from training.native_conversation_v5 import load_checkpoint

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(checkpoint, device=resolved_device)
    model.eval()
    rows: list[dict[str, Any]] = []
    case_list = list(cases)
    seed_list = [int(item) for item in seeds]
    if not seed_list:
        raise ValueError("at least one seed is required")

    for seed in seed_list:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for case in case_list:
            output = model.generate_text(
                case["prompt"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=1.10,
                no_repeat_ngram_size=5,
                enforce_valid_utf8=True,
            )
            metrics = score_generation(case["prompt"], output, case.get("target"))
            rows.append({
                "case_id": case["case_id"],
                "seed": seed,
                "prompt": case["prompt"],
                "response": output,
                "metrics": metrics,
            })

    metric_rows = [row["metrics"] for row in rows]
    exact_values = [row["exact_match"] for row in metric_rows if row["exact_match"] is not None]
    grounding_values = [
        row["prompt_keyword_coverage"] for row in metric_rows
        if row["prompt_keyword_coverage"] is not None
    ]
    summary = {
        "samples": len(rows),
        "cases": len(case_list),
        "seeds": seed_list,
        "nonempty_rate": round(sum(row["nonempty"] for row in metric_rows) / len(metric_rows), 4),
        "mean_prompt_keyword_coverage": (
            round(statistics.fmean(grounding_values), 4) if grounding_values else None
        ),
        "low_char_diversity_rate": round(
            sum(row["char_trigram_diversity"] < 0.35 for row in metric_rows) / len(metric_rows), 4
        ),
        "low_fivegram_diversity_rate": round(
            sum(row["word_fivegram_diversity"] < 0.70 for row in metric_rows) / len(metric_rows), 4
        ),
        "replacement_character_rate": round(
            sum(row["replacement_character_count"] > 0 for row in metric_rows) / len(metric_rows), 4
        ),
        "termination_rate": round(sum(row["terminated"] for row in metric_rows) / len(metric_rows), 4),
        "exact_match_rate": (
            round(sum(exact_values) / len(exact_values), 4) if exact_values else None
        ),
    }
    return {
        "evaluation": "native_free_running_diagnostics_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_format": payload.get("format"),
        "model_config": payload.get("config", {}),
        "device": resolved_device,
        "decoding": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "repetition_penalty": 1.10,
            "no_repeat_ngram_size": 5,
            "enforce_valid_utf8": True,
        },
        "summary": summary,
        "samples": rows,
        "limitations": [
            "Prompt keyword coverage and repetition statistics are lexical diagnostics, not semantic correctness.",
            "Exact match is reported only when a target is present and is not a substitute for semantic evaluation.",
            "A checkpoint comparison is valid only with the same frozen cases, seeds, and decoding settings.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True, help="Frozen JSONL with prompt and optional target fields")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or args.top_k < 1 or args.temperature < 0:
        parser.error("decoding limits must be positive and temperature non-negative")
    cases = load_prompt_cases(args.prompts, args.limit)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    report = evaluate_checkpoint(
        args.checkpoint,
        cases,
        seeds=seeds,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
