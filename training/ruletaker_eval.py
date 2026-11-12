"""Held-out RuleTaker evaluation for Mirror's native conversation checkpoint.

The evaluator is local-only: it reads an already downloaded test split and never
fetches data or trains on it. RuleTaker is controlled/template-generated
natural-language reasoning, not an organic web-text benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from training.free_running_eval import evaluate_checkpoint, sha256_file

_DATASET = "tasksource/ruletaker"
_MANIFEST = Path(__file__).with_name("ruletaker_v1.manifest.json")
_LABELS = {"entailment", "not entailment"}
_LABEL_PATTERN = re.compile(r"^\s*(not\s+entailment|entailment)(?=$|[\s.!?,:;])", re.IGNORECASE)
_DEPTH_PATTERN = re.compile(r"^depth-([1-8])$")


def load_manifest(path: str | Path = _MANIFEST) -> dict[str, Any]:
    """Load the pinned benchmark provenance manifest."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("dataset_id") != _DATASET or data.get("license") != "Apache-2.0":
        raise ValueError("unexpected RuleTaker manifest identity or license")
    return data


def _normalize_row(item: dict[str, Any], row_number: int) -> dict[str, str]:
    required = ("context", "question", "label", "config")
    missing = [key for key in required if not isinstance(item.get(key), str) or not item[key].strip()]
    if missing:
        raise ValueError(f"RuleTaker row {row_number} is missing required string fields: {missing}")

    label = item["label"].strip().casefold()
    if label not in _LABELS:
        raise ValueError(f"RuleTaker row {row_number} has unsupported label: {item['label']!r}")
    config = item["config"].strip().casefold()
    if not _DEPTH_PATTERN.fullmatch(config):
        raise ValueError(f"RuleTaker row {row_number} has unsupported config: {item['config']!r}")

    return {
        "context": item["context"].strip(),
        "question": item["question"].strip(),
        "label": label,
        "config": config,
    }


def load_ruletaker_rows(path: str | Path) -> list[dict[str, str]]:
    """Read JSONL, or Parquet when optional pyarrow is installed."""
    source = Path(path)
    suffix = source.suffix.casefold()
    if suffix in {".jsonl", ".ndjson"}:
        raw_rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            raw_rows.append(item)
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError(
                "Parquet input requires optional dependency pyarrow; install requirements-benchmark.txt"
            ) from exc
        raw_rows = parquet.read_table(source, columns=["context", "question", "label", "config"]).to_pylist()
    else:
        raise ValueError("input must be .parquet, .jsonl, or .ndjson")

    if not raw_rows:
        raise ValueError("RuleTaker input contains no rows")
    rows = [_normalize_row(item, index) for index, item in enumerate(raw_rows)]
    return rows


def build_prompt(row: dict[str, str]) -> str:
    return (
        "Use only the facts and rules in the context. Determine whether the statement "
        "is entailed by the context. Answer with exactly one label: entailment or "
        "not entailment.\n\nContext:\n"
        f"{row['context']}\n\nStatement:\n{row['question']}\n\nLabel:"
    )


def parse_prediction(response: str) -> str | None:
    """Parse only a leading, unambiguous benchmark label; all other outputs abstain."""
    match = _LABEL_PATTERN.match(str(response or ""))
    return match.group(1).casefold() if match else None


def _stratum_key(row: dict[str, str]) -> str:
    return f"{row['config']}|{row['label']}"


def select_test_cases(
    rows: list[dict[str, str]],
    *,
    max_prompt_bytes: int,
    examples_per_stratum: int,
    sample_seed: int = 17,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic balanced sample, excluding prompts that exceed the byte budget."""
    if max_prompt_bytes < 1:
        raise ValueError("max_prompt_bytes must be positive")
    if examples_per_stratum < 1:
        raise ValueError("examples_per_stratum must be positive")

    eligible: dict[str, list[tuple[int, dict[str, str], str]]] = defaultdict(list)
    oversized: dict[str, int] = defaultdict(int)
    for index, row in enumerate(rows):
        prompt = build_prompt(row)
        key = _stratum_key(row)
        if len(prompt.encode("utf-8")) > max_prompt_bytes:
            oversized[key] += 1
            continue
        eligible[key].append((index, row, prompt))

    expected_keys = [
        f"depth-{depth}|{label}"
        for depth in range(1, 9)
        for label in ("entailment", "not entailment")
    ]
    missing = [key for key in expected_keys if not eligible.get(key)]
    if missing:
        raise ValueError(f"test input is missing eligible benchmark strata: {missing}")

    rng = random.Random(sample_seed)
    selected: list[dict[str, Any]] = []
    eligible_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    for key in expected_keys:
        group = eligible[key]
        eligible_counts[key] = len(group)
        take = min(examples_per_stratum, len(group))
        chosen = rng.sample(group, take)
        selected_counts[key] = take
        for index, row, prompt in chosen:
            digest = hashlib.sha256(
                f"{index}\0{row['config']}\0{row['label']}\0{row['context']}\0{row['question']}".encode("utf-8")
            ).hexdigest()[:12]
            selected.append({
                "case_id": f"ruletaker-test-{index:06d}-{digest}",
                "prompt": prompt,
                "target": row["label"],
                "gold_label": row["label"],
                "config": row["config"],
                "source_row": index,
            })

    selected.sort(key=lambda item: item["source_row"])
    stats = {
        "input_rows": len(rows),
        "eligible_rows": sum(eligible_counts.values()),
        "excluded_over_prompt_budget": sum(oversized.values()),
        "eligible_by_stratum": eligible_counts,
        "excluded_over_budget_by_stratum": {
            key: oversized.get(key, 0) for key in expected_keys
        },
        "selected_by_stratum": selected_counts,
        "selected_rows": len(selected),
        "max_prompt_bytes": max_prompt_bytes,
        "examples_per_stratum": examples_per_stratum,
        "sampling_seed": sample_seed,
    }
    return selected, stats


def classification_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("metrics require at least one prediction")

    tp = tn = fp = fn = invalid = 0
    for row in rows:
        gold = row["gold_label"]
        predicted = row["predicted_label"]
        if predicted is None:
            invalid += 1
            if gold == "entailment":
                fn += 1
            else:
                fp += 1
        elif gold == "entailment" and predicted == "entailment":
            tp += 1
        elif gold == "not entailment" and predicted == "not entailment":
            tn += 1
        elif gold == "not entailment" and predicted == "entailment":
            fp += 1
        else:
            fn += 1

    total = len(rows)
    positive_total = sum(row["gold_label"] == "entailment" for row in rows)
    negative_total = total - positive_total
    recall_positive = tp / positive_total if positive_total else 0.0
    recall_negative = tn / negative_total if negative_total else 0.0
    precision_positive = tp / (tp + fp) if (tp + fp) else 0.0
    precision_negative = tn / (tn + fn) if (tn + fn) else 0.0

    def f1(precision: float, recall: float) -> float:
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "n": total,
        "accuracy": round((tp + tn) / total, 4),
        "accuracy_on_parsed_outputs": round(
            (tp + tn) / (total - invalid), 4
        ) if total > invalid else None,
        "coverage": round((total - invalid) / total, 4),
        "invalid_output_rate": round(invalid / total, 4),
        "balanced_accuracy": round((recall_positive + recall_negative) / 2, 4),
        "macro_f1": round((f1(precision_positive, recall_positive) + f1(precision_negative, recall_negative)) / 2, 4),
        "confusion": {
            "true_entailment_predicted_entailment": tp,
            "true_not_entailment_predicted_not_entailment": tn,
            "false_positive": fp,
            "false_negative": fn,
            "invalid_output": invalid,
        },
    }


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = classification_metrics(records)
    by_depth: dict[str, Any] = {}
    for depth in range(1, 9):
        name = f"depth-{depth}"
        selected = [item for item in records if item["config"] == name]
        if selected:
            by_depth[name] = classification_metrics(selected)
    summary["by_depth"] = by_depth
    return summary


def evaluate_ruletaker(
    checkpoint: str | Path,
    dataset_path: str | Path,
    *,
    max_prompt_bytes: int = 512,
    examples_per_stratum: int = 100,
    sample_seed: int = 17,
    decode_seeds: Iterable[int] = (17,),
    device: str | None = None,
    max_new_tokens: int = 24,
    temperature: float = 0.0,
    top_k: int = 40,
    expected_source_sha256: str | None = None,
    manifest_path: str | Path = _MANIFEST,
) -> dict[str, Any]:
    """Run the native checkpoint on a stratified sample of the official test split."""
    manifest = load_manifest(manifest_path)
    dataset_file = Path(dataset_path)
    dataset_sha256 = sha256_file(dataset_file)
    if expected_source_sha256 and dataset_sha256.casefold() != expected_source_sha256.casefold():
        raise ValueError("dataset file SHA-256 does not match the expected source hash")
    rows = load_ruletaker_rows(dataset_file)
    expected_rows = manifest["splits"]["test"]["num_examples"]
    if len(rows) != expected_rows:
        raise ValueError(
            f"expected the complete {expected_rows}-row RuleTaker test split; got {len(rows)}"
        )

    selected, selection = select_test_cases(
        rows,
        max_prompt_bytes=max_prompt_bytes,
        examples_per_stratum=examples_per_stratum,
        sample_seed=sample_seed,
    )
    cases = [
        {"case_id": item["case_id"], "prompt": item["prompt"], "target": item["target"]}
        for item in selected
    ]
    generation = evaluate_checkpoint(
        checkpoint,
        cases,
        seeds=decode_seeds,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        prompt_file=dataset_file,
    )

    metadata = {item["case_id"]: item for item in selected}
    scored: list[dict[str, Any]] = []
    for sample in generation["samples"]:
        meta = metadata[sample["case_id"]]
        scored.append({
            "case_id": sample["case_id"],
            "seed": sample["seed"],
            "config": meta["config"],
            "gold_label": meta["gold_label"],
            "predicted_label": parse_prediction(sample["response"]),
            "response": sample["response"],
        })

    by_seed: dict[str, Any] = {}
    for seed in generation["summary"]["seeds"]:
        seed_records = [item for item in scored if item["seed"] == seed]
        by_seed[str(seed)] = _summarize_records(seed_records)

    return {
        "evaluation": "mirror7_ruletaker_test_v1",
        "dataset": {
            **manifest,
            "split": "test",
            "source_file": str(dataset_file),
            "source_file_sha256": dataset_sha256,
        },
        "checkpoint": {
            "path": generation["checkpoint"],
            "sha256": generation["checkpoint_sha256"],
            "size_bytes": generation["checkpoint_size_bytes"],
            "format": generation["checkpoint_format"],
            "config": generation["model_config"],
        },
        "generation": {
            "device": generation["device"],
            "decoding": generation["decoding"],
            "seeds": generation["summary"]["seeds"],
            "seed_variation_is_sampling_only": generation["seed_sensitivity"]["measured"],
        },
        "selection": selection,
        "metrics": _summarize_records(scored),
        "metrics_by_seed": by_seed,
        "samples": scored,
        "limitations": [
            "RuleTaker uses controlled/template-generated natural-language theories and assertions; it is not an organic text benchmark.",
            "Only the official test split is evaluated; no test examples are used for training or tuning.",
            "Prompts longer than the configured byte budget are excluded rather than truncated; exclusions and strata counts are reported.",
            "Sampling seeds vary decoding only; they do not measure parameter-initialization or training-seed robustness.",
            "Exact-label parsing is strict; explanations before the required label count as invalid output.",
            "Results do not establish factual knowledge, unrestricted semantic understanding, or general intelligence.",
            "Training-corpus contamination must be checked separately before claiming an unseen benchmark.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-file", required=True, help="Complete RuleTaker test split (.parquet or JSONL export)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-prompt-bytes", type=int, default=512)
    parser.add_argument("--examples-per-stratum", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=17)
    parser.add_argument("--decode-seeds", default="17")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--expected-source-sha256", default=None)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or args.top_k < 1 or args.temperature < 0:
        parser.error("decoding limits must be positive and temperature non-negative")
    seeds = [int(value.strip()) for value in args.decode_seeds.split(",") if value.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("--decode-seeds must contain one or more unique integer IDs")

    report = evaluate_ruletaker(
        args.checkpoint,
        args.test_file,
        max_prompt_bytes=args.max_prompt_bytes,
        examples_per_stratum=args.examples_per_stratum,
        sample_seed=args.sample_seed,
        decode_seeds=seeds,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        expected_source_sha256=args.expected_source_sha256,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "metrics": report["metrics"],
        "selection": report["selection"],
        "checkpoint_sha256": report["checkpoint"]["sha256"],
        "dataset_sha256": report["dataset"]["source_file_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
