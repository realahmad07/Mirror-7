from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable
import hashlib
import json

from .schema import TrainingExample, example_from_dict


def load_jsonl(path: str | Path) -> list[TrainingExample]:
    path = Path(path)
    examples: list[TrainingExample] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
                example = example_from_dict(value)
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if example.example_id in seen_ids:
                raise ValueError(
                    f"{path}:{line_no}: duplicate example_id {example.example_id!r}"
                )
            seen_ids.add(example.example_id)
            examples.append(example)

    if not examples:
        raise ValueError(f"{path}: no training records found")

    return examples


def validate_split_isolation(examples: Iterable[TrainingExample]) -> None:
    owners: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    for example in examples:
        example.validate()
        prior = owners.setdefault(example.example_id, example.split)
        if prior != example.split:
            raise ValueError(
                f"example_id {example.example_id!r} occurs in multiple splits"
            )
        fingerprint = example.fingerprint()
        key = hashlib.sha256(
            example.user_text.strip().lower().encode("utf-8")
        ).hexdigest()
        previous = fingerprints.get(key)
        if previous is not None and previous != fingerprint:
            raise ValueError(
                f"possible user-input leakage across records: {example.example_id!r}"
            )
        fingerprints[key] = fingerprint


def summarize(examples: Iterable[TrainingExample]) -> dict[str, object]:
    items = list(examples)
    counts = Counter(example.split for example in items)
    return {
        "total": len(items),
        "splits": dict(sorted(counts.items())),
        "avg_user_bytes": (
            sum(len(x.user_text.encode("utf-8")) for x in items) / len(items)
        ),
        "avg_target_bytes": (
            sum(len(x.target_text.encode("utf-8")) for x in items) / len(items)
        ),
        "fingerprint": hashlib.sha256(
            "".join(sorted(x.fingerprint() for x in items)).encode("ascii")
        ).hexdigest(),
    }
