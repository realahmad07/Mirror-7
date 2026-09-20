from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

from .bytes import EOS, RESPONSE, VOCAB_SIZE, encode
from .dataset import load_jsonl, summarize, validate_split_isolation
from .schema import example_from_dict


def test_schema_and_byte_encoding_are_deterministic(tmp_path: Path):
    record = {
        "example_id": "x-1",
        "user_text": "hello",
        "target_text": "Hi!",
        "goal": "respond",
        "state": {"b": 2, "a": 1},
        "evidence": {"grounded": "demo"},
        "actions": ["reply"],
        "split": "train",
        "quality": 1.0,
    }
    example = example_from_dict(record)
    first = encode(example)
    second = encode(example)
    assert first == second
    assert VOCAB_SIZE > 256
    assert RESPONSE in first.input_ids
    assert EOS in first.input_ids
    assert first.loss_end == len(first.input_ids)
    assert first.loss_start < first.loss_end


def test_dataset_validation_and_fingerprint(tmp_path: Path):
    dataset = tmp_path / "data.jsonl"
    rows = [
        {"example_id":"train-a","user_text":"hello","target_text":"hi","split":"train"},
        {"example_id":"val-a","user_text":"bye","target_text":"bye","split":"validation"},
    ]
    dataset.write_text(
        "\n".join(json.dumps(x, sort_keys=True) for x in rows) + "\n",
        encoding="utf-8",
    )
    examples = load_jsonl(dataset)
    validate_split_isolation(examples)
    summary = summarize(examples)
    assert summary["total"] == 2
    assert summary["fingerprint"]


def test_smoke_dataset_builder(tmp_path: Path):
    completed = subprocess.run(
        [sys.executable, "-m", "training.make_smoke_dataset"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    generated = Path(completed.stdout.strip())
    try:
        assert generated.is_file()
        rows = load_jsonl(generated)
        validate_split_isolation(rows)
        assert len(rows) == 8
    finally:
        if generated.exists():
            generated.unlink()
