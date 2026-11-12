import json

from training.free_running_eval import (
    evaluate_checkpoint,
    fingerprint_cases,
    load_prompt_cases,
    score_generation,
    sha256_file,
)


def test_generation_diagnostics_measure_prompt_grounding_repetition_and_exact_match():
    metrics = score_generation(
        "Explain Python functions and classes",
        "Python functions define reusable behavior. Classes group related state and methods.",
        "Python functions define reusable behavior. Classes group related state and methods.",
    )
    assert metrics["nonempty"] is True
    assert metrics["prompt_keyword_coverage"] > 0.5
    assert metrics["exact_match"] is True
    assert metrics["replacement_character_count"] == 0


def test_empty_generation_is_not_counted_as_success():
    metrics = score_generation("Explain matrices", "")
    assert metrics["nonempty"] is False
    assert metrics["utf8_bytes"] == 0
    assert metrics["char_trigram_diversity"] == 0.0
    assert metrics["terminated"] is False



def test_file_and_selected_case_fingerprints_are_stable_and_content_sensitive(tmp_path):
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"checkpoint-v1")
    first_hash = sha256_file(artifact)
    assert first_hash == sha256_file(artifact)

    artifact.write_bytes(b"checkpoint-v2")
    assert sha256_file(artifact) != first_hash

    cases = [{"case_id": "a", "prompt": "Explain state", "target": "A state."}]
    assert fingerprint_cases(cases) == fingerprint_cases(list(cases))
    assert fingerprint_cases(cases) != fingerprint_cases([
        {"case_id": "a", "prompt": "Explain state", "target": "A hidden state."}
    ])


def test_evaluation_report_records_checkpoint_and_prompt_fingerprints(tmp_path, monkeypatch):
    import sys
    import types

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"minimal checkpoint fixture")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"case_id": "p1", "prompt": "Explain matrices", "target": "Matrices."}) + "\n",
        encoding="utf-8",
    )

    class FakeModel:
        def eval(self):
            return self

        def generate_text(self, prompt, **kwargs):
            return "Matrices organize values."

    fake_module = types.ModuleType("training.native_conversation_v5")
    fake_module.load_checkpoint = lambda path, device: (FakeModel(), {
        "format": "test-fixture",
        "config": {"test": True},
    })
    monkeypatch.setitem(sys.modules, "training.native_conversation_v5", fake_module)

    cases = load_prompt_cases(prompts)
    report = evaluate_checkpoint(
        checkpoint,
        cases,
        seeds=(17, 29, 43),
        device="cpu",
        prompt_file=prompts,
    )
    assert report["evaluation"] == "native_free_running_diagnostics_v2"
    assert report["checkpoint_sha256"] == sha256_file(checkpoint)
    assert report["checkpoint_size_bytes"] == checkpoint.stat().st_size
    assert report["prompt_file_sha256"] == sha256_file(prompts)
    assert report["selected_cases_sha256"] == fingerprint_cases(cases)
    assert len(report["samples"]) == 3


def test_evaluation_rejects_duplicate_seed_ids(tmp_path, monkeypatch):
    import sys
    import types
    import pytest

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"minimal checkpoint fixture")
    fake_module = types.ModuleType("training.native_conversation_v5")
    fake_module.load_checkpoint = lambda path, device: (object(), {"format": "test"})
    monkeypatch.setitem(sys.modules, "training.native_conversation_v5", fake_module)

    with pytest.raises(ValueError, match="seeds must be unique"):
        evaluate_checkpoint(
            checkpoint,
            [{"case_id": "x", "prompt": "Explain matrices", "target": None}],
            seeds=(17, 17, 29),
            device="cpu",
        )

def test_prompt_loader_accepts_prompt_and_target_aliases(tmp_path):
    data = tmp_path / "cases.jsonl"
    data.write_text(
        json.dumps({"case_id": "c1", "user_text": "Explain recursion", "target_text": "A function calls itself."}) + "\n",
        encoding="utf-8",
    )
    cases = load_prompt_cases(data)
    assert cases == [{
        "case_id": "c1",
        "prompt": "Explain recursion",
        "target": "A function calls itself.",
    }]
