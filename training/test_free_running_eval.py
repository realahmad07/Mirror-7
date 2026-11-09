import json

from training.free_running_eval import load_prompt_cases, score_generation


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
