import json

import pytest

from training.ruletaker_eval import (
    build_prompt,
    classification_metrics,
    load_manifest,
    load_ruletaker_rows,
    parse_prediction,
    select_test_cases,
)


def _balanced_rows():
    rows = []
    for depth in range(1, 9):
        for label in ("entailment", "not entailment"):
            rows.append({
                "context": f"Person {depth} is blue. Blue things are calm.",
                "question": f"Person {depth} is calm.",
                "label": label,
                "config": f"depth-{depth}",
            })
    return rows


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_rule_taker_manifest_is_pinned_and_license_approved():
    manifest = load_manifest()
    assert manifest["dataset_id"] == "tasksource/ruletaker"
    assert manifest["license"] == "Apache-2.0"
    assert manifest["splits"]["test"]["num_examples"] == 151911
    assert manifest["pinned_test_file"]["sha256"] == (
        "51dd62b07c199ba82870345e6c872eb0314770fbb0efbf30a632a1edbbbf3ffd"
    )


def test_load_jsonl_validates_required_fields_and_normalizes_labels(tmp_path):
    path = tmp_path / "test.jsonl"
    rows = _balanced_rows()
    rows[0]["label"] = "Entailment"
    _write_jsonl(path, rows)

    loaded = load_ruletaker_rows(path)
    assert len(loaded) == 16
    assert loaded[0]["label"] == "entailment"
    assert loaded[0]["config"] == "depth-1"


def test_loader_rejects_unknown_label_and_missing_required_fields(tmp_path):
    path = tmp_path / "bad.jsonl"
    rows = _balanced_rows()
    rows[0]["label"] = "unknown"
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="unsupported label"):
        load_ruletaker_rows(path)

    rows = _balanced_rows()
    del rows[0]["question"]
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="missing required string fields"):
        load_ruletaker_rows(path)


def test_prompt_contains_only_context_and_question_not_gold_label():
    row = {
        "context": "Ada is kind. Kind people are patient.",
        "question": "Ada is patient.",
        "label": "entailment",
        "config": "depth-2",
    }
    prompt = build_prompt(row)
    assert "Ada is kind." in prompt
    assert "Ada is patient." in prompt
    assert "Answer with exactly one label" in prompt
    assert "Gold label" not in prompt


def test_prediction_parser_is_strict_and_rejects_explanatory_prefixes():
    assert parse_prediction("entailment") == "entailment"
    assert parse_prediction("not entailment.") == "not entailment"
    assert parse_prediction("  NOT ENTAILMENT\n") == "not entailment"
    assert parse_prediction("The answer is entailment.") is None
    assert parse_prediction("uncertain") is None
    assert parse_prediction("") is None


def test_stratified_selection_is_deterministic_balanced_and_reports_byte_exclusions():
    rows = _balanced_rows()
    rows.append({
        "context": "x " * 1000,
        "question": "Extra statement.",
        "label": "entailment",
        "config": "depth-1",
    })

    first, stats = select_test_cases(
        rows, max_prompt_bytes=512, examples_per_stratum=1, sample_seed=29,
    )
    second, _ = select_test_cases(
        rows, max_prompt_bytes=512, examples_per_stratum=1, sample_seed=29,
    )

    assert [item["case_id"] for item in first] == [item["case_id"] for item in second]
    assert len(first) == 16
    assert stats["input_rows"] == 17
    assert stats["eligible_rows"] == 16
    assert stats["excluded_over_prompt_budget"] == 1
    assert set(stats["selected_by_stratum"].values()) == {1}


def test_stratified_selection_rejects_missing_depth_or_label_stratum():
    rows = _balanced_rows()[:-1]
    with pytest.raises(ValueError, match="missing eligible benchmark strata"):
        select_test_cases(
            rows, max_prompt_bytes=512, examples_per_stratum=1, sample_seed=17,
        )


def test_classification_metrics_count_invalid_outputs_as_incorrect():
    records = [
        {"gold_label": "entailment", "predicted_label": "entailment"},
        {"gold_label": "not entailment", "predicted_label": "not entailment"},
        {"gold_label": "entailment", "predicted_label": "not entailment"},
        {"gold_label": "not entailment", "predicted_label": None},
    ]
    metrics = classification_metrics(records)
    assert metrics["n"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["accuracy_on_parsed_outputs"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["coverage"] == 0.75
    assert metrics["invalid_output_rate"] == 0.25
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["confusion"] == {
        "true_entailment_predicted_entailment": 1,
        "true_not_entailment_predicted_not_entailment": 1,
        "false_positive": 1,
        "false_negative": 1,
        "invalid_output": 1,
    }


def test_manifest_and_loader_require_real_jsonl_content(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains no rows"):
        load_ruletaker_rows(path)
