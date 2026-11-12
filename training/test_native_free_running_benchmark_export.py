import json

from training.export_native_free_running_prompts import (
    build_prompt_cases,
    export_prompt_bundle,
)
from training.free_running_eval import load_prompt_cases


def test_exported_heldout_bundle_is_frozen_and_explicitly_not_promotion_grade(tmp_path):
    output = tmp_path / "native_heldout_v1.jsonl"
    manifest_path = tmp_path / "native_heldout_v1.manifest.json"

    manifest = export_prompt_bundle(
        output,
        partition="held_out",
        manifest_path=manifest_path,
    )
    cases = load_prompt_cases(output)
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(cases) == manifest["case_count"] == 10
    assert manifest["case_ids"] == [case["case_id"] for case in cases]
    assert manifest["prompt_file_sha256"] == saved_manifest["prompt_file_sha256"]
    assert manifest["promotion_eligible"] is False
    assert "not been audited for disjointness" in " ".join(manifest["limitations"])
    assert all(case["target"] for case in cases)


def test_exporter_keeps_dev_and_heldout_partitions_disjoint():
    dev = build_prompt_cases("dev")
    heldout = build_prompt_cases("held_out")
    dev_ids = {case["case_id"] for case in dev}
    heldout_ids = {case["case_id"] for case in heldout}

    assert len(dev) == 10
    assert len(heldout) == 10
    assert dev_ids.isdisjoint(heldout_ids)
    assert all(case["source_partition"] == "dev" for case in dev)
    assert all(case["source_partition"] == "held_out" for case in heldout)
