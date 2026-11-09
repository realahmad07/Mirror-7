from __future__ import annotations

import json
from pathlib import Path

from mirror_crystal import MirrorCrystalEngine, ZeroTrustSandboxJudge


def test_context_barcodes_discover_synonyms_without_weights():
    eng = MirrorCrystalEngine()
    eng.observe_unlabeled_text(
        "\n".join(
            [
                "configure gateway server_limit before production launch",
                "configure gateway traffic_cap before production launch",
                "monitor gateway server_limit during peak load",
                "monitor gateway traffic_cap during peak load",
            ]
        )
    )
    merged = eng.bit_table.fold_equivalences(min_shared_specific_frames=2)
    assert merged >= 1

    # Teach ONLY using "server_limit", then query using unseen phrasing "traffic_cap"!
    eng.ingest_pair(
        "How should we set gateway server_limit?",
        "Set gateway capacity to 80% of benchmarked saturation throughput.",
    )
    res = eng.respond("How should we set gateway traffic_cap?")
    assert "80% of benchmarked saturation throughput" in res.answer
    assert res.confidence > 0.55


def test_one_shot_slot_formula_generalizes_to_unseen_numbers():
    eng = MirrorCrystalEngine()
    # Teach ONLY ONE modular exponentiation example
    eng.ingest_pair(
        "Compute 17^19 mod 1000000007",
        "The exact modular result of 17^19 mod 1000000007 is 286765645.",
    )

    # Test on completely unseen large integers!
    res = eng.respond("Compute 31^43 mod 1000000007")
    expected = pow(31, 43, 1000000007)
    assert str(expected) in res.answer
    assert res.mode == "slot_formula_lock"


def test_multi_hop_lock_chaining_solves_unseen_combination():
    eng = MirrorCrystalEngine()
    eng.observe_unlabeled_text("ceramic_mug is a fragile_item.")
    eng.observe_unlabeled_text("if fragile_item, dropping_on_tile_shatters_it.")

    res = eng.respond("What happens to a ceramic_mug?")
    assert res.mode == "multi_hop_rule_chain"
    assert "dropping_on_tile_shatters_it" in res.answer


def test_multi_turn_session_memory_and_verified_rate_limiter_example():
    eng = MirrorCrystalEngine()
    eng.respond("My server limit is 120 requests/sec.", session_id="s1")

    r1 = eng.respond("Are 3 bursts of 45 safe? If not, give me a limiter.", session_id="s1")
    assert r1.mode == "session_constraint_lock"
    assert "Unsafe: 3 * 45 = 135" in r1.answer
    assert "exceeds your 120 limit by 15" in r1.answer
    assert "class TokenBucketLimiter" in r1.answer
    assert r1.sandbox_verified is True

    # Follow-up turn 20 minutes later without repeating the 120 limit:
    r2 = eng.respond("What about 3 bursts of 35?", session_id="s1")
    assert "Safe: 3 * 35 = 105" in r2.answer


def test_150_sequential_teachings_have_zero_catastrophic_forgetting():
    eng = MirrorCrystalEngine()
    for i in range(150):
        eng.ingest_pair(f"Lookup sensor channel {5000 + i}", f"Channel {5000 + i} reads {i * 3} Kelvin.")

    # Verify item #0, #75, and #149 are recalled with 100% precision
    assert "0 Kelvin" in eng.respond("Lookup sensor channel 5000").answer
    assert "225 Kelvin" in eng.respond("Lookup sensor channel 5075").answer
    assert "447 Kelvin" in eng.respond("Lookup sensor channel 5149").answer


def test_sandbox_reality_filter_rejects_broken_or_unsafe_code_and_persists(tmp_path: Path):
    eng = MirrorCrystalEngine()

    # Broken code fails unit test -> rejected automatically during ingestion!
    created_bad = eng.ingest_pair(
        "Write add_two(a, b)",
        "def add_two(a, b):\n  return a - b\n",
        test_code="assert add_two(2, 3) == 5",
    )
    assert created_bad is False
    assert eng.stats["sandbox_rejected_rules"] == 1

    #Unsafe code rejected by ZeroTrustSandboxJudge
    ok_unsafe, _ = ZeroTrustSandboxJudge.verify_code_and_test("import os\ndef h(): return os.getcwd()")
    assert ok_unsafe is False

    # Valid code passes unit test -> crystallized with verified bonus!
    jsonl_file = tmp_path / "train.jsonl"
    rows = [
        {
            "user_text": "Implement binary_search(arr, target) in Python.",
            "target_text": (
                "def binary_search(arr, target):\n"
                "  lo, hi = 0, len(arr) - 1\n"
                "  while lo <= hi:\n"
                "    mid = (lo + hi) // 2\n"
                "    if arr[mid] == target: return mid\n"
                "    if arr[mid] < target: lo = mid + 1\n"
                "    else: hi = mid - 1\n"
                "  return -1\n"
            ),
            "test": "assert binary_search([1, 4, 7, 9], 7) == 2\nassert binary_search([1, 4, 7], 5) == -1",
        }
    ]
    jsonl_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    stats = eng.ingest_jsonl(jsonl_file)
    assert stats["rows_ingested"] == 1

    eng.observe_unlabeled_text("obsidian_goblet is a fragile_artifact.")
    eng.observe_unlabeled_text("if fragile_artifact, dropping_shatters_immediately.")

    brain_path = tmp_path / "brain.mcrystal"
    size_b = eng.save(brain_path)
    assert size_b > 0

    reloaded = MirrorCrystalEngine.load(brain_path)
    ans = reloaded.respond("Implement binary_search(arr, target) in Python.")
    assert "def binary_search" in ans.answer
    assert ans.sandbox_verified is True

    # Reloaded brain also preserves multi-hop causal chaining & honest abstention
    chained_reloaded = reloaded.respond("What happens to an obsidian_goblet?")
    assert chained_reloaded.mode == "multi_hop_rule_chain"
    assert "dropping_shatters_immediately" in chained_reloaded.answer

    abstain = reloaded.respond("Zyxwvut unrelated alien telemetry 99999?")
    assert abstain.mode == "unmatched_honest_abstain"

def test_mcrystal_archive_zip_loads_and_answers_stored_rule(tmp_path: Path):
    """The distributable crystal_4 ZIP must behave identically to the .mcrystal file."""
    import zipfile

    repo_root = Path(__file__).resolve().parents[2]
    brain_path = repo_root / "artifacts" / "mirror_crystal_v1.mcrystal"
    assert brain_path.is_file()

    archive_path = tmp_path / "crystal_4.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(brain_path, "crystal_4/mirror_crystal_v1.mcrystal")

    engine = MirrorCrystalEngine.load(archive_path)
    stored = engine.respond("What is protocol code 1000?")
    assert "channel_0" in stored.answer
    assert stored.mode != "unmatched_honest_abstain"

    # Check the same archive-backed engine still handles a symbolic calculation
    # that is not dependent on a stored question/answer pair.
    math_result = engine.respond("Compute 31^43 mod 1000000007")
    assert str(pow(31, 43, 1000000007)) in math_result.answer

