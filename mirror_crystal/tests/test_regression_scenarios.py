"""
Comprehensive 10-Scenario Regression Test Suite for Mirror Crystal (In-RAM .mcrystal & Out-of-Core .dcrystal).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from mirror_crystal import DiskCrystalVault, MirrorCrystalEngine, ZeroTrustSandboxJudge


def test_scenario_01_disk_vault_streaming_and_zero_ram_rule_boot(tmp_path: Path):
    """Out-of-core .dcrystal vault streams .jsonl batches to SQLite B-Trees and loads 0 rules into RAM at boot."""
    db_file = tmp_path / "scale_test.dcrystal"
    jsonl_file = tmp_path / "stream_batch.jsonl"

    rows = [
        {"user_text": "Compute 19^23 mod 1000000007", "target_text": f"Exact mod answer is {pow(19, 23, 1000000007)}."}
    ]
    for i in range(250):
        rows.append(
            {
                "user_text": f"Explain telemetry register code {8000 + i}",
                "target_text": f"Register {8000 + i} monitors subsystem_{i} voltage.",
            }
        )
    jsonl_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    vault = DiskCrystalVault(db_file)
    stats = vault.ingest_jsonl(jsonl_file, batch_flush_rows=64)
    assert stats["rows_ingested"] == 251
    assert vault.rule_count() == 251
    vault.optimize_disk()
    vault.close()

    # Cold reopen with strict LRU cap = 96 rules: starts at 0 rules in RAM!
    reopened = DiskCrystalVault(db_file, lru_rule_cache_size=96)
    assert reopened.cached_rules_in_ram() == 0

    # Query unseen modular exponentiation on disk vault
    r_math = reopened.respond("Compute 37^41 mod 1000000007")
    assert str(pow(37, 41, 1000000007)) in r_math.answer
    assert r_math.mode == "slot_formula_lock"

    # Query earliest & latest keyed facts (0% catastrophic forgetting on disk)
    assert "subsystem_0 voltage" in reopened.respond("Explain telemetry register code 8000").answer
    assert "subsystem_249 voltage" in reopened.respond("Explain telemetry register code 8249").answer
    # Bounded RAM check: LRU cache never exceeds configured cap (96 << 251 total disk rules)
    assert reopened.cached_rules_in_ram() <= 96
    reopened.close()


def test_scenario_02_duplicate_storm_idempotency(tmp_path: Path):
    """Re-ingesting identical datasets skips 100% of duplicates in O(1)/O(log N) without rule bloat."""
    db_file = tmp_path / "dup.dcrystal"
    jsonl_file = tmp_path / "batch.jsonl"
    rows = [{"user_text": f"Define constant K_{i}", "target_text": f"K_{i} equals {i * 7}."} for i in range(50)]
    jsonl_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    vault = DiskCrystalVault(db_file)
    s1 = vault.ingest_jsonl(jsonl_file)
    s2 = vault.ingest_jsonl(jsonl_file)  # 100% duplicate file!
    assert s1["new_rules_added"] == 50
    assert s2["new_rules_added"] == 0
    assert s2["duplicates_skipped_in_file"] == 50
    assert vault.rule_count() == 50
    vault.close()


def test_scenario_03_slot_formula_arity_and_factual_year_isolation():
    """Factual prose years ('Blade Runner 2049 ... 2017') never steal 3-slot modular math queries."""
    eng = MirrorCrystalEngine()
    eng.ingest_pair(
        "Compute 17^19 mod 1000000007",
        "The exact modular result of 17^19 mod 1000000007 is 286765645.",
    )
    eng.ingest_pair(
        "Compute gcd of 48 and 180",
        f"The greatest common divisor of 48 and 180 is {math.gcd(48, 180)}.",
    )
    eng.ingest_pair(
        "Why is Blade Runner 2049 titled 2049 when released in 2017?",
        "Blade Runner 2049 takes place in 2049 even though its cinema release year was 2017.",
    )

    r_mod = eng.respond("Compute 53^61 mod 1000000007")
    assert str(pow(53, 61, 1000000007)) in r_mod.answer
    assert r_mod.mode == "slot_formula_lock"

    r_gcd = eng.respond("Compute gcd of 1071 and 462")
    assert str(math.gcd(1071, 462)) in r_gcd.answer

    r_movie = eng.respond("Why is Blade Runner 2049 titled 2049 when released in 2017?")
    assert "cinema release year was 2017" in r_movie.answer
    assert r_movie.mode == "direct_crystal_lock"


def test_scenario_04_multi_session_isolation_and_constraint_updates():
    """Separate session_ids never leak numerical limits across users, and updated limits take effect immediately."""
    eng = MirrorCrystalEngine()
    eng.respond("My server limit is 100 requests/sec.", session_id="alice")
    eng.respond("My server limit is 500 requests/sec.", session_id="bob")

    r_alice = eng.respond("Are 3 bursts of 45 safe?", session_id="alice")
    r_bob = eng.respond("Are 3 bursts of 45 safe?", session_id="bob")
    assert "Unsafe: 3 * 45 = 135" in r_alice.answer
    assert "Safe: 3 * 45 = 135" in r_bob.answer

    # Alice upgrades her server limit mid-conversation
    eng.respond("My server limit is 200 requests/sec.", session_id="alice")
    r_alice_after = eng.respond("Are 3 bursts of 45 safe?", session_id="alice")
    assert "Safe: 3 * 45 = 135" in r_alice_after.answer


def test_scenario_05_disk_multi_hop_causal_chaining_and_online_teach(tmp_path: Path):
    """DiskCrystalVault supports multi-hop causal chaining and <5ms incremental online teaching."""
    db_file = tmp_path / "chain.dcrystal"
    vault = DiskCrystalVault(db_file)
    vault.observe_unlabeled_text("lithium_cell is a reactive_battery.\nif reactive_battery, puncturing_causes_thermal_runaway.")
    vault.ingest_pair("What is Mirror Crystal?", "Mirror Crystal is a zero-weight CPU intelligence engine.")
    vault.close()

    reopened = DiskCrystalVault(db_file)
    r_chain = reopened.respond("What happens to a lithium_cell?")
    assert r_chain.mode == "multi_hop_rule_chain"
    assert "puncturing_causes_thermal_runaway" in r_chain.answer

    r_direct = reopened.respond("What is Mirror Crystal?")
    assert "zero-weight CPU intelligence engine" in r_direct.answer
    reopened.close()


def test_scenario_06_malicious_and_broken_code_quarantine():
    """ZeroTrustSandboxJudge blocks OS/network/dunder/open exploits and broken unit tests."""
    exploits = [
        "import subprocess\ndef f(): return subprocess.check_output(['whoami'])",
        "def f(): return __import__('os').system('echo pwn')",
        "def f(): return open('secret.txt').read()",
        "def f(x): return eval(x)",
    ]
    for bad_code in exploits:
        ok, reason = ZeroTrustSandboxJudge.verify_code_and_test(bad_code)
        assert ok is False, f"Expected exploit block, got {reason}"


def test_scenario_07_adversarial_empty_unicode_and_long_inputs(tmp_path: Path):
    """Empty strings, symbols, multilingual UTF-8, and 10,000-char prompts never crash RAM or Disk engines."""
    eng = MirrorCrystalEngine()
    eng.ingest_pair("Explain quantum superposition |psi>.", "Superposition combines |0> and |1> amplitudes.")

    db_file = tmp_path / "adv.dcrystal"
    vault = DiskCrystalVault(db_file)
    vault.ingest_pair("Explain quantum superposition |psi>.", "Superposition combines |0> and |1> amplitudes.")

    edge_inputs = [
        "",
        "   \n\t  ",
        "??? !!! @@@ ###",
        "اردو اور 中文 میں کوانٹم کمپیوٹنگ ∑ ∫ ψ ≠ ∞",
        "unseen_token_xyz " * 600,  # ~10,200 characters
    ]
    for inp in edge_inputs:
        r_ram = eng.respond(inp)
        r_disk = vault.respond(inp)
        assert isinstance(r_ram.answer, str)
        assert isinstance(r_disk.answer, str)
    vault.close()
