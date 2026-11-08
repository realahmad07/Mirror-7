"""
Regression tests covering independent-architecture quality fixes:
1. Vault epistemic abstention on missing entities/years & low query coverage
2. Concurrent multi-thread SQLite WAL safety in DiskCrystalVault
3. Non-vacuous AutonomousTDDEngine test verification, self-repair & security rejection
4. Compositional symbolic math & counterexample refutation in CerebrumCrystalBridge
5. Bounded FIFO cache & 4-band LSH near-duplicate detection in SixGateDataHygienePipeline
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest

from mirror_crystal import (
    AutonomousTDDEngine,
    CerebrumCrystalBridge,
    DiskCrystalVault,
    MirrorCrystalEngine,
    SixGateDataHygienePipeline,
)


def test_vault_abstains_on_missing_entities_years_and_low_coverage(tmp_path: Path):
    vault = DiskCrystalVault(tmp_path / "abstain_test.dcrystal", lru_rule_cache_size=32)
    try:
        vault.ingest_pair(
            "How does a Fenwick Binary Indexed Tree work?",
            "Fenwick Binary Indexed Tree updates prefix sums in O(log N) time complexity.",
            domain="algorithms",
        )
        vault.ingest_pair(
            "Explain Fast Fourier Transform complexity",
            "Fast Fourier Transform computes DFT in O(N log N) time complexity.",
            domain="algorithms",
        )
        vault.ingest_pair(
            "How often is the World Cup held?",
            "The FIFA World Cup tournament is held every four years.",
            domain="sports",
        )
        vault.ingest_pair(
            "Explain why the sky is blue.",
            "Rayleigh scattering scatters shorter blue wavelengths of sunlight in Earth's atmosphere.",
            domain="physics",
        )

        # True match succeeds cleanly
        r_sky = vault.respond("Why is the sky blue?")
        assert r_sky.epistemic_tier in {"verified_exact", "high_confidence"}
        assert "blue" in r_sky.answer.lower()

        # Missing algorithm entity ('Dijkstra') must abstain instead of returning Fenwick
        r_dijk = vault.respond("What is the time complexity of Dijkstra with a binary heap?")
        assert r_dijk.epistemic_tier == "honest_abstain"
        assert r_dijk.mode == "unmatched_honest_abstain"
        assert r_dijk.confidence == 0.0

        # Missing code synthesis ('nth prime') must abstain instead of returning FFT
        r_prime = vault.respond("Write a Python function that returns the nth prime.")
        assert r_prime.epistemic_tier == "honest_abstain"
        assert r_prime.mode == "unmatched_honest_abstain"

        # Missing explicit future year ('2026') must abstain instead of returning generic 4-year fact
        r_wc = vault.respond("What happened in the 2026 World Cup final?")
        assert r_wc.epistemic_tier == "honest_abstain"
        assert r_wc.mode == "unmatched_honest_abstain"
    finally:
        vault.close()


def test_in_memory_engine_also_abstains_on_partial_cousin_hallucination():
    eng = MirrorCrystalEngine()
    eng.ingest_pair(
        "What is the time complexity of Binary Indexed Tree?",
        "Binary Indexed Tree updates take O(log N) time complexity.",
        domain="algorithms",
    )
    res = eng.respond("What is the time complexity of Dijkstra with a binary heap?")
    assert res.epistemic_tier == "honest_abstain"
    assert res.confidence == 0.0


def test_disk_vault_concurrent_reads_and_writes(tmp_path: Path):
    vault = DiskCrystalVault(tmp_path / "concurrent.dcrystal", lru_rule_cache_size=32)
    try:
        for i in range(8):
            vault.ingest_pair(f"What is Protocol alpha_{i}?", f"Protocol alpha_{i} enforces factor {i + 2}.")

        errors: list[Exception] = []

        def worker(tid: int) -> None:
            try:
                for step in range(8):
                    if (tid + step) % 3 == 0:
                        vault.ingest_pair(
                            f"Define Zeta_{tid}_{step}",
                            f"Zeta_{tid}_{step} stabilizes subsystem {tid + step}.",
                            domain="concurrency",
                        )
                    else:
                        idx = (tid + step) % 8
                        r = vault.respond(f"What is Protocol alpha_{idx}?", session_id=f"sess_{tid}")
                        assert isinstance(r.answer, str)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(worker, range(16)))

        assert not errors
        assert vault.rule_count() >= 8
    finally:
        vault.close()


def test_tdd_engine_non_vacuous_and_self_repair():
    tdd = AutonomousTDDEngine(max_repairs=3)

    # 1. Unknown function with 0 assertions MUST NOT vacuously pass
    rep_unknown = tdd.run_tdd_cycle("def frobnicate(x):\n    return x + 1\n", spec_desc="mysterious transform")
    assert rep_unknown.verified is False
    assert rep_unknown.tests_generated == 0
    assert rep_unknown.tests_passed == 0

    # 2. Buggy two_sum (reuses same index i == j) repaired autonomously
    buggy_two_sum = (
        "def two_sum(nums, target):\n"
        "    for i in range(len(nums)):\n"
        "        for j in range(len(nums)):\n"
        "            if nums[i] + nums[j] == target:\n"
        "                return [i, j]\n"
    )
    rep_ts = tdd.run_tdd_cycle(buggy_two_sum, spec_desc="indices of two distinct numbers summing to target")
    assert rep_ts.verified is True
    assert rep_ts.tests_passed >= 4
    assert rep_ts.iterations_used >= 2

    # 3. Buggy merge_intervals stub discovers working sweep-line implementation
    rep_mi = tdd.run_tdd_cycle(
        "def merge_intervals(iv):\n    return iv\n",
        spec_desc="merge overlapping intervals",
    )
    assert rep_mi.verified is True
    assert rep_mi.tests_passed >= 4

    # 4. Buggy lcm (a * b) fails generated lcm tests on iter 1, repairs to gcd-based lcm
    rep_lcm = tdd.run_tdd_cycle("def lcm(a, b):\n    return a * b\n", spec_desc="least common multiple")
    assert rep_lcm.verified is True
    assert rep_lcm.tests_generated >= 4
    assert "gcd" in rep_lcm.final_code

    # 5. Unsafe code bypass attempt must stay rejected
    rep_unsafe = tdd.run_tdd_cycle(
        "def exploit():\n    return __import__('os').system('echo bad')\n",
        spec_desc="least common multiple",
        explicit_tests="assert exploit() is not None",
    )
    assert rep_unsafe.verified is False


def test_cerebrum_compositional_math_and_prm():
    bridge = CerebrumCrystalBridge()

    cases = [
        ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "72"),
        ("A bakery baked 80 loaves. It sold 25 in the morning and 15 in the afternoon. How many loaves remain?", "40"),
        ("What is 15% of 240?", "36"),
        ("Find the sum of all positive divisors of 36.", "91"),
        ("How many ways can 5 people be arranged in a line?", "120"),
        ("If a train travels 300 km in 2.5 hours, what is its average speed in km/h?", "120"),
        ("What is the remainder when 2^100 is divided by 7?", "2"),
        ("Solve x^2 - 5*x + 6 = 0", "2"),
        ("Integrate 3*x^2 + 2*x from 0 to 4", "80"),
    ]
    for prompt, expected_sub in cases:
        res = bridge.solve(prompt)
        assert res["ok"] is True, prompt
        assert expected_sub in str(res["answer"]), (prompt, res)
        assert res["prm_reward"] >= 0.85, (prompt, res)

    # Proof vs counterexample refutation
    r_irr = bridge.solve("Prove that the square root of 2 is irrational.")
    assert r_irr["ok"] is True and "Proof by contradiction" in r_irr["answer"]

    r_ref = bridge.solve("Prove that the square root of 9 is irrational.")
    assert r_ref["ok"] is True and "Counterexample: sqrt(9) = 3" in r_ref["answer"]


def test_six_gate_bounded_cache_and_near_duplicate_lsh():
    pipeline = SixGateDataHygienePipeline(max_seen_cache=3)
    base_doc = (
        "# Theorem on Merge Sort Complexity\n\n"
        "Proof: Divide-and-conquer splits an N-element array into two halves and merges in linear time,\n"
        "yielding recurrence $T(N) = 2T(N/2) + O(N)$, which resolves cleanly to O(N log N).\n\n"
        "```python\n"
        "def merge_sort(arr):\n"
        "    if len(arr) <= 1:\n"
        "        return arr\n"
        "    mid = len(arr) // 2\n"
        "    return sorted(arr[:mid] + arr[mid:])\n"
        "```\n"
    )
    r1 = pipeline.audit_record(base_doc, spdx_license="MIT")
    assert r1.passed is True

    # Near-duplicate (one tiny word tweaked) must be caught by 4-band LSH Gate 2
    near_dup = base_doc.replace("two halves", "two sublists")
    r_dup = pipeline.audit_record(near_dup, spdx_license="MIT")
    assert r_dup.passed is False
    assert r_dup.rejection_gate == "Gate 2 (Deduplication)"

    # Adding 3 completely distinct documents evicts the oldest key from FIFO cache of size 3
    for k, topic in enumerate(("Dijkstra", "Quicksort", "BoyerMoore"), 1):
        distinct = (
            f"# Analysis of {topic} Algorithm Part {k}\n\n"
            f"Step {k}: Demonstrates invariant {topic}_k{k} with complexity $O(N \\log N)$ and theorem proof.\n\n"
            f"```python\ndef fn_{topic.lower()}_{k}(seq_{k}):\n    return list(seq_{k})\n```\n"
        )
        res = pipeline.audit_record(distinct, spdx_license="MIT")
        assert res.passed is True
    assert len(pipeline._seen_minhashes) == 3

    # Clear backward compatibility (used by benchmark_qwen_opus)
    pipeline._seen_minhashes.clear()
    assert pipeline.audit_record(base_doc, spdx_license="MIT").passed is True
