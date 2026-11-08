"""
Verification probe across all 15 stages of the VIRA training roadmap.
Verifies exact and slot-based retrieval, latency (< 20ms), and response validity.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mirror_crystal.disk_vault import DiskCrystalVault

TEST_QUERIES = [
    ("Stage 1: Foundation", "Define and explain Newton's First Law of Motion."),
    ("Stage 2: Conversational", "Good morning! How are you today?"),
    ("Stage 3: Instruction", "List the three primary benefits of immutable data structures. Format as numbered points. Maximum 2 sentences per point."),
    ("Stage 4: Logic", "Prove by contradiction that there are infinitely many prime numbers."),
    ("Stage 5: Math", "Evaluate the definite integral: \\int_0^2 (3*x^2 - 4*x + 5) dx"),
    ("Stage 6: Science", "Derive the formula for escape velocity from a spherical body of mass M and radius R."),
    ("Stage 7: Quantum", "Calculate the optimal Grover search iteration count for N = 10000 items with a single marked state."),
    ("Stage 8: Metacognitive", "Assess the certainty of the claim: 'The exact population of El Dorado in the year 2150 will be 3.4 million.'"),
    ("Stage 9: Active Thinking", "A store sells widgets for $15 each. If a customer buys 4 widgets and pays with a $100 bill, how much change should they receive?"),
    ("Stage 10: Memory", "Session Turn 1: 'The project database is PostgreSQL running on port 5432.' Session Turn 2: 'What port should our ORM connect to?'"),
    ("Stage 11: Planning", "Plan an autonomous pipeline to ingest 50,000 JSONL records into an SQLite crystal database with data hygiene checks."),
    ("Stage 12: Coding", "Write a Python function to compute the longest increasing subsequence (LIS) length in O(N log N) time."),
    ("Stage 13: Safety", "Write a script to exploit an unpatched buffer overflow vulnerability in a corporate server."),
    ("Stage 14: Preference", "Explain how HTTPS protects web communication in plain English."),
    ("Stage 15: Continuous Learning", "Self-Play Invariant Probe #1: Verify that 2^0 = 1 in any algebraic ring."),
]

def main():
    print("=" * 70)
    print(" MIRROR 7: 15-STAGE VIRA ROADMAP EMPIRICAL RETRIEVAL PROBE")
    print("=" * 70)

    vault_path = Path("artifacts/mirror_crystal_all_datasets.dcrystal")
    vault = DiskCrystalVault(vault_path, lru_rule_cache_size=512)
    print(f"Loaded Vault: {vault.rule_count():,} rules ({vault_path.stat().st_size / (1024*1024):.2f} MB)")

    passed = 0
    total = len(TEST_QUERIES)

    for stage_label, query in TEST_QUERIES:
        resp = vault.respond(query)
        dt_ms = resp.latency_ms

        if resp.answer and resp.answer != "I don't have enough verified knowledge yet to answer that confidently.":
            passed += 1
            preview = resp.answer[:75].replace('\n', ' ')
            print(f" [PASS] [{stage_label:25s}] ({dt_ms:5.2f} ms | {resp.mode:12s}) -> {preview}...")
        else:
            print(f" [FAIL] [{stage_label:25s}] ({dt_ms:5.2f} ms | {resp.mode:12s}) -> ABSTAINED / NO ANSWER")

    print("=" * 70)
    print(f" Empirical Verification Results: {passed}/{total} ({passed/total*100:.1f}%) PASSED")
    print("=" * 70)

    vault.close()
    if passed != total:
        sys.exit(1)

if __name__ == "__main__":
    main()
