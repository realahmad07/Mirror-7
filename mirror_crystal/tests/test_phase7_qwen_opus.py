"""
Unit tests for Phase 7: Qwen-Math to Claude-Opus Frontier Cognitive Architecture.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from mirror_crystal.phase7_qwen_to_opus import (
    AutonomousTDDEngine,
    CerebrumCrystalBridge,
    HierarchicalProjectState,
    ProcessRewardVerifier,
    SixGateDataHygienePipeline,
)


def test_prm_process_reward_verifier():
    prm = ProcessRewardVerifier()

    # 1. Valid derivation step
    ev1 = prm.evaluate_step(1, "Let position s(t) = t^3 - 6*t^2 + 9*t", {})
    assert ev1.is_valid is True
    assert ev1.reward == 1.0

    # 2. Arithmetic equality step verified
    ev2 = prm.evaluate_step(2, "We have 3 * 4 + 5 = 17", {})
    assert ev2.is_valid is True
    assert "arithmetic_equality_verified" in ev2.invariants_checked

    # 3. Flawed arithmetic step flagged
    ev3 = prm.evaluate_step(3, "Then 2 + 2 = 5", {})
    assert ev3.is_valid is False
    assert ev3.reward <= 0.2
    assert "Arithmetic equality violation" in ev3.diagnostic

    # 4. Division by zero flagged
    ev4 = prm.evaluate_step(4, "Compute ratio y = (x + 1) / 0", {})
    assert ev4.is_valid is False
    assert ev4.reward == 0.0
    assert "Division by zero" in ev4.diagnostic

    # 5. Negative probability flagged
    ev5 = prm.evaluate_step(5, "The probability is P(A) = -0.25", {})
    assert ev5.is_valid is False
    assert "probability" in ev5.diagnostic.lower()

    # 6. Unbalanced parentheses flagged
    ev6 = prm.evaluate_step(6, "Expression (x + 3 * (y - 1)", {})
    assert ev6.is_valid is False
    assert ev6.reward < 1.0

    # 7. Exponential prefix before divisor is not misread as `2 = 1/2`
    ev7 = prm.evaluate_step(7, "Born rule P = (1)^2 / 2 = 1/2", {})
    assert ev7.is_valid is True
    assert ev7.reward == 1.0


def test_cerebrum_crystal_bridge_multi_step_math():
    bridge = CerebrumCrystalBridge()

    # 1. Direct CAS calculus
    res1 = bridge.solve("differentiate x^4 + 3*x^2 wrt x")
    assert res1["ok"] is True
    assert "4*x^3 + 6*x" in res1["answer"]
    assert res1["prm_reward"] > 0.8
    assert "\\boxed{" in res1["boxed_answer"]

    # 2. Kinematics multi-step word problem
    prompt = "A particle moves with position s(t) = t^3 - 6*t^2 + 9*t. Find acceleration when velocity is 0."
    res2 = bridge.solve(prompt)
    assert res2["ok"] is True
    assert res2["strategy"] == "kinematics_calculus_chain"
    assert "a(1)" in res2["answer"] or "a(3)" in res2["answer"]
    assert "-6" in res2["answer"] and "6" in res2["answer"]
    assert res2["prm_reward"] > 0.8

    # 3. Modular exponentiation
    res3 = bridge.solve("Compute 17^23 mod 1000000007")
    assert res3["ok"] is True
    assert res3["strategy"] == "modular_exponentiation"
    assert res3["answer"] == str(pow(17, 23, 1000000007))

    # 4. Rational arithmetic
    res4 = bridge.solve("compute: 3/4 + 5/6")
    assert res4["ok"] is True
    assert res4["answer"] == "19/12"

    # 5. Prime factorization: guard the Python 3.11 LaTeX formatting path.
    res5 = bridge.solve("Find the prime factorization of 84")
    assert res5["ok"] is True
    assert res5["strategy"] == "number_theory_cas"
    assert res5["answer"] == "2 * 2 * 3 * 7"
    assert r"84 = 2 \times 2 \times 3 \times 7" in res5["formatted_solution"]


def test_autonomous_tdd_engine_normal_and_self_repair():
    tdd = AutonomousTDDEngine(max_repairs=3)

    # 1. Clean code synthesis test (Fibonacci)
    clean_fib = (
        "def fib(n):\n"
        "    if n <= 0: return 0\n"
        "    if n == 1: return 1\n"
        "    a, b = 0, 1\n"
        "    for _ in range(2, n + 1):\n"
        "        a, b = b, a + b\n"
        "    return b\n"
    )
    rep1 = tdd.run_tdd_cycle(clean_fib, spec_desc="compute nth fibonacci number")
    assert rep1.verified is True
    assert rep1.iterations_used == 1
    assert rep1.tests_passed >= 5

    # 2. Buggy code requiring self-repair (missing base case in recursion)
    buggy_fib = (
        "def fib(n):\n"
        "    return fib(n - 1) + fib(n - 2)\n"
    )
    rep2 = tdd.run_tdd_cycle(buggy_fib, spec_desc="compute nth fibonacci number")
    assert rep2.verified is True
    assert rep2.iterations_used > 1
    assert "if n <= 0" in rep2.final_code
    assert rep2.tests_passed >= 5


def test_hierarchical_project_state():
    proj = HierarchicalProjectState(
        project_name="QuantumMirrorAgent",
        target_language="python",
        architecture_invariants=["Pure standard library", "Zero GPU"],
    )

    file_a = (
        "import math\n"
        "def compute_orbit(radius: float) -> float:\n"
        "    return 2 * math.pi * radius\n"
    )
    node_a = proj.register_file("src/orbit.py", file_a)
    assert "def compute_orbit" in node_a.exported_symbols
    assert "math" in node_a.dependencies
    assert len(proj.files) == 1

    proj.record_turn_diff(
        goal="Add orbital velocity calculation",
        modified_files=["src/orbit.py"],
        summary_notes="Added Kepler velocity formula",
    )
    assert proj.active_turn == 1

    manifest = proj.export_compact_manifest()
    assert manifest["project_name"] == "QuantumMirrorAgent"
    assert "src/orbit.py" in manifest["files"]
    assert manifest["total_files"] == 1


def test_six_gate_data_hygiene_pipeline():
    pipeline = SixGateDataHygienePipeline()

    # 1. Clean, permissive, high-pedagogy sample
    good_text = (
        "# Derivation of Fast Fourier Transform Complexity\n\n"
        "Theorem: The Cooley-Tukey FFT algorithm computes the Discrete Fourier Transform\n"
        "of sequence x[n] in O(N log N) time complexity.\n\n"
        "Step 1: Divide the sequence into even and odd indices:\n"
        "$$X[k] = E[k] + W_N^k O[k]$$\n\n"
        "```python\n"
        "def fft(x):\n"
        "    N = len(x)\n"
        "    if N <= 1: return x\n"
        "    even = fft(x[0::2])\n"
        "    odd = fft(x[1::2])\n"
        "    return [even[k] + odd[k] for k in range(N // 2)]\n"
        "```\n"
    )
    res_good = pipeline.audit_record(good_text, spdx_license="MIT")
    assert res_good.passed is True
    assert res_good.metrics["gate1_spdx_pii"] == "PASS"
    assert res_good.metrics["gate4_pedagogy"] == "PASS"

    # 2. Gate 1 Rejection: Non-permissive license
    res_bad_lic = pipeline.audit_record(good_text, spdx_license="GPL-3.0")
    assert res_bad_lic.passed is False
    assert "Gate 1 (SPDX)" in res_bad_lic.rejection_gate

    # 3. Gate 1 Rejection: PII leak
    pii_text = good_text + "\nContact developer at secret.admin@internal-corp.net for access keys."
    res_pii = pipeline.audit_record(pii_text, spdx_license="Apache-2.0")
    assert res_pii.passed is False
    assert "Gate 1 (PII)" in res_pii.rejection_gate

    # 4. Gate 2 Rejection: Duplicate MinHash
    res_dup1 = pipeline.audit_record(good_text, spdx_license="MIT")
    assert res_dup1.passed is False
    assert "Gate 2 (Deduplication)" in res_dup1.rejection_gate

    # 5. Gate 3 Rejection: Low-entropy spam
    spam = "spam " * 200
    res_spam = pipeline.audit_record(spam, spdx_license="MIT")
    assert res_spam.passed is False
    assert "Gate 3 (Entropy)" in res_spam.rejection_gate

    # 6. Gate 5 Rejection: Python SyntaxError
    bad_syntax = (
        "# Faulty Python Code\n"
        "```python\n"
        "def broken_function(\n"
        "    return 42\n"
        "```\n"
    )
    res_syntax = pipeline.audit_record(bad_syntax, spdx_license="MIT")
    assert res_syntax.passed is False
    assert "Gate 5 (AST Compiler)" in res_syntax.rejection_gate

    # 7. Gate 6 Rejection: Benchmark Decontamination
    contaminated = (
        "Solve this coding problem:\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        "    return True\n"
    )
    res_contam = pipeline.audit_record(contaminated, spdx_license="MIT")
    assert res_contam.passed is False
    assert "Gate 6 (Decontamination)" in res_contam.rejection_gate


def test_quantum_math_and_expanded_tdd_self_repair():
    bridge = CerebrumCrystalBridge()
    q_res1 = bridge.solve("How many Grover iterations are needed for N = 10000 items?")
    assert q_res1["ok"] is True
    assert q_res1["strategy"] == "quantum_computing_math"
    assert q_res1["answer"] == "78"

    q_res2 = bridge.solve("What is the Born probability of qubit amplitude 1/sqrt(2)?")
    assert q_res2["ok"] is True
    assert q_res2["answer"] == "1/2"
    assert q_res2["prm_reward"] == 1.0

    q_res3 = bridge.solve("What is the Born probability of qubit amplitude 3/5?")
    assert q_res3["ok"] is True
    assert q_res3["answer"] == "9/25"
    assert q_res3["prm_reward"] == 1.0

    q_res4 = bridge.solve("How many computational basis states does an 8-qubit register have?")
    assert q_res4["ok"] is True
    assert q_res4["answer"] == "256"
    assert q_res4["prm_reward"] == 1.0

    tdd = AutonomousTDDEngine(max_repairs=3)
    # Repair buggy sieve -> verified O(N log log N) sieve via ReasoningDiscoveryEngine
    rep_sieve = tdd.run_tdd_cycle("def sieve_of_eratosthenes(n):\n    return []\n", spec_desc="sieve primes")
    assert rep_sieve.verified is True
    assert "math.isqrt" in rep_sieve.final_code

    rep_parens = tdd.run_tdd_cycle("def is_valid_parentheses(s):\n    return True\n", spec_desc="balanced brackets")
    assert rep_parens.verified is True

    rep_dij = tdd.run_tdd_cycle(
        "def dijkstra_shortest_path(n, edges, src):\n    return [0] * n\n",
        spec_desc="dijkstra shortest path",
    )
    assert rep_dij.verified is True

    rep_mod = tdd.run_tdd_cycle(
        "def fast_mod_exp(a, b, m):\n    return (a * b) % m\n",
        spec_desc="modular exponentiation",
    )
    assert rep_mod.verified is True
