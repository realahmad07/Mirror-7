"""Comprehensive tests for the 4 Native CPU Intelligence Leaps (Zero 3rd-Party LLM):
1. Live 1-Shot Formula Teaching -> Verified UOL Rule + Auto-Inverses + Live Graph Composition
2. Native Symbolic Algebra & N x N Linear System Solver (self-verified LHS == RHS)
3. Formal Deductive Logic (DPLL SAT, Syllogisms, Transitive Ordering, Finite-Domain Constraints)
4. Compositional Multi-Function Native TDD Code Synthesis
"""
from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MIRROR_SANDBOX_MODE", "process")

from mirror_crystal.formal_logic_engine import solve_formal_logic_prompt, solve_sat
from mirror_crystal.nl_uol_router import NaturalLanguageUOLRouter
from mirror_crystal.phase7_qwen_to_opus import AutonomousTDDEngine
from mirror_crystal.symbolic_algebra_solver import solve_algebra_prompt
from mirror_crystal.uol_formula_teacher import compile_formula_to_uol
from mirror_crystal.uol_skill_store import load_uol_from_vault
from mirror7_backend.native_code_generation import NativeCodeGenerationProvider


def test_leap1_live_formula_teaching_inverse_and_multihop_composition(tmp_path: Path):
    router = NaturalLanguageUOLRouter()

    # 1. Teach a brand-new formula in plain English
    t = router.route("Learn formula: torque = force * lever_arm")
    assert t.matched is True
    assert "taught.torque" in t.rule_ids
    assert "taught.torque__solve_lever_arm" in t.rule_ids

    # 2. Immediately solve forward
    r_fwd = router.route("Find torque for force 50 and lever_arm 3")
    assert r_fwd.matched and math.isclose(float(r_fwd.output), 150.0, rel_tol=1e-6)

    # 3. Immediately solve inverse (auto-derived!)
    r_inv = router.route("Find lever_arm if torque is 150 and force is 50")
    assert r_inv.matched and math.isclose(float(r_inv.output), 3.0, rel_tol=1e-6)

    # 4. Immediately compose 2-hop with built-in Newton's Law: mass + accel -> force -> torque!
    r_2hop = router.route("Find torque for mass 10, acceleration 5, lever_arm 3")
    assert r_2hop.matched and math.isclose(float(r_2hop.output), 150.0, rel_tol=1e-6)
    assert r_2hop.rule_ids == ("physics.newton_force", "taught.torque")

    # 5. Safety: arbitrary Python injection MUST be rejected
    bad = router.route("Learn formula: evil = __import__('os').system('echo pwned')")
    assert bad.matched is False

    # 6. Persistence to .dcrystal vault
    vp = tmp_path / "taught.dcrystal"
    res = compile_formula_to_uol("escape_velocity", "sqrt(2 * g * r)", vault_path=vp)
    assert res.ok and res.persisted
    reloaded, rep = load_uol_from_vault(vp)
    assert rep.imported_count >= 1
    assert reloaded.rules.get("taught.escape_velocity") is not None


def test_leap2_native_symbolic_algebra_and_linear_systems():
    # Linear
    r1 = solve_algebra_prompt("Solve 3x + 7 = 22")
    assert r1.solved and r1.verified and r1.solutions == {"x": 5}

    r2 = solve_algebra_prompt("Solve 2(x - 3) + 4 = 10")
    assert r2.solved and r2.verified and r2.solutions == {"x": 6}

    # Quadratic (real)
    r3 = solve_algebra_prompt("Solve x^2 - 5x + 6 = 0")
    assert r3.solved and r3.verified and r3.solutions["x"] == [2, 3]

    # Quadratic (complex)
    r4 = solve_algebra_prompt("Solve x^2 + 1 = 0")
    assert r4.solved and r4.verified and len(r4.solutions["x"]) == 2

    # 2x2 Linear system
    r5 = solve_algebra_prompt("Solve system: 2x + y = 10 and x - y = 2")
    assert r5.solved and r5.verified and r5.solutions == {"x": 4, "y": 2}

    # 3x3 Linear system
    r6 = solve_algebra_prompt("Solve system: x + y + z = 6, 2x - y + z = 3, x + 2y - z = 2")
    assert r6.solved and r6.verified and r6.solutions == {"x": 1, "y": 2, "z": 3}

    # Singular / inconsistent system -> honest abstention
    r7 = solve_algebra_prompt("Solve system: x + y = 2 and 2x + 2y = 5")
    assert r7.solved is False

    # End-to-end through NaturalLanguageUOLRouter
    router = NaturalLanguageUOLRouter()
    rr = router.route("Solve 3x + 7 = 22")
    assert rr.matched and rr.output == 5


def test_leap3_formal_deductive_logic_sat_ordering_and_constraints():
    # 1. DPLL SAT solver
    model = solve_sat([[1, 2], [-1, 3], [-2, -3], [1]])
    assert model is not None and model[1] is True and model[3] is True and model[2] is False
    assert solve_sat([[1], [-1]]) is None  # UNSAT

    # 2. Syllogism prover
    s1 = solve_formal_logic_prompt(
        "All mammals are warm-blooded. All whales are mammals. Orca is a whale. Is Orca warm-blooded?"
    )
    assert s1.matched and s1.proven is True and len(s1.proof_steps) == 3

    s2 = solve_formal_logic_prompt("All birds fly. Rex is a dog. Is Rex a bird?")
    assert s2.matched and s2.proven is None  # UNDETERMINED, never hallucinates

    # 3. Transitive ordering
    o1 = solve_formal_logic_prompt("A > B, B > C, C = 5. Is A > 5?")
    assert o1.matched and o1.proven is True and len(o1.proof_steps) == 3

    # 4. Finite-domain integer constraint solver
    c1 = solve_formal_logic_prompt("Find x, y in 1..20 where x + y = 15 and x = 2 * y")
    assert c1.matched and c1.proven is True and c1.answer == {"x": 10, "y": 5}

    # End-to-end through NaturalLanguageUOLRouter
    router = NaturalLanguageUOLRouter()
    rl = router.route("A > B, B > C, C = 5. Is A > 5?")
    assert rl.matched and rl.output is True


def test_leap4_compositional_multifunction_tdd_code_synthesis():
    builder = NativeCodeGenerationProvider()
    tdd = AutonomousTDDEngine(max_repairs=1)

    compound_prompts = [
        ("Write Python code for the sum of primes up to n", "sum_of_primes"),
        ("Write Python code to count palindromes in a list", "count_palindromes"),
        ("Write Python code for even fibonacci numbers up to limit", "even_fibonacci"),
        ("Write Python code for gcd of a list of numbers", "gcd_list"),
        ("Write Python code for lcm of a list of numbers", "lcm_list"),
        ("Write Python code for sum of factorials", "sum_of_factorials"),
    ]
    for prompt, expected_kind in compound_prompts:
        gen = builder(prompt)
        assert gen is not None, f"failed to generate for: {prompt}"
        assert gen["generation_kind"] == expected_kind
        rep = tdd.run_tdd_cycle(gen["code"], spec_desc=gen["spec_desc"], explicit_tests=gen["tests"])
        assert rep.verified is True, f"TDD failed for {expected_kind}: {rep}"
        assert rep.tests_generated > 0 and rep.tests_passed == rep.tests_generated
