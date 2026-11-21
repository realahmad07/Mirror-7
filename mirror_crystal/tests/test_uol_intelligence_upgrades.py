"""Comprehensive verification for all 5 Mirror CPU UOL intelligence upgrades:
#1 NL -> UOL Auto-Router (forward, inverse, composed + Backend Orchestrator)
#2 Persistent UOL Skill Save in .dcrystal vault across sessions
#3 Bidirectional / Inverse UOL Rule Derivation (round-trip verified)
#4 Richer UOL Primitives (sqrt, log, exp, sin, cos, floor, ceil, round)
#5 Unit & Dimension-Guided Compositional Search
"""
from __future__ import annotations

import math
from pathlib import Path
import pytest

from mirror_crystal.capability_synthesizer import Example, SynthesisGoal
from mirror_crystal.disk_vault import DiskCrystalVault
from mirror_crystal.nl_uol_router import NaturalLanguageUOLRouter
from mirror_crystal.operational_language import (
    Call,
    Lit,
    Program,
    UOL,
    UOLExecutionError,
    Var,
    VariableSpec,
)
from mirror_crystal.uol_compositional_engine import UOLCompositionalEngine
from mirror_crystal.uol_inverse_deriver import derive_all_inverse_rules
from mirror_crystal.uol_rule_library import build_default_uol
from mirror_crystal.uol_skill_store import (
    load_uol_from_vault,
    persist_verified_program,
    seed_vault_with_uol_skills,
)
from mirror7_backend.cognitive_orchestrator import UnifiedCognitiveOrchestrator


def test_step4_richer_scientific_uol_primitives_and_domain_guards():
    uol = UOL()
    p_sqrt = Program("p:sqrt", "sqrt", (VariableSpec("x", "number"),), Call("sqrt", (Var("x"),)), "mathematics")
    assert math.isclose(uol.execute(p_sqrt, {"x": 25.0}), 5.0)
    with pytest.raises(UOLExecutionError):
        uol.execute(p_sqrt, {"x": -4.0})

    p_log = Program("p:log", "log", (VariableSpec("x", "number"),), Call("log", (Var("x"),)), "mathematics")
    assert math.isclose(uol.execute(p_log, {"x": math.e}), 1.0)
    with pytest.raises(UOLExecutionError):
        uol.execute(p_log, {"x": 0.0})

    p_trig = Program(
        "p:trig", "trig", (VariableSpec("x", "number"),),
        Call("add", (Call("power", (Call("sin", (Var("x"),)), Lit(2))), Call("power", (Call("cos", (Var("x"),)), Lit(2))))),
        "mathematics",
    )
    assert math.isclose(uol.execute(p_trig, {"x": 1.234}), 1.0)

    p_floor = Program("p:fl", "fl", (VariableSpec("x", "number"),), Call("floor", (Var("x"),)), "mathematics")
    assert uol.execute(p_floor, {"x": 3.9}) == 3


def test_step3_bidirectional_inverse_rule_derivation():
    base = build_default_uol()
    inverses = derive_all_inverse_rules(base.rules.values())
    assert len(inverses) >= 70
    full_uol = UOL(tuple(base.rules.values()) + inverses)

    # Solve velocity from kinetic energy: KE = 0.5 * m * v^2 -> v = sqrt(KE / (0.5 * m))
    rule = full_uol.rules.get("physics.kinetic_energy__solve_velocity")
    assert rule is not None
    prog = Program("p:inv_v", "v", rule.inputs, Call(rule.rule_id, tuple(Var(p.name) for p in rule.inputs)), "physics", rule_refs=(rule.rule_id,))
    assert math.isclose(full_uol.execute(prog, {"mass": 10.0, "result": 180.0}), 6.0)

    # Solve radius from circle area: A = pi * r^2 -> r = sqrt(A / pi)
    rule_r = full_uol.rules.get("math.circle_area__solve_r")
    assert rule_r is not None
    prog_r = Program("p:inv_r", "r", rule_r.inputs, Call(rule_r.rule_id, tuple(Var(p.name) for p in rule_r.inputs)), "mathematics", rule_refs=(rule_r.rule_id,))
    assert math.isclose(full_uol.execute(prog_r, {"result": math.pi * 49.0}), 7.0, rel_tol=1e-6)


def test_step2_persistent_skill_save_and_cross_session_reload(tmp_path: Path):
    vault_path = tmp_path / "skills.dcrystal"
    summary = seed_vault_with_uol_skills(vault_path, include_inverses=True)
    assert summary["forward_rules"] == 45
    assert summary["derived_inverse_rules"] >= 70
    assert summary["total_persisted_uol_rules"] == summary["forward_rules"] + summary["derived_inverse_rules"]
    assert summary["rejected"] == 0
    assert summary["conflicts"] == []

    # Synthesize a new composite skill (F -> K) and persist it durably to disk
    engine = UOLCompositionalEngine(domains=("physics",))
    f_to_k = lambda f: (f - 32.0) / 1.8 + 273.15
    goal = SynthesisGoal(
        name="fahrenheit_to_kelvin",
        domain="physics",
        inputs=(VariableSpec("f", "number"),),
        output_type="number",
        training=(Example({"f": 32.0}, f_to_k(32.0)), Example({"f": 212.0}, f_to_k(212.0))),
        validation=(Example({"f": 68.0}, f_to_k(68.0)),),
        constants=(),
        max_depth=2,
        candidate_budget=5000,
    )
    res = engine.solve_and_verify(goal, output_unit="kelvin")
    assert res.succeeded

    vault = DiskCrystalVault(vault_path)
    try:
        persist_verified_program(vault, res.program, rule_id="physics.fahrenheit_to_kelvin")
    finally:
        vault.close()

    # Fresh session reload from disk!
    reloaded_uol, report = load_uol_from_vault(vault_path)
    assert report.imported_count == summary["total_persisted_uol_rules"] + 1
    learned_rule = reloaded_uol.rules.get("physics.fahrenheit_to_kelvin")
    assert learned_rule is not None
    prog = Program("p:fk", "fk", learned_rule.inputs, Call(learned_rule.rule_id, (Var("f"),)), "physics", rule_refs=(learned_rule.rule_id,))
    assert math.isclose(reloaded_uol.execute(prog, {"f": 98.6}), 310.15, rel_tol=1e-6)


def test_step1_nl_uol_router_forward_inverse_composed_and_abstain():
    router = NaturalLanguageUOLRouter()

    # Composed temperature chain F -> C -> K
    r1 = router.route("Convert 98.6 F to Kelvin")
    assert r1.matched and math.isclose(float(r1.output), 310.15, rel_tol=1e-6)

    # Inverse + forward chain K -> C -> F
    r2 = router.route("Convert 300 K to Fahrenheit")
    assert r2.matched and math.isclose(float(r2.output), 80.33, rel_tol=1e-4)

    # Inverse physics solve (kinetic energy -> velocity)
    r3 = router.route("Find velocity if kinetic energy is 180 and mass is 10")
    assert r3.matched and math.isclose(float(r3.output), 6.0, rel_tol=1e-6)
    assert r3.rule_ids == ("physics.kinetic_energy__solve_velocity",)

    # Composed engineering (stress -> safety factor)
    r4 = router.route("Calculate safety factor for yield strength 400, force 1000, area 5")
    assert r4.matched and math.isclose(float(r4.output), 2.0, rel_tol=1e-6)
    assert r4.rule_ids == ("eng.axial_stress", "eng.safety_factor")

    # Missing inputs -> honest abstention, never guesses
    r5 = router.route("Calculate kinetic energy for mass 10 kg")
    assert not r5.matched
    assert "velocity" in r5.missing_inputs


def test_step1_backend_cognitive_orchestrator_routes_nl_to_verified_uol():
    class _Mem:
        def resolve_memory_context(self, session_id, prompt):
            return None

    class _Reas:
        def solve(self, prompt):
            return {"ok": False}

    class _Code:
        def run_tdd_cycle(self, *args, **kwargs):
            raise AssertionError("should not run code verifier")

    orch = UnifiedCognitiveOrchestrator(
        vault_provider=lambda: None,
        memory_provider=_Mem,
        reasoning_provider=_Reas,
        code_provider=_Code,
    )
    ans = orch.ask("Convert 98.6 F to Kelvin")
    assert ans["task_result_verified"] is True
    assert ans["abstained"] is False
    assert ans["uol_route"]["matched"] is True
    assert math.isclose(ans["uol_route"]["output"], 310.15, rel_tol=1e-6)


def test_step5_unit_guided_search_accelerates_large_catalog():
    base = build_default_uol()
    all_rules = tuple(base.rules.values()) + derive_all_inverse_rules(base.rules.values())
    engine = UOLCompositionalEngine(use_builtin_library=False, extra_rules=all_rules)
    assert engine.rule_count >= 120

    f_to_k = lambda f: (f - 32.0) / 1.8 + 273.15
    goal = SynthesisGoal(
        name="fahrenheit_to_kelvin_120",
        domain="physics",
        inputs=(VariableSpec("f", "number"),),
        output_type="number",
        training=(Example({"f": 32.0}, f_to_k(32.0)), Example({"f": 212.0}, f_to_k(212.0))),
        validation=(Example({"f": 68.0}, f_to_k(68.0)),),
        constants=(),
        max_depth=2,
        candidate_budget=5000,
    )
    guided = engine.solve_and_verify(goal, output_unit="kelvin")
    assert guided.succeeded
    assert guided.synthesis.tested < 50
