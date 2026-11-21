"""Tests for Phase 2 (Executable UOL Rule Library) and Phase 3 (Compositional CEGIS & Generalization)."""
from __future__ import annotations

import math

from mirror_crystal.capability_synthesizer import Example, SynthesisGoal
from mirror_crystal.operational_language import RuleStatus, VariableSpec
from mirror_crystal.uol_compositional_engine import UOLCompositionalEngine
from mirror_crystal.uol_rule_library import (
    build_default_uol,
    default_uol_knowledge_records,
    execute_library_rule,
    import_default_uol_rules,
)
from mirror7_backend.universal_intelligence import UniversalIntelligencePipeline


def test_phase2_all_45_structured_uol_rules_import_cleanly_and_verify():
    records = default_uol_knowledge_records()
    assert len(records) >= 45
    report = import_default_uol_rules()
    assert report.imported_count == len(records)
    assert report.skipped_count == 0
    assert report.rejected_count == 0
    assert report.conflicted_rule_ids == ()
    assert all(rule.status is RuleStatus.VERIFIED for rule in report.rules)


def test_phase2_executable_rules_generalize_across_five_domains():
    uol = build_default_uol()
    assert math.isclose(execute_library_rule("physics.kinetic_energy", {"mass": 10, "velocity": 6}, uol=uol), 180.0)
    assert math.isclose(execute_library_rule("eng.axial_stress", {"force": 2400, "area": 12}, uol=uol), 200.0)
    assert math.isclose(execute_library_rule("bio.michaelis_menten", {"vmax": 100, "substrate": 10, "km": 10}, uol=uol), 50.0)
    assert math.isclose(execute_library_rule("econ.compound_amount", {"principal": 1000, "rate": 0.1, "periods": 2}, uol=uol), 1210.0)
    assert math.isclose(execute_library_rule("math.quadratic_discriminant", {"a": 1, "b": -5, "c": 6}, uol=uol), 1.0)


def test_phase3_composes_two_physics_rules_fahrenheit_to_kelvin_and_generalizes():
    """Mirror was NEVER given a Fahrenheit->Kelvin rule. It composes F->C and C->K."""
    engine = UOLCompositionalEngine(domains=("physics",))
    f_to_k = lambda f: (f - 32.0) / 1.8 + 273.15
    goal = SynthesisGoal(
        name="fahrenheit_to_kelvin",
        domain="physics",
        inputs=(VariableSpec("f", "number"),),
        output_type="number",
        training=(
            Example({"f": 32.0}, f_to_k(32.0)),
            Example({"f": 212.0}, f_to_k(212.0)),
        ),
        validation=(
            Example({"f": 68.0}, f_to_k(68.0)),
            Example({"f": -40.0}, f_to_k(-40.0)),
        ),
        constants=(),
        max_depth=2,
        candidate_budget=5000,
    )
    unseen = [{"f": float(t)} for t in range(-100, 150, 5) if float(t) not in {32.0, 212.0, 68.0, -40.0}]
    result = engine.solve_and_verify(
        goal,
        unseen_probes=unseen,
        unseen_oracle=lambda inp: f_to_k(inp["f"]),
    )
    assert result.succeeded
    assert "physics.fahrenheit_to_celsius" in result.rule_ids_used
    assert "physics.celsius_to_kelvin" in result.rule_ids_used
    assert result.generalization is not None
    assert result.generalization.all_passed
    assert result.generalization.exact_accuracy == 1.0


def test_phase3_composes_engineering_stress_and_safety_factor_rules():
    """Compose axial_stress(force, area) inside safety_factor(yield_strength, stress)."""
    engine = UOLCompositionalEngine(domains=("engineering",))
    sf_oracle = lambda inp: inp["yield_strength"] / (inp["force"] / inp["area"])
    goal = SynthesisGoal(
        name="beam_safety_from_load",
        domain="engineering",
        inputs=(
            VariableSpec("yield_strength", "number"),
            VariableSpec("force", "number"),
            VariableSpec("area", "number"),
        ),
        output_type="number",
        training=(
            Example({"yield_strength": 400.0, "force": 1000.0, "area": 5.0}, 2.0),
            Example({"yield_strength": 600.0, "force": 1500.0, "area": 10.0}, 4.0),
        ),
        validation=(
            Example({"yield_strength": 500.0, "force": 2000.0, "area": 8.0}, 2.0),
        ),
        constants=(),
        max_depth=2,
        candidate_budget=8000,
    )
    unseen = [
        {"yield_strength": 250.0 + 50 * i, "force": 500.0 + 100 * j, "area": 2.0 + j}
        for i in range(5) for j in range(5)
    ]
    result = engine.solve_and_verify(goal, unseen_probes=unseen, unseen_oracle=sf_oracle)
    assert result.succeeded
    assert any(r in result.rule_ids_used for r in ("eng.safety_factor", "eng.axial_stress"))
    assert result.generalization is not None
    assert result.generalization.exact_accuracy == 1.0


def test_phase3_higher_order_map_rule_and_reduce_mean_squared():
    """Synthesize mean(map(xs, math.square)) — combining higher-order map + UOL rule + reduction."""
    engine = UOLCompositionalEngine(domains=("mathematics",))
    ms_oracle = lambda inp: sum(x * x for x in inp["xs"]) / len(inp["xs"])
    goal = SynthesisGoal(
        name="mean_squared_value",
        domain="mathematics",
        inputs=(VariableSpec("xs", "array"),),
        output_type="number",
        training=(
            Example({"xs": [1, 2, 3]}, 14 / 3),
            Example({"xs": [2, 4]}, 10.0),
        ),
        validation=(
            Example({"xs": [3, 4, 5]}, 50 / 3),
        ),
        constants=(),
        max_depth=2,
        candidate_budget=8000,
    )
    unseen = [
        {"xs": [1, 1, 1, 1]},
        {"xs": [0, 5, 10]},
        {"xs": [-2, 3, -4, 5]},
        {"xs": [10, 20, 30, 40]},
    ]
    result = engine.solve_and_verify(goal, unseen_probes=unseen, unseen_oracle=ms_oracle)
    assert result.succeeded
    assert "math.square" in result.rule_ids_used
    assert "math.mean" in result.rule_ids_used
    assert result.generalization is not None
    assert result.generalization.all_passed


def test_phase3_hierarchical_promotion_allows_stacking_learned_rules():
    """Synthesize a rule, promote it, then synthesize a higher-level rule on top of it."""
    engine = UOLCompositionalEngine(domains=("physics",))
    f_to_k = lambda f: (f - 32.0) / 1.8 + 273.15
    goal_1 = SynthesisGoal(
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
    res_1 = engine.solve_and_verify(goal_1)
    assert res_1.succeeded
    before_count = engine.rule_count
    engine.promote_program_as_rule(res_1.program, rule_id="physics.fahrenheit_to_kelvin")
    assert engine.rule_count == before_count + 1

    # Now compose a delta-kelvin rule using the newly promoted rule!
    goal_2 = SynthesisGoal(
        name="batch_fahrenheit_to_kelvin",
        domain="physics",
        inputs=(VariableSpec("temps", "array"),),
        output_type="array",
        training=(
            Example({"temps": [32.0, 212.0]}, [f_to_k(32.0), f_to_k(212.0)]),
            Example({"temps": [68.0]}, [f_to_k(68.0)]),
        ),
        validation=(
            Example({"temps": [-40.0, 104.0]}, [f_to_k(-40.0), f_to_k(104.0)]),
        ),
        constants=(),
        max_depth=1,
        candidate_budget=3000,
    )
    res_2 = engine.solve_and_verify(goal_2)
    assert res_2.succeeded
    assert "physics.fahrenheit_to_kelvin" in res_2.rule_ids_used


def test_phase3_backend_pipeline_uses_builtin_uol_rules_when_enabled():
    class EmptyVault:
        def list_knowledge_records(self, *, domain=None, limit=500):
            return ()

        def list_capabilities(self, *, domain=None, limit=500):
            return ()

    pipeline = UniversalIntelligencePipeline(
        vault_provider=EmptyVault,
        include_builtin_uol_rules=True,
    )
    f_to_k = lambda f: (f - 32.0) / 1.8 + 273.15
    result = pipeline.synthesize({
        "name": "fahrenheit_to_kelvin",
        "domain": "physics",
        "inputs": [{"name": "f", "value_type": "number"}],
        "output_type": "number",
        "training": [
            {"inputs": {"f": 32.0}, "expected": f_to_k(32.0)},
            {"inputs": {"f": 212.0}, "expected": f_to_k(212.0)},
        ],
        "validation": [
            {"inputs": {"f": 68.0}, "expected": f_to_k(68.0)},
        ],
        "constants": [],
        "max_depth": 2,
        "candidate_budget": 5000,
    })
    assert result["status"] == "candidate_unverified"
    assert result["rules_imported"] >= 10
    assert result["validation_passed"] == 1
