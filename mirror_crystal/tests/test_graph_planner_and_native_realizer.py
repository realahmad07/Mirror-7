"""Tests for Dynamic Unit-Graph Multi-Step Planner (#2) and Zero-3rd-Party Native Hybrid Realizer (#5)."""
from __future__ import annotations

import math

from mirror_crystal.native_realizer import NativeHybridRealizer
from mirror_crystal.nl_uol_router import NaturalLanguageUOLRouter
from mirror_crystal.uol_graph_planner import UOLGraphPlanner
from mirror7_backend.cognitive_orchestrator import UnifiedCognitiveOrchestrator


def test_graph_planner_3hop_physics_to_engineering_safety_factor():
    planner = UOLGraphPlanner()
    assert planner.edge_count >= 50

    res = planner.plan_and_execute(
        {"mass": 10.0, "acceleration": 5.0, "area": 2.0, "yield_strength": 100.0},
        "safety_factor",
    )
    assert res.solved
    assert math.isclose(res.output, 4.0, rel_tol=1e-6)
    assert res.rule_ids == ("physics.newton_force", "eng.axial_stress", "eng.safety_factor")
    assert len(res.steps) == 3


def test_graph_planner_2hop_electricity_power_from_current_and_resistance():
    planner = UOLGraphPlanner()
    res = planner.plan_and_execute({"current": 4.0, "resistance": 5.0}, "electrical_power")
    assert res.solved
    assert math.isclose(res.output, 80.0, rel_tol=1e-6)
    assert res.rule_ids == ("physics.ohms_voltage", "physics.electrical_power")


def test_graph_planner_3hop_mechanics_to_efficiency():
    planner = UOLGraphPlanner()
    res = planner.plan_and_execute(
        {"mass": 10.0, "acceleration": 4.0, "distance": 5.0, "input_energy": 400.0},
        "efficiency",
    )
    assert res.solved
    assert math.isclose(res.output, 0.5, rel_tol=1e-6)
    assert res.rule_ids == ("physics.newton_force", "physics.work_done", "eng.efficiency")


def test_graph_planner_inverse_then_forward_kinetic_energy_to_momentum():
    planner = UOLGraphPlanner()
    res = planner.plan_and_execute({"kinetic_energy": 180.0, "mass": 10.0}, "momentum")
    assert res.solved
    assert math.isclose(res.output, 60.0, rel_tol=1e-6)
    assert res.rule_ids == ("physics.kinetic_energy__solve_velocity", "physics.momentum")


def test_graph_planner_geometry_circle_area_to_cylinder_volume():
    planner = UOLGraphPlanner()
    res = planner.plan_and_execute({"circle_area": math.pi * 25.0, "height": 10.0}, "cylinder_volume")
    assert res.solved
    assert math.isclose(res.output, math.pi * 250.0, rel_tol=1e-5)
    assert res.rule_ids == ("math.circle_area__solve_r", "math.cylinder_volume")


def test_nl_router_and_orchestrator_solve_unseen_multistep_queries_zero_shot():
    router = NaturalLanguageUOLRouter()

    # 3-hop unseen query
    r1 = router.route("Find safety factor for mass 10, acceleration 5, area 2, yield strength 100")
    assert r1.matched
    assert math.isclose(float(r1.output), 4.0, rel_tol=1e-6)
    assert len(r1.steps) == 3

    # Inverse + forward unseen query
    r2 = router.route("Find momentum if kinetic energy is 180 and mass is 10")
    assert r2.matched
    assert math.isclose(float(r2.output), 60.0, rel_tol=1e-6)
    assert r2.rule_ids == ("physics.kinetic_energy__solve_velocity", "physics.momentum")

    # 2-hop electricity query
    r3 = router.route("Calculate electrical power for current 4 and resistance 5")
    assert r3.matched
    assert math.isclose(float(r3.output), 80.0, rel_tol=1e-6)

    # End-to-end through UnifiedCognitiveOrchestrator
    class _Mem:
        def resolve_memory_context(self, session_id, prompt):
            return None

    class _Reas:
        def solve(self, prompt):
            return {"ok": False}

    class _Code:
        def run_tdd_cycle(self, *args, **kwargs):
            raise AssertionError("not code")

    orch = UnifiedCognitiveOrchestrator(
        vault_provider=lambda: None,
        memory_provider=_Mem,
        reasoning_provider=_Reas,
        code_provider=_Code,
    )
    ans = orch.ask("Find safety factor for mass 10, acceleration 5, area 2, yield strength 100")
    assert ans["task_result_verified"] is True
    assert ans["abstained"] is False
    assert ans["uol_route"]["uses_third_party_model"] is False
    assert len(ans["uol_route"]["steps"]) == 3
    assert "4 ratio" in ans["answer"]


def test_native_hybrid_realizer_styles_truth_lock_and_zero_third_party():
    planner = UOLGraphPlanner()
    realizer = NativeHybridRealizer()
    plan = planner.plan_and_execute({"current": 4.0, "resistance": 5.0}, "electrical_power")

    for style in ("concise", "step_by_step", "technical"):
        out = realizer.realize_graph_plan(plan, style=style)
        assert out.verified is True
        assert out.truth_locked is True
        assert out.uses_third_party_model is False
        assert "80" in out.text

    # Unsolvable query must abstain honestly
    unsolved = planner.plan_and_execute({"mass": 5.0}, "kelvin")
    abstain_out = realizer.realize_graph_plan(unsolved)
    assert abstain_out.verified is False
    assert abstain_out.uses_third_party_model is False
    assert "could not derive" in abstain_out.text.lower()

    # Unverified candidate must NEVER be upgraded to verified
    unverified_resp = realizer.realize_orchestrator_response({
        "answer_qualification": "unverified_candidate",
        "candidate_answer": "Maybe 42",
    })
    assert unverified_resp.verified is False
    assert "unverified" in unverified_resp.text.lower()
