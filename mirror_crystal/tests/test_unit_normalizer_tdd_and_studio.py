"""Tests for:
1. Automatic SI Unit Normalization Layer (mixed metric/imperial -> SI in NL UOL Router)
2. Expanded Native TDD Python Algorithm Code Generator (11 new sandbox-verified algorithms)
3. Live Mirror Studio UI & Backend Auto-Load Integration
"""
from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MIRROR_SANDBOX_MODE", "process")

from mirror_crystal.nl_uol_router import NaturalLanguageUOLRouter
from mirror_crystal.phase7_qwen_to_opus import AutonomousTDDEngine
from mirror_crystal.unit_normalizer import normalize_to_si
from mirror7_backend.native_code_generation import NativeCodeGenerationProvider
from mirror7_backend.service import BackendService

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_step1_unit_normalizer_and_mixed_unit_word_problems():
    v_kg, conv = normalize_to_si(2000.0, "g", "mass")
    assert conv and math.isclose(v_kg, 2.0)
    v_ms, conv2 = normalize_to_si(72.0, "km/h", "velocity")
    assert conv2 and math.isclose(v_ms, 20.0)

    router = NaturalLanguageUOLRouter()

    # 2000 g = 2 kg, 72 km/h = 20 m/s -> KE = 0.5 * 2 * 400 = 400 J
    r1 = router.route("Calculate kinetic energy for mass 2000 g and velocity 72 km/h")
    assert r1.matched and math.isclose(float(r1.output), 400.0, rel_tol=1e-5)

    # 5 kN = 5000 N, 200 cm2 = 0.02 m^2 -> stress = 250,000 Pa
    r2 = router.route("Calculate axial stress for force 5 kN and area 200 cm2")
    assert r2.matched and math.isclose(float(r2.output), 250000.0, rel_tol=1e-5)

    # 70 kg, 175 cm = 1.75 m -> BMI = 22.8571
    r3 = router.route("Calculate BMI for mass 70 kg and height 175 cm")
    assert r3.matched and math.isclose(float(r3.output), 70.0 / (1.75 ** 2), rel_tol=1e-5)

    # Multi-hop with mixed units: 500 mA = 0.5 A, 2 kohm = 2000 Ohm -> V=1000 V, P=500 W
    r4 = router.route("Calculate electrical power for current 500 mA and resistance 2 kohm")
    assert r4.matched and math.isclose(float(r4.output), 500.0, rel_tol=1e-5)

    # 3-hop with mixed units: mass=5000 g (5 kg), accel=4 -> F=20 N; area=100 cm2 (0.01 m2) -> stress=2000 Pa; yield=10 kPa (10000 Pa) -> SF=5
    r5 = router.route("Find safety factor for mass 5000 g, acceleration 4, area 100 cm2, yield strength 10 kPa")
    assert r5.matched and math.isclose(float(r5.output), 5.0, rel_tol=1e-5)


def test_step2_expanded_native_tdd_code_generator_all_11_algorithms():
    builder = NativeCodeGenerationProvider()
    tdd = AutonomousTDDEngine(max_repairs=1)

    prompts = [
        ("Write a Python fibonacci function", "fibonacci"),
        ("Write Python code for gcd of two numbers", "gcd"),
        ("Write Python code for lcm of two numbers", "lcm"),
        ("Implement binary search in Python", "binary_search"),
        ("Write a Python two sum function", "two_sum"),
        ("Write Python code to check palindrome", "is_palindrome"),
        ("Write Python code to check anagram", "is_anagram"),
        ("Write Python code for prime factors", "prime_factors"),
        ("Write a Python fizzbuzz function", "fizzbuzz"),
        ("Write Python code to transpose a matrix", "matrix_transpose"),
        ("Write Python code to compute median", "median"),
    ]
    for prompt, expected_kind in prompts:
        gen = builder(prompt)
        assert gen is not None, f"failed to generate for: {prompt}"
        assert gen["generation_kind"] == expected_kind
        rep = tdd.run_tdd_cycle(gen["code"], spec_desc=gen["spec_desc"], explicit_tests=gen["tests"])
        assert rep.verified is True, f"TDD failed for {expected_kind}: {rep}"
        assert rep.tests_generated > 0 and rep.tests_passed == rep.tests_generated

    # Unsupported requests remain cleanly declined
    assert builder("Write a sorting function") is None


def test_step3_studio_ui_and_backend_auto_load_uol_vault():
    studio_html = (REPO_ROOT / "mirror7_backend/studio.html").read_text(encoding="utf-8")
    assert "VERIFIED UOL ENGINE" in studio_html
    assert "data.uol_route" in studio_html

    service = BackendService(enable_crystal_auto_load=True)
    try:
        assert service.crystal_vault is not None
        telemetry = service.crystal_coverage_map()
        assert telemetry["uses_third_party_llm"] is False
        organs = " ".join(telemetry["capability_organs"])
        assert "UOLGraphPlanner" in organs
        assert "UnitNormalizer" in organs
    finally:
        if service.crystal_vault is not None and hasattr(service.crystal_vault, "close"):
            service.crystal_vault.close()
