"""Tests for the 0-Token Byte-Level Intent Router, Compositional AST Code Synthesizer,
Constraint-Aware Adaptive Concept Realizer, and 34-Prompt Unseen Bilingual Stress Suite.
"""
from __future__ import annotations

import time
import tracemalloc

from mirror7_backend.native_code_generation import NativeCodeGenerationProvider
from mirror7_backend.service import BackendService
from mirror_crystal.byte_intent_router import (
    ByteLevelIntentRouter,
    extract_utf8_byte_ngrams,
    get_default_byte_intent_router,
)
from mirror_crystal.open_ended_concept_lattice import OpenEndedConceptLatticeEngine
from mirror_crystal.phase7_qwen_to_opus import AutonomousTDDEngine


def test_byte_level_intent_router_fit_speed_memory_and_online_learning():
    tracemalloc.start()
    t0 = time.perf_counter()
    router = ByteLevelIntentRouter()
    fit_ms = (time.perf_counter() - t0) * 1000.0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert fit_ms < 2000.0, f"ByteLevelIntentRouter fit took {fit_ms:.2f} ms (>= 2000 ms)"
    assert (peak_bytes / (1024 * 1024)) < 2.0, f"Peak memory {peak_bytes / (1024 * 1024):.3f} MB (>= 2.0 MB)"

    feats = extract_utf8_byte_ngrams("sql injection kia hta h")
    assert feats
    assert all(isinstance(k, bytes) for k in feats)

    # Vowel-dropped Roman Urdu & casual English classification
    p1 = router.predict_route("sql injection kia hta h")
    assert p1.intent == "concept_knowledge"
    assert p1.language == "roman_urdu"
    assert "definition" in p1.facets
    assert p1.uses_tokens is False and p1.uses_gpu is False

    p2 = router.predict_route("python k code likho jo Ahmad print kry")
    assert p2.intent == "code_generation"
    assert p2.language == "roman_urdu"
    assert p2.slots.get("print_target") == "Ahmad"

    p3 = router.predict_route("what is our server port?")
    assert p3.intent == "ood_abstain"

    # Online continual CPU learning on an unseen custom phrasing
    router.learn_online("zara python m loop chala kr dikhao", "code_generation", language="roman_urdu")
    p_learned = router.predict_route("zara python m loop chala kr dikhao")
    assert p_learned.intent == "code_generation"


def test_compositional_ast_synthesizer_pipelines_and_multifunction(monkeypatch):
    monkeypatch.setenv("MIRROR_SANDBOX_MODE", "process")
    synth = NativeCodeGenerationProvider()
    tdd = AutonomousTDDEngine()

    cases = [
        ("python k code likho jo Ahmad print kry", "print_text", "message = 'Ahmad'"),
        ("write python code to print Ahmad", "print_text", "message = 'Ahmad'"),
        ("bina quotes ke Ali print krne ka python program likho", "print_text", "message = 'Ali'"),
        ("Ahmad ko 5 bar print krne ka code do", "repeat_print", "message='Ahmad', times=5"),
        ("5 ka pahara 10 tak print krne ka python code likho", "multiplication_table", "def multiplication_table(n=5, upto=10):"),
        ("ek list mein se even numbers ka sum nikalne ka python function bnao", "sum_even_numbers", "x % 2 == 0"),
        ("python mein prime number aur factorial check krne ka code likho", "composed_is_prime_and_factorial", "def is_prime(n):"),
        ("write a python function to filter odd numbers and return their squares sorted", "process_odd_square_sort_asc_numbers", "x * x"),
        ("ek python function bnao jo list mein se positive numbers ka average nikale", "average_positive_numbers", "x > 0"),
        ("python code likho jo list mein se negative numbers count kry", "count_negative_numbers", "x < 0"),
        ("write a python function to filter numbers divisible by 3 and compute their sum", "sum_div_3_numbers", "x % 3 == 0"),
        ("create a python class for stack push and pop", "stack_class", "class Stack:"),
    ]

    for prompt, expected_kind, expected_snippet in cases:
        res = synth(prompt)
        assert res is not None, f"Failed to synthesize AST for: {prompt}"
        assert res["generation_kind"] == expected_kind, f"{prompt} -> {res['generation_kind']}"
        assert expected_snippet in res["code"], f"{prompt} -> {res['code']}"
        report = tdd.run_tdd_cycle(
            res["code"],
            spec_desc=res["spec_desc"],
            explicit_tests=res["tests"],
        )
        assert report.verified is True, f"TDD failed for {prompt}: {report.diagnostics}"
        assert report.tests_generated > 0 and report.tests_passed == report.tests_generated


def test_constraint_aware_adaptive_concept_realizer():
    eng = OpenEndedConceptLatticeEngine()
    try:
        # 1. Concise 2-line constraint ("2 lines mein btao")
        r_short = eng.route_open_ended("photosynthesis kya hta hy 2 lines mein btao")
        assert r_short.matched is True
        assert "Style: concise_2_lines" in r_short.response
        assert len(r_short.response.strip().splitlines()) == 2

        # 2. Simple / ELI5 constraint ("asaan lafzon mein")
        r_simple = eng.route_open_ended("kubernetes smjha do asaan lafzon mein")
        assert r_simple.matched is True
        assert "Style: simple_eli5" in r_simple.response
        assert "Asaan Lafzon Mein" in r_simple.response

        # 3. Strict single-facet constraint ("sirf nuqsan btao" / "only drawbacks")
        r_only_trd = eng.route_open_ended("blockchain ke sirf nuqsan btao")
        assert r_only_trd.matched is True
        assert r_only_trd.intents_resolved == ("tradeoffs",)
        assert "Nuqsan aur Hudood" in r_only_trd.response
        assert "Taaruf (Definition)" not in r_only_trd.response
        assert "Fayday aur Istemal" not in r_only_trd.response

        # 4. Code + explanation constraint ("with python code example")
        r_code = eng.route_open_ended("explain sql injection with python code example")
        assert r_code.matched is True
        assert "```python" in r_code.response
        assert "safe_parameterized_query" in r_code.response
    finally:
        eng.close()


def test_34_unseen_real_user_bilingual_stress_suite(monkeypatch):
    """End-to-end verification across 34 messy English & Roman-Urdu prompts (14 coding + 14 concept + 6 OOD traps)."""
    monkeypatch.setenv("MIRROR_SANDBOX_MODE", "process")
    svc = BackendService()

    coding_prompts = [
        ("python k code likho jo Ahmad print kry", "message = 'Ahmad'"),
        ("write python code to print Ahmad", "message = 'Ahmad'"),
        ("bina quotes ke Bilal print krne ka python program likho", "message = 'Bilal'"),
        ("Ahmad ko 5 bar print krne ka code do", "repeat_print(message='Ahmad', times=5)"),
        ("5 ka pahara 10 tak print krne ka python code likho", "def multiplication_table(n=5, upto=10):"),
        ("7 ka table 12 tak print krne ka python code do", "def multiplication_table(n=7, upto=12):"),
        ("1 se 15 tak numbers print kro python code", "def print_range(start=1, end=15):"),
        ("ek list mein se even numbers ka sum nikalne ka python function bnao", "def sum_even_numbers(nums):"),
        ("python mein prime number aur factorial check krne ka code likho", "def is_prime(n):"),
        ("write a python function to filter odd numbers and return their squares sorted", "def process_odd_square_sort_asc_numbers(nums):"),
        ("ek python function bnao jo list mein se positive numbers ka average nikale", "def average_positive_numbers(nums):"),
        ("python code likho jo list mein se negative numbers count kry", "def count_negative_numbers(nums):"),
        ("write python functions for gcd and lcm", "def gcd(a, b):"),
        ("create a python class for stack push and pop", "class Stack:"),
    ]
    for prompt, expected_code in coding_prompts:
        res = svc.cognitive_ask(prompt)
        assert res["abstained"] is False, f"Abstained on coding prompt: {prompt} -> {res}"
        assert res["task_code_verified"] is True, f"Code not verified for: {prompt}"
        assert expected_code in res["answer"], f"Missing '{expected_code}' in answer for {prompt}:\n{res['answer']}"

    concept_prompts = [
        ("sql injection kya hota hai", "injection"),
        ("sql injection kia hta h", "injection"),
        ("kubernetes samjha do", "kubernetes"),
        ("kubernetes smjha do asaan lafzon mein", "asaan lafzon mein"),
        ("photosynthesis btao", "photosynthesis"),
        ("photosynthesis kya hta hy 2 lines mein btao", "concise_2_lines"),
        ("blockchain ke sirf nuqsan btao", "nuqsan aur hudood"),
        ("only mechanism of cryptography", "mechanism"),
        ("explain sql injection with python code example", "safe_parameterized_query"),
        ("compare mitosis and meiosis", "mitosis"),
        ("what is privacy risk management", "privacy"),
        ("explain cwe-79", "79"),
        ("recursion kise kehte hain", "recursion"),
        ("deadlock kya hoti hai", "deadlock"),
    ]
    for prompt, expected_term in concept_prompts:
        res = svc.cognitive_ask(prompt)
        assert res["abstained"] is False, f"Abstained on concept prompt: {prompt} -> {res}"
        assert expected_term.casefold() in res["answer"].casefold(), f"{prompt} -> {res['answer'][:160]}"

    ood_trap_prompts = [
        "What is our server port?",
        "What will be the exact stock price of AAPL tomorrow?",
        "Who won the 2032 FIFA World Cup?",
        "What is the current population of Mars?",
        "What was decided in our secret board meeting yesterday?",
        "How many planets are there in the universe?",
    ]
    for trap in ood_trap_prompts:
        res = svc.cognitive_ask(trap)
        assert res["abstained"] is True, f"Failed to abstain on OOD trap: {trap} -> {res['answer']}"
        assert res["verified"] is False
