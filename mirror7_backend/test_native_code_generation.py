from __future__ import annotations

from mirror7_backend.native_code_generation import NativeCodeGenerationProvider
from mirror_crystal.phase7_qwen_to_opus import AutonomousTDDEngine


def test_native_builder_generates_and_tests_a_print_program():
    generated = NativeCodeGenerationProvider()(
        """Write Python code that prints 'Hello Mirror'"""
    )
    assert generated is not None
    assert generated["generation_kind"] == "print_text"
    assert "Hello Mirror" in generated["code"]
    assert "assert main()" in generated["tests"]

    report = AutonomousTDDEngine().run_tdd_cycle(
        generated["code"],
        spec_desc=generated["spec_desc"],
        explicit_tests=generated["tests"],
    )
    assert report.verified is True
    assert report.tests_generated == report.tests_passed == 1


def test_native_builder_generates_and_tests_simple_arithmetic():
    generated = NativeCodeGenerationProvider()(
        "Write Python code to add two numbers"
    )
    assert generated is not None
    assert "def add(a, b):" in generated["code"]
    report = AutonomousTDDEngine().run_tdd_cycle(
        generated["code"],
        spec_desc=generated["spec_desc"],
        explicit_tests=generated["tests"],
    )
    assert report.verified is True
    assert report.tests_generated == report.tests_passed == 3


def test_native_builder_asks_for_missing_personal_name():
    generated = NativeCodeGenerationProvider()(
        "Write Python code that prints my name"
    )
    assert generated["code"] is None
    assert "What name" in generated["needs_input"]


def test_native_builder_does_not_claim_to_support_arbitrary_programs():
    generated = NativeCodeGenerationProvider()("Write a complete sorting library")
    assert generated is None


def test_native_builder_supports_roman_urdu_and_unquoted_print():
    provider = NativeCodeGenerationProvider()
    for prompt, expected in (
        ("python k code likho jo Ahmad print kry", "Ahmad"),
        ("python ka code likho jo Ahmad print kare", "Ahmad"),
        ("write python code to print Ahmad", "Ahmad"),
        ("write a python program to print Hello World", "Hello World"),
    ):
        generated = provider(prompt)
        assert generated is not None, prompt
        assert generated["generation_kind"] == "print_text"
        assert f"message = {expected!r}" in generated["code"], generated["code"]
        report = AutonomousTDDEngine().run_tdd_cycle(
            generated["code"],
            spec_desc=generated["spec_desc"],
            explicit_tests=generated["tests"],
        )
        assert report.verified is True
        assert report.tests_generated == report.tests_passed == 1


def test_native_builder_supports_repeated_print_table_range_and_bilingual_ops():
    provider = NativeCodeGenerationProvider()

    rep = provider("python mein Ahmad ko 5 bar print kro")
    assert rep is not None
    assert rep["generation_kind"] == "repeat_print"
    assert "message='Ahmad', times=5" in rep["code"]
    assert AutonomousTDDEngine().run_tdd_cycle(
        rep["code"], spec_desc=rep["spec_desc"], explicit_tests=rep["tests"]
    ).verified is True

    for prompt, expected_kind in (
        ("5 ka table print kro python mein", "multiplication_table"),
        ("1 se 10 tak numbers print kro python code", "print_range"),
        ("python code likho jo 2 numbers ko add kry", "add"),
        ("python code banao calculator k liye", "calculator"),
        ("python function likho even ya odd check krne k liye", "check_even_odd"),
    ):
        generated = provider(prompt)
        assert generated is not None, prompt
        assert generated["generation_kind"] == expected_kind
        report = AutonomousTDDEngine().run_tdd_cycle(
            generated["code"],
            spec_desc=generated["spec_desc"],
            explicit_tests=generated["tests"],
        )
        assert report.verified is True, prompt


def test_native_builder_asks_for_missing_roman_urdu_name():
    generated = NativeCodeGenerationProvider()("python code likho jo mera naam print kry")
    assert generated is not None
    assert generated["code"] is None
    assert "What name" in generated["needs_input"]
