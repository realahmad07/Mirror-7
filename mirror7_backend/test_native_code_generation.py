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
