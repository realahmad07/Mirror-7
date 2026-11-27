from types import SimpleNamespace

from mirror7_backend.cognitive_orchestrator import UnifiedCognitiveOrchestrator


class Memory:
    def __init__(self, context=None):
        self.context = context
        self.extracted = []

    def resolve_memory_context(self, session_id, prompt):
        return self.context

    def auto_extract_and_remember(self, session_id, prompt):
        self.extracted.append((session_id, prompt))


class Metrics:
    def __init__(self):
        self.values = {}

    def increment(self, name):
        self.values[name] = self.values.get(name, 0) + 1


class Vault:
    def __init__(self, result):
        self.result = result

    def respond(self, prompt, session_id="default"):
        return self.result


class Reasoner:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def solve(self, prompt):
        self.calls.append(prompt)
        return self.result


class Code:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def run_tdd_cycle(self, candidate_code, spec_desc="", explicit_tests=""):
        self.calls.append((candidate_code, spec_desc, explicit_tests))
        return self.result


def make(
    memory=None,
    vault=None,
    reasoner=None,
    code=None,
    metrics=None,
    code_generation_provider=None,
    synthesis_provider=None,
):
    return UnifiedCognitiveOrchestrator(
        vault_provider=lambda: vault,
        memory_provider=lambda: memory if memory is not None else Memory(),
        reasoning_provider=lambda: reasoner if reasoner is not None else Reasoner({"ok": False}),
        code_provider=lambda: code if code is not None else Code({"verified": False}),
        metrics=metrics or Metrics(),
        code_generation_provider=code_generation_provider,
        synthesis_provider=synthesis_provider,
    )


def test_memory_is_context_not_an_answer_or_verified_evidence():
    memory = Memory("Remembered project context")
    vault = Vault(None)
    result = make(memory=memory, vault=vault).ask("Explain this", session_id="s")

    assert result["answer"] == "Insufficient independently verified evidence."
    assert result["abstained"] is True
    assert result["verified"] is False
    assert result["memory_context"] == "Remembered project context"
    assert result["memory_used_as_evidence"] is False
    assert result["evidence"] == []
    assert memory.extracted == []


def test_memory_is_not_auto_extracted_from_questions():
    memory = Memory()
    result = make(memory=memory).ask("What is our server port?")
    assert result["abstained"] is True
    assert memory.extracted == []


def test_explicit_configuration_statement_may_be_remembered():
    memory = Memory()
    make(memory=memory).ask("Our project server port is 8787")
    assert memory.extracted == [("default", "Our project server port is 8787")]


def test_only_independently_supported_crystal_modes_are_verified():
    vault = Vault(SimpleNamespace(
        answer="The calculation equals 42.",
        confidence=0.99,
        epistemic_tier="verified_exact",
        sandbox_verified=True,
        matched_rule_id=None,
        mode="symbolic_cas",
        reasoning_trace=["Exact symbolic operation"],
    ))
    result = make(vault=vault).ask("What is the answer?")
    assert result["answer"] == "The calculation equals 42."
    assert result["verified"] is True
    assert result["abstained"] is False
    assert result["synthesis"]["claims"][0]["sources"] == ["crystal_knowledge"]


def test_unreviewed_legacy_rule_does_not_become_verified_by_sandbox_flag():
    vault = Vault(SimpleNamespace(
        answer="Possibly true answer.",
        confidence=0.99,
        epistemic_tier="verified_exact",
        sandbox_verified=True,
        matched_rule_id=17,
        mode="exact_rule_match",
        reasoning_trace=[],
    ))
    result = make(vault=vault).ask("Question")
    assert result["verified"] is False
    assert result["abstained"] is True
    assert result["candidate_source"] == "crystal_knowledge"


def test_math_prm_result_is_a_candidate_not_independent_proof():
    reasoner = Reasoner({
        "ok": True,
        "prm_reward": 0.99,
        "formatted_solution": "42",
        "strategy": "rational_arithmetic",
        "prm_steps": [{"is_valid": True, "reward": 1.0}],
    })
    result = make(reasoner=reasoner).ask("Calculate 6 times 7")

    assert reasoner.calls == ["Calculate 6 times 7"]
    assert result["answer"].startswith("UNVERIFIED CANDIDATE — not independently verified.")
    assert "42" in result["answer"]
    assert result["answer_qualification"] == "unverified_candidate"
    assert result["verified"] is False
    assert result["abstained"] is True
    assert result["confidence"] == 0.0
    assert result["candidate_answer"] == "42"
    assert result["computed_candidates"][0]["epistemic_tier"] == "computed_candidate"


def test_code_verification_requires_non_vacuous_complete_tdd_report():
    code = Code({
        "verified": True,
        "tests_generated": 3,
        "tests_passed": 3,
        "final_code": "def f(): return 1",
    })
    result = make(code=code).ask(
        "Review this implementation", candidate_code="def f(): return 1"
    )
    assert result["code_verification"]["verified"] is True
    assert result["code_verification"]["tests_generated"] == 3
    assert result["verified"] is False  # code tests do not prove a factual response


def test_zero_assertion_tdd_report_is_never_verified():
    code = Code({
        "verified": True,
        "tests_generated": 0,
        "tests_passed": 0,
    })
    result = make(code=code).ask("Review this", candidate_code="pass")
    assert result["code_verification"]["verified"] is False


def test_natural_language_code_generation_is_connected_to_tdd():
    code = Code({
        "verified": True,
        "tests_generated": 2,
        "tests_passed": 2,
        "final_code": "def add(a, b): return a + b",
    })
    result = make(
        code=code,
        code_generation_provider=lambda prompt: {"code": "def add(a, b): return a + b"},
    ).ask(
        "Write code for an add function",
        spec_desc="Add two numbers",
        explicit_tests="assert add(2, 3) == 5\nassert add(-1, 1) == 0",
    )
    assert result["generated_code"] == "def add(a, b): return a + b"
    assert result["code_generation"]["available"] is True
    assert result["code_generation"]["generated"] is True
    assert result["code_verification"]["verified"] is True
    assert "code_generation" in result["sources_consulted"]
    assert "code_verification" in result["sources_consulted"]


def test_missing_code_generator_is_reported_not_fabricated():
    result = make().ask("Write code for a web scraper")
    assert result["code_generation"] == {"available": False, "generated": False}
    assert result["generated_code"] is None


def test_synthesis_generator_cannot_add_unsupported_claims():
    vault = Vault(SimpleNamespace(
        answer="Python is dynamically typed.",
        confidence=0.95,
        epistemic_tier="high_confidence",
        sandbox_verified=False,
        matched_rule_id=None,
        mode="blueprint_composition_reliable",
        reasoning_trace=[],
    ))
    result = make(
        vault=vault,
        synthesis_provider=lambda payload: (
            "Python is dynamically typed. Python always runs faster than C."
        ),
    ).ask("Describe Python typing")
    assert result["verified"] is True
    assert result["answer"] == "Python is dynamically typed."
    assert result["synthesis"]["generator_rejected"] is True
    assert result["synthesis"]["memory_used_as_evidence"] is False


def test_native_code_reply_includes_only_tdd_verified_source():
    code = Code({
        "verified": True,
        "tests_generated": 1,
        "tests_passed": 1,
        "final_code": "def main():\n    message = 'Hello Mirror'\n    print(message)\n    return message",
    })
    result = make(
        code=code,
        code_generation_provider=lambda prompt: {
            "code": "def main():\n    message = 'Hello Mirror'\n    print(message)\n    return message",
            "tests": "assert main() == 'Hello Mirror'\n",
            "spec_desc": "print Hello Mirror",
            "generation_kind": "print_text",
        },
    ).ask("Write Python code that prints 'Hello Mirror'")

    assert result["task_result_verified"] is True
    assert result["task_code_verified"] is True
    assert result["verified"] is False  # program checks do not certify factual claims
    assert "Generated and tested Python code:" in result["answer"]
    assert "def main()" in result["answer"]
    assert code.calls[0][2] == "assert main() == 'Hello Mirror'\n"


def test_native_code_generation_asks_for_missing_name_without_fabricating():
    result = make(
        code_generation_provider=lambda prompt: {
            "code": None,
            "needs_input": "What name should the program print?",
            "generation_kind": "print_text",
        },
    ).ask("Write Python code that prints my name")

    assert result["answer"] == "What name should the program print?"
    assert result["answer_qualification"] == "clarification_required"
    assert result["abstained"] is False
    assert result["generated_code"] is None


def test_unknown_factual_question_is_not_invented():
    result = make(vault=Vault(None)).ask("How many planets are there in the universe?")
    assert result["answer"] == "Insufficient independently verified evidence."
    assert result["verified"] is False
    assert result["abstained"] is True


def test_roman_urdu_code_prompt_triggers_native_builder_and_tdd():
    code = Code({
        "verified": True,
        "tests_generated": 1,
        "tests_passed": 1,
        "final_code": "def main():\n    message = 'Ahmad'\n    print(message)\n    return message",
    })
    result = make(
        code=code,
        code_generation_provider=lambda prompt: {
            "code": "def main():\n    message = 'Ahmad'\n    print(message)\n    return message",
            "tests": "assert main() == 'Ahmad'\n",
            "spec_desc": "print the requested text",
            "generation_kind": "print_text",
        },
    ).ask("python k code likho jo Ahmad print kry")

    assert result["task_result_verified"] is True
    assert result["task_code_verified"] is True
    assert result["abstained"] is False
    assert "message = 'Ahmad'" in result["answer"]
    assert "code_generation" in result["sources_consulted"]
    assert "code_verification" in result["sources_consulted"]


def test_missing_container_runtime_reports_transparent_sandbox_diagnostic():
    code = Code({
        "verified": False,
        "tests_generated": 1,
        "tests_passed": 0,
        "diagnostics": [
            "Iteration 1: FAILED: Docker or Podman is required; process fallback is disabled by default."
        ],
    })
    result = make(
        code=code,
        code_generation_provider=lambda prompt: {
            "code": "def main():\n    message = 'Ahmad'\n    print(message)\n    return message",
            "tests": "assert main() == 'Ahmad'\n",
            "spec_desc": "print the requested text",
            "generation_kind": "print_text",
        },
    ).ask("python k code likho jo Ahmad print kry")

    assert result["abstained"] is True
    assert result["verified"] is False
    assert result["task_code_verified"] is False
    assert result["answer_qualification"] == "sandbox_unavailable"
    assert "Docker or Podman" in result["answer"]
    assert "MIRROR_SANDBOX_MODE=process" in result["answer"]


def test_concept_lattice_fallback_respects_isolated_vault_precedence():
    vault = Vault(SimpleNamespace(
        answer="Custom isolated vault answer about quantum computing.",
        confidence=0.92,
        mode="direct_crystal_lock",
        matched_rule_id=7,
        sandbox_verified=False,
        epistemic_tier="unverified_knowledge",
        reasoning_trace=[],
    ))

    class FakeLatticeRouter:
        def route(self, prompt, session_id="default"):
            return SimpleNamespace(
                matched=True,
                rule_ids=["concept_lattice.quantum_computing"],
                inputs={},
                output="Concept Lattice encyclopedia explanation.",
                unit="",
                explanation="Concept Lattice encyclopedia explanation.",
                steps=["lattice_lookup"],
            )

    isolated = UnifiedCognitiveOrchestrator(
        vault_provider=lambda: vault,
        memory_provider=lambda: Memory(),
        reasoning_provider=lambda: Reasoner({"ok": False}),
        code_provider=lambda: Code({"verified": False}),
        uol_router_provider=lambda: FakeLatticeRouter(),
        prefer_vault_evidence_over_lattice=True,
    ).ask("What is quantum computing?")
    assert isolated["abstained"] is True
    assert isolated["answer_qualification"] == "unverified_candidate"
    assert "Custom isolated vault answer about quantum computing." in isolated["answer"]

    autoloaded = UnifiedCognitiveOrchestrator(
        vault_provider=lambda: vault,
        memory_provider=lambda: Memory(),
        reasoning_provider=lambda: Reasoner({"ok": False}),
        code_provider=lambda: Code({"verified": False}),
        uol_router_provider=lambda: FakeLatticeRouter(),
        prefer_vault_evidence_over_lattice=False,
    ).ask("What is quantum computing?")
    assert autoloaded["abstained"] is False
    assert autoloaded["task_result_verified"] is True
    assert "Concept Lattice encyclopedia explanation." in autoloaded["answer"]
