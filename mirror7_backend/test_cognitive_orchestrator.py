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
    assert result["code_generation"] == {"available": True, "generated": True}
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
