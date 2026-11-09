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

    def solve(self, prompt):
        return self.result


class Code:
    def __init__(self, result):
        self.result = result

    def run_tdd_cycle(self, candidate_code, spec_desc="", explicit_tests=""):
        return self.result


def make(memory=None, vault=None, reasoner=None, code=None, metrics=None):
    return UnifiedCognitiveOrchestrator(
        vault_provider=lambda: vault,
        memory_provider=lambda: memory or Memory(),
        reasoning_provider=lambda: reasoner or Reasoner({"ok": False}),
        code_provider=lambda: code or Code({"passed": False}),
        metrics=metrics or Metrics(),
    )


def test_collects_memory_and_crystal_without_short_circuiting():
    memory = Memory("Remembered project context")
    vault = Vault(SimpleNamespace(
        answer="Stored technical answer", confidence=0.8, epistemic_tier="moderate",
        sandbox_verified=False, matched_rule_id=7,
    ))
    result = make(memory=memory, vault=vault).ask("Explain this", session_id="s")
    assert result["answer"] == "Stored technical answer"
    assert {item["source"] for item in result["evidence"]} == {
        "cross_session_memory", "crystal_knowledge"
    }
    assert memory.extracted == [("s", "Explain this")]


def test_math_reasoner_contributes_verified_result():
    result = make(reasoner=Reasoner({
        "ok": True, "prm_reward": 0.95, "formatted_solution": "42",
        "strategy": "exact",
    })).ask("Calculate 6 times 7")
    assert result["answer"] == "42"
    assert result["verified"] is True
    assert "cerebrum_reasoning" in [item["source"] for item in result["evidence"]]


def test_conflicting_verified_sources_force_abstention():
    memory = Memory("Answer A")
    vault = Vault(SimpleNamespace(
        answer="Answer B", confidence=1.0, epistemic_tier="verified_exact",
        sandbox_verified=True, matched_rule_id=3,
    ))
    result = make(memory=memory, vault=vault).ask("What is the answer?")
    assert result["answer"] == "Answer B"
    assert result["conflict"] is False
    assert result["verified"] is True


def test_code_verification_requires_explicit_pass_flag():
    result = make(code=Code({"passed": True, "summary": "all tests pass"})).ask(
        "Review this implementation", candidate_code="print('ok')"
    )
    assert result["code_verification"]["verified"] is True
    assert result["verified"] is False  # code test result is not an answer proof


def test_missing_sources_abstain():
    result = make().ask("Unanswerable query")
    assert result["abstained"] is True
    assert result["answer"] == "Insufficient verified evidence."
