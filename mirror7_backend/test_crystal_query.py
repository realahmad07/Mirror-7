from __future__ import annotations

from types import SimpleNamespace

from mirror7_backend.crystal_query import CrystalQueryCoordinator


class FakeMetrics:
    def __init__(self):
        self.values = {}

    def increment(self, name):
        self.values[name] = self.values.get(name, 0) + 1


class FakeMemoryGraph:
    def __init__(self, context=None):
        self.context = context
        self.extracted = []

    def resolve_memory_context(self, session_id, prompt):
        return self.context

    def auto_extract_and_remember(self, session_id, prompt):
        self.extracted.append((session_id, prompt))


class FakeVault:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def respond(self, prompt, session_id="default"):
        self.calls.append((prompt, session_id))
        return self.response


class FakeBridge:
    def __init__(self, response=None):
        self.response = response or {"ok": False}
        self.calls = []

    def solve(self, prompt):
        self.calls.append(prompt)
        return self.response


def make_coordinator(vault, graph=None, bridge=None, metrics=None):
    graph = graph or FakeMemoryGraph()
    bridge = bridge or FakeBridge()
    metrics = metrics or FakeMetrics()
    coordinator = CrystalQueryCoordinator(
        vault_provider=lambda: vault,
        memory_graph_provider=lambda: graph,
        cerebrum_bridge_provider=lambda: bridge,
        metrics=metrics,
    )
    return coordinator, graph, bridge, metrics


def test_missing_vault_abstains_without_initializing_other_services():
    coordinator, graph, bridge, metrics = make_coordinator(None)
    result = coordinator.ask("What is this?")
    assert result["abstained"] is True
    assert result["epistemic_tier"] == "honest_abstain"
    assert result["teach_template"]["question"] == "What is this?"
    assert graph.extracted == []
    assert bridge.calls == []
    assert metrics.values == {}


def test_vault_answer_is_returned_and_telemetry_recorded():
    response = SimpleNamespace(
        answer="Stored fact", confidence=0.7, mode="direct_crystal_lock",
        matched_rule_id=12, latency_ms=0.4, sandbox_verified=False,
        reasoning_trace=["rule 12"], epistemic_tier="moderate_confidence",
    )
    vault = FakeVault(response)
    coordinator, graph, bridge, metrics = make_coordinator(vault)
    result = coordinator.ask("Explain stored fact", session_id="s-1")
    assert result["answer"] == "Stored fact"
    assert result["epistemic_tier"] == "moderate_confidence"
    assert vault.calls == [("Explain stored fact", "s-1")]
    assert graph.extracted == [("s-1", "Explain stored fact")]
    assert bridge.calls == []
    assert metrics.values == {"crystal_queries_total": 1, "crystal_queries_answered": 1}


def test_math_fallback_only_replaces_a_clean_abstention():
    abstention = SimpleNamespace(
        answer="No match", confidence=0.0, mode="unmatched_honest_abstain",
        matched_rule_id=None, latency_ms=1.0, sandbox_verified=False,
        reasoning_trace=["abstained"], epistemic_tier="honest_abstain",
    )
    vault = FakeVault(abstention)
    bridge = FakeBridge({
        "ok": True, "prm_reward": 0.91, "answer": "42",
        "formatted_solution": "The answer is 42.", "strategy": "exact_arithmetic",
        "latency_ms": 0.2,
    })
    coordinator, graph, _, metrics = make_coordinator(vault, bridge=bridge)
    result = coordinator.ask("What is 6 times 7?")
    assert result["answer"] == "The answer is 42."
    assert result["mode"] == "cerebrum_exact_arithmetic"
    assert result["epistemic_tier"] == "verified_exact"
    assert bridge.calls == ["What is 6 times 7?"]
    assert metrics.values == {"crystal_queries_total": 1, "crystal_queries_answered": 1}


def test_memory_recall_short_circuits_vault_lookup():
    vault = FakeVault(None)
    graph = FakeMemoryGraph(context="Project database is PostgreSQL.")
    coordinator, _, bridge, metrics = make_coordinator(vault, graph=graph)
    result = coordinator.ask("What is our database?", session_id="project")
    assert result["answer"] == "Project database is PostgreSQL."
    assert result["mode"] == "cross_session_memory_recall"
    assert vault.calls == []
    assert bridge.calls == []
    assert metrics.values == {"crystal_queries_total": 1, "crystal_queries_answered": 1}
