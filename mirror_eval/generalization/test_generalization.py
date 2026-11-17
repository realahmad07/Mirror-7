"""Tests for NovelGen generalization benchmark, ContextReasoner, and sealed evaluator."""
from __future__ import annotations

from mirror_crystal.native_intelligence import ContextReasoner
from mirror_eval.generalization import evaluate, generate, items_digest
from mirror_eval.generalization.sealed_runner import run_sealed_eval, verify_bundle
from mirror_eval.generalization.suts import native_chain


def test_generator_is_deterministic_and_uses_nonce_words():
    a = generate(seed=123, per_cell=2)
    b = generate(seed=123, per_cell=2)
    c = generate(seed=124, per_cell=2)
    assert items_digest(a) == items_digest(b)
    assert items_digest(a) != items_digest(c)
    assert len(a) == 6 * 6 * 2  # 6 families * 6 depths * 2


def test_context_reasoner_generalizes_across_all_families_and_ood_depths():
    items = generate(seed=20261117, per_cell=3)
    before = evaluate(items, native_chain(include_context_reasoner=False))
    after = evaluate(items, native_chain(include_context_reasoner=True))
    assert before["overall"]["accuracy"] < 0.20
    assert after["by_split"]["id"]["accuracy"] >= 0.98
    assert after["by_split"]["ood"]["accuracy"] >= 0.98
    assert after["overall"]["consistency"] >= 0.95
    assert after["overall"]["sources"].get("reasoner", 0) >= int(0.98 * after["overall"]["variants"])


def test_context_reasoner_rejects_affirming_the_consequent():
    r = ContextReasoner()
    ans = r.respond("Every glorp is a snarf. Blib is a snarf. Is Blib a glorp?")
    assert ans is not None
    assert "unknown" in ans.answer.lower()


def test_sealed_runner_signature_and_tamper_detection():
    bundle = run_sealed_eval(target="native", seed="unit-test-seed", per_cell=1)
    assert verify_bundle(bundle) is True
    assert bundle["payload"]["results"]["overall"]["accuracy"] >= 0.97
    # Tampering with any field invalidates the Ed25519 signature.
    bundle["payload"]["seed"] = "tampered"
    assert verify_bundle(bundle) is False


def test_orchestrator_routes_novel_concepts():
    from mirror7_backend.cognitive_orchestrator import UnifiedCognitiveOrchestrator

    orch = UnifiedCognitiveOrchestrator(
        vault_provider=lambda: None,
        memory_provider=lambda: None,
        reasoning_provider=lambda: None,
        code_provider=lambda: None,
    )
    r1 = orch.ask("Zim is plonker than Vex. Vex is plonker than Dax. Who is the plonkest?")
    assert "Zim" in r1["answer"] and r1["verified"] is True
    r2 = orch.ask("Define blarg(x, y) = 2 * x + y. What is blarg(3, blarg(1, 4))?")
    assert r2["answer"].strip() == "12" and r2["verified"] is True
