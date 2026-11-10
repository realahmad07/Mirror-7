from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import sqlite3

import pytest

from mirror_crystal import (
    Blueprint, BlueprintKind, BlueprintProvenance, DiskCrystalVault,
    EvidenceLevel, VerificationEvidence,
)
from mirror_crystal.answer_verification import BlueprintAnswerVerifier
from mirror_crystal.blueprint_runtime import BlueprintCompositionEngine, BlueprintCompositionError
from mirror_crystal.os_process_sandbox import run_python_isolated


def make_blueprint(
    blueprint_id: str,
    response: str,
    *,
    intent: str = "demo",
    domain: str = "testing",
    dependencies: tuple[str, ...] = (),
    preconditions: tuple[str, ...] = (),
    postconditions: tuple[str, ...] = (),
    evidence: EvidenceLevel = EvidenceLevel.UNVERIFIED,
    captured_at: str | None = None,
) -> Blueprint:
    prompt = "request {v1}"
    digest = Blueprint._digest(prompt, response)
    return Blueprint(
        blueprint_id=blueprint_id,
        kind=BlueprintKind.PROCEDURE,
        domain=domain,
        intent=intent,
        prompt_pattern=prompt,
        response_template=response,
        slots=("v1",) if "{v1}" in response else (),
        dependencies=dependencies,
        preconditions=preconditions,
        postconditions=postconditions,
        provenance=BlueprintProvenance(
            source="test-fixture",
            source_ref=f"fixture://{blueprint_id}",
            content_sha256=digest,
            captured_at=captured_at or datetime.now(timezone.utc).isoformat(),
        ),
        verification=VerificationEvidence(
            level=evidence,
            checks_passed=("fixture-check",) if evidence in {
                EvidenceLevel.EXECUTION_TESTED, EvidenceLevel.FORMALLY_VERIFIED,
                EvidenceLevel.INDEPENDENTLY_REPRODUCED
            } else (),
            verifier="independent-fixture-verifier" if evidence in {
                EvidenceLevel.FORMALLY_VERIFIED, EvidenceLevel.INDEPENDENTLY_REPRODUCED
            } else "",
        ),
    )


def test_dependency_order_and_template_rendering():
    prerequisite = make_blueprint("prep", "Prepare {v1}")
    root = make_blueprint("root", "Finish {v1}", dependencies=("prep",))
    result = BlueprintCompositionEngine().compose(
        "root", [root, prerequisite], values={"v1": "task"}
    )
    assert result.blueprint_ids == ("prep", "root")
    assert result.answer == "Prepare task\n\nFinish task"


def test_missing_dependency_and_precondition_fail_closed():
    root = make_blueprint("root", "Answer", dependencies=("missing",))
    with pytest.raises(BlueprintCompositionError, match="missing dependency"):
        BlueprintCompositionEngine().compose("root", [root])
    blocked = make_blueprint("blocked", "Answer", preconditions=("identity-confirmed",))
    with pytest.raises(BlueprintCompositionError, match="unmet preconditions"):
        BlueprintCompositionEngine().compose("blocked", [blocked])


def test_postcondition_requires_explicit_verifier():
    root = make_blueprint("root", "Result", postconditions=("result-is-valid",))
    engine = BlueprintCompositionEngine()
    with pytest.raises(BlueprintCompositionError, match="postcondition"):
        engine.compose("root", [root])
    result = engine.compose(
        "root", [root],
        postcondition_verifier=lambda blueprint, condition, output, state: condition == "result-is-valid" and output == "Result",
    )
    assert result.verified_postconditions == ("result-is-valid",)


def test_formula_requires_an_explicit_executor():
    root = make_blueprint("formula", "Result")
    object.__setattr__(root, "derived_expr", "x + 1")
    with pytest.raises(BlueprintCompositionError, match="safe executor"):
        BlueprintCompositionEngine().compose("formula", [root])
    result = BlueprintCompositionEngine().compose(
        "formula", [root], executor=lambda blueprint, state: "safe-result"
    )
    assert result.answer == "safe-result"


def test_legacy_blueprint_backfill_is_persisted_but_unverified(tmp_path):
    vault = DiskCrystalVault(tmp_path / "vault.dcrystal")
    try:
        assert vault.ingest_pair("What is the fixture answer?", "Fixture answer.")
        result = vault.backfill_legacy_blueprints(limit=10)
        assert result["migrated"] == 1
        blueprint = vault.get_blueprint("crystal:0")
        assert blueprint is not None
        assert blueprint.verification.level is EvidenceLevel.UNVERIFIED
    finally:
        vault.close()


def test_blueprint_storage_and_runtime_never_promote_unverified_source(tmp_path):
    vault = DiskCrystalVault(tmp_path / "vault.dcrystal")
    try:
        blueprint = make_blueprint("runtime", "The answer is {v1}.")
        vault.store_blueprint(blueprint)
        loaded = vault.get_blueprint("runtime")
        assert loaded is not None
        assert loaded.canonical_json() == blueprint.canonical_json()
        response = vault.respond_blueprint("runtime", values={"v1": "42"})
        assert response.answer == "The answer is 42."
        assert response.epistemic_tier == "unverified"
        assert response.mode == "blueprint_composition_unverified"
        assert any("source" in item.lower() for item in response.reasoning_trace)
    finally:
        vault.close()


def test_persisted_runtime_resolves_requested_chain_without_global_blueprint_scan(tmp_path):
    vault = DiskCrystalVault(tmp_path / "large-blueprint-vault.dcrystal")
    try:
        prerequisite = make_blueprint(
            "late:prerequisite", "Prepare {v1}", intent="prerequisite"
        )
        root = make_blueprint(
            "late:root", "Complete {v1}", intent="root",
            dependencies=("late:prerequisite",),
        )
        vault.store_blueprint(prerequisite)
        vault.store_blueprint(root)

        original_list = vault.list_blueprints

        def forbid_global_blueprint_scan(*, domain=None, intent=None, limit=10000):
            assert domain is not None and intent is not None, (
                "runtime must not enumerate the entire capped blueprint collection"
            )
            return original_list(domain=domain, intent=intent, limit=limit)

        vault.list_blueprints = forbid_global_blueprint_scan
        response = vault.respond_blueprint("late:root", values={"v1": "task"})
        assert response.mode == "blueprint_composition_unverified"
        assert response.epistemic_tier == "unverified"
        assert response.answer == "Prepare task\n\nComplete task"
    finally:
        vault.close()


def test_conflicting_blueprints_force_abstention(tmp_path):
    vault = DiskCrystalVault(tmp_path / "vault.dcrystal")
    try:
        vault.store_blueprint(make_blueprint("a", "Answer A"))
        vault.store_blueprint(make_blueprint("b", "Answer B"))
        response = vault.respond_blueprint("a")
        assert response.mode == "blueprint_contradiction_abstain"
        assert response.epistemic_tier == "honest_abstain"
    finally:
        vault.close()


def test_answer_verifier_rejects_stale_and_unverified_blueprints():
    stale = (datetime.now(timezone.utc) - timedelta(days=3650)).isoformat()
    blueprint = make_blueprint("stale", "Fact", captured_at=stale)
    report = BlueprintAnswerVerifier().verify(blueprint, max_age_days=30)
    assert not report.reliable
    assert report.freshness == "stale"
    assert any("evidence level" in reason for reason in report.reasons)


def test_process_sandbox_executes_in_distinct_bounded_process():
    result = run_python_isolated("print(1 + 1)")
    assert result.ok
    assert result.output == "2"
    assert result.pid is not None and result.pid != os.getpid()


def test_ast_gate_still_blocks_unsafe_imports():
    from mirror_crystal import ZeroTrustSandboxJudge
    ok, message = ZeroTrustSandboxJudge.verify_code_and_test("import os\nprint('x')")
    assert not ok
    assert "Blocked import" in message


def test_child_sandbox_accepts_typed_helper_and_generator_expression():
    from mirror_crystal import ZeroTrustSandboxJudge

    code = """def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for k in range(2, int(n ** 0.5) + 1):
        if n % k == 0:
            return False
    return True

def digit_sum(n: int) -> int:
    return sum(int(ch) for ch in str(abs(n)))
"""
    checks = "assert is_prime(2) and is_prime(13) and not is_prime(9)\nassert digit_sum(29) == 11"
    ok, message = ZeroTrustSandboxJudge.verify_code_and_test(code, test_code=checks)
    assert ok, message


def test_default_respond_path_prefers_canonical_blueprint_slot_match(tmp_path):
    vault = DiskCrystalVault(tmp_path / "vault.dcrystal")
    try:
        blueprint = make_blueprint("default-dispatch", "The answer is {v1}.")
        vault.store_blueprint(blueprint)
        response = vault.respond("request 42")
        assert response.mode == "blueprint_composition_unverified"
        assert response.answer == "The answer is 42."
        assert response.epistemic_tier == "unverified"
    finally:
        vault.close()


def test_default_respond_abstains_on_multiple_blueprint_matches(tmp_path):
    vault = DiskCrystalVault(tmp_path / "vault.dcrystal")
    try:
        vault.store_blueprint(make_blueprint("choice-a", "A"))
        vault.store_blueprint(make_blueprint("choice-b", "B"))
        response = vault.respond("request 42")
        assert response.mode == "blueprint_prompt_ambiguous"
        assert response.epistemic_tier == "honest_abstain"
    finally:
        vault.close()


def test_old_blueprint_table_is_additively_migrated_for_prompt_dispatch(tmp_path):
    db_path = tmp_path / "legacy-blueprints.dcrystal"
    blueprint = make_blueprint("preexisting", "Stored value {v1}.")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE blueprints (blueprint_id TEXT PRIMARY KEY, domain TEXT NOT NULL, "
            "intent TEXT NOT NULL, kind TEXT NOT NULL, canonical_json TEXT NOT NULL, updated_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO blueprints VALUES (?, ?, ?, ?, ?, ?)",
            (
                blueprint.blueprint_id,
                blueprint.domain,
                blueprint.intent,
                blueprint.kind.value,
                blueprint.canonical_json(),
                1.0,
            ),
        )
        conn.commit()

    vault = DiskCrystalVault(db_path)
    try:
        response = vault.respond("request migrated")
        assert response.mode == "blueprint_composition_unverified"
        assert response.answer == "Stored value migrated."
        columns = {row[1] for row in vault.conn.execute("PRAGMA table_info(blueprints)")}
        assert {"prompt_key", "prompt_prefix"}.issubset(columns)
    finally:
        vault.close()
