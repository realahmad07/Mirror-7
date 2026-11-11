from __future__ import annotations

import hashlib

import pytest

from mirror_crystal.autonomous_research import (
    AutonomousResearchEngine,
    ExperimentDirection,
    ExperimentObservation,
    ExperimentOutcomeStatus,
    ResearchError,
    ResearchExperimentDesign,
    ResearchHypothesis,
    ResearchSourceCandidate,
    ResearchStatus,
    evaluate_experiment,
)
from mirror_crystal.evidence_intelligence import EvidencePassage, SourceProvenance


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_candidate(
    source_id: str = "source-a",
    content: str = "The sample system returns deterministic output for identical inputs.",
    claim: str = "The sample system returns deterministic output for identical inputs.",
) -> ResearchSourceCandidate:
    provenance = SourceProvenance(
        source_id=source_id,
        title=f"Source {source_id}",
        url=f"https://research.example.org/{source_id}",
        publisher="Research Test Publisher",
        license_spdx="CC-BY-4.0",
        retrieved_at="2026-11-11T12:00:00+00:00",
        sha256_hash=digest(content),
        content=content,
    )
    passage = EvidencePassage(
        passage_id=f"{source_id}:p1",
        source_id=source_id,
        locator="Section 1, paragraph 1",
        text=content,
        sha256_hash=digest(content),
        authoritativeness_weight=1.0,
    )
    return ResearchSourceCandidate(provenance, (passage,), (claim,))


def design() -> ResearchExperimentDesign:
    return ResearchExperimentDesign(
        experiment_id="research-experiment-1",
        hypothesis_id="h1",
        protocol_version="protocol-v1",
        protocol_steps=("freeze dataset", "run baseline", "run candidate", "compare metric"),
        independent_variables=("implementation",),
        dependent_variables=("accuracy",),
        controls=("same held-out dataset", "same environment"),
        train_manifest_sha256=digest("train"),
        validation_manifest_sha256=digest("validation"),
        held_out_manifest_sha256=digest("held-out"),
        code_sha256=digest("code"),
        environment_sha256=digest("environment"),
        seeds=(7, 11, 19),
        baseline_name="baseline",
        candidate_name="candidate",
        primary_metric="accuracy",
        direction=ExperimentDirection.MAXIMIZE,
        minimum_effect=0.05,
    )


def observations(current_design: ResearchExperimentDesign) -> tuple[ExperimentObservation, ...]:
    result = []
    for seed in current_design.seeds:
        result.append(ExperimentObservation(
            run_id=f"run-{seed}-baseline",
            seed=seed,
            arm="baseline",
            value=0.70,
            artifact_sha256=digest(f"baseline-{seed}"),
            dataset_manifest_sha256=current_design.held_out_manifest_sha256,
            environment_sha256=current_design.environment_sha256,
        ))
        result.append(ExperimentObservation(
            run_id=f"run-{seed}-candidate",
            seed=seed,
            arm="candidate",
            value=0.80,
            artifact_sha256=digest(f"candidate-{seed}"),
            dataset_manifest_sha256=current_design.held_out_manifest_sha256,
            environment_sha256=current_design.environment_sha256,
        ))
    return tuple(result)


def test_stage18_question_synthesis_is_deterministic_and_falsifiable():
    hypothesis = ResearchHypothesis(
        "h1",
        "The candidate improves held-out accuracy by at least five percentage points.",
        "The candidate improves held-out accuracy by less than five percentage points.",
        "accuracy",
        ExperimentDirection.MAXIMIZE,
    )
    engine = AutonomousResearchEngine()
    first = engine.question_synthesizer.synthesize(
        "Does the candidate improve accuracy?", "Machine Learning", hypotheses=(hypothesis,),
    )
    second = engine.question_synthesizer.synthesize(
        "Does the candidate improve accuracy?", "machine learning", hypotheses=(hypothesis,),
    )
    assert first == second
    assert first.domain == "machine learning"
    assert len(first.subquestions) >= 3
    assert first.hypotheses[0].null_hypothesis.startswith("The candidate improves")


def test_stage18_research_synthesis_preserves_provenance_and_reports_grounded_claims():
    candidate = source_candidate()
    engine = AutonomousResearchEngine()
    report = engine.investigate(
        "Does the sample system return deterministic output for identical inputs?",
        "software verification",
        lambda plan: (candidate,),
    )
    assert report.status is ResearchStatus.EVIDENCE_GATHERED
    assert report.report_id == digest(__import__("mirror_crystal.universal_representation", fromlist=["canonical_json"]).canonical_json({
        "question_plan": report.question_plan.to_dict(),
        "status": report.status.value,
        "source_manifest": list(report.source_manifest),
        "discovery_errors": list(report.discovery_errors),
        "evidence": report.evidence.to_dict(),
        "experiment_design": None,
        "experiment_evaluation": None,
    }))
    assert report.source_manifest[0]["content_sha256"] == candidate.provenance.sha256_hash
    assert report.evidence.verified_facts
    assert report.evidence.provenance_summary[0]["source_provenance"][0]["source_id"] == "source-a"
    assert report.to_dict()["status"] == "evidence_gathered"


def test_stage18_discovery_exception_fails_closed_with_abstention():
    report = AutonomousResearchEngine().investigate(
        "What evidence supports this claim?",
        "general",
        lambda _plan: (_ for _ in ()).throw(RuntimeError("source adapter unavailable")),
    )
    assert report.status is ResearchStatus.ABSTAINED
    assert report.evidence.abstained
    assert report.discovery_errors
    assert "source adapter unavailable" in report.discovery_errors[0]


def test_stage18_rejects_malformed_source_and_duplicate_passage_identifiers():
    with pytest.raises(Exception, match="digest mismatch"):
        source_candidate(content="tampered", claim="tampered")
    first = source_candidate("source-1")
    second = source_candidate("source-2")
    reused_passage = EvidencePassage(
        passage_id=first.passages[0].passage_id,
        source_id=second.provenance.source_id,
        locator="Section 1",
        text=second.passages[0].text,
        sha256_hash=digest(second.passages[0].text),
    )
    second = ResearchSourceCandidate(second.provenance, (reused_passage,), second.claims)
    with pytest.raises(ResearchError, match="globally unique"):
        AutonomousResearchEngine().investigate(
            "Any claim?", "general", lambda _plan: (first, second),
        )


def test_stage18_experiment_evaluation_requires_paired_fresh_seed_runs_and_minimum_effect():
    current = design()
    evaluation = evaluate_experiment(current, observations(current))
    assert evaluation.status is ExperimentOutcomeStatus.SUPPORTED
    assert len(evaluation.per_seed_deltas) == 3
    assert evaluation.mean_delta == pytest.approx(0.1)
    assert len(evaluation.observation_fingerprints) == 6

    weaker = tuple(
        ExperimentObservation(
            item.run_id, item.seed, item.arm,
            0.70 if item.arm == "baseline" else 0.72,
            item.artifact_sha256, item.dataset_manifest_sha256, item.environment_sha256,
        )
        for item in observations(current)
    )
    rejected = evaluate_experiment(current, weaker)
    assert rejected.status is ExperimentOutcomeStatus.REJECTED


def test_stage18_experiment_evaluation_rejects_missing_seed_duplicate_run_and_wrong_environment():
    current = design()
    items = observations(current)
    with pytest.raises(ResearchError, match="every declared seed"):
        evaluate_experiment(current, items[:-1])
    with pytest.raises(ResearchError, match="run identifiers"):
        evaluate_experiment(current, items[:-1] + (items[0],))
    wrong_environment = (
        *items[:-1],
        ExperimentObservation(
            items[-1].run_id, items[-1].seed, items[-1].arm, items[-1].value,
            items[-1].artifact_sha256, items[-1].dataset_manifest_sha256, digest("wrong"),
        ),
    )
    with pytest.raises(ResearchError, match="environment"):
        evaluate_experiment(current, wrong_environment)


def test_stage18_experiment_design_rejects_reused_splits_and_unseeded_runs():
    base = design().to_dict()
    base["validation_manifest_sha256"] = base["train_manifest_sha256"]
    with pytest.raises(ResearchError, match="must be distinct"):
        ResearchExperimentDesign(**base)
    base = design().to_dict()
    base["seeds"] = (7, 7, 11)
    with pytest.raises(ResearchError, match="distinct"):
        ResearchExperimentDesign(**base)


def test_stage18_research_requires_hypothesis_link_for_experiment_design():
    current = design()
    with pytest.raises(ResearchError, match="hypothesis must be present"):
        AutonomousResearchEngine().investigate(
            "Does the candidate improve?",
            "machine learning",
            lambda _plan: (),
            experiment_design=current,
        )
