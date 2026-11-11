"""Stage 18: auditable autonomous research planning and evidence synthesis.

Research-source discovery is an adapter boundary: the engine does not silently
crawl the network. It accepts retrieved source records, verifies their content
hashes and passage bindings, then delegates claim grounding and contradiction
screening to the Stage 14 evidence engine. Experiment designs and result records
are frozen and reproducible; finite benchmark results are not proofs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
import statistics
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from .evidence_intelligence import (
    AtomicClaim,
    EvidenceBasedIntelligenceEngine,
    EvidencePassage,
    GroundedSynthesisReport,
    SourceProvenance,
)
from .universal_representation import canonical_json


class ResearchError(ValueError):
    """Raised when a research question, source bundle or experiment contract is invalid."""


class ResearchStatus(str, Enum):
    ABSTAINED = "abstained"
    EVIDENCE_GATHERED = "evidence_gathered"
    CONFLICTED = "conflicted"


class ExperimentDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ExperimentOutcomeStatus(str, Enum):
    SUPPORTED = "supported"
    REJECTED = "rejected"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_text(value: str, field: str, *, max_length: int = 10_000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchError(f"{field} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > max_length:
        raise ResearchError(f"{field} exceeds the {max_length}-character limit")
    return normalized


def _bounded_tuple(values: Iterable[Any], field: str, *, maximum: int) -> tuple[Any, ...]:
    """Consume a caller-provided iterable without allowing unbounded materialization."""
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise TypeError(f"{field} must be iterable") from exc
    items: list[Any] = []
    for item in iterator:
        if len(items) >= maximum:
            raise ResearchError(f"{field} exceeds the {maximum}-item limit")
        items.append(item)
    return tuple(items)


@dataclass(frozen=True)
class ResearchHypothesis:
    hypothesis_id: str
    statement: str
    null_hypothesis: str
    metric: str
    expected_direction: ExperimentDirection

    def __post_init__(self) -> None:
        for field_name in ("hypothesis_id", "statement", "null_hypothesis", "metric"):
            limit = 128 if field_name == "hypothesis_id" else (256 if field_name == "metric" else 4_000)
            object.__setattr__(self, field_name, _require_text(
                getattr(self, field_name), field_name, max_length=limit,
            ))
        object.__setattr__(self, "expected_direction", ExperimentDirection(self.expected_direction))


@dataclass(frozen=True)
class ResearchQuestionPlan:
    research_id: str
    domain: str
    primary_question: str
    objectives: tuple[str, ...]
    subquestions: tuple[str, ...]
    hypotheses: tuple[ResearchHypothesis, ...]
    version: str = "stage18-question-plan-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "research_id": self.research_id,
            "domain": self.domain,
            "primary_question": self.primary_question,
            "objectives": list(self.objectives),
            "subquestions": list(self.subquestions),
            "hypotheses": [
                {
                    **asdict(item),
                    "expected_direction": item.expected_direction.value,
                }
                for item in self.hypotheses
            ],
            "version": self.version,
        }


class ResearchQuestionSynthesizer:
    """Turn a broad research request into an explicit question and testable work plan."""

    DEFAULT_OBJECTIVES = (
        "identify source-backed evidence directly relevant to the question",
        "compare independent sources and expose material contradictions",
        "separate directly grounded claims from inference and assumption",
        "define a falsifiable experiment with reproducible inputs and metrics",
    )

    def synthesize(
        self,
        query: str,
        domain: str,
        *,
        objectives: Iterable[str] = (),
        hypotheses: Iterable[ResearchHypothesis] = (),
    ) -> ResearchQuestionPlan:
        question = _require_text(query, "query", max_length=10_000)
        area = _require_text(domain, "domain", max_length=256).casefold()
        supplied_objectives = _bounded_tuple(objectives, "objectives", maximum=12)
        chosen_objectives = tuple(
            _require_text(item, "objective", max_length=512)
            for item in (supplied_objectives or self.DEFAULT_OBJECTIVES)
        )
        if len(set(chosen_objectives)) != len(chosen_objectives):
            raise ResearchError("objectives must be unique")
        hypothesis_items = _bounded_tuple(hypotheses, "hypotheses", maximum=32)
        if any(not isinstance(item, ResearchHypothesis) for item in hypothesis_items):
            raise TypeError("hypotheses must contain ResearchHypothesis values")
        if len({item.hypothesis_id for item in hypothesis_items}) != len(hypothesis_items):
            raise ResearchError("hypothesis identifiers must be unique")
        subquestions = (
            f"What primary-source evidence directly addresses: {question}",
            f"Which retrieved sources independently corroborate or contradict claims about: {question}",
            f"What assumptions, limitations and alternative explanations affect the evidence in {area}",
            f"What observable result would falsify the stated hypothesis about: {question}",
        )
        payload = {
            "domain": area,
            "question": question,
            "objectives": list(chosen_objectives),
            "subquestions": list(subquestions),
            "hypotheses": [
                {
                    "id": item.hypothesis_id,
                    "statement": item.statement,
                    "null": item.null_hypothesis,
                    "metric": item.metric,
                    "direction": item.expected_direction.value,
                }
                for item in hypothesis_items
            ],
        }
        return ResearchQuestionPlan(
            _sha256(canonical_json(payload)), area, question,
            chosen_objectives, subquestions, hypothesis_items,
        )


@dataclass(frozen=True)
class ResearchSourceCandidate:
    provenance: SourceProvenance
    passages: tuple[EvidencePassage, ...]
    claims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, SourceProvenance):
            raise TypeError("provenance must be SourceProvenance")
        passages = _bounded_tuple(self.passages, "source passages", maximum=100)
        if any(not isinstance(item, EvidencePassage) for item in passages):
            raise TypeError("passages must contain EvidencePassage values")
        if any(item.source_id != self.provenance.source_id for item in passages):
            raise ResearchError("every passage must be bound to the candidate source")
        claims = tuple(
            _require_text(item, "claim", max_length=5_000)
            for item in _bounded_tuple(self.claims, "source claims", maximum=100)
        )
        if len(set(claims)) != len(claims):
            raise ResearchError("source claim statements must be unique")
        object.__setattr__(self, "passages", passages)
        object.__setattr__(self, "claims", claims)


@dataclass(frozen=True)
class ResearchExperimentDesign:
    experiment_id: str
    hypothesis_id: str
    protocol_version: str
    protocol_steps: tuple[str, ...]
    independent_variables: tuple[str, ...]
    dependent_variables: tuple[str, ...]
    controls: tuple[str, ...]
    train_manifest_sha256: str
    validation_manifest_sha256: str
    held_out_manifest_sha256: str
    code_sha256: str
    environment_sha256: str
    seeds: tuple[int, ...]
    baseline_name: str
    candidate_name: str
    primary_metric: str
    direction: ExperimentDirection
    minimum_effect: float

    def __post_init__(self) -> None:
        for name in (
            "experiment_id", "hypothesis_id", "protocol_version",
            "baseline_name", "candidate_name", "primary_metric",
        ):
            object.__setattr__(self, name, _require_text(
                getattr(self, name), name, max_length=512,
            ))
        for name in ("protocol_steps", "independent_variables", "dependent_variables", "controls"):
            values = tuple(
                _require_text(item, name, max_length=1_024)
                for item in _bounded_tuple(getattr(self, name), name, maximum=64)
            )
            if not values or len(set(values)) != len(values):
                raise ResearchError(f"{name} must be non-empty and unique")
            object.__setattr__(self, name, values)
        for name in (
            "train_manifest_sha256", "validation_manifest_sha256",
            "held_out_manifest_sha256", "code_sha256", "environment_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ResearchError(f"{name} must be a lowercase SHA-256 digest")
        split_digests = (
            self.train_manifest_sha256,
            self.validation_manifest_sha256,
            self.held_out_manifest_sha256,
        )
        if len(set(split_digests)) != 3:
            raise ResearchError("train, validation and held-out manifests must be distinct")
        seeds = _bounded_tuple(self.seeds, "seeds", maximum=64)
        if len(seeds) < 3 or any(isinstance(item, bool) or not isinstance(item, int) for item in seeds):
            raise ResearchError("at least three integer random seeds are required")
        if len(set(seeds)) != len(seeds):
            raise ResearchError("experiment random seeds must be distinct")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "direction", ExperimentDirection(self.direction))
        if isinstance(self.minimum_effect, bool) or not isinstance(self.minimum_effect, (int, float)):
            raise ResearchError("minimum_effect must be numeric")
        if not math.isfinite(float(self.minimum_effect)) or self.minimum_effect < 0:
            raise ResearchError("minimum_effect must be finite and non-negative")

    @property
    def fingerprint(self) -> str:
        return _sha256(canonical_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["direction"] = self.direction.value
        value["seeds"] = list(self.seeds)
        return value


@dataclass(frozen=True)
class ExperimentObservation:
    run_id: str
    seed: int
    arm: str
    value: float
    artifact_sha256: str
    dataset_manifest_sha256: str
    environment_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id", max_length=256))
        arm = _require_text(self.arm, "arm").casefold()
        if arm not in {"baseline", "candidate"}:
            raise ResearchError("arm must be baseline or candidate")
        object.__setattr__(self, "arm", arm)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ResearchError("seed must be an integer")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)) or not math.isfinite(float(self.value)):
            raise ResearchError("observation value must be finite numeric")
        object.__setattr__(self, "value", float(self.value))
        for name in ("artifact_sha256", "dataset_manifest_sha256", "environment_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ResearchError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ExperimentEvaluation:
    experiment_id: str
    design_fingerprint: str
    status: ExperimentOutcomeStatus
    per_seed_deltas: tuple[tuple[int, float], ...]
    mean_delta: float
    sample_stddev: float
    minimum_effect: float
    observation_fingerprints: tuple[str, ...]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "design_fingerprint": self.design_fingerprint,
            "status": self.status.value,
            "per_seed_deltas": [list(item) for item in self.per_seed_deltas],
            "mean_delta": self.mean_delta,
            "sample_stddev": self.sample_stddev,
            "minimum_effect": self.minimum_effect,
            "observation_fingerprints": list(self.observation_fingerprints),
            "explanation": self.explanation,
        }


def evaluate_experiment(
    design: ResearchExperimentDesign,
    observations: Iterable[ExperimentObservation],
) -> ExperimentEvaluation:
    if not isinstance(design, ResearchExperimentDesign):
        raise TypeError("design must be ResearchExperimentDesign")
    items = _bounded_tuple(observations, "experiment observations", maximum=128)
    if any(not isinstance(item, ExperimentObservation) for item in items):
        raise TypeError("observations must contain ExperimentObservation values")
    expected_keys = {(seed, arm) for seed in design.seeds for arm in ("baseline", "candidate")}
    found_keys = [(item.seed, item.arm) for item in items]
    if len(found_keys) != len(set(found_keys)):
        raise ResearchError("each seed/arm pair must appear exactly once")
    if set(found_keys) != expected_keys:
        raise ResearchError("experiment observations must cover every declared seed and both arms")
    if len({item.run_id for item in items}) != len(items):
        raise ResearchError("experiment run identifiers must be unique")
    if len({item.artifact_sha256 for item in items}) != len(items):
        raise ResearchError("experiment artifacts must have unique SHA-256 digests")
    for item in items:
        if item.dataset_manifest_sha256 != design.held_out_manifest_sha256:
            raise ResearchError("experiment observation uses an unexpected held-out dataset manifest")
        if item.environment_sha256 != design.environment_sha256:
            raise ResearchError("experiment observation environment does not match the design")
    indexed = {(item.seed, item.arm): item for item in items}
    deltas = []
    for seed in design.seeds:
        baseline = indexed[(seed, "baseline")].value
        candidate = indexed[(seed, "candidate")].value
        delta = candidate - baseline if design.direction is ExperimentDirection.MAXIMIZE else baseline - candidate
        deltas.append((seed, round(delta, 12)))
    values = [value for _, value in deltas]
    mean_delta = statistics.mean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    passed = all(value >= design.minimum_effect for value in values)
    return ExperimentEvaluation(
        design.experiment_id,
        design.fingerprint,
        ExperimentOutcomeStatus.SUPPORTED if passed else ExperimentOutcomeStatus.REJECTED,
        tuple(deltas),
        round(mean_delta, 12),
        round(stddev, 12),
        float(design.minimum_effect),
        tuple(sorted(item.artifact_sha256 for item in items)),
        (
            "Candidate met the predeclared minimum effect for every paired random seed."
            if passed else
            "Candidate failed the predeclared minimum effect for at least one paired random seed."
        ),
    )


@dataclass(frozen=True)
class ResearchReport:
    report_id: str
    question_plan: ResearchQuestionPlan
    status: ResearchStatus
    evidence: GroundedSynthesisReport
    source_manifest: tuple[Mapping[str, Any], ...]
    discovery_errors: tuple[str, ...]
    experiment_design: ResearchExperimentDesign | None
    experiment_evaluation: ExperimentEvaluation | None
    created_at: str
    version: str = "stage18-research-report-v1"

    def __post_init__(self) -> None:
        frozen_entries = []
        for entry in self.source_manifest:
            if not isinstance(entry, Mapping):
                raise TypeError("source_manifest entries must be mappings")
            copied = dict(entry)
            copied["passage_ids"] = tuple(copied.get("passage_ids", ()))
            frozen_entries.append(MappingProxyType(copied))
        object.__setattr__(self, "source_manifest", tuple(frozen_entries))
        object.__setattr__(self, "discovery_errors", tuple(self.discovery_errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "question_plan": self.question_plan.to_dict(),
            "status": self.status.value,
            "evidence": self.evidence.to_dict(),
            "source_manifest": [
                {**dict(item), "passage_ids": list(item.get("passage_ids", ()))}
                for item in self.source_manifest
            ],
            "discovery_errors": list(self.discovery_errors),
            "experiment_design": self.experiment_design.to_dict() if self.experiment_design else None,
            "experiment_evaluation": self.experiment_evaluation.to_dict() if self.experiment_evaluation else None,
            "created_at": self.created_at,
            "version": self.version,
        }


class AutonomousResearchEngine:
    """Question framing, adapter-backed source discovery and auditable research synthesis."""

    MAX_SOURCES = 50
    MAX_SOURCE_BYTES = 5_000_000
    MAX_PASSAGES = 500
    MAX_CLAIMS = 500
    MAX_EVIDENCE_BYTES = 10_000_000

    def __init__(
        self,
        evidence_engine: EvidenceBasedIntelligenceEngine | None = None,
        question_synthesizer: ResearchQuestionSynthesizer | None = None,
    ) -> None:
        self.evidence_engine = evidence_engine or EvidenceBasedIntelligenceEngine()
        self.question_synthesizer = question_synthesizer or ResearchQuestionSynthesizer()

    def investigate(
        self,
        query: str,
        domain: str,
        source_discoverer: Callable[[ResearchQuestionPlan], Iterable[ResearchSourceCandidate]],
        *,
        objectives: Iterable[str] = (),
        hypotheses: Iterable[ResearchHypothesis] = (),
        experiment_design: ResearchExperimentDesign | None = None,
        experiment_observations: Iterable[ExperimentObservation] = (),
        confidence_threshold: float = 0.55,
    ) -> ResearchReport:
        if not callable(source_discoverer):
            raise TypeError("source_discoverer must be callable")
        if isinstance(confidence_threshold, bool) or not isinstance(confidence_threshold, (int, float)):
            raise ResearchError("confidence_threshold must be numeric")
        if not math.isfinite(float(confidence_threshold)) or not 0 <= confidence_threshold <= 1:
            raise ResearchError("confidence_threshold must be between zero and one")
        plan = self.question_synthesizer.synthesize(
            query, domain, objectives=objectives, hypotheses=hypotheses,
        )
        discovery_errors: list[str] = []
        candidates_list: list[ResearchSourceCandidate] = []
        total_source_bytes = 0
        total_evidence_bytes = 0
        total_passages = 0
        total_claims = 0
        try:
            discovered = source_discoverer(plan)
            iterator = iter(discovered)
        except Exception as exc:
            iterator = iter(())
            discovery_errors.append(f"{type(exc).__name__}: {str(exc)[:500]}")
        while True:
            try:
                candidate = next(iterator)
            except StopIteration:
                break
            except Exception as exc:
                discovery_errors.append(f"{type(exc).__name__}: {str(exc)[:500]}")
                break
            if len(candidates_list) >= self.MAX_SOURCES:
                raise ResearchError("source discovery exceeded the 50-source budget")
            if not isinstance(candidate, ResearchSourceCandidate):
                raise TypeError("source discovery must return ResearchSourceCandidate values")
            source_content_bytes = len(candidate.provenance.content.encode("utf-8"))
            passage_bytes = sum(len(item.text.encode("utf-8")) for item in candidate.passages)
            claim_bytes = sum(len(item.encode("utf-8")) for item in candidate.claims)
            total_source_bytes += source_content_bytes
            total_evidence_bytes += passage_bytes + claim_bytes
            total_passages += len(candidate.passages)
            total_claims += len(candidate.claims)
            if total_source_bytes > self.MAX_SOURCE_BYTES:
                raise ResearchError("source discovery exceeded the 5 MB source-content budget")
            if total_source_bytes + total_evidence_bytes > self.MAX_EVIDENCE_BYTES:
                raise ResearchError("source discovery exceeded the 10 MB total evidence-input budget")
            if total_passages > self.MAX_PASSAGES:
                raise ResearchError("source discovery exceeded the 500-passage budget")
            if total_claims > self.MAX_CLAIMS:
                raise ResearchError("source discovery exceeded the 500-explicit-claim budget")
            candidates_list.append(candidate)
        candidates = tuple(candidates_list)
        source_ids = tuple(item.provenance.source_id for item in candidates)
        if len(set(source_ids)) != len(source_ids):
            raise ResearchError("source discovery returned duplicate source identifiers")

        sources = [item.provenance for item in candidates]
        passages = [passage for item in candidates for passage in item.passages]
        passage_ids = tuple(item.passage_id for item in passages)
        if len(set(passage_ids)) != len(passage_ids):
            raise ResearchError("passage identifiers must be globally unique")

        claims: list[AtomicClaim] = []
        claim_counts: dict[str, int] = {}
        for candidate in candidates:
            statements = candidate.claims
            if not statements:
                statements = tuple(
                    claim.statement
                    for passage in candidate.passages
                    for claim in self.evidence_engine.decompose_statements(passage.text)
                )
            if len(claims) + len(statements) > self.MAX_CLAIMS:
                raise ResearchError("research claim extraction exceeded the 500-claim budget")
            claim_counts[candidate.provenance.source_id] = len(statements)
            for index, statement in enumerate(statements):
                claims.append(AtomicClaim(
                    claim_id=f"{candidate.provenance.source_id}:claim:{index + 1}",
                    statement=statement,
                ))

        evidence = self.evidence_engine.process_and_synthesize(
            plan.primary_question,
            claims,
            sources,
            passages,
            confidence_threshold=float(confidence_threshold),
        )
        if not candidates or evidence.abstained:
            status = ResearchStatus.ABSTAINED
        elif evidence.contradictions_detected:
            status = ResearchStatus.CONFLICTED
        else:
            status = ResearchStatus.EVIDENCE_GATHERED

        if experiment_design is not None:
            if not isinstance(experiment_design, ResearchExperimentDesign):
                raise TypeError("experiment_design must be ResearchExperimentDesign")
            if not any(item.hypothesis_id == experiment_design.hypothesis_id for item in plan.hypotheses):
                raise ResearchError("experiment design hypothesis must be present in the question plan")
            observations = _bounded_tuple(
                experiment_observations, "experiment observations", maximum=128,
            )
            experiment_evaluation = (
                evaluate_experiment(experiment_design, observations) if observations else None
            )
        else:
            if _bounded_tuple(
                experiment_observations, "experiment observations", maximum=128,
            ):
                raise ResearchError("experiment observations require an experiment design")
            experiment_evaluation = None

        source_manifest = tuple({
            "source_id": item.provenance.source_id,
            "title": item.provenance.title,
            "url": item.provenance.url,
            "publisher": item.provenance.publisher,
            "license_spdx": item.provenance.license_spdx,
            "retrieved_at": item.provenance.retrieved_at,
            "content_sha256": item.provenance.sha256_hash,
            "passage_ids": [passage.passage_id for passage in item.passages],
            "claim_count": claim_counts[item.provenance.source_id],
        } for item in candidates)
        report_payload = {
            "question_plan": plan.to_dict(),
            "status": status.value,
            "source_manifest": list(source_manifest),
            "discovery_errors": list(discovery_errors),
            "evidence": evidence.to_dict(),
            "experiment_design": experiment_design.to_dict() if experiment_design else None,
            "experiment_evaluation": experiment_evaluation.to_dict() if experiment_evaluation else None,
        }
        from datetime import datetime, timezone
        created_at = datetime.now(timezone.utc).isoformat()
        report_id = _sha256(canonical_json(report_payload))
        return ResearchReport(
            report_id, plan, status, evidence, source_manifest,
            tuple(discovery_errors), experiment_design, experiment_evaluation, created_at,
        )
