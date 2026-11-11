"""Phase 8: sealed, verifier-backed cross-domain transfer acceptance.

This harness composes the existing transfer and verified-learning gates. Every
target-domain candidate is independently checked and learned separately; source
trust is never copied into a target domain. It reports a bounded benchmark result,
not unrestricted semantic transfer.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Callable, Iterable

from .capability_synthesizer import Example
from .cross_domain_transfer import (
    CrossDomainTransferEngine,
    CrossDomainTransferError,
    CrossDomainTransferProposal,
)
from .operational_language import UOL, _OPS, _walk
from .universal_representation import stable_representation_id
from .verified_program_learning import (
    UOLProgramLearningProposal,
    VerifiedProgramLearningEngine,
    VerifiedProgramObservation,
)


class TransferBenchmarkError(ValueError):
    """Raised when a transfer benchmark definition is malformed."""


class TransferBenchmarkStatus(str, Enum):
    PASS = "pass"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class TransferTargetBenchmark:
    domain: str
    training: tuple[Example, ...]
    held_out: tuple[Example, ...]
    postconditions: tuple[str, ...]
    observations: Callable[[CrossDomainTransferProposal], Iterable[VerifiedProgramObservation]]

    def __post_init__(self) -> None:
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise TransferBenchmarkError("target domain must be non-empty")
        if not callable(self.observations):
            raise TypeError("observations must be a target-verifier callback")
        object.__setattr__(self, "domain", self.domain.strip().casefold())
        object.__setattr__(self, "training", tuple(self.training))
        object.__setattr__(self, "held_out", tuple(self.held_out))
        object.__setattr__(self, "postconditions", tuple(self.postconditions))


@dataclass(frozen=True)
class TransferTargetResult:
    domain: str
    transfer: CrossDomainTransferProposal | None
    learned: UOLProgramLearningProposal | None
    task_ids: tuple[str, ...]
    execution_fingerprints: tuple[str, ...]
    blocker: str = ""

    @property
    def passed(self) -> bool:
        return self.transfer is not None and self.learned is not None and not self.blocker


@dataclass(frozen=True)
class CrossDomainTransferBenchmarkReport:
    benchmark_id: str
    source_program_fingerprint: str
    source_domain: str
    status: TransferBenchmarkStatus
    targets: tuple[TransferTargetResult, ...]
    minimum_target_domains: int
    manifest_sha256: str
    blockers: tuple[str, ...]
    version: str = "strategic-phase8-v1"

    @property
    def passed_domains(self) -> tuple[str, ...]:
        return tuple(item.domain for item in self.targets if item.passed)

    @property
    def learning_proposals(self) -> tuple[UOLProgramLearningProposal, ...]:
        return tuple(item.learned for item in self.targets if item.learned is not None)


def _valid_digest(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_source(source: UOLProgramLearningProposal, language: UOL) -> None:
    """Reject manually assembled or stale source proposals before transferring trust."""
    if not isinstance(source, UOLProgramLearningProposal) or not isinstance(language, UOL):
        raise TypeError("source and language must be a learning proposal and UOL")
    capability, program, handler = source.capability, source.program, source.handler
    if not source.requires_explicit_promotion or not capability.executable:
        raise TransferBenchmarkError("source must be an independently learned capability proposal")
    if capability.verification.level.value not in {"formally_verified", "independently_reproduced"}:
        raise TransferBenchmarkError("source capability is not independently verified")
    if capability.provenance.source != "mirror_verified_uol_program_learning":
        raise TransferBenchmarkError("source capability lacks verified-program-learning provenance")
    if capability.provenance.source_ref != program.fingerprint:
        raise TransferBenchmarkError("source capability provenance does not bind the exact program")
    if program.domain not in capability.domains:
        raise TransferBenchmarkError("source program domain is not declared by its capability")
    if getattr(handler, "program", None) != program:
        raise TransferBenchmarkError("source handler is not bound to the exact source program")
    if (
        isinstance(source.minimum_successes, bool)
        or not isinstance(source.minimum_successes, int)
        or source.minimum_successes < 3
    ):
        raise TransferBenchmarkError("source evidence must require at least three distinct observations")
    if (
        isinstance(source.minimum_verifiers, bool)
        or not isinstance(source.minimum_verifiers, int)
        or source.minimum_verifiers < 2
    ):
        raise TransferBenchmarkError("source evidence must require at least two verifier identities")
    if len(source.observation_task_ids) < source.minimum_successes:
        raise TransferBenchmarkError("source proposal is missing required task observations")
    if len(set(source.observation_task_ids)) != len(source.observation_task_ids):
        raise TransferBenchmarkError("source task observations must be distinct")
    if len(source.execution_fingerprints) < source.minimum_successes:
        raise TransferBenchmarkError("source proposal is missing execution fingerprints")
    if len(set(source.execution_fingerprints)) != len(source.execution_fingerprints):
        raise TransferBenchmarkError("source execution fingerprints must be distinct")
    if any(not _valid_digest(item) for item in source.execution_fingerprints):
        raise TransferBenchmarkError("source execution fingerprints must be SHA-256 digests")
    if len(source.evidence_fingerprints) < source.minimum_successes:
        raise TransferBenchmarkError("source proposal is missing verification evidence fingerprints")
    if any(not _valid_digest(item) for item in source.evidence_fingerprints):
        raise TransferBenchmarkError("source evidence fingerprints must be SHA-256 digests")
    if len(set(source.verifier_ids)) < source.minimum_verifiers:
        raise TransferBenchmarkError("source proposal lacks distinct verifier identities")
    if not capability.verification.checks_passed or not capability.verification.verifier.strip():
        raise TransferBenchmarkError("source capability has no recorded passing verification")
    language.validate(program)
    handler_rules = {
        item.rule_id: item for item in getattr(handler, "rules", ())
    }
    language_rules = {
        item.rule_id: item for item in language.rules.values()
    }
    pending = list(program.rule_refs)
    visited: set[str] = set()
    while pending:
        rule_id = pending.pop()
        if rule_id in visited:
            continue
        visited.add(rule_id)
        source_rule = handler_rules.get(rule_id)
        transfer_rule = language_rules.get(rule_id)
        if source_rule is None or transfer_rule is None or source_rule != transfer_rule:
            raise TransferBenchmarkError(
                f"source handler and transfer language disagree about rule: {rule_id}"
            )
        if source_rule.status.value != "verified":
            raise TransferBenchmarkError("source program rule dependency is not independently verified")
        for node in _walk(source_rule.body):
            operator = getattr(node, "operator", None)
            if operator and operator not in _OPS:
                pending.append(operator)


class CrossDomainTransferBenchmarkEngine:
    """Evaluate one verified source across multiple separately verified target domains."""

    VERSION = "strategic-phase8-v1"
    MAX_TARGET_DOMAINS = 5
    MAX_EXAMPLES_PER_TARGET = 10000

    def __init__(
        self,
        transfer_engine: CrossDomainTransferEngine | None = None,
        learning_engine: VerifiedProgramLearningEngine | None = None,
    ) -> None:
        self.transfer_engine = transfer_engine or CrossDomainTransferEngine()
        self.learning_engine = learning_engine or VerifiedProgramLearningEngine()

    def evaluate(
        self,
        benchmark_id: str,
        source: UOLProgramLearningProposal,
        language: UOL,
        targets: Iterable[TransferTargetBenchmark],
        *,
        minimum_target_domains: int = 2,
        minimum_successes: int = 3,
        minimum_verifiers: int = 2,
    ) -> CrossDomainTransferBenchmarkReport:
        if not isinstance(benchmark_id, str) or not benchmark_id.strip():
            raise TransferBenchmarkError("benchmark_id must be non-empty")
        if isinstance(minimum_target_domains, bool) or not isinstance(minimum_target_domains, int) or minimum_target_domains < 1:
            raise ValueError("minimum_target_domains must be a positive integer")
        if isinstance(minimum_successes, bool) or not isinstance(minimum_successes, int) or minimum_successes < 3:
            raise ValueError("minimum_successes must be at least 3")
        if isinstance(minimum_verifiers, bool) or not isinstance(minimum_verifiers, int) or minimum_verifiers < 2:
            raise ValueError("minimum_verifiers must be at least 2")
        _validate_source(source, language)

        target_items = tuple(targets)
        domains = tuple(item.domain for item in target_items if isinstance(item, TransferTargetBenchmark))
        if len(target_items) != len(domains):
            raise TypeError("targets must contain TransferTargetBenchmark values")
        if not minimum_target_domains <= len(target_items) <= self.MAX_TARGET_DOMAINS:
            raise TransferBenchmarkError("target count is outside the configured benchmark bounds")
        if len(set(domains)) != len(domains):
            raise TransferBenchmarkError("each target domain may appear only once")
        if source.program.domain in domains:
            raise TransferBenchmarkError("a target domain cannot equal the source domain")

        manifest_targets = []
        for target in target_items:
            if len(target.training) + len(target.held_out) > self.MAX_EXAMPLES_PER_TARGET:
                raise TransferBenchmarkError("target benchmark exceeds the per-domain example budget")
            manifest_targets.append({
                "domain": target.domain,
                "training": [
                    stable_representation_id("phase8-example", {"inputs": dict(item.inputs), "expected": item.expected})
                    for item in target.training
                ],
                "held_out": [
                    stable_representation_id("phase8-example", {"inputs": dict(item.inputs), "expected": item.expected})
                    for item in target.held_out
                ],
                "postconditions": list(target.postconditions),
            })
        manifest_sha256 = stable_representation_id(
            "phase8-transfer-manifest",
            {
                "version": self.VERSION,
                "benchmark_id": benchmark_id.strip(),
                "source_program": source.program.fingerprint,
                "source_domain": source.program.domain,
                "targets": manifest_targets,
            },
        )

        results: list[TransferTargetResult] = []
        all_task_ids: set[str] = set()
        all_execution_ids: set[str] = set()
        blockers: list[str] = []
        all_sample_ids: set[str] = set()
        for target in target_items:
            try:
                transfer = self.transfer_engine.propose(
                    source,
                    language,
                    target_domain=target.domain,
                    training=target.training,
                    validation=target.held_out,
                    target_postconditions=target.postconditions,
                )
                observations = tuple(target.observations(transfer))
                task_ids = tuple(item.task_id for item in observations)
                execution_ids = tuple(item.execution.fingerprint for item in observations)
                if any(not item or item in all_task_ids for item in task_ids):
                    raise TransferBenchmarkError("task identifiers must be non-empty and unique across target domains")
                if any(not item or item in all_execution_ids for item in execution_ids):
                    raise TransferBenchmarkError("execution fingerprints must be unique across target domains")
                used_inputs = {
                    stable_representation_id("transfer-input", dict(item.inputs))
                    for item in (*target.training, *target.held_out)
                }
                for observation in observations:
                    if len(observation.execution.steps) != 1:
                        raise TransferBenchmarkError("each target verification observation must cover exactly one step")
                    fresh_inputs = {
                        key: value
                        for key, value in observation.execution.steps[0].inputs.items()
                        if key != "__mirror_verification__"
                    }
                    sample_id = stable_representation_id("transfer-input", fresh_inputs)
                    if sample_id in used_inputs or sample_id in all_sample_ids:
                        raise TransferBenchmarkError(
                            "target verification observations must be disjoint from transfer train/held-out data"
                        )
                    used_inputs.add(sample_id)
                    all_sample_ids.add(sample_id)
                all_task_ids.update(task_ids)
                all_execution_ids.update(execution_ids)
                learned = self.learning_engine.propose(
                    transfer.program,
                    observations,
                    language,
                    intents=source.capability.intents,
                    minimum_successes=minimum_successes,
                    minimum_verifiers=minimum_verifiers,
                )
                if learned.capability.domains != (target.domain,):
                    raise TransferBenchmarkError("learned target capability has the wrong domain")
                results.append(TransferTargetResult(
                    target.domain, transfer, learned, task_ids, execution_ids,
                ))
            except Exception as exc:
                reason = f"{target.domain}: {type(exc).__name__}: {str(exc)[:400]}"
                blockers.append(reason)
                results.append(TransferTargetResult(
                    target.domain, None, None, (), (), reason,
                ))

        passed = len(results) == len(target_items) and all(item.passed for item in results)
        if len(results) < minimum_target_domains:
            blockers.append("too few target domains were evaluated")
            passed = False
        if not passed and not blockers:
            blockers.append("one or more target-domain transfer gates failed")
        return CrossDomainTransferBenchmarkReport(
            benchmark_id.strip(),
            source.program.fingerprint,
            source.program.domain,
            TransferBenchmarkStatus.PASS if passed else TransferBenchmarkStatus.BLOCKED,
            tuple(results),
            minimum_target_domains,
            manifest_sha256,
            tuple(blockers),
            self.VERSION,
        )
