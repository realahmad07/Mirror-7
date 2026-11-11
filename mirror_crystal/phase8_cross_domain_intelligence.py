"""Sealed cross-domain transfer evaluation with target-domain verification (Phase 8)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping
from .capability_synthesizer import Example
from .cross_domain_transfer import CrossDomainTransferEngine, CrossDomainTransferError, CrossDomainTransferProposal
from .operational_language import UOL
from .universal_representation import stable_representation_id
from .verified_program_learning import (
    ProgramLearningError, UOLProgramLearningProposal, VerifiedProgramLearningEngine,
    VerifiedProgramObservation,
)

class CrossDomainBenchmarkError(ValueError):
    """Malformed or insufficient cross-domain benchmark."""

@dataclass(frozen=True)
class CrossDomainBenchmarkCase:
    case_id: str
    source: UOLProgramLearningProposal
    language: UOL
    target_domain: str
    training: tuple[Example, ...]
    validation: tuple[Example, ...]
    target_postconditions: tuple[str, ...]
    expect_transfer: bool = True
    expected_rejection: str = ""

    def __post_init__(self):
        if not self.case_id.strip() or not self.target_domain.strip():
            raise CrossDomainBenchmarkError("case and target identifiers must not be empty")
        if not isinstance(self.source, UOLProgramLearningProposal) or not isinstance(self.language, UOL):
            raise TypeError("source and language have invalid types")
        training, validation = tuple(self.training), tuple(self.validation)
        if not training or not validation or any(not isinstance(x, Example) for x in training + validation):
            raise CrossDomainBenchmarkError("each case needs typed, non-empty train and held-out examples")
        if not isinstance(self.expect_transfer, bool):
            raise CrossDomainBenchmarkError("expect_transfer must be boolean")
        post = tuple(str(x).strip() for x in self.target_postconditions)
        if self.expect_transfer and (not post or any(not x for x in post)):
            raise CrossDomainBenchmarkError("positive cases require target postconditions")
        if not self.expect_transfer and not self.expected_rejection.strip():
            raise CrossDomainBenchmarkError("negative controls require an expected rejection")
        object.__setattr__(self, "case_id", self.case_id.strip())
        object.__setattr__(self, "target_domain", self.target_domain.strip().casefold())
        object.__setattr__(self, "training", training)
        object.__setattr__(self, "validation", validation)
        object.__setattr__(self, "target_postconditions", post)
        object.__setattr__(self, "expected_rejection", self.expected_rejection.strip())

    @property
    def source_domain(self):
        return self.source.program.domain

    @property
    def fingerprint(self):
        return stable_representation_id("phase8-case-v1", {
            "id": self.case_id, "source": self.source.program.fingerprint,
            "capability": self.source.capability.fingerprint, "target": self.target_domain,
            "train": [{"inputs": dict(x.inputs), "expected": x.expected} for x in self.training],
            "heldout": [{"inputs": dict(x.inputs), "expected": x.expected} for x in self.validation],
            "postconditions": list(self.target_postconditions), "positive": self.expect_transfer,
            "expected_rejection": self.expected_rejection,
        })

@dataclass(frozen=True)
class CrossDomainBenchmarkManifest:
    benchmark_id: str
    cases: tuple[CrossDomainBenchmarkCase, ...]
    minimum_positive_cases: int = 3
    minimum_negative_controls: int = 1
    minimum_source_domains: int = 2
    minimum_target_domains: int = 2
    minimum_domain_pairs: int = 2

    def __post_init__(self):
        cases = tuple(self.cases)
        if not self.benchmark_id.strip() or not cases or any(not isinstance(x, CrossDomainBenchmarkCase) for x in cases):
            raise CrossDomainBenchmarkError("manifest requires an ID and typed cases")
        if len({x.case_id for x in cases}) != len(cases):
            raise CrossDomainBenchmarkError("case IDs must be unique")
        for name in ("minimum_positive_cases", "minimum_negative_controls", "minimum_source_domains", "minimum_target_domains", "minimum_domain_pairs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise CrossDomainBenchmarkError(f"{name} must be a positive integer")
        positives = tuple(x for x in cases if x.expect_transfer)
        negatives = tuple(x for x in cases if not x.expect_transfer)
        if len(positives) < self.minimum_positive_cases or len(negatives) < self.minimum_negative_controls:
            raise CrossDomainBenchmarkError("manifest lacks required positive cases or negative controls")
        if len({x.source_domain for x in positives}) < self.minimum_source_domains:
            raise CrossDomainBenchmarkError("manifest lacks source-domain coverage")
        if len({x.target_domain for x in positives}) < self.minimum_target_domains:
            raise CrossDomainBenchmarkError("manifest lacks target-domain coverage")
        if len({(x.source_domain, x.target_domain) for x in positives}) < self.minimum_domain_pairs:
            raise CrossDomainBenchmarkError("manifest lacks source-target pair coverage")
        object.__setattr__(self, "benchmark_id", self.benchmark_id.strip())
        object.__setattr__(self, "cases", cases)

    @property
    def fingerprint(self):
        return stable_representation_id("phase8-manifest-v1", {
            "id": self.benchmark_id, "cases": [x.fingerprint for x in self.cases],
            "thresholds": [self.minimum_positive_cases, self.minimum_negative_controls,
                           self.minimum_source_domains, self.minimum_target_domains, self.minimum_domain_pairs],
        })

@dataclass(frozen=True)
class CrossDomainCaseResult:
    case_id: str
    passed: bool
    transferred: bool
    source_domain: str
    target_domain: str
    learned_proposal: UOLProgramLearningProposal | None = None
    error: str = ""

@dataclass(frozen=True)
class CrossDomainBenchmarkReport:
    benchmark_id: str
    benchmark_fingerprint: str
    passed: bool
    results: tuple[CrossDomainCaseResult, ...]
    source_domains: tuple[str, ...]
    target_domains: tuple[str, ...]
    learned_proposals: tuple[UOLProgramLearningProposal, ...]

class CrossDomainBenchmarkRunner:
    """A positive transfer passes only after target evidence and repeated learning pass."""
    def run(self, manifest: CrossDomainBenchmarkManifest, *, target_verifiers: Mapping[str, Callable[[CrossDomainTransferProposal], Iterable[VerifiedProgramObservation]]]):
        if not isinstance(manifest, CrossDomainBenchmarkManifest) or not isinstance(target_verifiers, Mapping):
            raise TypeError("manifest and target_verifiers have invalid types")
        results, learned = [], []
        for case in manifest.cases:
            try:
                candidate = CrossDomainTransferEngine().propose(
                    case.source, case.language, target_domain=case.target_domain,
                    training=case.training, validation=case.validation,
                    target_postconditions=case.target_postconditions,
                )
            except (CrossDomainTransferError, TypeError, ValueError) as exc:
                expected = (not case.expect_transfer and case.expected_rejection.casefold() in str(exc).casefold())
                results.append(CrossDomainCaseResult(case.case_id, expected, False, case.source_domain, case.target_domain, error=f"{type(exc).__name__}: {exc}"))
                continue
            if not case.expect_transfer:
                results.append(CrossDomainCaseResult(case.case_id, False, True, case.source_domain, case.target_domain, error="negative control unexpectedly transferred"))
                continue
            factory = target_verifiers.get(case.case_id)
            if not callable(factory):
                results.append(CrossDomainCaseResult(case.case_id, False, True, case.source_domain, case.target_domain, error="missing independent target-verifier factory"))
                continue
            try:
                observations = tuple(factory(candidate))
                proposal = VerifiedProgramLearningEngine().propose(
                    candidate.program, observations, case.language, intents=case.source.capability.intents,
                )
                if proposal.capability.domains != (case.target_domain,):
                    raise ProgramLearningError("learned capability is not restricted to target domain")
                learned.append(proposal)
                results.append(CrossDomainCaseResult(case.case_id, True, True, case.source_domain, case.target_domain, proposal))
            except Exception as exc:
                results.append(CrossDomainCaseResult(case.case_id, False, True, case.source_domain, case.target_domain, error=f"{type(exc).__name__}: {exc}"))
        return CrossDomainBenchmarkReport(
            manifest.benchmark_id, manifest.fingerprint, bool(results) and all(x.passed for x in results),
            tuple(results), tuple(sorted({x.source_domain for x in manifest.cases})),
            tuple(sorted({x.target_domain for x in manifest.cases})), tuple(learned),
        )
