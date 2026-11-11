"""Multi-cycle, evidence-gated continuous expansion (Strategic Phase 9).

The engine admits only repeated Phase-8 successes over distinct frozen benchmarks,
unseen task/case IDs, a fixed baseline, adversarial controls and independent review.
It returns a proposal for explicit Phase-5 promotion; it never mutates registries.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Iterable
from .phase8_cross_domain_intelligence import CrossDomainBenchmarkReport
from .universal_representation import stable_representation_id
from .verified_program_learning import UOLProgramLearningProposal

class ContinuousExpansionError(ValueError):
    """Insufficient or inconsistent continuous-expansion evidence."""

@dataclass(frozen=True)
class ExpansionCycleEvidence:
    cycle_id: str
    benchmark: CrossDomainBenchmarkReport
    task_ids: tuple[str, ...]
    unseen_case_ids: tuple[str, ...]
    reviewer_ids: tuple[str, ...]
    negative_controls_passed: bool
    frozen_baseline_fingerprint: str

    def __post_init__(self):
        if not self.cycle_id.strip() or not isinstance(self.benchmark, CrossDomainBenchmarkReport):
            raise ContinuousExpansionError("cycle ID and Phase-8 report are required")
        if not self.benchmark.passed:
            raise ContinuousExpansionError("a failed Phase-8 benchmark blocks expansion")
        if not any(result.passed and not result.transferred for result in self.benchmark.results):
            raise ContinuousExpansionError("each cycle must contain a passed adversarial negative control")
        tasks, unseen, reviewers = tuple(self.task_ids), tuple(self.unseen_case_ids), tuple(self.reviewer_ids)
        if len(tasks) < 3 or any(not x for x in tasks) or len(set(tasks)) != len(tasks):
            raise ContinuousExpansionError("each cycle needs at least three distinct task IDs")
        if not unseen or any(not x for x in unseen) or len(set(unseen)) != len(unseen):
            raise ContinuousExpansionError("each cycle needs distinct unseen-case IDs")
        if len(reviewers) < 2 or any(not x for x in reviewers) or len(set(reviewers)) != len(reviewers):
            raise ContinuousExpansionError("each cycle needs at least two independent reviewers")
        if self.negative_controls_passed is not True:
            raise ContinuousExpansionError("all adversarial negative controls must pass")
        if not isinstance(self.frozen_baseline_fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", self.frozen_baseline_fingerprint):
            raise ContinuousExpansionError("baseline fingerprint must be a lowercase SHA-256 digest")
        object.__setattr__(self, "cycle_id", self.cycle_id.strip())
        object.__setattr__(self, "task_ids", tasks)
        object.__setattr__(self, "unseen_case_ids", unseen)
        object.__setattr__(self, "reviewer_ids", reviewers)

@dataclass(frozen=True)
class ContinuousExpansionProposal:
    generation_id: str
    cycle_ids: tuple[str, ...]
    benchmark_fingerprints: tuple[str, ...]
    task_ids: tuple[str, ...]
    unseen_case_ids: tuple[str, ...]
    target_domains: tuple[str, ...]
    reviewer_ids: tuple[str, ...]
    learned_proposals: tuple[UOLProgramLearningProposal, ...]
    frozen_baseline_fingerprint: str
    requires_explicit_promotion: bool = True

class ContinuousExpansionEngine:
    """Construct a promotion proposal only after three fresh, reviewed cycles."""
    def propose(self, cycles: Iterable[ExpansionCycleEvidence], *, minimum_cycles: int = 3, minimum_target_domains: int = 2):
        if isinstance(minimum_cycles, bool) or not isinstance(minimum_cycles, int) or minimum_cycles < 3:
            raise ValueError("minimum_cycles must be at least 3")
        if isinstance(minimum_target_domains, bool) or not isinstance(minimum_target_domains, int) or minimum_target_domains < 2:
            raise ValueError("minimum_target_domains must be at least 2")
        items = tuple(cycles)
        if len(items) < minimum_cycles or any(not isinstance(x, ExpansionCycleEvidence) for x in items):
            raise ContinuousExpansionError(f"need at least {minimum_cycles} valid cycles")
        if len({x.cycle_id for x in items}) != len(items):
            raise ContinuousExpansionError("cycle IDs must be unique")
        fingerprints = tuple(x.benchmark.benchmark_fingerprint for x in items)
        if len(set(fingerprints)) != len(fingerprints):
            raise ContinuousExpansionError("every cycle requires a distinct frozen benchmark")
        baselines = {x.frozen_baseline_fingerprint for x in items}
        if len(baselines) != 1:
            raise ContinuousExpansionError("all cycles must use the same frozen baseline")
        tasks = tuple(task for x in items for task in x.task_ids)
        unseen = tuple(case for x in items for case in x.unseen_case_ids)
        if len(set(tasks)) != len(tasks):
            raise ContinuousExpansionError("task IDs must be disjoint across cycles")
        if len(set(unseen)) != len(unseen):
            raise ContinuousExpansionError("unseen-case IDs must be disjoint across cycles")
        domains = tuple(sorted({d for x in items for d in x.benchmark.target_domains}))
        if len(domains) < minimum_target_domains:
            raise ContinuousExpansionError("insufficient target-domain coverage")
        reviewers = tuple(sorted({r for x in items for r in x.reviewer_ids}))
        if len(reviewers) < 2:
            raise ContinuousExpansionError("independent reviewer coverage is insufficient")
        by_id = {}
        observed_tasks, observed_executions, observed_evidence = set(), set(), set()
        for cycle in items:
            cycle_tasks, cycle_executions, cycle_evidence = set(), set(), set()
            for proposal in cycle.benchmark.learned_proposals:
                if len(proposal.observation_task_ids) < 3 or len(set(proposal.observation_task_ids)) != len(proposal.observation_task_ids):
                    raise ContinuousExpansionError("each learned capability needs three distinct verified observation tasks")
                if len(proposal.execution_fingerprints) < 3 or len(set(proposal.execution_fingerprints)) != len(proposal.execution_fingerprints):
                    raise ContinuousExpansionError("each learned capability needs three distinct execution fingerprints")
                if len(proposal.evidence_fingerprints) < 3 or len(set(proposal.evidence_fingerprints)) != len(proposal.evidence_fingerprints):
                    raise ContinuousExpansionError("each learned capability needs three distinct evidence fingerprints")
                cycle_tasks.update(proposal.observation_task_ids)
                cycle_executions.update(proposal.execution_fingerprints)
                cycle_evidence.update(proposal.evidence_fingerprints)
            if cycle_tasks & observed_tasks:
                raise ContinuousExpansionError("verified observation task IDs must be fresh across cycles")
            if cycle_executions & observed_executions:
                raise ContinuousExpansionError("verified execution fingerprints must be fresh across cycles")
            if cycle_evidence & observed_evidence:
                raise ContinuousExpansionError("verification evidence fingerprints must be fresh across cycles")
            observed_tasks.update(cycle_tasks)
            observed_executions.update(cycle_executions)
            observed_evidence.update(cycle_evidence)
            for proposal in cycle.benchmark.learned_proposals:
                if proposal.capability.domains[0] not in cycle.benchmark.target_domains:
                    raise ContinuousExpansionError("learned capability is outside benchmark target coverage")
                previous = by_id.get(proposal.capability.capability_id)
                if previous and previous.capability.fingerprint != proposal.capability.fingerprint:
                    raise ContinuousExpansionError("conflicting capability ID across cycles")
                by_id[proposal.capability.capability_id] = proposal
        if not by_id:
            raise ContinuousExpansionError("no independently learned capabilities are available")
        baseline = next(iter(baselines))
        learned = tuple(by_id[k] for k in sorted(by_id))
        generation = stable_representation_id("mirror-expansion-v1", {
            "cycles": [x.cycle_id for x in items], "benchmarks": list(fingerprints),
            "tasks": list(tasks), "unseen": list(unseen), "domains": list(domains),
            "reviewers": list(reviewers), "baseline": baseline,
            "capabilities": [x.capability.fingerprint for x in learned],
        })
        return ContinuousExpansionProposal(
            generation, tuple(x.cycle_id for x in items), fingerprints, tasks, unseen,
            domains, reviewers, learned, baseline, True,
        )
