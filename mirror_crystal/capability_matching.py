"""Deterministic task-to-capability matching (Phase 5).

This layer ranks declarative capability candidates. It never executes capabilities
and treats missing required inputs separately from type conflicts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .capability_model import CapabilitySpec
from .task_understanding import TaskIntent, TaskSpec, TaskStatus


@dataclass(frozen=True)
class CapabilityMatch:
    subtask_id: str
    capability: CapabilitySpec
    score: int
    eligible: bool
    matched_inputs: tuple[str, ...] = ()
    missing_inputs: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityMatchReport:
    task_id: str
    matches: tuple[CapabilityMatch, ...]
    unmatched_subtasks: tuple[str, ...]
    task_ready: bool

    def for_subtask(self, subtask_id: str) -> tuple[CapabilityMatch, ...]:
        return tuple(item for item in self.matches if item.subtask_id == subtask_id)

    def eligible_for(self, subtask_id: str) -> tuple[CapabilityMatch, ...]:
        return tuple(item for item in self.for_subtask(subtask_id) if item.eligible)


class CapabilityMatcher:
    """Match each task unit against the same universal capability contract."""

    VERSION = "capability-matcher-v1"

    def match(
        self,
        task: TaskSpec,
        capabilities: Iterable[CapabilitySpec],
        *,
        limit_per_subtask: int = 100,
    ) -> CapabilityMatchReport:
        if not isinstance(task, TaskSpec):
            raise TypeError("task must be a TaskSpec")
        if isinstance(limit_per_subtask, bool) or not isinstance(limit_per_subtask, int) or limit_per_subtask < 1:
            raise ValueError("limit_per_subtask must be a positive integer")
        unique: dict[str, CapabilitySpec] = {}
        for capability in capabilities:
            if not isinstance(capability, CapabilitySpec):
                raise TypeError("capabilities must contain CapabilitySpec values")
            previous = unique.get(capability.capability_id)
            if previous is not None and previous.fingerprint != capability.fingerprint:
                raise ValueError(f"conflicting capability id: {capability.capability_id}")
            unique[capability.capability_id] = capability

        units = (
            tuple(
                (subtask.subtask_id, subtask.intent, subtask.goal)
                for subtask in task.subtasks
            )
            if task.subtasks
            else ((f"task:{task.task_id}", task.intent, task.goal),)
        )
        all_matches: list[CapabilityMatch] = []
        unmatched: list[str] = []
        known = dict(task.known_inputs)
        context = dict(task.context)

        for subtask_id, declared_intent, _goal in units:
            intent = declared_intent if declared_intent is not TaskIntent.UNKNOWN else task.intent
            candidates: list[CapabilityMatch] = []
            for capability in unique.values():
                exact_domain = task.domain in capability.domains
                generic_domain = "general" in capability.domains or "*" in capability.domains
                if not (exact_domain or generic_domain):
                    continue
                accepts_intent = (
                    "any" in capability.intents
                    or "*" in capability.intents
                    or intent.value in capability.intents
                )
                if not accepts_intent:
                    continue

                matched_inputs: list[str] = []
                missing_inputs: list[str] = []
                incompatible_inputs: list[str] = []
                for port in capability.inputs:
                    if port.name in known:
                        value = known[port.name]
                    elif port.name in context:
                        value = context[port.name]
                    elif port.required:
                        missing_inputs.append(port.name)
                        continue
                    else:
                        continue
                    if port.accepts_value(value):
                        matched_inputs.append(port.name)
                    else:
                        incompatible_inputs.append(port.name)
                if incompatible_inputs:
                    # A named value with the wrong type is a real conflict, not a
                    # missing value which a later dependency might provide.
                    continue

                blockers = () if capability.executable else (capability.evidence_blocker,)
                score = (40 if exact_domain else 15) + (35 if intent.value in capability.intents else 10)
                score += min(20, 5 * len(matched_inputs))
                if capability.outputs:
                    score += 5
                candidates.append(CapabilityMatch(
                    subtask_id=subtask_id,
                    capability=capability,
                    score=score,
                    eligible=capability.executable,
                    matched_inputs=tuple(sorted(matched_inputs)),
                    missing_inputs=tuple(sorted(missing_inputs)),
                    blockers=blockers,
                ))
            candidates.sort(key=lambda item: (
                not item.eligible, -item.score, item.capability.capability_id
            ))
            if not candidates:
                unmatched.append(subtask_id)
            all_matches.extend(candidates[:limit_per_subtask])

        return CapabilityMatchReport(
            task_id=task.task_id,
            matches=tuple(all_matches),
            unmatched_subtasks=tuple(unmatched),
            task_ready=task.status is TaskStatus.READY,
        )
