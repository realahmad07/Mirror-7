"""Typed, evidence-gated capability composition planning (Phase 6).

The planner constructs a proposed dependency graph only. It never runs capabilities,
claims postconditions are true, or converts plan completion into verification evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Iterable, Mapping

from .capability_matching import CapabilityMatchReport, CapabilityMatcher
from .capability_model import CapabilitySpec
from .task_understanding import TaskSpec, TaskStatus
from .universal_representation import stable_representation_id


class CompositionPlanningError(ValueError):
    """Raised when composition inputs are malformed."""


class PlanStatus(str, Enum):
    READY = "ready"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class InputBinding:
    input_name: str
    source: str
    source_name: str
    source_step_id: str = ""

    def __post_init__(self) -> None:
        if not self.input_name.strip() or not self.source.strip():
            raise CompositionPlanningError("binding input_name and source are required")
        if self.source not in {"task_input", "task_context", "step_output"}:
            raise CompositionPlanningError(f"unsupported binding source: {self.source}")
        if self.source in {"task_input", "task_context"} and not self.source_name.strip():
            raise CompositionPlanningError("task binding source_name is required")
        if self.source == "step_output" and (
            not self.source_name.strip() or not self.source_step_id.strip()
        ):
            raise CompositionPlanningError("step output bindings require a source name and step id")


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    subtask_id: str
    capability_id: str
    dependency_step_ids: tuple[str, ...] = ()
    input_bindings: tuple[InputBinding, ...] = ()
    expected_postconditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompositionPlan:
    plan_id: str
    task_id: str
    status: PlanStatus
    steps: tuple[PlanStep, ...]
    blockers: tuple[str, ...] = ()
    verification_requirements: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.status is PlanStatus.READY

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "steps": [
                {
                    "step_id": step.step_id,
                    "subtask_id": step.subtask_id,
                    "capability_id": step.capability_id,
                    "dependency_step_ids": list(step.dependency_step_ids),
                    "input_bindings": [
                        {
                            "input_name": binding.input_name,
                            "source": binding.source,
                            "source_name": binding.source_name,
                            "source_step_id": binding.source_step_id,
                        }
                        for binding in step.input_bindings
                    ],
                    "expected_postconditions": list(step.expected_postconditions),
                }
                for step in self.steps
            ],
            "blockers": list(self.blockers),
            "verification_requirements": list(self.verification_requirements),
        }


class CapabilityCompositionPlanner:
    """Build a bounded deterministic plan from matched capabilities."""

    VERSION = "capability-composer-v1"

    @staticmethod
    def _units(task: TaskSpec):
        if task.subtasks:
            by_id = {item.subtask_id: item for item in task.subtasks}
            return tuple(
                (identifier, by_id[identifier].goal, by_id[identifier].depends_on)
                for identifier in task.topological_order()
            )
        return ((f"task:{task.task_id}", task.goal, ()),)

    @staticmethod
    def _ancestors(units) -> dict[str, set[str]]:
        direct = {identifier: set(dependencies) for identifier, _goal, dependencies in units}
        result: dict[str, set[str]] = {}
        for identifier, _goal, _dependencies in units:
            found: set[str] = set()
            pending = list(direct[identifier])
            while pending:
                item = pending.pop()
                if item in found:
                    continue
                found.add(item)
                pending.extend(direct.get(item, ()))
            result[identifier] = found
        return result

    @staticmethod
    def _has_cycle(dependencies: Mapping[str, set[str]]) -> bool:
        remaining = set(dependencies)
        complete: set[str] = set()
        while remaining:
            ready = {
                item for item in remaining
                if dependencies[item].issubset(complete)
            }
            if not ready:
                return True
            complete.update(ready)
            remaining.difference_update(ready)
        return False

    @staticmethod
    def _plan_id(task_id: str, selected: Mapping[str, CapabilitySpec]) -> str:
        return stable_representation_id("composition-plan", {
            "task_id": task_id,
            "selection": {
                key: value.capability_id for key, value in sorted(selected.items())
            },
        })

    def _build_candidate(
        self,
        task: TaskSpec,
        units,
        selected: Mapping[str, CapabilitySpec],
        *,
        verified_conditions: set[str],
    ) -> CompositionPlan:
        ancestors = self._ancestors(units)
        step_ids = {
            subtask_id: stable_representation_id("plan-step", {
                "task_id": task.task_id,
                "subtask_id": subtask_id,
                "capability_id": selected[subtask_id].capability_id,
            })
            for subtask_id, _goal, _deps in units
        }
        capability_owners: dict[str, str] = {}
        for subtask_id, capability in selected.items():
            capability_owners.setdefault(capability.capability_id, subtask_id)

        dependencies: dict[str, set[str]] = {}
        blockers: list[str] = []
        steps: list[PlanStep] = []
        requirements: list[str] = []

        if task.status is not TaskStatus.READY:
            blockers.append("task requires clarification before capability composition")

        for subtask_id, _goal, task_dependencies in units:
            capability = selected[subtask_id]
            step_id = step_ids[subtask_id]
            dependency_subtasks = set(ancestors[subtask_id])
            dependency_subtasks.update(task_dependencies)
            direct_step_dependencies = {
                step_ids[item] for item in task_dependencies if item in step_ids
            }
            for required_capability in capability.dependencies:
                owner = capability_owners.get(required_capability)
                if owner is None:
                    blockers.append(
                        f"{capability.capability_id} requires missing capability dependency {required_capability}"
                    )
                    continue
                if owner == subtask_id:
                    blockers.append(
                        f"{capability.capability_id} depends on itself within {subtask_id}"
                    )
                    continue
                dependency_subtasks.add(owner)
                direct_step_dependencies.add(step_ids[owner])

            dependencies[step_id] = set(direct_step_dependencies)
            missing_preconditions = sorted(set(capability.preconditions) - verified_conditions)
            for condition in missing_preconditions:
                blockers.append(
                    f"{subtask_id} / {capability.capability_id} requires verified precondition: {condition}"
                )

            bindings: list[InputBinding] = []
            for port in capability.inputs:
                if port.name in task.known_inputs:
                    value = task.known_inputs[port.name]
                    if port.accepts_value(value):
                        bindings.append(InputBinding(port.name, "task_input", port.name))
                        continue
                    blockers.append(
                        f"{subtask_id} input {port.name} has an incompatible task input type"
                    )
                    continue
                if port.name in task.context:
                    value = task.context[port.name]
                    if port.accepts_value(value):
                        bindings.append(InputBinding(port.name, "task_context", port.name))
                        continue
                    blockers.append(
                        f"{subtask_id} input {port.name} has an incompatible context type"
                    )
                    continue
                if not port.required:
                    continue

                possible_sources: list[tuple[str, str, object]] = []
                for source_subtask in sorted(dependency_subtasks):
                    source_capability = selected.get(source_subtask)
                    if source_capability is None:
                        continue
                    for output in source_capability.outputs:
                        if output.name == port.name and port.accepts_output(output):
                            possible_sources.append((
                                step_ids[source_subtask], output.name, output
                            ))
                # Distinct dependency outputs with the same port name are ambiguous.
                unique_sources = {
                    (source_step_id, output_name): output
                    for source_step_id, output_name, output in possible_sources
                }
                if len(unique_sources) == 1:
                    (source_step_id, output_name), _output = next(iter(unique_sources.items()))
                    bindings.append(InputBinding(
                        port.name, "step_output", output_name, source_step_id
                    ))
                elif len(unique_sources) > 1:
                    blockers.append(
                        f"{subtask_id} input {port.name} has multiple compatible dependency outputs"
                    )
                else:
                    blockers.append(f"{subtask_id} required input is unbound: {port.name}")

            requirements.extend(
                f"{capability.capability_id}:{condition}"
                for condition in capability.postconditions
            )
            steps.append(PlanStep(
                step_id=step_id,
                subtask_id=subtask_id,
                capability_id=capability.capability_id,
                dependency_step_ids=tuple(sorted(direct_step_dependencies)),
                input_bindings=tuple(bindings),
                expected_postconditions=capability.postconditions,
            ))

        if self._has_cycle(dependencies):
            blockers.append("combined task and capability dependencies contain a cycle")

        deduped_blockers = tuple(dict.fromkeys(blockers))
        return CompositionPlan(
            plan_id=self._plan_id(task.task_id, selected),
            task_id=task.task_id,
            status=PlanStatus.BLOCKED if deduped_blockers else PlanStatus.READY,
            steps=tuple(steps),
            blockers=deduped_blockers,
            verification_requirements=tuple(dict.fromkeys(requirements)),
        )

    def compose(
        self,
        task: TaskSpec,
        capabilities: Iterable[CapabilitySpec],
        *,
        match_report: CapabilityMatchReport | None = None,
        verified_conditions: Iterable[str] = (),
        max_candidates_per_subtask: int = 8,
        max_search_states: int = 256,
    ) -> CompositionPlan:
        if not isinstance(task, TaskSpec):
            raise TypeError("task must be a TaskSpec")
        if isinstance(max_candidates_per_subtask, bool) or not isinstance(max_candidates_per_subtask, int) or max_candidates_per_subtask < 1:
            raise ValueError("max_candidates_per_subtask must be positive")
        if isinstance(max_search_states, bool) or not isinstance(max_search_states, int) or max_search_states < 1:
            raise ValueError("max_search_states must be positive")
        items = tuple(capabilities)
        if match_report is None:
            match_report = CapabilityMatcher().match(task, items)
        if match_report.task_id != task.task_id:
            raise CompositionPlanningError("match report belongs to a different task")
        units = self._units(task)
        choices: list[tuple[str, tuple[CapabilitySpec, ...]]] = []
        for subtask_id, _goal, _deps in units:
            eligible = [
                item for item in match_report.eligible_for(subtask_id)
            ]
            eligible.sort(key=lambda item: (-item.score, item.capability.capability_id))
            choices.append((
                subtask_id,
                tuple(item.capability for item in eligible[:max_candidates_per_subtask]),
            ))

        missing = [
            subtask_id for subtask_id, candidates in choices if not candidates
        ]
        if missing:
            blockers = []
            for subtask_id in missing:
                observed = match_report.for_subtask(subtask_id)
                if observed:
                    blockers.append(
                        f"{subtask_id} has no evidence-eligible capability; "
                        + "; ".join(sorted({reason for item in observed for reason in item.blockers if reason}))
                    )
                else:
                    blockers.append(f"{subtask_id} has no domain/intent-compatible capability")
            empty_id = stable_representation_id("composition-plan", {
                "task_id": task.task_id, "missing": missing
            })
            return CompositionPlan(
                plan_id=empty_id,
                task_id=task.task_id,
                status=PlanStatus.BLOCKED,
                steps=(),
                blockers=tuple(blockers),
            )

        verified = {str(item).strip() for item in verified_conditions}
        best: CompositionPlan | None = None
        best_key: tuple[int, int, str] | None = None
        states = 0
        pools = [candidates for _subtask_id, candidates in choices]
        for combination in product(*pools):
            states += 1
            if states > max_search_states:
                break
            selected = {
                choices[index][0]: capability
                for index, capability in enumerate(combination)
            }
            candidate_plan = self._build_candidate(
                task, units, selected, verified_conditions=verified
            )
            if candidate_plan.ready:
                return candidate_plan
            score = sum(
                next(
                    (match.score for match in match_report.for_subtask(subtask_id)
                     if match.capability.capability_id == selected[subtask_id].capability_id),
                    0,
                )
                for subtask_id in selected
            )
            candidate_key = (len(candidate_plan.blockers), -score, candidate_plan.plan_id)
            if best_key is None or candidate_key < best_key:
                best = candidate_plan
                best_key = candidate_key

        if best is not None and states > max_search_states:
            extra = f"composition search reached its {max_search_states}-state bound"
            return CompositionPlan(
                plan_id=best.plan_id,
                task_id=best.task_id,
                status=PlanStatus.BLOCKED,
                steps=best.steps,
                blockers=tuple(dict.fromkeys(best.blockers + (extra,))),
                verification_requirements=best.verification_requirements,
            )
        return best or CompositionPlan(
            plan_id=stable_representation_id("composition-plan", {"task_id": task.task_id}),
            task_id=task.task_id,
            status=PlanStatus.BLOCKED,
            steps=(),
            blockers=("no valid capability composition was found",),
        )
