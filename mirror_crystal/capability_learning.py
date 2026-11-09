"""Evidence-gated learning of reusable composite capabilities (Phase 9).

A composite capability is proposed only after the same typed plan succeeds and is
independently verified on multiple distinct tasks by multiple named verifiers.
Proposals are not silently promoted or persisted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .capability_composition import CompositionPlan
from .capability_model import CapabilitySpec, PortContract
from .execution import PlanExecutionReport
from .task_understanding import TaskSpec
from .blueprint_contracts import EvidenceLevel, VerificationEvidence
from .universal_representation import RepresentationProvenance, stable_representation_id
from .verification import VerificationReport, VerificationStatus


class CapabilityLearningError(ValueError):
    """Raised when successful observations cannot justify reusable learning."""


@dataclass(frozen=True)
class CapabilityLearningObservation:
    task: TaskSpec
    plan: CompositionPlan
    execution: PlanExecutionReport
    verification: VerificationReport

    def __post_init__(self) -> None:
        if self.plan.task_id != self.task.task_id:
            raise CapabilityLearningError("composition plan belongs to a different task")
        if self.execution.task_id != self.task.task_id or self.execution.plan_id != self.plan.plan_id:
            raise CapabilityLearningError("execution report does not match task and plan")
        if self.verification.task_id != self.task.task_id or self.verification.plan_id != self.plan.plan_id:
            raise CapabilityLearningError("verification report does not match task and plan")
        if self.verification.execution_fingerprint != self.execution.fingerprint:
            raise CapabilityLearningError("verification report is bound to a different execution")


@dataclass(frozen=True)
class CapabilityLearningProposal:
    capability: CapabilitySpec
    observation_task_ids: tuple[str, ...]
    verifier_ids: tuple[str, ...]
    plan_signature: str
    minimum_successes: int
    minimum_verifiers: int
    requires_explicit_promotion: bool = True


class CapabilityLearningEngine:
    """Infer a reusable typed composite only from repeated independently verified runs."""

    VERSION = "capability-learning-v1"

    @staticmethod
    def _structure(
        plan: CompositionPlan,
        capability_map: dict[str, CapabilitySpec],
    ) -> dict[str, object]:
        if not plan.ready or len(plan.steps) < 2:
            raise CapabilityLearningError("learning requires a ready composition with at least two steps")
        indices = {step.step_id: index for index, step in enumerate(plan.steps)}
        if len(indices) != len(plan.steps):
            raise CapabilityLearningError("plan contains duplicate step identifiers")
        normalized_steps = []
        for index, step in enumerate(plan.steps):
            capability = capability_map.get(step.capability_id)
            if capability is None:
                raise CapabilityLearningError(f"missing capability contract: {step.capability_id}")
            dependency_indices = []
            for dependency in step.dependency_step_ids:
                if dependency not in indices or indices[dependency] >= index:
                    raise CapabilityLearningError("composition steps are not in dependency-first order")
                dependency_indices.append(indices[dependency])
            normalized_steps.append({
                "capability_id": capability.capability_id,
                "capability_fingerprint": capability.fingerprint,
                "dependencies": sorted(dependency_indices),
                "bindings": [
                    {
                        "input_name": binding.input_name,
                        "source": binding.source,
                        "source_name": binding.source_name,
                        "source_step_index": indices.get(binding.source_step_id, -1)
                        if binding.source == "step_output" else -1,
                    }
                    for binding in step.input_bindings
                ],
                "expected_postconditions": list(step.expected_postconditions),
            })
        return {"steps": normalized_steps}

    @staticmethod
    def _boundary_ports(
        plan: CompositionPlan,
        capability_map: dict[str, CapabilitySpec],
    ) -> tuple[tuple[PortContract, ...], tuple[PortContract, ...]]:
        by_step = {step.step_id: step for step in plan.steps}
        inputs: dict[str, PortContract] = {}
        outputs: dict[str, PortContract] = {}
        consumed: set[tuple[str, str]] = {
            (binding.source_step_id, binding.source_name)
            for step in plan.steps
            for binding in step.input_bindings
            if binding.source == "step_output"
        }
        for step in plan.steps:
            capability = capability_map[step.capability_id]
            input_ports = {port.name: port for port in capability.inputs}
            for binding in step.input_bindings:
                if binding.source not in {"task_input", "task_context"}:
                    continue
                port = input_ports.get(binding.input_name)
                if port is None:
                    raise CapabilityLearningError(
                        f"composition binding references unknown input {binding.input_name}"
                    )
                external_name = binding.source_name
                external_port = PortContract(
                    name=external_name,
                    value_type=port.value_type,
                    required=True,
                    unit=port.unit,
                    dimensions=dict(port.dimensions),
                )
                previous = inputs.get(external_name)
                if previous is not None and previous != external_port:
                    raise CapabilityLearningError(
                        f"incompatible external input contract for {external_name}"
                    )
                inputs[external_name] = external_port
            for port in capability.outputs:
                if (step.step_id, port.name) in consumed:
                    continue
                previous = outputs.get(port.name)
                if previous is not None and previous != port:
                    raise CapabilityLearningError(
                        f"incompatible external output contract for {port.name}"
                    )
                outputs[port.name] = port
        if not outputs:
            raise CapabilityLearningError("composition has no externally visible output ports")
        return (
            tuple(inputs[key] for key in sorted(inputs)),
            tuple(outputs[key] for key in sorted(outputs)),
        )

    def propose(
        self,
        observations: Iterable[CapabilityLearningObservation],
        capabilities: Iterable[CapabilitySpec],
        *,
        minimum_successes: int = 3,
        minimum_verifiers: int = 2,
    ) -> CapabilityLearningProposal:
        if isinstance(minimum_successes, bool) or not isinstance(minimum_successes, int) or minimum_successes < 2:
            raise ValueError("minimum_successes must be an integer of at least 2")
        if isinstance(minimum_verifiers, bool) or not isinstance(minimum_verifiers, int) or minimum_verifiers < 2:
            raise ValueError("minimum_verifiers must be an integer of at least 2")
        catalog = tuple(capabilities)
        if any(not isinstance(item, CapabilitySpec) for item in catalog):
            raise TypeError("capabilities must contain CapabilitySpec values")
        capability_map = {item.capability_id: item for item in catalog}
        if len(capability_map) != len(catalog):
            raise CapabilityLearningError("capability catalog contains duplicate identifiers")
        samples = tuple(observations)
        if len(samples) < minimum_successes:
            raise CapabilityLearningError(
                f"need at least {minimum_successes} successful independent observations"
            )
        if any(not isinstance(item, CapabilityLearningObservation) for item in samples):
            raise TypeError("observations must contain CapabilityLearningObservation values")

        task_ids = [item.task.task_id for item in samples]
        fingerprints = [item.task.fingerprint for item in samples]
        if len(set(task_ids)) != len(task_ids) or len(set(fingerprints)) != len(fingerprints):
            raise CapabilityLearningError("learning observations must come from distinct tasks")

        signatures: list[str] = []
        verifiers: set[str] = set()
        checks_passed: set[str] = set()
        for observation in samples:
            if not observation.execution.succeeded:
                raise CapabilityLearningError("failed execution cannot support capability learning")
            if observation.verification.status is not VerificationStatus.VERIFIED:
                raise CapabilityLearningError("observation must be independently verified before capability learning")
            if observation.verification.blockers or any(not item.passed for item in observation.verification.capabilities):
                raise CapabilityLearningError("verification report contains a blocker or failed capability")
            if not observation.verification.results or any(not item.passed for item in observation.verification.results):
                raise CapabilityLearningError("every independent verification check must pass")
            expected_step_ids = {step.step_id for step in observation.execution.steps}
            verified_step_ids = {item.step_id for item in observation.verification.results if item.passed}
            if verified_step_ids != expected_step_ids:
                raise CapabilityLearningError("every executed step must have independent verification results")
            if not observation.verification.capabilities or any(not item.passed for item in observation.verification.capabilities):
                raise CapabilityLearningError("every selected capability must have a passing verification record")
            signatures.append(stable_representation_id(
                "plan-structure",
                self._structure(observation.plan, capability_map),
            ))
            verifiers.update(observation.verification.verifier_ids)
            checks_passed.update(
                f"{result.verifier_id}:{result.check_id}"
                for result in observation.verification.results
            )
        if len(set(signatures)) != 1:
            raise CapabilityLearningError("verified observations do not share the same typed composition structure")
        if len(verifiers) < minimum_verifiers:
            raise CapabilityLearningError(
                f"need at least {minimum_verifiers} distinct independent verifier identities"
            )
        if not checks_passed:
            raise CapabilityLearningError("no passing independent verification evidence was supplied")

        first = samples[0]
        input_ports, output_ports = self._boundary_ports(first.plan, capability_map)
        selected = tuple(dict.fromkeys(step.capability_id for step in first.plan.steps))
        selected_specs = tuple(capability_map[item] for item in selected)
        domains = tuple(sorted({item.task.domain for item in samples}))
        intents = tuple(sorted({item.task.intent.value for item in samples}))
        preconditions = tuple(sorted({
            condition
            for item in selected_specs
            for condition in item.preconditions
        }))
        postconditions = tuple(sorted({
            condition
            for step in first.plan.steps
            for condition in step.expected_postconditions
        }))
        structure = self._structure(first.plan, capability_map)
        capability_id = "learned:" + stable_representation_id(
            "composite", {"structure": structure, "domains": domains, "intents": intents}
        )
        evidence = VerificationEvidence(
            level=EvidenceLevel.INDEPENDENTLY_REPRODUCED,
            checks_passed=tuple(sorted(checks_passed)),
            checks_failed=(),
            verifier="independent verifiers: " + ", ".join(sorted(verifiers)),
        )
        learned = CapabilitySpec(
            capability_id=capability_id,
            name="Learned composite: " + " + ".join(item.name for item in selected_specs),
            description=(
                "A reusable composition learned from "
                f"{len(samples)} distinct independently verified tasks. "
                "A native handler must be explicitly registered before execution."
            ),
            domains=domains,
            intents=intents,
            inputs=input_ports,
            outputs=output_ports,
            preconditions=preconditions,
            postconditions=postconditions,
            dependencies=(),
            provenance=RepresentationProvenance(
                source="mirror_verified_composition_learning",
                source_ref=stable_representation_id("learning-observations", sorted(fingerprints)),
                parent_ids=selected,
            ),
            verification=evidence,
        )
        return CapabilityLearningProposal(
            capability=learned,
            observation_task_ids=tuple(task_ids),
            verifier_ids=tuple(sorted(verifiers)),
            plan_signature=signatures[0],
            minimum_successes=minimum_successes,
            minimum_verifiers=minimum_verifiers,
            requires_explicit_promotion=True,
        )

    @staticmethod
    def persist(vault: Any, proposal: CapabilityLearningProposal) -> str:
        """Explicitly promote a reviewed proposal into the durable capability catalog."""
        if not isinstance(proposal, CapabilityLearningProposal):
            raise TypeError("proposal must be CapabilityLearningProposal")
        store = getattr(vault, "store_capability", None)
        if not callable(store):
            raise TypeError("vault must expose store_capability")
        return store(proposal.capability)
