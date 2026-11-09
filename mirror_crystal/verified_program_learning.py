"""Evidence-gated extraction of verified UOL programs into reusable capabilities.

This phase learns only after the same immutable UOL program has passed multiple
execution-bound domain-verification reports over distinct tasks and executions.
Promotion into registries is an explicit host action.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import pickle
from typing import Any, Iterable, Mapping

from .blueprint_contracts import EvidenceLevel, VerificationEvidence
from .capability_model import CapabilityRegistry, CapabilitySpec, PortContract
from .domain_verification import DomainVerificationReport, DomainVerificationStatus
from .execution import NativeOperationRegistry, PlanExecutionReport, StepExecutionResult
from .operational_language import Program, UOL, RuleStatus, _OPS, _walk
from .universal_representation import RepresentationProvenance, stable_representation_id


class ProgramLearningError(ValueError):
    """Raised when repeated evidence cannot justify reusable program learning."""


@dataclass(frozen=True)
class VerifiedProgramObservation:
    task_id: str
    capability: CapabilitySpec
    execution: PlanExecutionReport
    verification: DomainVerificationReport

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ProgramLearningError("task_id must not be empty")
        if not isinstance(self.capability, CapabilitySpec):
            raise TypeError("capability must be CapabilitySpec")
        if not isinstance(self.execution, PlanExecutionReport):
            raise TypeError("execution must be PlanExecutionReport")
        if not isinstance(self.verification, DomainVerificationReport):
            raise TypeError("verification must be DomainVerificationReport")
        if self.execution.task_id != self.task_id or self.verification.task_id != self.task_id:
            raise ProgramLearningError("observation task identifiers do not match")
        if self.execution.plan_id != self.verification.plan_id:
            raise ProgramLearningError("domain verification belongs to a different plan")
        if self.execution.fingerprint != self.verification.execution_fingerprint:
            raise ProgramLearningError("domain verification belongs to a different execution")


@dataclass(frozen=True)
class UOLProgramHandler:
    """Pickle-safe native handler for a reviewed UOL program and its fixed rule catalog."""

    program: Program
    rules: tuple[Any, ...]
    output_name: str = "result"

    def __call__(self, values: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(values, Mapping):
            raise ProgramLearningError("program handler inputs must be a mapping")
        inputs = {key: value for key, value in values.items() if key != "__mirror_verification__"}
        expected = {item.name for item in self.program.inputs}
        if set(inputs) != expected:
            raise ProgramLearningError("runtime inputs do not match the learned program schema")
        language = UOL(self.rules)
        return {self.output_name: language.execute(self.program, inputs)}


@dataclass(frozen=True)
class UOLProgramLearningProposal:
    capability: CapabilitySpec
    program: Program
    handler: UOLProgramHandler
    observation_task_ids: tuple[str, ...]
    execution_fingerprints: tuple[str, ...]
    verifier_ids: tuple[str, ...]
    evidence_fingerprints: tuple[str, ...]
    minimum_successes: int
    minimum_verifiers: int
    requires_explicit_promotion: bool = True


class VerifiedProgramLearningEngine:
    """Learn a reusable operation from repeated independent domain verification."""

    VERSION = "verified-uol-program-learning-v1"
    _CANDIDATE_SOURCES = {"mirror_uol_synthesizer", "mirror_cross_domain_transfer"}

    @staticmethod
    def _close(left: Any, right: Any) -> bool:
        if isinstance(left, bool) or isinstance(right, bool):
            return type(left) is type(right) and left == right
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
        return left == right

    def propose(
        self,
        program: Program,
        observations: Iterable[VerifiedProgramObservation],
        language: UOL,
        *,
        output_name: str = "result",
        intents: Iterable[str] = ("solve",),
        minimum_successes: int = 3,
        minimum_verifiers: int = 2,
    ) -> UOLProgramLearningProposal:
        if not isinstance(program, Program) or not isinstance(language, UOL):
            raise TypeError("program and language must be Program and UOL")
        if isinstance(minimum_successes, bool) or not isinstance(minimum_successes, int) or minimum_successes < 2:
            raise ValueError("minimum_successes must be an integer of at least 2")
        if isinstance(minimum_verifiers, bool) or not isinstance(minimum_verifiers, int) or minimum_verifiers < 2:
            raise ValueError("minimum_verifiers must be an integer of at least 2")
        if not isinstance(output_name, str) or not output_name.strip():
            raise ValueError("output_name must not be empty")
        intent_values = tuple(sorted({str(item).strip().casefold() for item in intents}))
        if not intent_values or any(not item for item in intent_values):
            raise ValueError("at least one non-empty intent is required")

        samples = tuple(observations)
        if any(not isinstance(item, VerifiedProgramObservation) for item in samples):
            raise TypeError("observations must contain VerifiedProgramObservation values")
        if len(samples) < minimum_successes:
            raise ProgramLearningError(f"need at least {minimum_successes} verified executions")
        task_ids = tuple(item.task_id for item in samples)
        execution_fingerprints = tuple(item.execution.fingerprint for item in samples)
        if len(set(task_ids)) != len(task_ids):
            raise ProgramLearningError("learning requires distinct task identifiers")
        if len(set(execution_fingerprints)) != len(execution_fingerprints):
            raise ProgramLearningError("learning requires distinct execution fingerprints")

        output_type = language.validate(program)
        if any(item.name == "__mirror_verification__" for item in program.inputs):
            raise ProgramLearningError("the reserved verification envelope cannot be a UOL program input")
        rule_ids = set(program.rule_refs)
        pending_rules = list(rule_ids)
        visited_rules: set[str] = set()
        while pending_rules:
            rule_id = pending_rules.pop()
            if rule_id in visited_rules:
                continue
            visited_rules.add(rule_id)
            rule = language.rules.get(rule_id)
            if rule is None:
                raise ProgramLearningError(f"program references an unavailable operational rule: {rule_id}")
            if rule.status is not RuleStatus.VERIFIED:
                raise ProgramLearningError(f"program depends on a rule without independent verification: {rule_id}")
            if program.domain not in rule.domains and "general" not in rule.domains and "*" not in rule.domains:
                raise ProgramLearningError(f"program depends on a rule outside its domain: {rule_id}")
            for node in _walk(rule.body):
                if getattr(node, "operator", None) and node.operator not in _OPS:
                    pending_rules.append(node.operator)

        postconditions: tuple[str, ...] | None = None
        verifier_ids: set[str] = set()
        checks_passed: set[str] = set()
        evidence_fingerprints: list[str] = []
        for observation in samples:
            candidate = observation.capability
            if candidate.provenance.source not in self._CANDIDATE_SOURCES or candidate.executable:
                raise ProgramLearningError("observations must verify an unpromoted UOL synthesis/transfer candidate")
            if candidate.provenance.source_ref != program.fingerprint:
                raise ProgramLearningError("candidate provenance is not bound to the exact UOL program")
            if candidate.domains != (program.domain,):
                raise ProgramLearningError("candidate domain does not match the UOL program domain")
            if candidate.capability_id != observation.execution.steps[0].capability_id:
                raise ProgramLearningError("execution did not run the candidate capability")
            if len(observation.execution.steps) != 1 or not observation.execution.succeeded:
                raise ProgramLearningError("program learning requires one successful candidate execution per task")
            report = observation.verification
            if report.status is not DomainVerificationStatus.VERIFIED or report.blockers:
                raise ProgramLearningError("every observation must have a fully verified domain report")
            if len(report.capabilities) != 1 or not report.capabilities[0].passed:
                raise ProgramLearningError("domain report must verify the candidate capability")
            step = observation.execution.steps[0]
            if report.capabilities[0].step_id != step.step_id:
                raise ProgramLearningError("domain report does not cover the executed candidate step")
            if any(
                item.step_id != step.step_id or item.capability_id != candidate.capability_id
                for item in report.results
            ):
                raise ProgramLearningError("domain evidence does not bind to the exact candidate step")
            if not report.results or any(item.decision.value != "pass" for item in report.results):
                raise ProgramLearningError("all independent domain checks must pass")
            if len(report.verifier_ids) < minimum_verifiers:
                raise ProgramLearningError("each observation needs distinct independent verifier identities")
            actual_conditions = tuple(sorted(report.verified_conditions))
            if not actual_conditions:
                raise ProgramLearningError("domain evidence must cover at least one postcondition")
            if postconditions is None:
                postconditions = actual_conditions
            elif postconditions != actual_conditions:
                raise ProgramLearningError("verified postconditions changed across observations")

            actual_inputs = {
                key: value for key, value in step.inputs.items()
                if key != "__mirror_verification__"
            }
            try:
                expected = language.execute(program, actual_inputs)
            except Exception as exc:
                raise ProgramLearningError(f"candidate replay failed: {type(exc).__name__}: {exc}") from exc
            if set(step.outputs) != {output_name} or not self._close(step.outputs[output_name], expected):
                raise ProgramLearningError("execution output does not match the immutable UOL program replay")
            if candidate.postconditions and set(candidate.postconditions) != set(actual_conditions):
                raise ProgramLearningError("candidate postconditions do not match verified evidence")

            verifier_ids.update(report.verifier_ids)
            checks_passed.update(f"{item.verifier_id}:{item.check_id}" for item in report.results)
            evidence_fingerprints.append(report.fingerprint)

        if len(verifier_ids) < minimum_verifiers:
            raise ProgramLearningError("insufficient independent verifier identities across observations")
        if not checks_passed or postconditions is None:
            raise ProgramLearningError("no passing domain-verification evidence was collected")

        first = samples[0].capability
        program_ports = tuple(PortContract(item.name, item.value_type) for item in program.inputs)
        source_ports = {item.name: item for item in first.inputs}
        extra_input_ports = set(source_ports) - {item.name for item in program_ports}
        if extra_input_ports - {"__mirror_verification__"}:
            raise ProgramLearningError("candidate contains undeclared semantic input ports")
        for port in program_ports:
            if source_ports.get(port.name) is None or source_ports[port.name].value_type != port.value_type:
                raise ProgramLearningError("candidate input contract does not match the UOL program")
        if output_name not in {item.name for item in first.outputs}:
            raise ProgramLearningError("candidate output contract is missing the program result port")
        output_port = next(item for item in first.outputs if item.name == output_name)
        if output_port.value_type not in {output_type, "any"} and not (
            output_type == "integer" and output_port.value_type == "number"
        ):
            raise ProgramLearningError("candidate output contract is incompatible with the UOL program result")
        # Preserve the complete source port metadata (units and dimensions); do not
        # silently strip physical meaning while converting the candidate to a capability.
        inputs = [source_ports[port.name] for port in program_ports]
        if "__mirror_verification__" in source_ports:
            reserved = source_ports["__mirror_verification__"]
            if reserved.value_type not in {"object", "record", "mapping", "any"}:
                raise ProgramLearningError("reserved verification port must accept an object")
        else:
            reserved = PortContract("__mirror_verification__", "object", required=False)
        inputs.append(reserved)

        identity = {
            "program": program.fingerprint,
            "domain": program.domain,
            "tasks": sorted(task_ids),
            "executions": sorted(execution_fingerprints),
            "evidence": sorted(evidence_fingerprints),
            "verifiers": sorted(verifier_ids),
        }
        capability_id = "learned-uol:" + stable_representation_id("program", identity)
        verification = VerificationEvidence(
            level=EvidenceLevel.INDEPENDENTLY_REPRODUCED,
            checks_passed=tuple(sorted(checks_passed)),
            verifier="independent domain verifiers: " + ", ".join(sorted(verifier_ids)),
        )
        capability = CapabilitySpec(
            capability_id=capability_id,
            name="Learned UOL operation: " + program.name,
            description=(
                f"UOL program learned from {len(samples)} distinct executions with "
                "execution-bound independent domain verification and regression replay."
            ),
            domains=(program.domain,),
            intents=intent_values,
            inputs=tuple(inputs),
            outputs=(output_port,),
            preconditions=first.preconditions,
            postconditions=postconditions,
            provenance=RepresentationProvenance(
                source="mirror_verified_uol_program_learning",
                source_ref=program.fingerprint,
                parent_ids=tuple(sorted(rule_ids)),
            ),
            verification=verification,
        )
        rules = tuple(language.rules.values())
        handler = UOLProgramHandler(program, rules, output_name)
        try:
            pickle.dumps(handler)
        except Exception as exc:
            raise ProgramLearningError("learned UOL handler is not process-serializable") from exc
        return UOLProgramLearningProposal(
            capability, program, handler, task_ids, execution_fingerprints,
            tuple(sorted(verifier_ids)), tuple(sorted(evidence_fingerprints)),
            minimum_successes, minimum_verifiers, True,
        )

    @staticmethod
    def promote(
        proposal: UOLProgramLearningProposal,
        capability_registry: CapabilityRegistry,
        operation_registry: NativeOperationRegistry,
    ) -> str:
        """Explicitly register an accepted capability and its process-safe UOL handler."""
        if not isinstance(proposal, UOLProgramLearningProposal):
            raise TypeError("proposal must be UOLProgramLearningProposal")
        if not isinstance(capability_registry, CapabilityRegistry) or not isinstance(operation_registry, NativeOperationRegistry):
            raise TypeError("promotion requires CapabilityRegistry and NativeOperationRegistry")
        if not proposal.requires_explicit_promotion or not proposal.capability.executable:
            raise ProgramLearningError("proposal is not eligible for explicit promotion")
        existing_capability = capability_registry.get(proposal.capability.capability_id)
        existing_handler = operation_registry.get(proposal.capability.capability_id)
        if existing_capability is not None and existing_capability.fingerprint != proposal.capability.fingerprint:
            raise ProgramLearningError("capability identifier conflicts with an existing registry entry")
        if existing_handler is not None and existing_handler != proposal.handler:
            raise ProgramLearningError("a different native handler is already registered")
        if existing_capability is not None and existing_handler is not None:
            return proposal.capability.capability_id
        operation_registry.register(proposal.capability.capability_id, proposal.handler)
        capability_registry.add(proposal.capability)
        return proposal.capability.capability_id

    @staticmethod
    def persist(vault: Any, proposal: UOLProgramLearningProposal) -> str:
        if not isinstance(proposal, UOLProgramLearningProposal):
            raise TypeError("proposal must be UOLProgramLearningProposal")
        store = getattr(vault, "store_capability", None)
        if not callable(store):
            raise TypeError("vault must expose store_capability")
        return store(proposal.capability)
