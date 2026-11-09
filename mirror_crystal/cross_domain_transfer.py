"""Conservative cross-domain transfer of verified, domain-neutral UOL procedures.

Transfer never copies source-domain trust into a target domain. It requires globally scoped,
independently verified UOL dependencies, unchanged typed/unit interfaces, and fresh target
training and held-out examples. The result remains UNVERIFIED until target-domain evidence
is produced and passed through candidate verification plus Phase 5 learning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .blueprint_contracts import EvidenceLevel, VerificationEvidence
from .capability_model import CapabilitySpec, PortContract
from .capability_synthesizer import Example
from .operational_language import Program, RuleStatus, UOL, UOLError, _OPS, _walk
from .universal_representation import RepresentationProvenance, stable_representation_id
from .verified_program_learning import UOLProgramLearningProposal


class CrossDomainTransferError(ValueError):
    """Raised when a procedure cannot safely transfer into a target domain."""


@dataclass(frozen=True)
class CrossDomainTransferProposal:
    capability: CapabilitySpec
    program: Program
    source_capability_id: str
    source_domain: str
    target_domain: str
    training_passed: int
    validation_passed: int
    rule_ids: tuple[str, ...]
    requires_target_verification: bool = True


class CrossDomainTransferEngine:
    """Transfer only domain-neutral, verified operation structure with target regression."""

    VERSION = "cross-domain-transfer-v1"

    @staticmethod
    def _validate_samples(
        samples: tuple[Example, ...],
        input_names: set[str],
        *,
        label: str,
        used: set[str],
    ) -> None:
        for item in samples:
            if not isinstance(item, Example):
                raise TypeError(f"{label} must contain Example values")
            if set(item.inputs) != input_names:
                raise CrossDomainTransferError(f"{label} inputs do not match the program schema")
            key = stable_representation_id("transfer-input", dict(item.inputs))
            if key in used:
                raise CrossDomainTransferError("target training and held-out inputs must be globally disjoint")
            used.add(key)

    @staticmethod
    def _domain_neutral_rules(program: Program, language: UOL) -> tuple[str, ...]:
        pending = list(program.rule_refs)
        visited: set[str] = set()
        while pending:
            rule_id = pending.pop()
            if rule_id in visited:
                continue
            visited.add(rule_id)
            rule = language.rules.get(rule_id)
            if rule is None:
                raise CrossDomainTransferError(f"missing UOL dependency: {rule_id}")
            if rule.status is not RuleStatus.VERIFIED:
                raise CrossDomainTransferError(f"UOL dependency is not independently verified: {rule_id}")
            if not ({"general", "*"} & set(rule.domains)):
                raise CrossDomainTransferError(
                    f"domain-specific rule cannot be transferred without a target-domain derivation: {rule_id}"
                )
            for node in _walk(rule.body):
                if getattr(node, "operator", None) and node.operator not in _OPS:
                    pending.append(node.operator)
        return tuple(sorted(visited))

    def propose(
        self,
        source: UOLProgramLearningProposal,
        language: UOL,
        *,
        target_domain: str,
        training: Iterable[Example],
        validation: Iterable[Example],
        target_postconditions: Iterable[str],
        minimum_training: int = 3,
        minimum_validation: int = 2,
    ) -> CrossDomainTransferProposal:
        if not isinstance(source, UOLProgramLearningProposal) or not isinstance(language, UOL):
            raise TypeError("source and language must be UOLProgramLearningProposal and UOL")
        target = str(target_domain).strip().casefold()
        if not target:
            raise CrossDomainTransferError("target_domain must not be empty")
        if target == source.program.domain:
            raise CrossDomainTransferError("source and target domains must differ")
        source_capability = source.capability
        source_program = source.program
        if not source_capability.executable or source_capability.verification.level not in {
            EvidenceLevel.FORMALLY_VERIFIED, EvidenceLevel.INDEPENDENTLY_REPRODUCED,
        }:
            raise CrossDomainTransferError("source program must already be independently verified")
        if source_program.domain not in source_capability.domains:
            raise CrossDomainTransferError("source program domain is not declared by its capability")
        if (
            source_capability.provenance.source != "mirror_verified_uol_program_learning"
            or source_capability.provenance.source_ref != source_program.fingerprint
        ):
            raise CrossDomainTransferError("source capability provenance is not bound to this program")
        if isinstance(minimum_training, bool) or not isinstance(minimum_training, int) or minimum_training < 2:
            raise ValueError("minimum_training must be at least 2")
        if isinstance(minimum_validation, bool) or not isinstance(minimum_validation, int) or minimum_validation < 1:
            raise ValueError("minimum_validation must be at least 1")

        training_examples = tuple(training)
        validation_examples = tuple(validation)
        if len(training_examples) < minimum_training or len(validation_examples) < minimum_validation:
            raise CrossDomainTransferError("insufficient fresh target-domain training/held-out examples")
        names = {item.name for item in source_program.inputs}
        used: set[str] = set()
        self._validate_samples(training_examples, names, label="training", used=used)
        self._validate_samples(validation_examples, names, label="validation", used=used)

        source_ports = {item.name: item for item in source_capability.inputs}
        for spec in source_program.inputs:
            port = source_ports.get(spec.name)
            if port is None or port.value_type != spec.value_type:
                raise CrossDomainTransferError("source capability input contract differs from its UOL program")
        extra_ports = set(source_ports) - names
        if extra_ports - {"__mirror_verification__"}:
            raise CrossDomainTransferError("source capability has undeclared semantic input ports")
        if "__mirror_verification__" in source_ports:
            reserved = source_ports["__mirror_verification__"]
            if reserved.required or reserved.value_type not in {"object", "record", "mapping", "any"}:
                raise CrossDomainTransferError("reserved verification envelope must be optional and accept an object")
        if len(source_capability.outputs) != 1:
            raise CrossDomainTransferError("transfer currently requires one explicitly typed result port")

        dependencies = self._domain_neutral_rules(source_program, language)
        transferred = Program(
            program_id=stable_representation_id(
                "uol-transfer",
                {"source_program": source_program.fingerprint, "target_domain": target},
            ),
            name=f"{source_program.name} [{target}]",
            inputs=source_program.inputs,
            body=source_program.body,
            domain=target,
            preconditions=source_program.preconditions,
            postconditions=source_program.postconditions,
            rule_refs=source_program.rule_refs,
        )
        try:
            output_type = language.validate(transferred)
        except UOLError as exc:
            raise CrossDomainTransferError(f"target UOL validation failed: {exc}") from exc

        for label, samples in (("training", training_examples), ("held-out validation", validation_examples)):
            for sample in samples:
                try:
                    result = language.execute(transferred, sample.inputs)
                except Exception as exc:
                    raise CrossDomainTransferError(
                        f"target {label} execution failed: {type(exc).__name__}: {exc}"
                    ) from exc
                if not _same_value(result, sample.expected):
                    raise CrossDomainTransferError(
                        f"transferred procedure failed a target-domain {label} example"
                    )

        postconditions = tuple(sorted({str(item).strip() for item in target_postconditions}))
        if not postconditions or any(not item for item in postconditions):
            raise CrossDomainTransferError("target-specific postconditions must be explicit and non-empty")
        output_port = source_capability.outputs[0]
        if output_port.value_type not in {output_type, "any"} and not (
            output_type == "integer" and output_port.value_type == "number"
        ):
            raise CrossDomainTransferError("transferred UOL output type differs from source capability contract")

        # Carry forward the exact typed/unit/dimension interface. No conversion is inferred.
        inputs = [source_ports[item.name] for item in transferred.inputs]
        if "__mirror_verification__" in source_ports:
            inputs.append(source_ports["__mirror_verification__"])
        else:
            inputs.append(PortContract("__mirror_verification__", "object", required=False))
        transfer_id = "transfer-uol:" + stable_representation_id(
            "target-capability",
            {
                "source_capability": source_capability.fingerprint,
                "target_program": transferred.fingerprint,
                "training": sorted(stable_representation_id("example", {"inputs": dict(item.inputs), "expected": item.expected}) for item in training_examples),
                "validation": sorted(stable_representation_id("example", {"inputs": dict(item.inputs), "expected": item.expected}) for item in validation_examples),
                "postconditions": list(postconditions),
            },
        )
        capability = CapabilitySpec(
            capability_id=transfer_id,
            name=f"Transferred: {source_capability.name} → {target}",
            description=(
                "Cross-domain UOL candidate; validated on separate target training and held-out examples. "
                "Requires target-domain independent verification before promotion."
            ),
            domains=(target,),
            intents=source_capability.intents,
            inputs=tuple(inputs),
            outputs=(output_port,),
            preconditions=source_capability.preconditions,
            postconditions=postconditions,
            provenance=RepresentationProvenance(
                source="mirror_cross_domain_transfer",
                source_ref=transferred.fingerprint,
                parent_ids=(source_capability.capability_id, *dependencies),
            ),
            verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
        )
        return CrossDomainTransferProposal(
            capability, transferred, source_capability.capability_id,
            source_program.domain, target, len(training_examples),
            len(validation_examples), dependencies, True,
        )


def _same_value(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        import math
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    return left == right
