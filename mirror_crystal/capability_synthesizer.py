"""Bounded example-guided and counterexample-guided discovery of UOL programs."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import itertools
import json
import math
from typing import Any, Callable, Iterable, Mapping

from .operational_language import (
    UOL, UOLError, Lit, Var, Call, If, Program, RuleStatus, VariableSpec,
    expr_dict, _infer, _type_name, _freeze, _matches,
)
from .blueprint_contracts import EvidenceLevel, VerificationEvidence
from .capability_model import CapabilitySpec, PortContract
from .universal_representation import RepresentationProvenance


class SynthesisError(ValueError):
    pass


class SynthesisStatus(str, Enum):
    SYNTHESIZED = "synthesized"
    NO_SOLUTION = "no_solution"
    BUDGET_EXHAUSTED = "budget_exhausted"
    COUNTEREXAMPLE_LIMIT = "counterexample_limit"


@dataclass(frozen=True)
class Example:
    inputs: Mapping[str, Any]
    expected: Any


@dataclass(frozen=True)
class Counterexample:
    inputs: Mapping[str, Any]
    expected: Any
    actual: Any = None
    round_found: int = 1
    reason: str = "candidate disagreed with the supplied oracle"


@dataclass(frozen=True)
class SynthesisGoal:
    name: str
    domain: str
    inputs: tuple[Any, ...]
    output_type: str
    training: tuple[Example, ...]
    validation: tuple[Example, ...]
    constants: tuple[Any, ...] = (0, 1, 2, -1)
    max_depth: int = 2
    candidate_budget: int = 20000
    include_assumed_rules: bool = True

    def __post_init__(self):
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if any(not isinstance(item, VariableSpec) for item in self.inputs):
            raise SynthesisError("inputs must be VariableSpec items")
        object.__setattr__(self, "training", tuple(self.training))
        object.__setattr__(self, "validation", tuple(self.validation))
        object.__setattr__(self, "constants", tuple(self.constants))
        object.__setattr__(self, "domain", self.domain.casefold())
        object.__setattr__(self, "output_type", _type_name(self.output_type))
        if len(self.training) < 2 or not self.validation:
            raise SynthesisError("need at least two training and one held-out example")
        if self.max_depth < 0 or self.max_depth > 4 or not 1 <= self.candidate_budget <= 250000:
            raise SynthesisError("synthesis search limits out of range")
        names = [item.name for item in self.inputs]
        if len(names) != len(set(names)):
            raise SynthesisError("input names must be unique")
        required = set(names)
        seen = set()
        for example in self.training + self.validation:
            if set(example.inputs) != required:
                raise SynthesisError("example inputs must match synthesis schema")
            key = json.dumps(dict(example.inputs), sort_keys=True, separators=(",", ":"))
            if key in seen:
                raise SynthesisError("training and validation inputs must be disjoint")
            seen.add(key)


@dataclass(frozen=True)
class SynthesisReport:
    status: SynthesisStatus
    program: Any | None
    tested: int
    train_passed: int
    validation_passed: int
    rule_ids: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    rounds: int = 1
    counterexamples: tuple[Counterexample, ...] = ()
    probes_passed: int = 0

    @property
    def succeeded(self):
        return self.status is SynthesisStatus.SYNTHESIZED and self.program is not None


class CapabilitySynthesizer:
    VERSION = "capability-synthesizer-v2-cegis"

    def __init__(self, language: UOL | None = None):
        self.language = language or UOL()

    def _terminals(self, goal):
        out = []
        for port in goal.inputs:
            out.append((Var(port.name), port.value_type))
        for value in goal.constants:
            if isinstance(value, bool):
                kind = "boolean"
            elif isinstance(value, int):
                kind = "integer"
            elif isinstance(value, float):
                kind = "number"
            elif isinstance(value, str):
                kind = "text"
            elif value is None:
                kind = "null"
            else:
                continue
            out.append((Lit(value), kind))
        return out

    def _eligible_rules(self, goal):
        return tuple(
            rule for rule in self.language.rules.values()
            if (goal.domain in rule.domains or "general" in rule.domains or "*" in rule.domains)
            and rule.status not in {RuleStatus.UNKNOWN, RuleStatus.DISPUTED}
            and (goal.include_assumed_rules or rule.status is not RuleStatus.ASSUMED)
        )

    def _generate(self, goal, budget):
        terminals = self._terminals(goal)
        seen = {
            json.dumps(expr_dict(expr), sort_keys=True): (expr, value_type, 0)
            for expr, value_type in terminals
        }
        front = list(seen.values())

        def collected(additions):
            merged = dict(seen)
            for item in additions:
                merged.setdefault(json.dumps(expr_dict(item[0]), sort_keys=True), item)
            return list(merged.values())

        for depth in range(1, goal.max_depth + 1):
            additions = []
            for expression, _, _ in front:
                for operator in ("negate", "absolute", "not", "length"):
                    if budget[0] >= goal.candidate_budget:
                        return collected(additions), True
                    budget[0] += 1
                    try:
                        output_type = _infer(
                            Call(operator, (expression,)),
                            {item.name: item.value_type for item in goal.inputs},
                            self.language.rules, set(),
                        )
                    except UOLError:
                        continue
                    candidate = Call(operator, (expression,))
                    key = json.dumps(expr_dict(candidate), sort_keys=True)
                    if key not in seen:
                        additions.append((candidate, output_type, depth))

            pool = list(seen.values())
            for left in pool:
                for right in pool:
                    if budget[0] >= goal.candidate_budget:
                        return collected(additions), True
                    for operator in (
                        "add", "subtract", "multiply", "divide", "modulo", "power",
                        "minimum", "maximum", "equal", "less_than", "less_equal",
                        "greater_than", "greater_equal", "and", "or", "concat",
                    ):
                        budget[0] += 1
                        candidate = Call(operator, (left[0], right[0]))
                        try:
                            output_type = _infer(
                                candidate,
                                {item.name: item.value_type for item in goal.inputs},
                                self.language.rules, set(),
                            )
                        except UOLError:
                            continue
                        if 1 + max(left[2], right[2]) > depth:
                            continue
                        key = json.dumps(expr_dict(candidate), sort_keys=True)
                        if key not in seen:
                            additions.append((candidate, output_type, depth))
                    if budget[0] >= goal.candidate_budget:
                        return collected(additions), True

            for rule in self._eligible_rules(goal):
                choices = []
                for port in rule.inputs:
                    matching = [
                        item for item in pool
                        if port.value_type in {"any", item[1]}
                        or (port.value_type == "number" and item[1] == "integer")
                    ]
                    if not matching:
                        choices = []
                        break
                    choices.append(matching)
                selections = itertools.product(*choices) if choices else [()]
                for selection in selections:
                    if budget[0] >= goal.candidate_budget:
                        return collected(additions), True
                    budget[0] += 1
                    candidate = Call(rule.rule_id, tuple(item[0] for item in selection))
                    try:
                        output_type = _infer(
                            candidate,
                            {item.name: item.value_type for item in goal.inputs},
                            self.language.rules, set(),
                        )
                    except UOLError:
                        continue
                    if 1 + max((item[2] for item in selection), default=0) > depth:
                        continue
                    key = json.dumps(expr_dict(candidate), sort_keys=True)
                    if key not in seen:
                        additions.append((candidate, output_type, depth))

            for item in additions:
                key = json.dumps(expr_dict(item[0]), sort_keys=True)
                if key not in seen:
                    seen[key] = item
            front = additions
            if not front:
                break
        return list(seen.values()), False

    @staticmethod
    def _rule_refs(expression, known):
        refs = set()
        def visit(node):
            if isinstance(node, Call):
                if node.operator in known:
                    refs.add(node.operator)
                for argument in node.args:
                    visit(argument)
            elif isinstance(node, If):
                visit(node.condition)
                visit(node.yes)
                visit(node.no)
        visit(expression)
        return tuple(sorted(refs))

    @staticmethod
    def _equal(left, right):
        if isinstance(left, bool) or isinstance(right, bool):
            return type(left) is type(right) and left == right
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
        return left == right

    def _synthesize_once(self, goal):
        budget = [0]
        candidates, exhausted = self._generate(goal, budget)
        tested = 0
        for expression, value_type, depth in sorted(
            candidates,
            key=lambda item: (item[2], len(json.dumps(expr_dict(item[0]), sort_keys=True))),
        ):
            if value_type not in {goal.output_type, "any"} and not (
                goal.output_type == "number" and value_type == "integer"
            ):
                continue
            refs = self._rule_refs(
                expression, {rule.rule_id for rule in self.language.rules.values()}
            )
            program = Program(
                "uol-candidate:" + hashlib.sha256(
                    json.dumps(expr_dict(expression), sort_keys=True).encode()
                ).hexdigest()[:20],
                goal.name, goal.inputs, expression, goal.domain, rule_refs=refs,
            )
            try:
                self.language.validate(program)
            except UOLError:
                continue
            tested += 1
            passed = True
            for sample in goal.training:
                try:
                    actual = self.language.execute(program, sample.inputs)
                except (UOLError, RuntimeError, TypeError, ValueError, OverflowError):
                    passed = False
                    break
                if not self._equal(actual, sample.expected):
                    passed = False
                    break
            if not passed:
                continue
            validation_passed = 0
            for sample in goal.validation:
                try:
                    actual = self.language.execute(program, sample.inputs)
                except (UOLError, RuntimeError, TypeError, ValueError, OverflowError):
                    passed = False
                    break
                if not self._equal(actual, sample.expected):
                    passed = False
                    break
                validation_passed += 1
            if passed and validation_passed == len(goal.validation):
                return SynthesisReport(
                    SynthesisStatus.SYNTHESIZED, program, tested,
                    len(goal.training), validation_passed, refs,
                ), budget[0]
        status = SynthesisStatus.BUDGET_EXHAUSTED if exhausted else SynthesisStatus.NO_SOLUTION
        blocker = "search budget exhausted" if exhausted else "no candidate passed all training and held-out examples"
        return SynthesisReport(status, None, tested, 0, 0, blockers=(blocker,)), budget[0]

    def synthesize(
        self,
        goal: SynthesisGoal,
        *,
        counterexample_inputs: Iterable[Mapping[str, Any]] = (),
        counterexample_oracle: Callable[[Mapping[str, Any]], Any] | None = None,
        max_rounds: int = 8,
        counterexample_budget: int = 32,
    ) -> SynthesisReport:
        """Run bounded CEGIS using an explicit host-supplied oracle and probe pool."""
        if not isinstance(goal, SynthesisGoal):
            raise TypeError("goal must be SynthesisGoal")
        probes = tuple(dict(item) for item in counterexample_inputs)
        if bool(probes) != (counterexample_oracle is not None):
            raise SynthesisError("counterexample probes and an explicit oracle must be supplied together")
        if not 1 <= max_rounds <= 16 or not 0 <= counterexample_budget <= 256:
            raise SynthesisError("CEGIS round or counterexample budget out of range")
        if len(probes) > 1000:
            raise SynthesisError("counterexample probe pool exceeds 1000 inputs")

        required = {item.name for item in goal.inputs}
        used_inputs = {
            json.dumps(dict(item.inputs), sort_keys=True, separators=(",", ":"))
            for item in goal.training + goal.validation
        }
        seen_probes = set()
        for probe in probes:
            if set(probe) != required:
                raise SynthesisError("counterexample probe inputs must match synthesis schema")
            key = json.dumps(probe, sort_keys=True, separators=(",", ":"))
            if key in used_inputs or key in seen_probes:
                raise SynthesisError("counterexample probes must be unique and disjoint from train/validation")
            seen_probes.add(key)
            for spec in goal.inputs:
                try:
                    if not _matches(_freeze(probe[spec.name]), spec.value_type):
                        raise SynthesisError(f"counterexample input {spec.name!r} has the wrong type")
                except (TypeError, ValueError) as exc:
                    raise SynthesisError(f"invalid counterexample input {spec.name!r}: {exc}") from exc

        if not probes:
            report, _ = self._synthesize_once(goal)
            return report

        active_goal = goal
        remaining = list(probes)
        counterexamples: list[Counterexample] = []
        total_tested = 0
        total_budget_used = 0
        probes_passed = 0
        for round_number in range(1, max_rounds + 1):
            # The per-round search limit is fixed. The total work is still bounded
            # by candidate_budget multiplied by the maximum number of rounds.
            round_goal = replace(active_goal, candidate_budget=goal.candidate_budget)
            report, budget_used = self._synthesize_once(round_goal)
            total_budget_used += budget_used
            total_tested += report.tested
            if not report.succeeded:
                return SynthesisReport(
                    report.status, None, total_tested, 0, 0,
                    blockers=report.blockers, rounds=round_number,
                    counterexamples=tuple(counterexamples), probes_passed=probes_passed,
                )

            discovered = None
            for probe in remaining:
                try:
                    expected = _freeze(counterexample_oracle(probe))
                    if not _matches(expected, goal.output_type):
                        raise SynthesisError("counterexample oracle returned a value with the wrong output type")
                except SynthesisError:
                    raise
                except Exception as exc:
                    raise SynthesisError(f"counterexample oracle failed: {exc}") from exc
                try:
                    actual = self.language.execute(report.program, probe)
                    matches = self._equal(actual, expected)
                except (UOLError, RuntimeError, TypeError, ValueError, OverflowError) as exc:
                    actual = None
                    matches = False
                    reason = f"candidate execution failed: {type(exc).__name__}: {exc}"
                else:
                    reason = "candidate output disagreed with the supplied oracle"
                if not matches:
                    discovered = Counterexample(
                        inputs=probe, expected=expected, actual=actual,
                        round_found=round_number, reason=reason,
                    )
                    break
                probes_passed += 1

            if discovered is None:
                return SynthesisReport(
                    SynthesisStatus.SYNTHESIZED, report.program, total_tested,
                    len(active_goal.training), len(goal.validation), report.rule_ids,
                    rounds=round_number, counterexamples=tuple(counterexamples),
                    probes_passed=probes_passed,
                )

            counterexamples.append(discovered)
            remaining.remove(discovered.inputs)
            if len(counterexamples) > counterexample_budget or round_number >= max_rounds:
                return SynthesisReport(
                    SynthesisStatus.COUNTEREXAMPLE_LIMIT, None, total_tested, 0, 0,
                    blockers=("counterexample/round limit reached before a candidate passed all probes",),
                    rounds=round_number, counterexamples=tuple(counterexamples),
                    probes_passed=probes_passed,
                )
            if not remaining:
                return SynthesisReport(
                    SynthesisStatus.COUNTEREXAMPLE_LIMIT, None, total_tested, 0, 0,
                    blockers=("all probes were consumed by counterexamples; provide a fresh held-out probe",),
                    rounds=round_number, counterexamples=tuple(counterexamples),
                    probes_passed=probes_passed,
                )
            active_goal = replace(
                active_goal,
                training=active_goal.training + (Example(discovered.inputs, discovered.expected),),
            )

        return SynthesisReport(
            SynthesisStatus.COUNTEREXAMPLE_LIMIT, None, total_tested, 0, 0,
            blockers=("CEGIS round limit reached",), rounds=max_rounds,
            counterexamples=tuple(counterexamples), probes_passed=probes_passed,
        )

    def to_candidate_capability(self, report: SynthesisReport, *, intent="any", output_name="result"):
        if not report.succeeded:
            raise SynthesisError("only a synthesized report can become a capability candidate")
        program = report.program
        return CapabilitySpec(
            capability_id=program.program_id,
            name=program.name,
            description="Synthesized UOL candidate; requires independent verification",
            domains=(program.domain,),
            intents=(intent,),
            inputs=tuple(PortContract(item.name, item.value_type) for item in program.inputs),
            outputs=(PortContract(output_name, "any"),),
            preconditions=("UOL preconditions",) if program.preconditions else (),
            postconditions=("UOL postconditions",) if program.postconditions else (),
            provenance=RepresentationProvenance(
                source="mirror_uol_synthesizer",
                source_ref=program.fingerprint,
                parent_ids=report.rule_ids,
            ),
            verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
        )
