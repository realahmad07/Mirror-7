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
    UOL, UOLError, Lit, Var, Call, If, MapExpr, FoldExpr, Program,
    RuleStatus, VariableSpec, expr_dict, _walk, _infer, _type_name, _freeze, _matches,
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
    enable_conditionals: bool = True
    enable_map: bool = True
    enable_fold: bool = True

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
        if any(type(value) is not bool for value in (
            self.include_assumed_rules, self.enable_conditionals, self.enable_map, self.enable_fold
        )):
            raise SynthesisError("synthesis grammar switches must be booleans")
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
    VERSION = "capability-synthesizer-v3-oe-cegis"
    _COMMUTATIVE_OPS = frozenset({
        "add", "multiply", "minimum", "maximum", "equal", "and", "or",
    })

    def __init__(self, language: UOL | None = None):
        self.language = language or UOL()

    @staticmethod
    def _norm_obs(value: Any) -> Any:
        if isinstance(value, bool):
            return ("b", value)
        if isinstance(value, int):
            return ("i", value)
        if isinstance(value, float):
            if not math.isfinite(value):
                return ("f", "nan")
            # Keep integer-valued floats distinct from ints only by type tag in _obs_key
            return ("f", round(value, 9))
        if isinstance(value, str):
            return ("s", value)
        if value is None:
            return ("null",)
        if isinstance(value, (tuple, list)):
            return ("arr", tuple(CapabilitySynthesizer._norm_obs(v) for v in value))
        if isinstance(value, Mapping):
            return ("rec", tuple((k, CapabilitySynthesizer._norm_obs(v)) for k, v in sorted(value.items())))
        return ("other", repr(value))

    def _obs_key(
        self,
        expr: Any,
        value_type: str,
        training_envs: tuple[dict[str, Any], ...],
    ) -> Any:
        """Compute observational equivalence signature on training examples."""
        outs = []
        for idx, env in enumerate(training_envs):
            try:
                val = self.language._eval(expr, dict(env), [0], 0, ())
                outs.append(self._norm_obs(val))
            except Exception:
                outs.append(("__err__", idx))
        return (value_type, tuple(outs))

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
        rules = [
            rule for rule in self.language.rules.values()
            if (goal.domain in rule.domains or "general" in rule.domains or "*" in rule.domains)
            and rule.status not in {RuleStatus.UNKNOWN, RuleStatus.DISPUTED}
            and (goal.include_assumed_rules or rule.status is not RuleStatus.ASSUMED)
        ]
        # Order by (domain_match_priority, arity, rule_id) so exact-domain and low-arity rules
        # are explored first and high-arity rules never starve composition search.
        rules.sort(key=lambda r: (0 if goal.domain in r.domains else 1, len(r.inputs), r.rule_id))
        return tuple(rules)

    def _generate_layer(
        self,
        goal: SynthesisGoal,
        depth: int,
        seen: dict[str, tuple[Any, str, int]],
        seen_obs: set[Any],
        front: list[tuple[Any, str, int]],
        budget: list[int],
        training_envs: tuple[dict[str, Any], ...],
        input_env: dict[str, str],
        eligible_rules: tuple[Any, ...],
    ) -> tuple[list[tuple[Any, str, int]], bool]:
        additions: list[tuple[Any, str, int]] = []
        pool = list(seen.values())

        def try_add(expr: Any, output_type: str, expr_depth: int) -> None:
            key = json.dumps(expr_dict(expr), sort_keys=True)
            if key in seen:
                return
            obs = self._obs_key(expr, output_type, training_envs)
            if obs in seen_obs:
                return
            seen_obs.add(obs)
            seen[key] = (expr, output_type, expr_depth)
            additions.append((expr, output_type, expr_depth))

        # 1. Eligible UOL rules FIRST (prioritizes rule composition over raw primitives),
        # while reserving budget for primitive ops when the rule catalog is large.
        per_high_arity_cap = max(64, goal.candidate_budget // max(4, len(eligible_rules) * 2))
        rule_phase_limit = (
            budget[0] + int((goal.candidate_budget - budget[0]) * 0.6)
            if len(eligible_rules) > 10
            else goal.candidate_budget
        )
        for rule in eligible_rules:
            if budget[0] >= rule_phase_limit:
                break
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
            if not choices and rule.inputs:
                continue
            selections = itertools.product(*choices) if choices else [()]
            rule_tried = 0
            for selection in selections:
                if 1 + max((item[2] for item in selection), default=0) > depth:
                    continue
                if len(rule.inputs) >= 3 and rule_tried >= per_high_arity_cap:
                    break
                if budget[0] >= rule_phase_limit:
                    break
                if budget[0] >= goal.candidate_budget:
                    return additions, True
                budget[0] += 1
                rule_tried += 1
                candidate = Call(rule.rule_id, tuple(item[0] for item in selection))
                try:
                    output_type = _infer(candidate, input_env, self.language.rules, set())
                except UOLError:
                    continue
                try_add(candidate, output_type, depth)

        # 2. Unary primitives (including array sum and product)
        for expression, _, _ in front:
            for operator in ("negate", "absolute", "not", "length", "sum", "product"):
                if budget[0] >= goal.candidate_budget:
                    return additions, True
                budget[0] += 1
                candidate = Call(operator, (expression,))
                try:
                    output_type = _infer(candidate, input_env, self.language.rules, set())
                except UOLError:
                    continue
                try_add(candidate, output_type, depth)

        # 3. Binary primitives (with commutative deduplication)
        for left_idx, left in enumerate(pool):
            for right_idx, right in enumerate(pool):
                if 1 + max(left[2], right[2]) > depth:
                    continue
                if budget[0] >= goal.candidate_budget:
                    return additions, True
                for operator in (
                    "add", "subtract", "multiply", "divide", "modulo", "power",
                    "minimum", "maximum", "equal", "less_than", "less_equal",
                    "greater_than", "greater_equal", "and", "or", "concat",
                ):
                    if operator in self._COMMUTATIVE_OPS and right_idx < left_idx:
                        continue
                    budget[0] += 1
                    candidate = Call(operator, (left[0], right[0]))
                    try:
                        output_type = _infer(candidate, input_env, self.language.rules, set())
                    except UOLError:
                        continue
                    try_add(candidate, output_type, depth)
                    if budget[0] >= goal.candidate_budget:
                        return additions, True

        # 4. Higher-order collection construction (Map + Fold)
        array_ports = [item for item in goal.inputs if item.value_type == "array"]
        scalar_ports = [item for item in goal.inputs if item.value_type in {"integer", "number", "boolean", "text"}]
        constants = [(Var(port.name), port.value_type) for port in scalar_ports]
        for value in goal.constants:
            if isinstance(value, bool):
                value_type = "boolean"
            elif isinstance(value, int):
                value_type = "integer"
            elif isinstance(value, float):
                value_type = "number"
            elif isinstance(value, str):
                value_type = "text"
            elif value is None:
                value_type = "null"
            else:
                continue
            constants.append((Lit(value), value_type))
        element = Var("__mirror_item")
        accumulator = Var("__mirror_acc")
        map_env = {**input_env, "__mirror_item": "any"}

        # Allow MapExpr for array output OR as intermediate for sum(map(...)) / mean(map(...))
        allow_map_here = goal.enable_map and array_ports and (
            goal.output_type in {"array", "any"} or depth < goal.max_depth
        )
        if allow_map_here:
            # 4a. Map with 1-arg UOL rules (available starting at depth 1 so sum(map(rule)) works at depth 2!)
            unary_rules = [r for r in eligible_rules if len(r.inputs) == 1]
            for array_port in array_ports:
                for rule in unary_rules:
                    if budget[0] >= goal.candidate_budget:
                        return additions, True
                    body = Call(rule.rule_id, (element,))
                    try:
                        _infer(body, map_env, self.language.rules, set())
                    except UOLError:
                        continue
                    expr = MapExpr(Var(array_port.name), "__mirror_item", body)
                    budget[0] += 1
                    try_add(expr, "array", depth)

            # 4b. Map with primitive ops
            if depth >= 2:
                for array_port in array_ports:
                    body_terms = [(element, "any"), *constants]
                    for term, _ in body_terms:
                        for operator in ("negate", "absolute", "not", "length"):
                            if budget[0] >= goal.candidate_budget:
                                return additions, True
                            body = Call(operator, (term,))
                            try:
                                _infer(body, map_env, self.language.rules, set())
                            except UOLError:
                                continue
                            expr = MapExpr(Var(array_port.name), "__mirror_item", body)
                            budget[0] += 1
                            try_add(expr, "array", depth)
                        for other, _ in body_terms:
                            for operator in ("add", "subtract", "multiply", "minimum", "maximum", "equal", "less_than", "greater_than"):
                                if budget[0] >= goal.candidate_budget:
                                    return additions, True
                                body = Call(operator, (term, other))
                                try:
                                    _infer(body, map_env, self.language.rules, set())
                                except UOLError:
                                    continue
                                expr = MapExpr(Var(array_port.name), "__mirror_item", body)
                                budget[0] += 1
                                try_add(expr, "array", depth)

        if goal.enable_fold and array_ports and depth >= 2 and goal.output_type in {"integer", "number", "any"}:
            for array_port in array_ports:
                for initial in (0, 1):
                    for operator in ("add", "multiply", "minimum", "maximum"):
                        if budget[0] >= goal.candidate_budget:
                            return additions, True
                        body = Call(operator, (accumulator, element))
                        expr = FoldExpr(
                            Var(array_port.name), Lit(initial),
                            "__mirror_acc", "__mirror_item", body,
                        )
                        try:
                            output_type = _infer(expr, input_env, self.language.rules, set())
                        except UOLError:
                            continue
                        budget[0] += 1
                        try_add(expr, output_type, depth)

        # 5. Conditional construction
        if goal.enable_conditionals:
            cond_pool = list(seen.values())
            conditions = [item for item in cond_pool if item[1] in {"boolean", "any"} and item[2] <= 1][:16]
            branches = [item for item in cond_pool if item[1] in {"integer", "number", "text", "boolean", "any"} and item[2] <= 1][:24]
            for condition in conditions:
                for yes in branches:
                    for no in branches:
                        if yes[0] == no[0]:
                            continue
                        if budget[0] >= goal.candidate_budget:
                            return additions, True
                        if yes[1] != no[1] and not {yes[1], no[1]} <= {"integer", "number", "any"}:
                            continue
                        expr = If(condition[0], yes[0], no[0])
                        expr_depth = 1 + max(condition[2], yes[2], no[2])
                        if expr_depth > depth:
                            continue
                        budget[0] += 1
                        try:
                            output_type = _infer(expr, input_env, self.language.rules, set())
                        except UOLError:
                            continue
                        try_add(expr, output_type, expr_depth)

        return additions, False

    def _generate(self, goal, budget):
        training_envs = tuple({k: _freeze(v) for k, v in s.inputs.items()} for s in goal.training)
        input_env = {item.name: item.value_type for item in goal.inputs}
        eligible_rules = self._eligible_rules(goal)
        terminals = self._terminals(goal)
        seen: dict[str, tuple[Any, str, int]] = {}
        seen_obs: set[Any] = set()
        for expr, value_type in terminals:
            key = json.dumps(expr_dict(expr), sort_keys=True)
            if key not in seen:
                seen[key] = (expr, value_type, 0)
                seen_obs.add(self._obs_key(expr, value_type, training_envs))
        front = list(seen.values())
        for depth in range(1, goal.max_depth + 1):
            additions, exhausted = self._generate_layer(
                goal, depth, seen, seen_obs, front, budget, training_envs, input_env, eligible_rules,
            )
            if exhausted:
                return list(seen.values()), True
            front = additions
            if not front:
                break
        return list(seen.values()), False

    @staticmethod
    def _rule_refs(expression, known):
        refs = set()
        for node in _walk(expression):
            if isinstance(node, Call) and node.operator in known:
                refs.add(node.operator)
        return tuple(sorted(refs))

    @staticmethod
    def _equal(left, right):
        if isinstance(left, bool) or isinstance(right, bool):
            return type(left) is type(right) and left == right
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(
                CapabilitySynthesizer._equal(a, b) for a, b in zip(left, right)
            )
        return left == right

    def _check_candidates(self, goal: SynthesisGoal, layer: list[tuple[Any, str, int]], known_rule_ids: set[str], tested: list[int]):
        for expression, value_type, _depth in sorted(
            layer,
            key=lambda item: (item[2], len(json.dumps(expr_dict(item[0]), sort_keys=True))),
        ):
            if value_type not in {goal.output_type, "any"} and not (
                goal.output_type == "number" and value_type == "integer"
            ):
                continue
            refs = self._rule_refs(expression, known_rule_ids)
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
            tested[0] += 1
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
                    SynthesisStatus.SYNTHESIZED, program, tested[0],
                    len(goal.training), validation_passed, refs,
                )
        return None

    def _synthesize_once(self, goal):
        budget = [0]
        tested = [0]
        known_rule_ids = {rule.rule_id for rule in self.language.rules.values()}
        training_envs = tuple({k: _freeze(v) for k, v in s.inputs.items()} for s in goal.training)
        input_env = {item.name: item.value_type for item in goal.inputs}
        eligible_rules = self._eligible_rules(goal)

        terminals = self._terminals(goal)
        seen: dict[str, tuple[Any, str, int]] = {}
        seen_obs: set[Any] = set()
        for expr, value_type in terminals:
            key = json.dumps(expr_dict(expr), sort_keys=True)
            if key not in seen:
                seen[key] = (expr, value_type, 0)
                seen_obs.add(self._obs_key(expr, value_type, training_envs))
        front = list(seen.values())

        hit = self._check_candidates(goal, front, known_rule_ids, tested)
        if hit is not None:
            return hit, budget[0]

        exhausted = False
        for depth in range(1, goal.max_depth + 1):
            additions, exhausted = self._generate_layer(
                goal, depth, seen, seen_obs, front, budget, training_envs, input_env, eligible_rules,
            )
            hit = self._check_candidates(goal, additions, known_rule_ids, tested)
            if hit is not None:
                return hit, budget[0]
            if exhausted:
                break
            front = additions
            if not front:
                break

        status = SynthesisStatus.BUDGET_EXHAUSTED if exhausted else SynthesisStatus.NO_SOLUTION
        blocker = "search budget exhausted" if exhausted else "no candidate passed all training and held-out examples"
        return SynthesisReport(status, None, tested[0], 0, 0, blockers=(blocker,)), budget[0]

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
                # A passing probe is not final evidence until a candidate passes every
                # remaining probe. Counterexamples are added to training and are
                # revalidated by the next synthesis round, so count the complete
                # original probe pool only when the loop finishes successfully.

            if discovered is None:
                return SynthesisReport(
                    SynthesisStatus.SYNTHESIZED, report.program, total_tested,
                    len(active_goal.training), len(goal.validation), report.rule_ids,
                    rounds=round_number, counterexamples=tuple(counterexamples),
                    probes_passed=len(probes),
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
