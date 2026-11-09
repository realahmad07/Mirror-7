"""Universal capability planning, composition, execution, verification and learning.

This module is domain-neutral. Domain adapters contribute typed operators and
independent verifiers; the planner, graph composer, execution contract, evidence
ledger and learning gate are shared by every domain. No external model is called.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from decimal import Decimal, InvalidOperation
import hashlib
import hmac
import marshal
import secrets
import re
from typing import Any, Callable, Mapping

from .universal_representation import (
    CapabilitySpec,
    EvidenceRecord,
    LearningGateReport,
    PlannedStep,
    ProvenanceRecord,
    RepresentationKind,
    RepresentationNode,
    SubtaskSpec,
    TaskPlan,
    TaskSpec,
    VerificationLevel,
    VerificationReport,
    canonical_json,
    stable_id,
    type_compatible,
    value_matches_type,
)


class CapabilityError(RuntimeError):
    """Raised when a capability cannot safely be planned or executed."""


class CapabilityRegistry:
    """Registry for typed capabilities, executable handlers, and non-executable knowledge."""

    def __init__(self) -> None:
        self._specs: dict[str, CapabilitySpec] = {}
        self._handlers: dict[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]] = {}
        self._knowledge: dict[str, RepresentationNode] = {}

    def register(
        self,
        spec: CapabilitySpec,
        handler: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        *,
        self_test: Callable[[], bool] | None = None,
    ) -> CapabilitySpec:
        if spec.capability_id in self._specs:
            raise ValueError(f"capability already registered: {spec.capability_id}")
        if handler is not None and not callable(handler):
            raise TypeError("capability handler must be callable")
        effective = spec
        if handler is not None and self_test is not None:
            passed = bool(self_test())
            if passed and not spec.evidence.trusted_for_execution:
                effective = replace(
                    spec,
                    evidence=EvidenceRecord(
                        level=VerificationLevel.EXECUTION_TESTED,
                        verifier="Mirror capability registration self-test",
                        checks_passed=(f"self-test:{spec.capability_id}",),
                        details={"origin": "local_builtin_or_adapter"},
                    ),
                )
            elif not passed:
                effective = replace(
                    spec,
                    evidence=EvidenceRecord(
                        level=VerificationLevel.UNVERIFIED,
                        checks_failed=(f"self-test:{spec.capability_id}",),
                        details={"reason": "registration self-test failed"},
                    ),
                )
        if handler is not None and not effective.trusted:
            # An unverified proposal may be stored, but cannot become executable.
            self._specs[effective.capability_id] = effective
            return effective
        self._specs[effective.capability_id] = effective
        if handler is not None:
            self._handlers[effective.capability_id] = handler
        return effective

    def register_knowledge(self, node: RepresentationNode) -> None:
        if node.node_id in self._knowledge:
            raise ValueError(f"representation node already registered: {node.node_id}")
        self._knowledge[node.node_id] = node

    def get(self, capability_id: str) -> CapabilitySpec | None:
        return self._specs.get(capability_id)

    def handler(self, capability_id: str):
        return self._handlers.get(capability_id)

    def load_learned_blueprints(
        self,
        blueprints: list[Any] | tuple[Any, ...],
        handler_factory: Callable[[CapabilitySpec], Callable[[Mapping[str, Any]], Mapping[str, Any]] | None],
    ) -> int:
        """Load only independently reproduced universal capabilities with a replayable implementation."""
        loaded = 0
        for blueprint in blueprints:
            if str(getattr(blueprint, "intent", "")) != "universal_capability.v1":
                continue
            blueprint_evidence = getattr(blueprint, "verification", None)
            blueprint_level = getattr(
                getattr(blueprint_evidence, "level", VerificationLevel.UNVERIFIED),
                "value",
                getattr(blueprint_evidence, "level", VerificationLevel.UNVERIFIED),
            )
            if (
                blueprint_level != VerificationLevel.INDEPENDENTLY_REPRODUCED.value
                or not str(getattr(blueprint_evidence, "verifier", "")).strip()
            ):
                continue
            try:
                import json
                spec = CapabilitySpec.from_dict(json.loads(str(blueprint.response_template)))
            except Exception:
                continue
            if (
                not spec.evidence.independently_reproduced
                or not spec.implementation
                or blueprint.blueprint_id != f"universal:{spec.capability_id}"
                or blueprint.domain not in spec.domains
            ):
                continue
            handler = handler_factory(spec)
            if handler is None:
                continue
            try:
                self.register(spec, handler)
            except ValueError:
                continue
            loaded += 1
        return loaded

    def list_capabilities(self, *, domain: str | None = None) -> tuple[CapabilitySpec, ...]:
        items = list(self._specs.values())
        if domain:
            target = domain.casefold()
            items = [x for x in items if target in x.domains or "general" in x.domains]
        return tuple(sorted(items, key=lambda x: x.capability_id))

    def executable(self, *, domain: str | None = None) -> tuple[CapabilitySpec, ...]:
        return tuple(
            item for item in self.list_capabilities(domain=domain)
            if item.capability_id in self._handlers and item.trusted
        )

    def add_legacy_blueprints(self, blueprints: list[Any] | tuple[Any, ...]) -> int:
        """Index blueprint knowledge without upgrading its evidence or inventing an executor."""
        added = 0
        for blueprint in blueprints:
            provenance = getattr(blueprint, "provenance", None)
            verification = getattr(blueprint, "verification", None)
            level = getattr(verification, "level", VerificationLevel.UNVERIFIED)
            evidence = EvidenceRecord(
                level=VerificationLevel(getattr(level, "value", level)),
                verifier=str(getattr(verification, "verifier", "")),
                checks_passed=tuple(getattr(verification, "checks_passed", ()) or ()),
                checks_failed=tuple(getattr(verification, "checks_failed", ()) or ()),
            )
            node = RepresentationNode(
                node_id=str(getattr(blueprint, "blueprint_id")),
                kind=RepresentationKind.PROCEDURE if str(getattr(getattr(blueprint, "kind", None), "value", "")) == "procedure" else RepresentationKind.RULE,
                domain=str(getattr(blueprint, "domain", "general")),
                label=str(getattr(blueprint, "intent", "legacy blueprint")),
                content={
                    "prompt_pattern": str(getattr(blueprint, "prompt_pattern", "")),
                    "response_template": str(getattr(blueprint, "response_template", "")),
                    "derived_expr": str(getattr(blueprint, "derived_expr", "")),
                },
                attributes={
                    "executable": False,
                    "reason": "legacy blueprint has no registered typed executor",
                },
                dependencies=tuple(getattr(blueprint, "dependencies", ()) or ()),
                provenance=ProvenanceRecord(
                    source=str(getattr(provenance, "source", "legacy_blueprint")),
                    source_ref=str(getattr(provenance, "source_ref", "")),
                    license_spdx=str(getattr(provenance, "license_spdx", "NOASSERTION")),
                    content_sha256=str(getattr(provenance, "content_sha256", "")),
                    captured_at=str(getattr(provenance, "captured_at", "")),
                ),
                evidence=evidence,
            )
            if node.node_id not in self._knowledge:
                self.register_knowledge(node)
                added += 1
        return added

    def knowledge_nodes(self, *, domain: str | None = None) -> tuple[RepresentationNode, ...]:
        values = list(self._knowledge.values())
        if domain:
            values = [x for x in values if x.domain == domain or x.domain == "general"]
        return tuple(sorted(values, key=lambda x: x.node_id))

    def search_knowledge(
        self,
        query: str,
        *,
        domain: str | None = None,
        kinds: tuple[RepresentationKind, ...] = (),
        limit: int = 20,
    ) -> tuple[RepresentationNode, ...]:
        """Rank stored domain knowledge by token overlap without changing its trust level."""
        terms = {
            token for token in re.findall(r"[a-z0-9_]+", query.casefold())
            if len(token) > 2
        }
        candidates = list(self.knowledge_nodes(domain=domain))
        if kinds:
            allowed = {RepresentationKind(x) for x in kinds}
            candidates = [node for node in candidates if node.kind in allowed]
        ranked = []
        for node in candidates:
            text = " ".join((node.label, node.domain, str(node.content))).casefold()
            node_terms = set(re.findall(r"[a-z0-9_]+", text))
            score = len(terms & node_terms)
            if score:
                ranked.append((-score, node.node_id, node))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return tuple(item[2] for item in ranked[:max(1, min(int(limit), 100))])


class UniversalTaskInterpreter:
    """Native request interpreter with injectable in-process domain decomposition rules.

    No network/model call is made. Domain packs may register deterministic or native
    Mirror-owned interpretation callbacks that return a typed TaskSpec.
    """

    def __init__(self) -> None:
        self._domain_interpreters: dict[str, Callable[..., Any]] = {}

    def register_domain_interpreter(self, domain: str, interpreter: Callable[..., Any]) -> None:
        key = domain.casefold().strip()
        if not key or not callable(interpreter):
            raise ValueError("domain and callable interpreter are required")
        if key in self._domain_interpreters:
            raise ValueError(f"domain interpreter already registered: {key}")
        self._domain_interpreters[key] = interpreter

    _ARITHMETIC = re.compile(
        r"(?P<left>-?\d+(?:\.\d+)?)\s*"
        r"(?P<op>\+|\-|\*|/|plus|add(?:ed)?|minus|subtract(?:ed)?|"
        r"times|multiplied\s+by|divided\s+by)\s*"
        r"(?P<right>-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    _DOMAIN_TERMS = {
        "programming": ("code", "function", "program", "calculator", "api", "script", "software"),
        "mathematics": ("math", "calculate", "equation", "matrix", "integral", "derivative", "probability"),
        "physics": ("physics", "force", "velocity", "energy", "momentum", "quantum"),
        "biology": ("biology", "cell", "gene", "protein", "organism", "ecosystem"),
        "chemistry": ("chemistry", "molecule", "reaction", "compound", "atom", "ph"),
        "engineering": ("engineering", "circuit", "beam", "mechanism", "control system"),
    }

    def interpret(
        self,
        prompt: str,
        *,
        domain_hint: str | None = None,
        inputs: Mapping[str, Any] | None = None,
        acceptance_criteria: tuple[Mapping[str, Any], ...] = (),
    ) -> TaskSpec:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        text = prompt.strip()
        lower = text.casefold()
        domain = domain_hint.casefold() if domain_hint else "general"
        if not domain_hint:
            for candidate, terms in self._DOMAIN_TERMS.items():
                if any(re.search(r"\b" + re.escape(term) + r"\b", lower) for term in terms):
                    domain = candidate
                    break

        domain_interpreter = self._domain_interpreters.get(domain)
        if domain_interpreter is not None:
            interpreted = domain_interpreter(text, inputs=dict(inputs or {}))
            if isinstance(interpreted, Mapping):
                interpreted = TaskSpec.from_dict(interpreted)
            if not isinstance(interpreted, TaskSpec):
                raise TypeError(f"{domain} interpreter must return TaskSpec or mapping")
            if interpreted.domain != domain:
                raise ValueError("domain interpreter returned a TaskSpec for the wrong domain")
            return interpreted

        if domain == "programming" and re.search(r"\\b(build|create|make|write|implement)\\b", lower) and re.search(r"\\bcalculator\\b", lower):
            tests = (
                'assert calculate(2, 3, "add") == 5\\n'
                'assert calculate(5, 2, "subtract") == 3\\n'
                'assert calculate(3, 4, "multiply") == 12\\n'
                'assert calculate(8, 2, "divide") == 4\\n'
            )
            subtask = SubtaskSpec(
                subtask_id="synthesize_calculator",
                goal="Construct a calculator program from Mirror-native typed operations.",
                domain="programming",
                capability_tags=("program.calculator.synthesize",),
                capability_id="program.calculator.synthesize",
                output_schema={"code": "string", "program_ir": "mapping"},
            )
            return TaskSpec(
                task_id=stable_id("task", {"request": text, "artifact": "native_calculator"}),
                request=text,
                goal="Build an executable calculator supporting addition, subtraction, multiplication and division.",
                domain="programming",
                required_outputs={"code": "string", "program_ir": "mapping"},
                required_capability_tags=("program.calculator.synthesize",),
                subtasks=(subtask,),
                context={"tests": tests, "artifact_type": "python_source"},
            )

        match = self._ARITHMETIC.search(text)
        if match:
            left_text, right_text = match.group("left"), match.group("right")
            left = float(left_text) if "." in left_text else int(left_text)
            right = float(right_text) if "." in right_text else int(right_text)
            op = re.sub(r"\s+", " ", match.group("op").casefold())
            capability = {
                "+": "numeric.add", "plus": "numeric.add", "add": "numeric.add", "added": "numeric.add",
                "-": "numeric.subtract", "minus": "numeric.subtract", "subtract": "numeric.subtract", "subtracted": "numeric.subtract",
                "*": "numeric.multiply", "times": "numeric.multiply", "multiplied by": "numeric.multiply",
                "/": "numeric.divide", "divided by": "numeric.divide",
            }[op]
            task_inputs = {"left": left, "right": right}
            if inputs:
                task_inputs.update(dict(inputs))
            subtask = SubtaskSpec(
                subtask_id="compute",
                goal=f"Compute {left} {op} {right}",
                domain="mathematics",
                capability_tags=(capability,),
                capability_id=capability,
                input_bindings={"left": "input:left", "right": "input:right"},
                output_schema={"result": "number"},
            )
            return TaskSpec(
                task_id=stable_id("task", {"request": text, "inputs": task_inputs}),
                request=text,
                goal="Evaluate the requested arithmetic expression exactly.",
                domain="mathematics",
                inputs=task_inputs,
                input_schema={"left": "number", "right": "number"},
                required_outputs={"result": "number"},
                required_capability_tags=(capability,),
                subtasks=(subtask,),
                acceptance_criteria=tuple(acceptance_criteria),
            )

        task_inputs = dict(inputs or {})
        task_id = stable_id("task", {"request": text, "domain": domain, "inputs": task_inputs})
        tags: tuple[str, ...] = ()
        if domain == "programming":
            tags = ("program.structure", "program.construct")
        elif domain in {"physics", "biology", "chemistry", "engineering"}:
            tags = (f"{domain}.model", f"{domain}.verify")
        elif domain == "mathematics":
            tags = ("mathematics.reason", "mathematics.verify")
        return TaskSpec(
            task_id=task_id,
            request=text,
            goal=text,
            domain=domain,
            inputs=task_inputs,
            required_capability_tags=tags,
            acceptance_criteria=tuple(acceptance_criteria),
        )


class UniversalTaskPlanner:
    """Build a deterministic task DAG and match each subtask to typed capabilities."""

    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry

    @staticmethod
    def _ordered(subtasks: tuple[SubtaskSpec, ...]) -> tuple[SubtaskSpec, ...]:
        by_id = {item.subtask_id: item for item in subtasks}
        ordered: list[SubtaskSpec] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise CapabilityError(f"task dependency cycle at {step_id}")
            if step_id in visited:
                return
            item = by_id.get(step_id)
            if item is None:
                raise CapabilityError(f"unknown task dependency: {step_id}")
            visiting.add(step_id)
            for dep in item.dependencies:
                visit(dep)
            visiting.remove(step_id)
            visited.add(step_id)
            ordered.append(item)

        for item in sorted(subtasks, key=lambda x: x.subtask_id):
            visit(item.subtask_id)
        return tuple(ordered)

    @staticmethod
    def _domain_matches(capability: CapabilitySpec, domain: str) -> bool:
        return domain in capability.domains or "general" in capability.domains

    def _select(self, subtask: SubtaskSpec) -> CapabilitySpec | None:
        candidates = []
        for capability in self.registry.executable(domain=subtask.domain):
            if not self._domain_matches(capability, subtask.domain):
                continue
            if subtask.capability_id and capability.capability_id != subtask.capability_id:
                continue
            if subtask.capability_tags and not set(subtask.capability_tags).issubset(set(capability.tags) | {capability.capability_id}):
                continue
            if subtask.output_schema:
                if set(subtask.output_schema) != set(capability.output_schema):
                    continue
                if any(
                    not type_compatible(capability.output_schema[key], expected)
                    for key, expected in subtask.output_schema.items()
                ):
                    continue
            if subtask.input_bindings and set(subtask.input_bindings) != set(capability.input_schema):
                continue
            score = (
                100 if subtask.capability_id == capability.capability_id and subtask.capability_id else 0
            ) + 20 * len(set(subtask.capability_tags) & (set(capability.tags) | {capability.capability_id}))
            score += 10 if subtask.domain in capability.domains else 0
            candidates.append((score, capability.capability_id, capability))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1]))
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            raise CapabilityError(
                f"ambiguous capability match for subtask {subtask.subtask_id}: "
                f"{candidates[0][1]}, {candidates[1][1]}"
            )
        return candidates[0][2]

    def plan(self, task: TaskSpec) -> TaskPlan:
        subtasks = task.subtasks
        if not subtasks:
            subtasks = (
                SubtaskSpec(
                    subtask_id="goal",
                    goal=task.goal,
                    domain=task.domain,
                    capability_tags=task.required_capability_tags,
                    input_bindings={
                        key: f"input:{key}" for key in task.inputs
                    },
                    output_schema=task.required_outputs,
                ),
            )
        ordered = self._ordered(tuple(subtasks))
        planned: list[PlannedStep] = []
        unresolved: list[str] = []
        for subtask in ordered:
            capability = self._select(subtask)
            if capability is None:
                tags = ", ".join(subtask.capability_tags) or "compatible typed capability"
                unresolved.append(
                    f"{subtask.subtask_id}: no trusted executable capability for {subtask.domain} ({tags})"
                )
                continue
            bindings = dict(subtask.input_bindings)
            if not bindings:
                bindings = {
                    name: f"input:{name}"
                    for name in capability.input_schema
                }
            missing = [
                name for name, source in bindings.items()
                if not source.startswith("input:") and not source.startswith("step:")
            ]
            if missing:
                unresolved.append(f"{subtask.subtask_id}: invalid input binding(s): {', '.join(missing)}")
                continue
            planned.append(PlannedStep(
                step_id=subtask.subtask_id,
                subtask=subtask,
                capability_id=capability.capability_id,
                input_bindings=bindings,
            ))
        plan_payload = {
            "task": task.fingerprint,
            "steps": [(x.step_id, x.capability_id, dict(x.input_bindings)) for x in planned],
            "unresolved": unresolved,
        }
        return TaskPlan(
            plan_id=stable_id("plan", plan_payload),
            task_id=task.task_id,
            task_fingerprint=task.fingerprint,
            steps=tuple(planned),
            unresolved=tuple(unresolved),
        )


class UniversalExecutionEngine:
    """Execute a capability DAG with strict inputs, output schemas, and fail-closed errors."""

    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry

    @staticmethod
    def _resolve_source(
        source: str,
        task: TaskSpec,
        outputs: Mapping[str, Mapping[str, Any]],
    ) -> Any:
        if source.startswith("input:"):
            key = source[len("input:"):]
            if key not in task.inputs:
                raise CapabilityError(f"missing task input: {key}")
            return task.inputs[key]
        if source.startswith("step:"):
            path = source[len("step:"):]
            step_id, separator, output_name = path.partition(".")
            if not separator or step_id not in outputs or output_name not in outputs[step_id]:
                raise CapabilityError(f"unavailable dependency output: {source}")
            return outputs[step_id][output_name]
        raise CapabilityError(f"unsupported binding expression: {source}")

    def execute(self, task: TaskSpec, plan: TaskPlan) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if plan.task_fingerprint != task.fingerprint:
            raise CapabilityError("task changed after planning; refusing stale plan execution")
        if plan.unresolved or not plan.steps:
            raise CapabilityError("cannot execute an incomplete task plan")
        executions = []
        step_outputs: dict[str, Mapping[str, Any]] = {}
        for planned in plan.steps:
            spec = self.registry.get(planned.capability_id)
            handler = self.registry.handler(planned.capability_id)
            if spec is None or handler is None or not spec.trusted:
                raise CapabilityError(f"capability is no longer executable: {planned.capability_id}")
            resolved = {
                input_name: self._resolve_source(source, task, step_outputs)
                for input_name, source in planned.input_bindings.items()
            }
            if set(resolved) != set(spec.input_schema):
                raise CapabilityError(f"{spec.capability_id}: resolved input schema mismatch")
            for name, expected in spec.input_schema.items():
                if not value_matches_type(resolved[name], expected):
                    raise CapabilityError(
                        f"{spec.capability_id}: input {name} expected {expected}, got {type(resolved[name]).__name__}"
                    )
            result = handler(resolved)
            if not isinstance(result, Mapping):
                raise CapabilityError(f"{spec.capability_id}: handler must return a mapping")
            produced = dict(result)
            if set(produced) != set(spec.output_schema):
                raise CapabilityError(f"{spec.capability_id}: output schema mismatch")
            for name, expected in spec.output_schema.items():
                if not value_matches_type(produced[name], expected):
                    raise CapabilityError(
                        f"{spec.capability_id}: output {name} expected {expected}, got {type(produced[name]).__name__}"
                    )
            step_outputs[planned.step_id] = produced
            executions.append({
                "step_id": planned.step_id,
                "capability_id": spec.capability_id,
                "inputs": resolved,
                "outputs": produced,
                "succeeded": True,
            })
        flattened = {
            f"{step_id}.{name}": value
            for step_id, values in step_outputs.items()
            for name, value in values.items()
        }
        final_outputs = dict(step_outputs[plan.steps[-1].step_id])
        return tuple(executions), {"steps": step_outputs, "flat": flattened, "final": final_outputs}


class UniversalVerifier:
    """Independent contract verifier with pluggable domain-specific proof adapters."""

    def __init__(self) -> None:
        self._domain_verifiers: dict[str, Callable[[TaskSpec, TaskPlan, tuple[Any, ...], Mapping[str, Any]], tuple[bool, tuple[str, ...], tuple[str, ...]]]] = {}
        self.register_domain_verifier("mathematics", self._verify_arithmetic)
        self.register_domain_verifier("programming", self._verify_program)
        self.register_domain_verifier("general", self._verify_generic)

    def register_domain_verifier(self, domain: str, verifier: Callable) -> None:
        self._domain_verifiers[str(domain).casefold()] = verifier

    @staticmethod
    def _lookup(field: str, outputs: Mapping[str, Any]) -> Any:
        if field in outputs.get("flat", {}):
            return outputs["flat"][field]
        if field in outputs.get("final", {}):
            return outputs["final"][field]
        raise KeyError(f"verification field not found: {field}")

    @staticmethod
    def _check_criteria(criteria: tuple[Mapping[str, Any], ...], outputs: Mapping[str, Any]):
        passed: list[str] = []
        failed: list[str] = []
        for index, criterion in enumerate(criteria):
            field = str(criterion.get("field", ""))
            operation = str(criterion.get("op", "equals"))
            expected = criterion.get("expected")
            try:
                actual = UniversalVerifier._lookup(field, outputs)
                ok = {
                    "equals": lambda: actual == expected,
                    "not_equals": lambda: actual != expected,
                    "greater_than": lambda: actual > expected,
                    "greater_or_equal": lambda: actual >= expected,
                    "less_than": lambda: actual < expected,
                    "less_or_equal": lambda: actual <= expected,
                    "contains": lambda: expected in actual,
                    "non_empty": lambda: bool(actual),
                    "type": lambda: value_matches_type(actual, str(expected)),
                    "truthy": lambda: bool(actual),
                }.get(operation)
                if ok is None:
                    raise ValueError(f"unsupported criterion operation: {operation}")
                if ok():
                    passed.append(f"criterion:{index}:{field}:{operation}")
                else:
                    failed.append(f"criterion:{index}:{field}:{operation}")
            except Exception as exc:
                failed.append(f"criterion:{index}:{field}:{operation}:{type(exc).__name__}")
        return tuple(passed), tuple(failed)

    @staticmethod
    def _verify_program(task, plan, executions, outputs):
        """Syntax-check and TDD-test native program artifacts inside Mirror's sandbox."""
        code = outputs.get("final", {}).get("code")
        if not isinstance(code, str) or not code.strip():
            return False, (), ("program_source_missing",)
        try:
            from .phase7_qwen_to_opus import AutonomousTDDEngine
            test_code = str(task.context.get("tests", "") or "")
            if not test_code.strip():
                return False, (), ("program_verification_tests_missing",)
            report = AutonomousTDDEngine().run_tdd_cycle(
                code,
                spec_desc=task.goal,
                explicit_tests=test_code,
            )
            count = int(getattr(report, "tests_generated", 0) or 0)
            passed = int(getattr(report, "tests_passed", 0) or 0)
            if report.verified and count > 0 and passed == count:
                return True, (
                    "python_source_compiled",
                    f"tdd_assertions_passed:{passed}/{count}",
                    "sandbox_execution_passed",
                ), ()
            return False, (), (
                f"tdd_verification_failed:{passed}/{count}",
                *tuple(getattr(report, "diagnostics", ()) or ()),
            )
        except Exception as exc:
            return False, (), (f"program_verifier_error:{type(exc).__name__}",)

    @staticmethod
    def _verify_generic(task, plan, executions, outputs):
        criteria = task.acceptance_criteria + tuple(
            criterion
            for subtask in task.subtasks
            for criterion in subtask.acceptance_criteria
        )
        if not criteria:
            return False, (), ("generic_verification_requires_acceptance_criteria",)
        passed, failed = UniversalVerifier._check_criteria(criteria, outputs)
        return not failed, passed, failed

    @staticmethod
    def _verify_arithmetic(task, plan, executions, outputs):
        from decimal import Decimal
        independent: list[str] = []
        failed: list[str] = []
        criterion_passed, criterion_failed = UniversalVerifier._check_criteria(
            task.acceptance_criteria, outputs
        )
        independent.extend(criterion_passed)
        failed.extend(criterion_failed)
        operation_map = {
            "numeric.add": lambda a, b: Decimal(str(a)) + Decimal(str(b)),
            "numeric.subtract": lambda a, b: Decimal(str(a)) - Decimal(str(b)),
            "numeric.multiply": lambda a, b: Decimal(str(a)) * Decimal(str(b)),
            "numeric.divide": lambda a, b: Decimal(str(a)) / Decimal(str(b)),
        }
        for execution in executions:
            capability_id = execution["capability_id"]
            if capability_id not in operation_map:
                continue
            try:
                args = execution["inputs"]
                expected = operation_map[capability_id](args["left"], args["right"])
                actual = Decimal(str(execution["outputs"]["result"]))
                tolerance = max(abs(expected), Decimal("1")) * Decimal("1e-12")
                if abs(actual - expected) <= tolerance:
                    independent.append(f"independent_decimal_recompute:{capability_id}")
                else:
                    failed.append(f"independent_decimal_recompute_failed:{capability_id}")
            except (KeyError, InvalidOperation, ZeroDivisionError, ValueError):
                failed.append(f"independent_decimal_recompute_error:{capability_id}")
        if not any(x.startswith("independent_decimal_recompute:") for x in independent):
            failed.append("no_independent_domain_verifier_applied")
        return not failed, tuple(independent), tuple(failed)

    def verify(
        self,
        task: TaskSpec,
        plan: TaskPlan,
        executions: tuple[Any, ...],
        outputs: Mapping[str, Any],
    ) -> VerificationReport:
        if not executions:
            return VerificationReport(
                passed=False,
                level=VerificationLevel.UNVERIFIED,
                checks_failed=("no_executions",),
                verifier="Mirror UniversalVerifier",
            )
        domain_verifier = self._domain_verifiers.get(task.domain)
        if domain_verifier is None:
            domain_verifier = self._domain_verifiers.get("general", self._verify_generic)
            domain_name = "generic_contract"
        else:
            domain_name = task.domain
        domain_passed, domain_checks_passed, domain_checks_failed = domain_verifier(
            task, plan, executions, outputs
        )

        # Mandatory output contracts and acceptance criteria cannot be bypassed by
        # a domain adapter. The adapter adds an independent check; it never replaces
        # the shared type/contract gate.
        contract_passed: list[str] = []
        contract_failed: list[str] = []
        final_values = outputs.get("final", {})
        for name, expected_type in task.required_outputs.items():
            if name not in final_values:
                contract_failed.append(f"required_output_missing:{name}")
            elif not value_matches_type(final_values[name], expected_type):
                contract_failed.append(f"required_output_wrong_type:{name}:{expected_type}")
            else:
                contract_passed.append(f"required_output_type:{name}:{expected_type}")
        all_criteria = task.acceptance_criteria + tuple(
            criterion
            for subtask in task.subtasks
            for criterion in subtask.acceptance_criteria
        )
        criterion_passed, criterion_failed = self._check_criteria(all_criteria, outputs)
        contract_passed.extend(criterion_passed)
        contract_failed.extend(criterion_failed)

        passed = domain_passed and not contract_failed
        checks_passed = tuple(contract_passed) + tuple(domain_checks_passed)
        checks_failed = tuple(contract_failed) + tuple(domain_checks_failed)
        level = (
            VerificationLevel.FORMALLY_VERIFIED
            if passed and domain_name == "mathematics"
            else VerificationLevel.EXECUTION_TESTED if passed
            else VerificationLevel.UNVERIFIED
        )
        return VerificationReport(
            passed=passed,
            level=level,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            verifier=f"Mirror UniversalVerifier/{domain_name}",
            verified_outputs=dict(outputs.get("final", {})) if passed else {},
            notes=(
                "Generic contract checks do not establish scientific or factual truth.",
                "Domains require independent adapters before results can be formally verified.",
            ),
        )


class CapabilityLearningEngine:
    """Promote a reusable composition only after multi-seed, held-out and adversarial gates."""

    def __init__(
        self,
        registry: CapabilityRegistry,
        gate_authorizer: Callable[[LearningGateReport], bool] | None = None,
    ) -> None:
        self.registry = registry
        self._gate_authorizer = gate_authorizer

    def promote(
        self,
        spec: CapabilitySpec,
        handler: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        gate: LearningGateReport,
        *,
        persist: Callable[[CapabilitySpec], Any] | None = None,
    ) -> CapabilitySpec:
        if self._gate_authorizer is None or not self._gate_authorizer(gate):
            raise CapabilityError(
                "capability learning requires a Mirror-attested run of >=3 cases across "
                ">=3 seeds, held-out and adversarial evaluations, and an independent verifier"
            )
        learned = replace(
            spec,
            evidence=EvidenceRecord(
                level=VerificationLevel.INDEPENDENTLY_REPRODUCED,
                verifier=gate.independent_verifier,
                checks_passed=tuple(
                    f"case_{index + 1}_passed" for index in range(gate.cases_passed)
                ) + ("held_out_passed", "adversarial_controls_passed"),
                held_out_passed=True,
                adversarial_passed=True,
                seed_count=gate.seed_count,
                details={
                    "case_count": gate.cases_generated,
                    "base_case_count": gate.base_case_count,
                    "held_out_case_count": gate.held_out_case_count,
                    "adversarial_case_count": gate.adversarial_case_count,
                    "candidate_runner_sha256": gate.candidate_runner_sha256,
                    "independent_verifier_sha256": gate.independent_verifier_sha256,
                    "learning_gate": "v2",
                    "run_digest": gate.run_digest,
                },
            ),
        )
        self.registry.register(learned, handler)
        if persist is not None:
            persist(learned)
        return learned


class CrossDomainTransferEngine:
    """Transfer capability structure, never source-domain proof, into a new domain."""

    @staticmethod
    def propose(source: CapabilitySpec, target_domain: str) -> CapabilitySpec:
        if not source.transferable:
            raise CapabilityError(f"capability is not marked transferable: {source.capability_id}")
        target = target_domain.casefold().strip()
        if not target or target in source.domains:
            raise ValueError("target domain must be non-empty and different from source domains")
        return replace(
            source,
            capability_id=stable_id("transfer", {"source": source.capability_id, "target": target}),
            name=f"{source.name} (transfer proposal: {target})",
            domains=(target,),
            evidence=EvidenceRecord(
                level=VerificationLevel.UNVERIFIED,
                details={
                    "source_capability": source.capability_id,
                    "target_domain": target,
                    "reason": "transferred structure requires target-domain verification",
                },
            ),
            provenance=ProvenanceRecord(
                source="cross_domain_transfer_proposal",
                source_ref=source.capability_id,
                parent_ids=(source.capability_id,),
            ),
            lineage=source.lineage + (source.capability_id,),
        )


class UniversalTaskCurriculum:
    """Expose domain/capability gaps as explicit curriculum items, not fake successes."""

    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry

    def gaps(self, target_domains: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
        result = []
        for domain in target_domains:
            known = [
                x for x in self.registry.list_capabilities(domain=domain)
                if x.trusted and self.registry.handler(x.capability_id) is not None
            ]
            result.append({
                "domain": domain,
                "trusted_executable_capabilities": [x.capability_id for x in known],
                "capability_count": len(known),
                "status": "capabilities_available" if known else "capability_gap",
            })
        return tuple(result)


class UniversalIntelligenceEngine:
    """Knowledge -> capabilities -> composition -> verification -> learning."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry | None = None,
        verifier: UniversalVerifier | None = None,
        install_primitives: bool = True,
    ) -> None:
        self.registry = registry or CapabilityRegistry()
        self.planner = UniversalTaskPlanner(self.registry)
        self.executor = UniversalExecutionEngine(self.registry)
        self.verifier = verifier or UniversalVerifier()
        self._learning_gate_key = secrets.token_bytes(32)
        self.learner = CapabilityLearningEngine(self.registry, self._is_learning_gate_attested)
        self.transfer = CrossDomainTransferEngine()
        self.curriculum = UniversalTaskCurriculum(self.registry)
        self.interpreter = UniversalTaskInterpreter()
        self._domain_adapters: set[str] = set()
        if install_primitives:
            self._install_primitives()

    @staticmethod
    def _callable_fingerprint(function: Callable[..., Any]) -> str:
        code = getattr(function, "__code__", None)
        if code is None:
            code = getattr(getattr(type(function), "__call__", None), "__code__", None)
        if code is None:
            identity = f"{type(function).__module__}.{type(function).__qualname__}"
            return hashlib.sha256(identity.encode("utf-8")).hexdigest()
        defaults = repr(getattr(function, "__defaults__", None))
        kwdefaults = repr(getattr(function, "__kwdefaults__", None))
        closure = getattr(function, "__closure__", None) or ()
        closure_values = []
        for cell in closure:
            try:
                closure_values.append(repr(cell.cell_contents))
            except ValueError:
                closure_values.append("<empty-cell>")
        material = (
            marshal.dumps(code)
            + defaults.encode("utf-8")
            + kwdefaults.encode("utf-8")
            + repr(tuple(closure_values)).encode("utf-8")
        )
        return hashlib.sha256(material).hexdigest()

    @staticmethod
    def _learning_gate_payload(report: LearningGateReport) -> str:
        payload = asdict(report)
        payload.pop("attestation", None)
        return canonical_json(payload)

    def _sign_learning_gate(self, report: LearningGateReport) -> str:
        return hmac.new(
            self._learning_gate_key,
            self._learning_gate_payload(report).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _is_learning_gate_attested(self, report: LearningGateReport) -> bool:
        if not isinstance(report, LearningGateReport) or not report.passed:
            return False
        expected = self._sign_learning_gate(report)
        return hmac.compare_digest(report.attestation, expected)

    def _require_learning_gate(self, report: LearningGateReport) -> None:
        if not self._is_learning_gate_attested(report):
            raise CapabilityError(
                "learning gate is missing a valid Mirror run attestation; "
                "construct it with evaluate_learning_gate() on this engine instance"
            )

    def evaluate_learning_gate(
        self,
        candidate: Any,
        cases: tuple[Any, ...] | list[Any],
        seeds: tuple[int, ...] | list[int],
        held_out_cases: tuple[Any, ...] | list[Any],
        adversarial_cases: tuple[Any, ...] | list[Any],
        *,
        independent_verifier: str,
        candidate_runner: Callable[[Any, Any, int], Any],
        independent_check: Callable[[Any, Any, Any, str], bool | tuple[bool, str]],
    ) -> LearningGateReport:
        """Run the candidate through base, held-out and adversarial cases, then attest outcomes.

        candidate_runner(candidate, case, seed) executes the proposed capability.
        independent_check(candidate, case, output, suite_name) independently judges
        the result. Any exception, failed judgment, missing suite, or insufficient
        seeds produces a non-promotable report.
        """
        if not callable(candidate_runner) or not callable(independent_check):
            raise TypeError("candidate_runner and independent_check must be callable")
        seed_values = tuple(sorted({int(seed) for seed in seeds}))
        base_cases = tuple(cases)
        held_cases = tuple(held_out_cases)
        adversarial = tuple(adversarial_cases)
        failures: list[str] = []
        passed_count = 0
        total_count = 0
        held_out_passed = bool(held_cases) and bool(seed_values)
        adversarial_passed = bool(adversarial) and bool(seed_values)
        outcomes: list[dict[str, Any]] = []

        def run_suite(name: str, suite: tuple[Any, ...]) -> bool:
            suite_ok = bool(suite) and len(seed_values) >= 1
            for case_index, case in enumerate(suite):
                for seed in seed_values:
                    try:
                        output = candidate_runner(candidate, case, seed)
                        verdict = independent_check(candidate, case, output, name)
                        if isinstance(verdict, tuple):
                            accepted = bool(verdict[0])
                            detail = str(verdict[1]) if len(verdict) > 1 else ""
                        else:
                            accepted = bool(verdict)
                            detail = ""
                    except Exception as exc:
                        accepted = False
                        detail = type(exc).__name__
                    outcomes.append({
                        "suite": name,
                        "case_index": case_index,
                        "seed": seed,
                        "passed": accepted,
                        "detail": detail,
                    })
                    if not accepted:
                        suite_ok = False
                        failures.append(f"{name}:case_{case_index}:seed_{seed}:{detail or 'failed'}")
                    if name == "base":
                        nonlocal_passed[0] += int(accepted)
                        nonlocal_total[0] += 1
            return suite_ok

        nonlocal_passed = [0]
        nonlocal_total = [0]
        run_suite("base", base_cases) if len(seed_values) >= 3 else False
        held_out_passed = run_suite("held_out", held_cases) if len(seed_values) >= 3 else False
        adversarial_passed = run_suite("adversarial", adversarial) if len(seed_values) >= 3 else False
        if len(base_cases) < 3:
            failures.append("base_suite_requires_at_least_3_distinct_cases")
        if len(seed_values) < 3:
            failures.append("learning_gate_requires_at_least_3_seeds")
        if not held_cases:
            failures.append("held_out_suite_missing")
        if not adversarial:
            failures.append("adversarial_suite_missing")
        if not str(independent_verifier).strip():
            failures.append("independent_verifier_identity_missing")

        run_digest = hashlib.sha256(
            canonical_json({
                "candidate": candidate if isinstance(candidate, (str, int, float, bool, dict, list, tuple, type(None))) else repr(candidate),
                "base_cases": base_cases,
                "held_out_cases": held_cases,
                "adversarial_cases": adversarial,
                "seeds": seed_values,
                "outcomes": outcomes,
                "independent_verifier": str(independent_verifier),
                "base_case_count": len(base_cases),
                "held_out_case_count": len(held_cases),
                "adversarial_case_count": len(adversarial),
                "candidate_runner_sha256": self._callable_fingerprint(candidate_runner),
                "independent_verifier_sha256": self._callable_fingerprint(independent_check),
            }).encode("utf-8")
        ).hexdigest()
        unsigned = LearningGateReport(
            cases_generated=nonlocal_total[0],
            cases_passed=nonlocal_passed[0],
            seed_count=len(seed_values),
            held_out_passed=held_out_passed,
            adversarial_passed=adversarial_passed,
            independent_verifier=str(independent_verifier),
            failures=tuple(failures),
            run_digest=run_digest,
            attestation="",
            base_case_count=len(base_cases),
            held_out_case_count=len(held_cases),
            adversarial_case_count=len(adversarial),
            candidate_runner_sha256=self._callable_fingerprint(candidate_runner),
            independent_verifier_sha256=self._callable_fingerprint(independent_check),
        )
        signed = replace(unsigned, attestation=self._sign_learning_gate(unsigned))
        # A signed failed report stays failed; attestation proves execution, not success.
        return signed

    @staticmethod
    def _numeric(value: Any) -> Decimal:
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            raise TypeError("numeric primitive requires a number")
        return Decimal(str(value))

    def _install_primitives(self) -> None:
        definitions = [
            ("numeric.add", "Add two numbers", "mathematics", {"left": "number", "right": "number"}, {"result": "number"}, ("numeric.add",), lambda x: {"result": float(self._numeric(x["left"]) + self._numeric(x["right"]))}),
            ("numeric.subtract", "Subtract two numbers", "mathematics", {"left": "number", "right": "number"}, {"result": "number"}, ("numeric.subtract",), lambda x: {"result": float(self._numeric(x["left"]) - self._numeric(x["right"]))}),
            ("numeric.multiply", "Multiply two numbers", "mathematics", {"left": "number", "right": "number"}, {"result": "number"}, ("numeric.multiply",), lambda x: {"result": float(self._numeric(x["left"]) * self._numeric(x["right"]))}),
            ("numeric.divide", "Divide two numbers", "mathematics", {"left": "number", "right": "number"}, {"result": "number"}, ("numeric.divide",), self._divide),
            ("compare.equal", "Compare two values for equality", "general", {"left": "any", "right": "any"}, {"result": "boolean"}, ("compare.equal", "logic"), lambda x: {"result": x["left"] == x["right"]}),
            ("compare.less_than", "Compare ordered values", "general", {"left": "number", "right": "number"}, {"result": "boolean"}, ("compare.less_than", "logic"), lambda x: {"result": x["left"] < x["right"]}),
            ("logic.and", "Conjoin boolean values", "general", {"left": "boolean", "right": "boolean"}, {"result": "boolean"}, ("logic.and", "logic"), lambda x: {"result": x["left"] and x["right"]}),
            ("logic.or", "Disjoin boolean values", "general", {"left": "boolean", "right": "boolean"}, {"result": "boolean"}, ("logic.or", "logic"), lambda x: {"result": x["left"] or x["right"]}),
            ("logic.not", "Negate a boolean value", "general", {"value": "boolean"}, {"result": "boolean"}, ("logic.not", "logic"), lambda x: {"result": not x["value"]}),
            ("sequence.length", "Measure sequence length", "general", {"items": "sequence"}, {"result": "integer"}, ("sequence.length", "sequence"), lambda x: {"result": len(x["items"])}),
            ("sequence.sum", "Sum a numeric sequence", "mathematics", {"items": "sequence"}, {"result": "number"}, ("sequence.sum", "sequence", "numeric"), lambda x: {"result": float(sum((self._numeric(v) for v in x["items"]), Decimal("0")))}),
            ("text.concatenate", "Concatenate text values", "general", {"left": "string", "right": "string"}, {"result": "string"}, ("text.concatenate", "text"), lambda x: {"result": x["left"] + x["right"]}),
        ]
        from .native_program_synthesis import NativeProgramSynthesizer

        program_synthesizer = NativeProgramSynthesizer()
        definitions.append((
            "program.calculator.synthesize",
            "Synthesize a calculator from Mirror-native typed syntax operations",
            "programming",
            {},
            {"code": "string", "program_ir": "mapping"},
            ("program.calculator.synthesize", "program.construct", "program.ast"),
            lambda _values: program_synthesizer.build_calculator(),
        ))
        for capability_id, name, domain, inputs, outputs, tags, handler in definitions:
            spec = CapabilitySpec(
                capability_id=capability_id,
                name=name,
                domains=(domain, "general") if domain != "general" else ("general",),
                input_schema=inputs,
                output_schema=outputs,
                tags=tags,
                executor_key=capability_id,
                evidence=EvidenceRecord(level=VerificationLevel.UNVERIFIED),
                provenance=ProvenanceRecord(source="mirror_native_primitive", source_ref=capability_id),
                transferable=domain == "general",
            )
            def self_test(fn=handler, cid=capability_id):
                # Earn execution-tested status only when the operation's value,
                # not just its output shape, matches an independent expected result.
                probes = {
                    "numeric.add": ({"left": 2, "right": 3}, {"result": 5.0}),
                    "numeric.subtract": ({"left": 5, "right": 2}, {"result": 3.0}),
                    "numeric.multiply": ({"left": 3, "right": 4}, {"result": 12.0}),
                    "numeric.divide": ({"left": 8, "right": 2}, {"result": 4.0}),
                    "compare.equal": ({"left": "a", "right": "a"}, {"result": True}),
                    "compare.less_than": ({"left": 1, "right": 2}, {"result": True}),
                    "logic.and": ({"left": True, "right": True}, {"result": True}),
                    "logic.or": ({"left": False, "right": True}, {"result": True}),
                    "logic.not": ({"value": False}, {"result": True}),
                    "sequence.length": ({"items": [1, 2]}, {"result": 2}),
                    "sequence.sum": ({"items": [1, 2, 3]}, {"result": 6.0}),
                    "text.concatenate": ({"left": "mi", "right": "rror"}, {"result": "mirror"}),
                    "program.calculator.synthesize": ({}, None),
                }
                probe, expected = probes[cid]
                output = fn(probe)
                if not isinstance(output, Mapping) or set(output) != set(spec.output_schema):
                    return False
                if cid == "program.calculator.synthesize":
                    return (
                        isinstance(output.get("code"), str)
                        and "def calculate" in output["code"]
                        and set(output.get("program_ir", {}).get("operations", ()))
                        == {"add", "subtract", "multiply", "divide"}
                    )
                return dict(output) == expected
            self.registry.register(spec, handler, self_test=self_test)

    @staticmethod
    def _divide(values: Mapping[str, Any]) -> Mapping[str, Any]:
        denominator = UniversalIntelligenceEngine._numeric(values["right"])
        if denominator == 0:
            raise ZeroDivisionError("division by zero")
        return {"result": float(UniversalIntelligenceEngine._numeric(values["left"]) / denominator)}

    def register_domain_adapter(
        self,
        domain: str,
        *,
        capabilities: tuple[Any, ...] = (),
        verifier: Callable[..., Any] | None = None,
        interpreter: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Attach a domain pack to the shared engine without creating a separate reasoner.

        Each capability entry is (CapabilitySpec, handler) or
        (CapabilitySpec, handler, deterministic_self_test).
        """
        key = domain.casefold().strip()
        if not key or key == "general" or key in self._domain_adapters:
            raise ValueError(f"invalid or already registered domain adapter: {domain}")
        registered = []
        for entry in capabilities:
            if len(entry) == 2:
                spec, handler = entry
                self_test = None
            elif len(entry) == 3:
                spec, handler, self_test = entry
            else:
                raise ValueError("capability entry must contain 2 or 3 items")
            if key not in spec.domains:
                raise ValueError(f"capability {spec.capability_id} is not declared for {key}")
            self.registry.register(spec, handler, self_test=self_test)
            registered.append(spec.capability_id)
        if verifier is not None:
            self.verifier.register_domain_verifier(key, verifier)
        if interpreter is not None:
            self.interpreter.register_domain_interpreter(key, interpreter)
        self._domain_adapters.add(key)
        return {
            "domain": key,
            "capabilities_registered": registered,
            "verifier_registered": verifier is not None,
            "interpreter_registered": interpreter is not None,
        }

    def ingest_blueprints(self, blueprints: list[Any] | tuple[Any, ...]) -> int:
        return self.registry.add_legacy_blueprints(blueprints)

    def solve(self, task: TaskSpec | Mapping[str, Any] | str) -> dict[str, Any]:
        if isinstance(task, str):
            task = self.interpreter.interpret(task)
        elif isinstance(task, Mapping):
            task = TaskSpec.from_dict(task)
        if not isinstance(task, TaskSpec):
            raise TypeError("task must be a TaskSpec, mapping, or request string")
        try:
            plan = self.planner.plan(task)
        except CapabilityError as exc:
            return {
                "status": "planning_failed",
                "verified": False,
                "answer": "Mirror could not construct a safe task plan.",
                "task": task.to_dict(),
                "plan": None,
                "errors": [str(exc)],
            }
        if not plan.complete:
            return {
                "status": "capability_gap",
                "verified": False,
                "answer": "Mirror has represented the goal, but lacks a trusted executable capability for at least one required step.",
                "task": task.to_dict(),
                "plan": {
                    "plan_id": plan.plan_id,
                    "steps": [
                        {"step_id": item.step_id, "capability_id": item.capability_id}
                        for item in plan.steps
                    ],
                    "unresolved": list(plan.unresolved),
                },
                "capability_gaps": list(plan.unresolved),
                "learning_candidate": None,
            }
        try:
            executions, outputs = self.executor.execute(task, plan)
        except Exception as exc:
            return {
                "status": "execution_failed",
                "verified": False,
                "answer": "Mirror constructed a plan but execution failed; no result was promoted.",
                "task": task.to_dict(),
                "plan": {"plan_id": plan.plan_id},
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        report = self.verifier.verify(task, plan, executions, outputs)
        final_outputs = dict(outputs.get("final", {}))
        answer = (
            final_outputs["code"]
            if task.domain == "programming" and isinstance(final_outputs.get("code"), str)
            else (
                str(next(iter(final_outputs.values())))
                if len(final_outputs) == 1
                else canonical_json(final_outputs)
            )
        )
        status = "verified" if report.passed else "verification_failed"
        learning_candidate = None
        if report.passed:
            learning_candidate = {
                "task_id": task.task_id,
                "plan_id": plan.plan_id,
                "domain": task.domain,
                "capability_ids": [item.capability_id for item in plan.steps],
                "required_learning_gate": {
                    "minimum_cases": 3,
                    "minimum_seeds": 3,
                    "held_out_pass": True,
                    "adversarial_pass": True,
                    "independent_verifier": True,
                },
                "promoted": False,
            }
        return {
            "status": status,
            "verified": report.passed,
            "answer": answer if report.passed else "A candidate result was produced, but verification did not pass.",
            "candidate_outputs": final_outputs,
            "verification": {
                "passed": report.passed,
                "level": report.level.value,
                "verifier": report.verifier,
                "checks_passed": list(report.checks_passed),
                "checks_failed": list(report.checks_failed),
                "notes": list(report.notes),
            },
            "task": task.to_dict(),
            "plan": {
                "plan_id": plan.plan_id,
                "steps": [
                    {
                        "step_id": item.step_id,
                        "capability_id": item.capability_id,
                        "input_bindings": dict(item.input_bindings),
                    }
                    for item in plan.steps
                ],
            },
            "executions": [
                {
                    "step_id": item["step_id"],
                    "capability_id": item["capability_id"],
                    "inputs": dict(item["inputs"]),
                    "outputs": dict(item["outputs"]),
                    "succeeded": bool(item["succeeded"]),
                }
                for item in executions
            ],
            "learning_candidate": learning_candidate,
        }

    def solve_prompt(self, prompt: str, *, domain_hint: str | None = None) -> dict[str, Any]:
        return self.solve(self.interpreter.interpret(prompt, domain_hint=domain_hint))

    @staticmethod
    def _infer_value_type(value: Any) -> str:
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, (float, Decimal)):
            return "real"
        if isinstance(value, str):
            return "string"
        if isinstance(value, (list, tuple)):
            return "sequence"
        if isinstance(value, Mapping):
            return "mapping"
        return "any"

    @staticmethod
    def _serialize_subtask(item: SubtaskSpec) -> dict[str, Any]:
        return {
            "subtask_id": item.subtask_id,
            "goal": item.goal,
            "domain": item.domain,
            "capability_tags": list(item.capability_tags),
            "capability_id": item.capability_id,
            "input_bindings": dict(item.input_bindings),
            "output_schema": dict(item.output_schema),
            "dependencies": list(item.dependencies),
            "acceptance_criteria": list(item.acceptance_criteria),
        }

    def _build_handler_from_implementation(self, spec: CapabilitySpec):
        implementation = dict(spec.implementation or {})
        if implementation.get("kind") != "mirror_composition_v1":
            return None
        template_data = implementation.get("task_template")
        if not isinstance(template_data, Mapping):
            return None
        try:
            template = TaskSpec.from_dict(template_data)
        except Exception:
            return None

        def replay(values: Mapping[str, Any]) -> Mapping[str, Any]:
            if set(values) != set(spec.input_schema):
                raise CapabilityError("learned capability input keys do not match its contract")
            for key, expected in spec.input_schema.items():
                if not value_matches_type(values[key], expected):
                    raise CapabilityError(f"learned capability input {key} has the wrong type")
            task = replace(template, inputs=dict(values))
            plan = self.planner.plan(task)
            if not plan.complete:
                raise CapabilityError("learned composition dependencies are no longer available")
            executions, outputs = self.executor.execute(task, plan)
            report = self.verifier.verify(task, plan, executions, outputs)
            if not report.passed:
                raise CapabilityError("learned composition failed replay verification")
            final = dict(outputs["final"])
            if set(final) != set(spec.output_schema):
                raise CapabilityError("learned composition output contract changed")
            return final

        return replay

    def load_learned_blueprints(self, blueprints: list[Any] | tuple[Any, ...]) -> int:
        return self.registry.load_learned_blueprints(
            blueprints,
            self._build_handler_from_implementation,
        )

    def promote_verified_composition(
        self,
        task: TaskSpec | Mapping[str, Any],
        result: Mapping[str, Any],
        gate: LearningGateReport,
        *,
        persist: Callable[[CapabilitySpec], Any] | None = None,
    ) -> CapabilitySpec:
        """Turn a verified task composition into a replayable capability after independent gates."""
        if isinstance(task, Mapping):
            task = TaskSpec.from_dict(task)
        if not isinstance(task, TaskSpec):
            raise TypeError("task must be TaskSpec or a mapping")
        if result.get("verified") is not True or result.get("status") != "verified":
            raise CapabilityError("only independently verified compositions can be proposed for learning")
        self._require_learning_gate(gate)
        plan_data = result.get("plan") or {}
        plan_steps = plan_data.get("steps") or []
        if not plan_steps:
            raise CapabilityError("verified result has no composition steps")

        # Rebuild a task template without retaining the original input values.
        subtasks = []
        for step in plan_steps:
            step_id = str(step["step_id"])
            original = next(
                (item for item in task.subtasks if item.subtask_id == step_id),
                None,
            )
            if original is None:
                # Planner-generated single-goal subtask.
                original = SubtaskSpec(
                    subtask_id=step_id,
                    goal=task.goal,
                    domain=task.domain,
                    capability_tags=task.required_capability_tags,
                    capability_id=str(step["capability_id"]),
                    input_bindings=dict(step.get("input_bindings", {})),
                    output_schema=dict(
                        self.registry.get(str(step["capability_id"])).output_schema
                    ),
                )
            subtasks.append(original)

        input_schema = dict(task.input_schema) or {
            key: self._infer_value_type(value) for key, value in task.inputs.items()
        }
        required_outputs = dict(task.required_outputs)
        last_capability = self.registry.get(str(plan_steps[-1]["capability_id"]))
        if last_capability is None:
            raise CapabilityError("last composed capability is no longer registered")
        output_schema = required_outputs or dict(last_capability.output_schema)
        capability_id = stable_id("learned", {
            "task": task.goal,
            "domain": task.domain,
            "steps": [(x["step_id"], x["capability_id"]) for x in plan_steps],
            "inputs": input_schema,
            "outputs": output_schema,
        })
        task_template = replace(
            task,
            inputs={},
            input_schema=input_schema,
            subtasks=tuple(subtasks),
        )
        spec = CapabilitySpec(
            capability_id=capability_id,
            name=task.goal[:120],
            domains=(task.domain,),
            input_schema=input_schema,
            output_schema=output_schema,
            tags=tuple(set(task.required_capability_tags) | {"composed", "learned"}),
            dependencies=tuple(str(x["capability_id"]) for x in plan_steps),
            executor_key=capability_id,
            evidence=EvidenceRecord(
                level=VerificationLevel.UNVERIFIED,
                details={"promotion_pending": True},
            ),
            provenance=ProvenanceRecord(
                source="mirror_verified_composition",
                source_ref=str(plan_data.get("plan_id", "")),
            ),
            transferable=False,
            implementation={
                "kind": "mirror_composition_v1",
                "task_template": task_template.to_dict(),
                "source_plan_id": str(plan_data.get("plan_id", "")),
            },
        )
        handler = self._build_handler_from_implementation(
            replace(
                spec,
                evidence=EvidenceRecord(
                    level=VerificationLevel.INDEPENDENTLY_REPRODUCED,
                    verifier=gate.independent_verifier,
                    checks_passed=("promotion_preflight",),
                    held_out_passed=True,
                    adversarial_passed=True,
                    seed_count=gate.seed_count,
                ),
            )
        )
        if handler is None:
            raise CapabilityError("could not construct a replayable learned capability")
        return self.learner.promote(spec, handler, gate, persist=persist)

    def propose_cross_domain_transfer(
        self,
        capability_id: str,
        target_domain: str,
    ) -> CapabilitySpec:
        source = self.registry.get(capability_id)
        if source is None:
            raise CapabilityError(f"unknown source capability: {capability_id}")
        return self.transfer.propose(source, target_domain)

    def promote_cross_domain_transfer(
        self,
        proposal: CapabilitySpec,
        gate: LearningGateReport,
    ) -> CapabilitySpec:
        if proposal.evidence.level != VerificationLevel.UNVERIFIED:
            raise CapabilityError("transfer proposal must start unverified")
        source_id = proposal.lineage[-1] if proposal.lineage else ""
        source = self.registry.get(source_id)
        if source is None or self.registry.handler(source_id) is None:
            raise CapabilityError("source executor is unavailable for transfer retesting")
        # Reuse is permitted only after this engine has executed the target-domain gate.
        self._require_learning_gate(gate)
        promoted = replace(
            proposal,
            evidence=EvidenceRecord(
                level=VerificationLevel.INDEPENDENTLY_REPRODUCED,
                verifier=gate.independent_verifier,
                checks_passed=tuple(
                    f"target_case_{i + 1}_passed" for i in range(gate.cases_passed)
                ) + ("target_held_out_passed", "target_adversarial_passed"),
                held_out_passed=True,
                adversarial_passed=True,
                seed_count=gate.seed_count,
                details={"transferred_from": source_id, "target_domain_retested": True},
            ),
            executor_key=source.executor_key,
        )
        source_handler = self.registry.handler(source_id)
        self.registry.register(promoted, source_handler)
        return promoted

    def promote_learned_capability(
        self,
        spec: CapabilitySpec,
        handler: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        gate: LearningGateReport,
        *,
        persist: Callable[[CapabilitySpec], Any] | None = None,
    ) -> CapabilitySpec:
        return self.learner.promote(spec, handler, gate, persist=persist)

    def capability_gaps(self, domains: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
        return self.curriculum.gaps(domains)
