"""Native, bounded reference verification engines for Mirror's five initial domains.

The suite supplies concrete deterministic checks, not general-purpose scientific truth
engines. Every check consumes an explicit structured verification payload attached to the
execution inputs and emits evidence bound to that exact execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Any, Callable, Mapping

from .domain_verification import (
    DomainDecision, DomainEvidenceArtifact, DomainVerificationCheck,
    DomainVerificationRequirement, DomainVerifierRegistry,
)
from .universal_representation import canonical_json


class NativeVerificationError(ValueError):
    """Raised for malformed domain-specific verification material."""


@dataclass(frozen=True)
class _CheckSpec:
    domain: str
    check_id: str
    verifier_id: str
    evidence_kind: str
    condition: str
    verify: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativeVerificationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise NativeVerificationError(f"{name} must be finite")
    return result


def _fraction(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise NativeVerificationError("boolean is not an exact numeric literal")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError) as exc:
            raise NativeVerificationError("invalid rational literal") from exc
    raise NativeVerificationError("exact arithmetic accepts integers or rational strings")


def _poly(node: Any, variables: tuple[str, ...], *, budget: list[int], depth: int = 0):
    """Normalize a small polynomial expression to exact rational coefficients."""
    budget[0] += 1
    if budget[0] > 4096 or depth > 32:
        raise NativeVerificationError("polynomial proof budget exceeded")
    if not isinstance(node, Mapping):
        raise NativeVerificationError("polynomial node must be an object")
    op = node.get("op")
    zero = (0,) * len(variables)
    if op == "lit":
        return {zero: _fraction(node.get("value"))}
    if op == "var":
        name = node.get("name")
        if name not in variables:
            raise NativeVerificationError(f"unknown polynomial variable: {name}")
        exponent = [0] * len(variables)
        exponent[variables.index(name)] = 1
        return {tuple(exponent): Fraction(1)}
    if op not in {"add", "sub", "mul", "pow"}:
        raise NativeVerificationError(f"unsupported polynomial operator: {op}")
    left = _poly(node.get("left"), variables, budget=budget, depth=depth + 1)
    right_node = node.get("right")
    if op == "pow":
        exponent = _fraction(right_node.get("value")) if isinstance(right_node, Mapping) and right_node.get("op") == "lit" else None
        if exponent is None or exponent.denominator != 1 or not 0 <= exponent <= 8:
            raise NativeVerificationError("polynomial powers must be integer literals in 0..8")
        result = {zero: Fraction(1)}
        for _ in range(int(exponent)):
            result = _poly_multiply(result, left)
            if len(result) > 4096:
                raise NativeVerificationError("polynomial term budget exceeded")
        return result
    right = _poly(right_node, variables, budget=budget, depth=depth + 1)
    if op in {"add", "sub"}:
        result = dict(left)
        sign = 1 if op == "add" else -1
        for powers, coefficient in right.items():
            result[powers] = result.get(powers, Fraction(0)) + sign * coefficient
        return {key: value for key, value in result.items() if value}
    return _poly_multiply(left, right)


def _poly_multiply(left, right):
    result = {}
    for a, ca in left.items():
        for b, cb in right.items():
            powers = tuple(x + y for x, y in zip(a, b))
            result[powers] = result.get(powers, Fraction(0)) + ca * cb
            if len(result) > 4096:
                raise NativeVerificationError("polynomial term budget exceeded")
    return {key: value for key, value in result.items() if value}


def _math_eval(node: Any, variables: Mapping[str, Any], *, budget: list[int], depth: int = 0) -> Fraction:
    """A second exact evaluator, independent of the polynomial normalizer."""
    budget[0] += 1
    if budget[0] > 4096 or depth > 32 or not isinstance(node, Mapping):
        raise NativeVerificationError("exact expression evaluation limit or malformed node")
    op = node.get("op")
    if op == "lit":
        return _fraction(node.get("value"))
    if op == "var":
        if node.get("name") not in variables:
            raise NativeVerificationError("sample is missing an expression variable")
        return _fraction(variables[node["name"]])
    left = _math_eval(node.get("left"), variables, budget=budget, depth=depth + 1)
    right = _math_eval(node.get("right"), variables, budget=budget, depth=depth + 1)
    if op == "add":
        return left + right
    if op == "sub":
        return left - right
    if op == "mul":
        return left * right
    if op == "div":
        if right == 0:
            raise NativeVerificationError("division by zero in exact sample")
        return left / right
    if op == "pow":
        if right.denominator != 1 or not -8 <= right <= 8:
            raise NativeVerificationError("sample exponent must be an integer in -8..8")
        return left ** int(right)
    raise NativeVerificationError(f"unsupported exact sample operator: {op}")


def _math_identity(_context, payload):
    math_data = payload.get("mathematics")
    if not isinstance(math_data, Mapping):
        raise NativeVerificationError("mathematics payload is required")
    variables = tuple(math_data.get("variables", ()))
    if not variables or len(set(variables)) != len(variables) or any(not isinstance(x, str) or not x for x in variables):
        raise NativeVerificationError("identity needs unique named variables")
    left = _poly(math_data.get("left"), variables, budget=[0])
    right = _poly(math_data.get("right"), variables, budget=[0])
    if left != right:
        raise NativeVerificationError("left and right polynomials are not identical")
    return {"variables": list(variables), "normalized_term_count": len(left), "proof_method": "exact_polynomial_normal_form"}


def _math_samples(_context, payload):
    math_data = payload.get("mathematics")
    if not isinstance(math_data, Mapping):
        raise NativeVerificationError("mathematics payload is required")
    left, right = math_data.get("left"), math_data.get("right")
    samples = math_data.get("samples")
    if not isinstance(samples, (tuple, list)) or not samples:
        raise NativeVerificationError("at least one independent rational sample is required")
    for sample in samples:
        if not isinstance(sample, Mapping) or not isinstance(sample.get("variables"), Mapping):
            raise NativeVerificationError("each sample requires a variables object")
        a = _math_eval(left, sample["variables"], budget=[0])
        b = _math_eval(right, sample["variables"], budget=[0])
        if a != b:
            raise NativeVerificationError("exact rational sample mismatch")
    return {"sample_count": len(samples), "proof_method": "exact_rational_sample_replay"}


def _finite_json(value: Any, *, depth: int = 0, budget: list[int] | None = None):
    budget = budget if budget is not None else [0]
    budget[0] += 1
    if budget[0] > 10000 or depth > 32:
        raise NativeVerificationError("program contract structure limit exceeded")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        if abs(value) > 10**100:
            raise NativeVerificationError("integer magnitude limit exceeded")
        return
    if isinstance(value, float):
        if not math.isfinite(value) or abs(value) > 1e100:
            raise NativeVerificationError("non-finite or out-of-range program data")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise NativeVerificationError("program record keys must be strings")
            _finite_json(child, depth=depth + 1, budget=budget)
        return
    if isinstance(value, (tuple, list)):
        if len(value) > 10000:
            raise NativeVerificationError("program collection limit exceeded")
        for child in value:
            _finite_json(child, depth=depth + 1, budget=budget)
        return
    raise NativeVerificationError(f"unsupported program value: {type(value).__name__}")


def _reference_program_eval(program: Mapping[str, Any], inputs: Mapping[str, Any]):
    """Small independent UOL reference semantics; no eval, imports or host callbacks."""
    if not isinstance(program, Mapping) or not isinstance(program.get("body"), Mapping):
        raise NativeVerificationError("program must contain a serialized UOL body")
    input_specs = program.get("inputs")
    if not isinstance(input_specs, (tuple, list)):
        raise NativeVerificationError("program input schema is missing")
    expected = {item.get("name") for item in input_specs if isinstance(item, Mapping)}
    if len(expected) != len(input_specs) or set(inputs) != expected:
        raise NativeVerificationError("program inputs do not match the serialized schema")
    rules = {}
    for item in program.get("rules", ()):
        if not isinstance(item, Mapping) or not item.get("rule_id"):
            raise NativeVerificationError("invalid program rule catalog")
        if item["rule_id"] in rules:
            raise NativeVerificationError("duplicate program rule identifier")
        rules[item["rule_id"]] = item
    budget = [0]

    def run(node, env, active=(), depth=0):
        budget[0] += 1
        if budget[0] > 10000 or depth > 64 or not isinstance(node, Mapping):
            raise NativeVerificationError("reference interpreter resource limit exceeded")
        op = node.get("op")
        if op == "lit":
            value = node.get("value")
            _finite_json(value)
            return value
        if op == "var":
            name = node.get("name")
            if name not in env:
                raise NativeVerificationError(f"undefined variable: {name}")
            return env[name]
        if op == "call":
            name = node.get("name")
            args = [run(child, env, active, depth + 1) for child in node.get("args", ())]
            builtins = {
                "add": lambda a, b: a + b, "subtract": lambda a, b: a - b,
                "multiply": lambda a, b: a * b, "divide": lambda a, b: a / b,
                "modulo": lambda a, b: a % b, "power": lambda a, b: a ** b,
                "minimum": min, "maximum": max, "equal": lambda a, b: a == b,
                "less_than": lambda a, b: a < b, "less_equal": lambda a, b: a <= b,
                "greater_than": lambda a, b: a > b, "greater_equal": lambda a, b: a >= b,
                "and": lambda a, b: a and b, "or": lambda a, b: a or b,
                "not": lambda a: not a, "negate": lambda a: -a,
                "absolute": abs, "concat": lambda a, b: a + b, "length": len,
                "sum": sum, "product": lambda values: math.prod(values),
            }
            if name in builtins:
                if name in {"add", "subtract", "multiply", "divide", "modulo", "power",
                            "minimum", "maximum", "equal", "less_than", "less_equal",
                            "greater_than", "greater_equal", "and", "or", "concat"} and len(args) != 2:
                    raise NativeVerificationError("reference operator arity mismatch")
                if name in {"not", "negate", "absolute", "length", "sum", "product"} and len(args) != 1:
                    raise NativeVerificationError("reference operator arity mismatch")
                if name in {"divide", "modulo"} and args[1] == 0:
                    raise NativeVerificationError("division by zero")
                numeric_ops = {"add", "subtract", "multiply", "divide", "modulo", "power", "minimum", "maximum", "negate", "absolute", "sum", "product"}
                if name in numeric_ops:
                    numeric_values = args[0] if name in {"sum", "product"} else args
                    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in numeric_values):
                        raise NativeVerificationError("numeric operator received a non-number")
                if name in {"and", "or", "not"} and any(not isinstance(value, bool) for value in args):
                    raise NativeVerificationError("boolean operator received a non-boolean")
                if name == "power" and (abs(args[1]) > 16 or abs(args[0]) > 1e6):
                    raise NativeVerificationError("reference power bounds exceeded")
                result = builtins[name](*args)
                _finite_json(result)
                return result
            rule = rules.get(name)
            if rule is None or name in active:
                raise NativeVerificationError(f"unavailable or recursive rule: {name}")
            ports = rule.get("inputs", ())
            if len(args) != len(ports):
                raise NativeVerificationError("reference rule arity mismatch")
            local = {port["name"]: value for port, value in zip(ports, args)}
            return run(rule["body"], local, active + (name,), depth + 1)
        if op == "if":
            condition = run(node.get("condition"), env, active, depth + 1)
            if not isinstance(condition, bool):
                raise NativeVerificationError("conditional guard must be boolean")
            return run(node.get("yes") if condition else node.get("no"), env, active, depth + 1)
        if op == "array":
            return tuple(run(child, env, active, depth + 1) for child in node.get("items", ()))
        if op == "record":
            fields = node.get("fields", {})
            if not isinstance(fields, Mapping):
                raise NativeVerificationError("record expression fields must be an object")
            return {key: run(child, env, active, depth + 1) for key, child in fields.items()}
        if op == "field":
            target = run(node.get("target"), env, active, depth + 1)
            if not isinstance(target, Mapping) or node.get("name") not in target:
                raise NativeVerificationError("record field not found")
            return target[node["name"]]
        if op == "index":
            target = run(node.get("target"), env, active, depth + 1)
            index = run(node.get("index"), env, active, depth + 1)
            if not isinstance(target, (tuple, list)) or isinstance(index, bool) or not isinstance(index, int):
                raise NativeVerificationError("invalid array index expression")
            return target[index]
        if op == "let":
            local = dict(env)
            local[node["name"]] = run(node.get("value"), env, active, depth + 1)
            return run(node.get("body"), local, active, depth + 1)
        if op == "map":
            values = run(node.get("items"), env, active, depth + 1)
            if not isinstance(values, (tuple, list)) or len(values) > 10000:
                raise NativeVerificationError("map source must be a bounded array")
            result = []
            for value in values:
                local = dict(env)
                local[node["variable"]] = value
                result.append(run(node.get("body"), local, active, depth + 1))
            return tuple(result)
        if op == "fold":
            values = run(node.get("items"), env, active, depth + 1)
            if not isinstance(values, (tuple, list)) or len(values) > 10000:
                raise NativeVerificationError("fold source must be a bounded array")
            accumulator = run(node.get("initial"), env, active, depth + 1)
            for value in values:
                local = dict(env)
                local[node["accumulator"]] = accumulator
                local[node["variable"]] = value
                accumulator = run(node.get("body"), local, active, depth + 1)
            return accumulator
        raise NativeVerificationError(f"unsupported UOL reference opcode: {op}")

    env = dict(inputs)
    result = run(program["body"], env)
    _finite_json(result)
    return result


def _program_replay(context, payload):
    verification = payload.get("programming")
    if not isinstance(verification, Mapping) or not isinstance(verification.get("program"), Mapping):
        raise NativeVerificationError("programming payload requires a serialized program")
    program = verification["program"]
    actual_inputs = {key: value for key, value in context.step.inputs.items() if key != "__mirror_verification__"}
    expected = _reference_program_eval(program, actual_inputs)
    output_name = verification.get("output_name", "result")
    if output_name not in context.step.outputs:
        raise NativeVerificationError("execution output is missing the program result port")
    actual = context.step.outputs[output_name]
    if isinstance(expected, (int, float)) and not isinstance(expected, bool) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        matches = math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-9)
    else:
        matches = expected == actual
    if not matches:
        raise NativeVerificationError("independent UOL reference replay disagrees with the execution output")
    return {"output_name": output_name, "reference_output": expected, "observed_output": actual, "method": "independent_reference_interpreter"}


def _program_contract(context, _payload):
    declared_inputs = {port.name: port for port in context.capability.inputs}
    observed_inputs = dict(context.step.inputs)
    if set(observed_inputs) != set(declared_inputs):
        raise NativeVerificationError("execution input names do not match the capability contract")
    for name, value in observed_inputs.items():
        if not declared_inputs[name].accepts_value(value):
            raise NativeVerificationError(f"input port {name} has an incompatible runtime type")
    declared_outputs = {port.name: port for port in context.capability.outputs}
    observed_outputs = dict(context.step.outputs)
    if set(observed_outputs) != set(declared_outputs):
        raise NativeVerificationError("execution output names do not match the capability contract")
    for name, value in observed_outputs.items():
        if not declared_outputs[name].accepts_value(value):
            raise NativeVerificationError(f"output port {name} has an incompatible runtime type")
        _finite_json(value)
    return {"input_ports_checked": sorted(declared_inputs), "output_ports_checked": sorted(declared_outputs), "method": "independent_contract_validation"}


def _dimensions(value, name):
    if not isinstance(value, Mapping) or not value:
        raise NativeVerificationError(f"{name} must be a non-empty dimension vector")
    result = {}
    for key, exponent in value.items():
        if not isinstance(key, str) or not key or isinstance(exponent, bool) or not isinstance(exponent, int):
            raise NativeVerificationError(f"{name} must contain string dimension names and integer exponents")
        if exponent:
            result[key.casefold()] = exponent
    return result


def _physics_dimensions(_context, payload):
    data = payload.get("physics")
    if not isinstance(data, Mapping):
        raise NativeVerificationError("physics payload is required")
    terms = data.get("terms")
    if not isinstance(terms, (tuple, list)) or len(terms) < 2:
        raise NativeVerificationError("dimension check needs at least two equation terms")
    vectors = [_dimensions(item.get("dimensions"), "term dimensions") for item in terms if isinstance(item, Mapping)]
    if len(vectors) != len(terms) or any(vector != vectors[0] for vector in vectors[1:]):
        raise NativeVerificationError("equation terms are not dimensionally homogeneous")
    return {"term_count": len(terms), "dimension_vector": vectors[0], "method": "dimension_vector_equivalence"}


def _physics_conservation(_context, payload):
    data = payload.get("physics")
    if not isinstance(data, Mapping):
        raise NativeVerificationError("physics payload is required")
    before, after = data.get("before"), data.get("after")
    tolerance = _number(data.get("tolerance", 0), "conservation tolerance")
    if tolerance < 0 or not isinstance(before, (tuple, list)) or not isinstance(after, (tuple, list)) or not before or not after:
        raise NativeVerificationError("conservation check requires non-empty before/after arrays and non-negative tolerance")
    initial = sum(_number(value, "before value") for value in before)
    final = sum(_number(value, "after value") for value in after)
    if abs(initial - final) > tolerance:
        raise NativeVerificationError("declared conserved quantity changed beyond tolerance")
    return {"before_total": initial, "after_total": final, "absolute_error": abs(initial - final), "method": "independent_balance_sum"}


def _biology_ranges(_context, payload):
    data = payload.get("biology")
    if not isinstance(data, Mapping):
        raise NativeVerificationError("biology payload is required")
    measurements = data.get("measurements")
    if not isinstance(measurements, (tuple, list)) or not measurements:
        raise NativeVerificationError("biology check requires bounded measurements")
    for item in measurements:
        if not isinstance(item, Mapping) or not isinstance(item.get("unit"), str) or not item["unit"].strip():
            raise NativeVerificationError("each measurement needs a unit")
        value = _number(item.get("value"), "measurement value")
        lower = _number(item.get("minimum"), "measurement minimum")
        upper = _number(item.get("maximum"), "measurement maximum")
        if lower > upper or not lower <= value <= upper:
            raise NativeVerificationError(f"measurement {item.get('name', '<unnamed>')} is outside its declared bounds")
    return {"measurement_count": len(measurements), "method": "bounded_measurement_validation"}


def _biology_mass_balance(_context, payload):
    data = payload.get("biology")
    if not isinstance(data, Mapping):
        raise NativeVerificationError("biology payload is required")
    total = _number(data.get("initial_mass"), "initial mass")
    components = data.get("component_masses")
    tolerance = _number(data.get("mass_tolerance", 0), "mass tolerance")
    if total < 0 or tolerance < 0 or not isinstance(components, (tuple, list)) or not components:
        raise NativeVerificationError("mass balance requires a non-negative total, tolerance and components")
    values = [_number(value, "component mass") for value in components]
    if any(value < 0 for value in values) or abs(total - sum(values)) > tolerance:
        raise NativeVerificationError("biological mass balance failed")
    return {"initial_mass": total, "component_mass_sum": sum(values), "method": "independent_mass_balance"}


def _engineering_stress(context, payload):
    data = payload.get("engineering")
    if not isinstance(data, Mapping):
        raise NativeVerificationError("engineering payload is required")
    load = _number(data.get("load_n"), "load_n")
    area = _number(data.get("area_mm2"), "area_mm2")
    if load < 0 or area <= 0:
        raise NativeVerificationError("engineering load must be non-negative and area positive")
    expected = load / area
    observed = _number(context.step.outputs.get("stress_mpa"), "stress_mpa output")
    if not math.isclose(expected, observed, rel_tol=1e-8, abs_tol=1e-8):
        raise NativeVerificationError("reported stress does not equal load/area in MPa")
    return {"computed_stress_mpa": expected, "reported_stress_mpa": observed, "method": "load_area_equilibrium"}


def _engineering_safety(context, payload):
    data = payload.get("engineering")
    if not isinstance(data, Mapping):
        raise NativeVerificationError("engineering payload is required")
    allowable = _number(data.get("allowable_stress_mpa"), "allowable_stress_mpa")
    required = _number(data.get("required_safety_factor"), "required_safety_factor")
    stress = _number(context.step.outputs.get("stress_mpa"), "stress_mpa output")
    observed = _number(context.step.outputs.get("safety_factor"), "safety_factor output")
    if allowable <= 0 or required <= 0 or stress <= 0:
        raise NativeVerificationError("allowable stress, required factor and actual stress must be positive")
    expected = allowable / stress
    if not math.isclose(expected, observed, rel_tol=1e-8, abs_tol=1e-8):
        raise NativeVerificationError("reported safety factor does not equal allowable/actual stress")
    if observed < required:
        raise NativeVerificationError("engineering safety factor is below the required minimum")
    return {"computed_safety_factor": expected, "required_safety_factor": required, "method": "allowable_over_actual_stress"}


class NativeDomainVerifierSuite:
    """Register two distinct deterministic verifiers for each currently supported domain."""

    VERSION = "native-domain-verifiers-v1"

    _DEFINITIONS = (
        _CheckSpec("mathematics", "exact_polynomial_identity", "mirror.math.polynomial_normal_form.v1", "exact_identity", "math.identity_exact", _math_identity),
        _CheckSpec("mathematics", "exact_rational_samples", "mirror.math.rational_replay.v1", "sample_replay", "math.samples_exact", _math_samples),
        _CheckSpec("programming", "independent_program_replay", "mirror.program.reference_interpreter.v1", "reference_replay", "program.output_matches_reference", _program_replay),
        _CheckSpec("programming", "program_contract_validation", "mirror.program.contract_checker.v1", "contract_validation", "program.ports_match_contract", _program_contract),
        _CheckSpec("physics", "dimensional_homogeneity", "mirror.physics.dimension_checker.v1", "dimensional_homogeneity", "physics.dimensions_homogeneous", _physics_dimensions),
        _CheckSpec("physics", "conservation_balance", "mirror.physics.conservation_checker.v1", "conservation_balance", "physics.conservation_balanced", _physics_conservation),
        _CheckSpec("biology", "measurement_bounds", "mirror.biology.measurement_bounds.v1", "measurement_bounds", "biology.measurements_within_bounds", _biology_ranges),
        _CheckSpec("biology", "mass_balance", "mirror.biology.mass_balance.v1", "mass_balance", "biology.mass_balanced", _biology_mass_balance),
        _CheckSpec("engineering", "load_area_stress", "mirror.engineering.stress_calculator.v1", "stress_calculation", "engineering.stress_matches_load_area", _engineering_stress),
        _CheckSpec("engineering", "safety_factor", "mirror.engineering.safety_factor.v1", "safety_factor", "engineering.safety_factor_meets_requirement", _engineering_safety),
    )

    def __init__(self, registry: DomainVerifierRegistry | None = None):
        self.registry = registry or DomainVerifierRegistry()
        for spec in self._DEFINITIONS:
            self.registry.register(
                DomainVerificationCheck(
                    check_id=spec.check_id,
                    verifier_id=spec.verifier_id,
                    domain=spec.domain,
                    evidence_kind=spec.evidence_kind,
                    callback=self._callback(spec),
                ),
                independent=True,
            )

    def _callback(self, spec: _CheckSpec):
        def callback(context):
            decision = DomainDecision.PASS
            detail = ""
            observation = {}
            try:
                envelope = context.step.inputs.get("__mirror_verification__")
                if not isinstance(envelope, Mapping) or type(envelope.get("schema_version")) is not int or envelope.get("schema_version") != 1:
                    raise NativeVerificationError("step input __mirror_verification__ must be a schema_version 1 object")
                if envelope.get("domain") != spec.domain:
                    raise NativeVerificationError("verification envelope domain does not match the registered verifier")
                payload = envelope.get("payload")
                if not isinstance(payload, Mapping):
                    raise NativeVerificationError("verification envelope payload must be an object")
                observation = dict(spec.verify(context, payload))
            except Exception as exc:
                decision = DomainDecision.FAIL
                detail = f"{type(exc).__name__}: {str(exc)[:400]}"
                envelope = context.step.inputs.get("__mirror_verification__", {})
                payload = dict(envelope) if isinstance(envelope, Mapping) else {"invalid": True}
            binding = {
                "execution_fingerprint": context.execution_fingerprint,
                "step_id": context.step.step_id,
                "capability_id": context.capability.capability_id,
                "check_id": spec.check_id,
            }
            digest = hashlib.sha256(canonical_json({
                "binding": binding, "payload": payload, "observation": observation,
                "decision": decision.value,
            }).encode("utf-8")).hexdigest()
            return DomainEvidenceArtifact(
                evidence_id=f"native:{digest[:32]}",
                domain=spec.domain,
                evidence_kind=spec.evidence_kind,
                decision=decision,
                step_id=context.step.step_id,
                capability_id=context.capability.capability_id,
                execution_fingerprint=context.execution_fingerprint,
                verifier_id=spec.verifier_id,
                evidence_ref=f"mirror-native://{spec.check_id}/{digest}",
                evidence_payload={
                    "input_payload_sha256": hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest(),
                    "observation": observation,
                    "decision": decision.value,
                    "verifier_version": self.VERSION,
                },
                verified_conditions=(spec.condition,) if decision is DomainDecision.PASS else (),
                detail=detail,
            )
        return callback

    @classmethod
    def supported_domains(cls) -> tuple[str, ...]:
        return tuple(sorted({item.domain for item in cls._DEFINITIONS}))

    @classmethod
    def requirement(cls, domain: str) -> DomainVerificationRequirement:
        specs = tuple(item for item in cls._DEFINITIONS if item.domain == domain.casefold())
        if not specs:
            raise NativeVerificationError(f"no native verification policy for domain: {domain}")
        return DomainVerificationRequirement(domain, tuple(item.evidence_kind for item in specs), 2)

    @classmethod
    def required_checks(cls, domain: str) -> dict[str, str]:
        specs = tuple(item for item in cls._DEFINITIONS if item.domain == domain.casefold())
        if not specs:
            raise NativeVerificationError(f"no native verification checks for domain: {domain}")
        return {item.evidence_kind: item.check_id for item in specs}

    @classmethod
    def postconditions(cls, domain: str) -> tuple[str, ...]:
        specs = tuple(item for item in cls._DEFINITIONS if item.domain == domain.casefold())
        if not specs:
            raise NativeVerificationError(f"no native verification postconditions for domain: {domain}")
        return tuple(item.condition for item in specs)
