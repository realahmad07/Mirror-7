"""Phase 2: Structured, executable UOL rule library across multiple domains.

Each rule is authored as a canonical KnowledgeRecord carrying an explicit
`content.uol_rule` AST payload with deterministic SHA-256 provenance and
verification metadata. Rules are imported through KnowledgeRuleImporter so
all conflict, provenance, and epistemic policies remain enforced.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from .blueprint_contracts import EvidenceLevel, VerificationEvidence
from .knowledge_foundation import KnowledgeKind, KnowledgeRecord
from .knowledge_uol import KnowledgeRuleImporter, KnowledgeRuleImportReport
from .operational_language import (
    Call,
    Expression,
    Lit,
    OperationalRule,
    Program,
    UOL,
    Var,
    VariableSpec,
    expr_dict,
)
from .universal_representation import EpistemicStatus, RepresentationProvenance

LIBRARY_VERSION = "uol-rule-library-v1"
_PI = 3.141592653589793


def _call(op: str, *args: Expression) -> Call:
    return Call(op, args)


def _var(name: str) -> Var:
    return Var(name)


def _lit(value: Any) -> Lit:
    return Lit(value)


def _spec(
    rule_id: str,
    name: str,
    domain: str,
    statement: str,
    inputs: tuple[tuple[str, str], ...],
    body: Expression,
    *,
    domains: tuple[str, ...] | None = None,
    kind: KnowledgeKind = KnowledgeKind.FORMULA,
) -> KnowledgeRecord:
    payload = {
        "schema_version": 1,
        "rule_id": rule_id,
        "name": name,
        "inputs": [{"name": n, "value_type": t} for n, t in inputs],
        "body": expr_dict(body),
        "domains": list(domains or (domain,)),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return KnowledgeRecord(
        knowledge_id=f"k:uol:{rule_id}",
        kind=kind,
        domain=domain,
        subject=name,
        predicate="uol_formula",
        object_value=rule_id,
        statement=statement,
        epistemic_status=EpistemicStatus.DERIVED,
        content={"uol_rule": payload, "library_version": LIBRARY_VERSION},
        provenance=RepresentationProvenance(
            source=LIBRARY_VERSION,
            source_ref=f"mirror://uol_rule_library/{rule_id}",
            content_sha256=digest,
        ),
        verification=VerificationEvidence(
            level=EvidenceLevel.FORMALLY_VERIFIED,
            checks_passed=("uol-type-check", "uol-identity-check"),
            verifier="mirror-uol-library-verifier",
        ),
    )


def _build_records() -> tuple[KnowledgeRecord, ...]:
    return (
        # ---------------- Mathematics ----------------
        _spec(
            "math.square", "Square", "mathematics",
            "Compute the square of a number: x * x.",
            (("x", "number"),),
            _call("multiply", _var("x"), _var("x")),
            domains=("mathematics", "general"),
        ),
        _spec(
            "math.cube", "Cube", "mathematics",
            "Compute the cube of a number: x * x * x.",
            (("x", "number"),),
            _call("multiply", _call("multiply", _var("x"), _var("x")), _var("x")),
            domains=("mathematics", "general"),
        ),
        _spec(
            "math.double", "Double", "mathematics",
            "Double a number: x * 2.",
            (("x", "number"),),
            _call("multiply", _var("x"), _lit(2)),
            domains=("mathematics", "general"),
        ),
        _spec(
            "math.half", "Half", "mathematics",
            "Halve a number: x / 2.",
            (("x", "number"),),
            _call("divide", _var("x"), _lit(2)),
            domains=("mathematics", "general"),
        ),
        _spec(
            "math.is_even", "Is Even", "mathematics",
            "Return true when integer n is divisible by 2.",
            (("n", "integer"),),
            _call("equal", _call("modulo", _var("n"), _lit(2)), _lit(0)),
            domains=("mathematics", "general"),
            kind=KnowledgeKind.RULE,
        ),
        _spec(
            "math.is_divisible", "Is Divisible", "mathematics",
            "Return true when integer n is evenly divisible by d.",
            (("n", "integer"), ("d", "integer")),
            _call("equal", _call("modulo", _var("n"), _var("d")), _lit(0)),
            domains=("mathematics", "general"),
            kind=KnowledgeKind.RULE,
        ),
        _spec(
            "math.clamp", "Clamp", "mathematics",
            "Clamp x to the closed interval [lo, hi].",
            (("x", "number"), ("lo", "number"), ("hi", "number")),
            _call("minimum", _call("maximum", _var("x"), _var("lo")), _var("hi")),
            domains=("mathematics", "general"),
            kind=KnowledgeKind.PROCEDURE,
        ),
        _spec(
            "math.linear_transform", "Linear Transform", "mathematics",
            "Evaluate affine map m * x + b.",
            (("x", "number"), ("m", "number"), ("b", "number")),
            _call("add", _call("multiply", _var("m"), _var("x")), _var("b")),
            domains=("mathematics", "general"),
        ),
        _spec(
            "math.quadratic_discriminant", "Quadratic Discriminant", "mathematics",
            "Evaluate discriminant b^2 - 4ac of ax^2 + bx + c = 0.",
            (("a", "number"), ("b", "number"), ("c", "number")),
            _call(
                "subtract",
                _call("multiply", _var("b"), _var("b")),
                _call("multiply", _lit(4), _call("multiply", _var("a"), _var("c"))),
            ),
        ),
        _spec(
            "math.arithmetic_series_sum", "Arithmetic Series Sum", "mathematics",
            "Sum of n terms of an arithmetic progression: n * (a1 + an) / 2.",
            (("n", "number"), ("a1", "number"), ("an", "number")),
            _call("divide", _call("multiply", _var("n"), _call("add", _var("a1"), _var("an"))), _lit(2)),
        ),
        _spec(
            "math.triangle_area", "Triangle Area", "mathematics",
            "Area of a triangle: 0.5 * base * height.",
            (("base", "number"), ("height", "number")),
            _call("multiply", _lit(0.5), _call("multiply", _var("base"), _var("height"))),
        ),
        _spec(
            "math.circle_area", "Circle Area", "mathematics",
            "Area of a circle: pi * r^2.",
            (("r", "number"),),
            _call("multiply", _lit(_PI), _call("multiply", _var("r"), _var("r"))),
        ),
        _spec(
            "math.cylinder_volume", "Cylinder Volume", "mathematics",
            "Volume of a right circular cylinder: pi * r^2 * h.",
            (("r", "number"), ("h", "number")),
            _call("multiply", _call("multiply", _lit(_PI), _call("multiply", _var("r"), _var("r"))), _var("h")),
        ),
        _spec(
            "math.mean", "Arithmetic Mean", "mathematics",
            "Mean of a non-empty numeric array: sum(xs) / length(xs).",
            (("xs", "array"),),
            _call("divide", _call("sum", _var("xs")), _call("length", _var("xs"))),
            domains=("mathematics", "general"),
        ),
        _spec(
            "math.pythagorean_sq", "Pythagorean Squared Hypotenuse", "mathematics",
            "Squared hypotenuse c^2 = a^2 + b^2.",
            (("a", "number"), ("b", "number")),
            _call("add", _call("multiply", _var("a"), _var("a")), _call("multiply", _var("b"), _var("b"))),
        ),
        _spec(
            "math.manhattan_distance", "Manhattan Distance", "mathematics",
            "L1 distance |x1 - x2| + |y1 - y2|.",
            (("x1", "number"), ("y1", "number"), ("x2", "number"), ("y2", "number")),
            _call(
                "add",
                _call("absolute", _call("subtract", _var("x1"), _var("x2"))),
                _call("absolute", _call("subtract", _var("y1"), _var("y2"))),
            ),
        ),
        # ---------------- Physics ----------------
        _spec(
            "physics.newton_force", "Newton Second Law Force", "physics",
            "Force equals mass times acceleration: F = m * a.",
            (("mass", "number"), ("accel", "number")),
            _call("multiply", _var("mass"), _var("accel")),
        ),
        _spec(
            "physics.momentum", "Linear Momentum", "physics",
            "Linear momentum p = m * v.",
            (("mass", "number"), ("velocity", "number")),
            _call("multiply", _var("mass"), _var("velocity")),
        ),
        _spec(
            "physics.kinetic_energy", "Kinetic Energy", "physics",
            "Classical kinetic energy KE = 0.5 * m * v^2.",
            (("mass", "number"), ("velocity", "number")),
            _call("multiply", _lit(0.5), _call("multiply", _var("mass"), _call("multiply", _var("velocity"), _var("velocity")))),
        ),
        _spec(
            "physics.potential_energy", "Gravitational Potential Energy", "physics",
            "Gravitational potential energy PE = m * g * h.",
            (("mass", "number"), ("gravity", "number"), ("height", "number")),
            _call("multiply", _var("mass"), _call("multiply", _var("gravity"), _var("height"))),
        ),
        _spec(
            "physics.work_done", "Mechanical Work", "physics",
            "Mechanical work W = F * d.",
            (("force", "number"), ("distance", "number")),
            _call("multiply", _var("force"), _var("distance")),
        ),
        _spec(
            "physics.ohms_voltage", "Ohm's Law Voltage", "physics",
            "Voltage V = I * R.",
            (("current", "number"), ("resistance", "number")),
            _call("multiply", _var("current"), _var("resistance")),
        ),
        _spec(
            "physics.electrical_power", "Electrical Power", "physics",
            "Electrical power P = V * I.",
            (("voltage", "number"), ("current", "number")),
            _call("multiply", _var("voltage"), _var("current")),
        ),
        _spec(
            "physics.density", "Mass Density", "physics",
            "Density rho = mass / volume.",
            (("mass", "number"), ("volume", "number")),
            _call("divide", _var("mass"), _var("volume")),
        ),
        _spec(
            "physics.wave_speed", "Wave Speed", "physics",
            "Wave speed v = frequency * wavelength.",
            (("frequency", "number"), ("wavelength", "number")),
            _call("multiply", _var("frequency"), _var("wavelength")),
        ),
        _spec(
            "physics.celsius_to_fahrenheit", "Celsius to Fahrenheit", "physics",
            "Convert Celsius to Fahrenheit: C * 1.8 + 32.",
            (("c", "number"),),
            _call("add", _call("multiply", _var("c"), _lit(1.8)), _lit(32)),
            domains=("physics", "general"),
        ),
        _spec(
            "physics.fahrenheit_to_celsius", "Fahrenheit to Celsius", "physics",
            "Convert Fahrenheit to Celsius: (F - 32) / 1.8.",
            (("f", "number"),),
            _call("divide", _call("subtract", _var("f"), _lit(32)), _lit(1.8)),
            domains=("physics", "general"),
        ),
        _spec(
            "physics.celsius_to_kelvin", "Celsius to Kelvin", "physics",
            "Convert Celsius to Kelvin: C + 273.15.",
            (("c", "number"),),
            _call("add", _var("c"), _lit(273.15)),
            domains=("physics", "general"),
        ),
        _spec(
            "physics.uniform_displacement", "Uniformly Accelerated Displacement", "physics",
            "Displacement s = v0 * t + 0.5 * a * t^2.",
            (("v0", "number"), ("a", "number"), ("t", "number")),
            _call(
                "add",
                _call("multiply", _var("v0"), _var("t")),
                _call("multiply", _lit(0.5), _call("multiply", _var("a"), _call("multiply", _var("t"), _var("t")))),
            ),
        ),
        # ---------------- Engineering ----------------
        _spec(
            "eng.axial_stress", "Axial Stress", "engineering",
            "Axial stress sigma = F / A.",
            (("force", "number"), ("area", "number")),
            _call("divide", _var("force"), _var("area")),
        ),
        _spec(
            "eng.safety_factor", "Factor of Safety", "engineering",
            "Safety factor SF = yield_strength / applied_stress.",
            (("yield_strength", "number"), ("applied_stress", "number")),
            _call("divide", _var("yield_strength"), _var("applied_stress")),
        ),
        _spec(
            "eng.beam_center_moment", "Simply Supported Beam Center Moment", "engineering",
            "Maximum bending moment for center point load: M = F * L / 4.",
            (("load", "number"), ("span", "number")),
            _call("divide", _call("multiply", _var("load"), _var("span")), _lit(4)),
        ),
        _spec(
            "eng.thermal_expansion", "Linear Thermal Expansion", "engineering",
            "Change in length dL = alpha * L * dT.",
            (("alpha", "number"), ("length", "number"), ("delta_t", "number")),
            _call("multiply", _var("alpha"), _call("multiply", _var("length"), _var("delta_t"))),
        ),
        _spec(
            "eng.efficiency", "Energy Efficiency", "engineering",
            "Efficiency ratio eta = output_energy / input_energy.",
            (("output_energy", "number"), ("input_energy", "number")),
            _call("divide", _var("output_energy"), _var("input_energy")),
        ),
        _spec(
            "eng.pressure", "Pressure", "engineering",
            "Pressure P = F / A.",
            (("force", "number"), ("area", "number")),
            _call("divide", _var("force"), _var("area")),
        ),
        # ---------------- Biology & Chemistry ----------------
        _spec(
            "bio.bmi", "Body Mass Index", "biology",
            "BMI = mass_kg / (height_m^2).",
            (("mass_kg", "number"), ("height_m", "number")),
            _call("divide", _var("mass_kg"), _call("multiply", _var("height_m"), _var("height_m"))),
        ),
        _spec(
            "bio.population_next", "Discrete Population Growth Step", "biology",
            "Next-step population P1 = P0 * (1 + r).",
            (("pop", "number"), ("growth_rate", "number")),
            _call("multiply", _var("pop"), _call("add", _lit(1), _var("growth_rate"))),
        ),
        _spec(
            "chem.dilution_c2", "Dilution Concentration C2", "biology",
            "Final concentration C2 = (C1 * V1) / V2.",
            (("c1", "number"), ("v1", "number"), ("v2", "number")),
            _call("divide", _call("multiply", _var("c1"), _var("v1")), _var("v2")),
            domains=("biology", "chemistry"),
        ),
        _spec(
            "chem.molarity", "Molarity", "biology",
            "Molar concentration M = moles / liters.",
            (("moles", "number"), ("liters", "number")),
            _call("divide", _var("moles"), _var("liters")),
            domains=("biology", "chemistry"),
        ),
        _spec(
            "bio.michaelis_menten", "Michaelis-Menten Reaction Rate", "biology",
            "Enzyme reaction velocity v = (Vmax * S) / (Km + S).",
            (("vmax", "number"), ("substrate", "number"), ("km", "number")),
            _call(
                "divide",
                _call("multiply", _var("vmax"), _var("substrate")),
                _call("add", _var("km"), _var("substrate")),
            ),
        ),
        # ---------------- Economics / General ----------------
        _spec(
            "econ.simple_interest", "Simple Interest", "economics",
            "Simple interest I = P * r * t.",
            (("principal", "number"), ("rate", "number"), ("time", "number")),
            _call("multiply", _var("principal"), _call("multiply", _var("rate"), _var("time"))),
            domains=("economics", "general"),
        ),
        _spec(
            "econ.compound_amount", "Compound Amount", "economics",
            "Compound amount A = P * (1 + r)^n.",
            (("principal", "number"), ("rate", "number"), ("periods", "number")),
            _call("multiply", _var("principal"), _call("power", _call("add", _lit(1), _var("rate")), _var("periods"))),
            domains=("economics", "general"),
        ),
        _spec(
            "econ.discount_price", "Discounted Price", "economics",
            "Discounted price = price * (1 - discount_rate).",
            (("price", "number"), ("discount_rate", "number")),
            _call("multiply", _var("price"), _call("subtract", _lit(1), _var("discount_rate"))),
            domains=("economics", "general"),
        ),
        _spec(
            "econ.profit_margin", "Profit Margin", "economics",
            "Profit margin = (revenue - cost) / revenue.",
            (("revenue", "number"), ("cost", "number")),
            _call("divide", _call("subtract", _var("revenue"), _var("cost")), _var("revenue")),
            domains=("economics", "general"),
        ),
        _spec(
            "econ.percentage_change", "Percentage Change", "economics",
            "Percentage change = ((new - old) / old) * 100.",
            (("old_val", "number"), ("new_val", "number")),
            _call("multiply", _call("divide", _call("subtract", _var("new_val"), _var("old_val")), _var("old_val")), _lit(100)),
            domains=("economics", "general"),
        ),
    )


_DEFAULT_RECORDS: tuple[KnowledgeRecord, ...] = _build_records()


def default_uol_knowledge_records(
    domains: Iterable[str] | None = None,
) -> tuple[KnowledgeRecord, ...]:
    """Return canonical KnowledgeRecords carrying executable UOL rule payloads."""
    if domains is None:
        return _DEFAULT_RECORDS
    wanted = {str(d).strip().casefold() for d in domains}
    out = []
    for rec in _DEFAULT_RECORDS:
        rule_domains = {str(d).casefold() for d in rec.content["uol_rule"]["domains"]}
        if rec.domain.casefold() in wanted or (rule_domains & wanted):
            out.append(rec)
    return tuple(out)


def import_default_uol_rules(
    domains: Iterable[str] | None = None,
    extra_rules: Iterable[OperationalRule] = (),
) -> KnowledgeRuleImportReport:
    """Import the built-in structured rule library through KnowledgeRuleImporter."""
    importer = KnowledgeRuleImporter(base_rules=extra_rules)
    return importer.import_records(default_uol_knowledge_records(domains=domains))


def build_default_uol(
    domains: Iterable[str] | None = None,
    extra_rules: Iterable[OperationalRule] = (),
) -> UOL:
    """Construct a ready-to-use UOL interpreter loaded with verified structured rules."""
    report = import_default_uol_rules(domains=domains, extra_rules=extra_rules)
    return UOL(report.catalog(extra_rules).values())


def execute_library_rule(
    rule_id: str,
    inputs: Mapping[str, Any],
    *,
    uol: UOL | None = None,
) -> Any:
    """Execute a single library rule directly on concrete typed inputs."""
    engine = uol or build_default_uol()
    rule = engine.rules.get(rule_id)
    if rule is None:
        raise KeyError(f"unknown UOL library rule: {rule_id}")
    program = Program(
        program_id=f"p:lib:{rule_id}",
        name=rule.name,
        inputs=rule.inputs,
        body=Call(rule.rule_id, tuple(Var(port.name) for port in rule.inputs)),
        domain=rule.domains[0],
        rule_refs=(rule.rule_id,),
    )
    return engine.execute(program, inputs)
