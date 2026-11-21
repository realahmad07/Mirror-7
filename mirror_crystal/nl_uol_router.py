"""Step 1: Natural Language -> UOL Auto-Router (Forward, Inverse & Multi-Step Composed).

Maps plain-text quantitative questions into verified UOL rule executions or
multi-step UOL compositions without guessing or hallucinating. If required
inputs are missing or no verified rule matches, it abstains cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from .operational_language import (
    Call,
    Program,
    UOL,
    UOLError,
    UOLExecutionError,
    Var,
    VariableSpec,
)
from .uol_inverse_deriver import derive_all_inverse_rules
from .uol_rule_library import build_default_uol

_NUM = r"(-?\d+(?:\.\d+)?)"


@dataclass(frozen=True)
class NLRouteResult:
    matched: bool
    rule_ids: tuple[str, ...] = ()
    inputs: Mapping[str, float] = None  # type: ignore[assignment]
    output: float | int | bool | None = None
    unit: str = ""
    explanation: str = ""
    missing_inputs: tuple[str, ...] = ()

    def __post_init__(self):
        if self.inputs is None:
            object.__setattr__(self, "inputs", {})


def _extract_param(text: str, aliases: tuple[str, ...]) -> float | None:
    """Extract a numeric value associated with any alias, e.g. 'mass = 10', 'mass of 10 kg', 'm=5'."""
    for alias in aliases:
        # Pattern 1: "alias [=|:|is|of] NUMBER"
        p1 = rf"\b{re.escape(alias)}\s*(?:=|:|\bis\b|\bof\b)?\s*{_NUM}\b"
        m = re.search(p1, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
        # Pattern 2: "NUMBER alias" (e.g. "10 kg mass" or "5 meters radius")
        p2 = rf"{_NUM}\s*(?:kg|g|m/s|m|s|n|pa|j|w|v|a|ohm|ohms|mol|l|%|k|c|f)?\s+{re.escape(alias)}\b"
        m2 = re.search(p2, text, flags=re.IGNORECASE)
        if m2:
            return float(m2.group(1))
    return None


@dataclass(frozen=True)
class _RuleBinding:
    rule_id: str
    keywords: tuple[str, ...]
    param_aliases: Mapping[str, tuple[str, ...]]
    unit: str = ""
    inverse_targets: Mapping[str, tuple[str, ...]] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.inverse_targets is None:
            object.__setattr__(self, "inverse_targets", {})


_BINDINGS: tuple[_RuleBinding, ...] = (
    _RuleBinding(
        "physics.kinetic_energy",
        ("kinetic energy", "ke"),
        {"mass": ("mass", "m"), "velocity": ("velocity", "speed", "v")},
        unit="J",
        inverse_targets={
            "velocity": ("find velocity", "calculate velocity", "solve for velocity", "what is velocity", "find speed", "what is speed"),
            "mass": ("find mass", "calculate mass", "solve for mass", "what is mass"),
        },
    ),
    _RuleBinding(
        "physics.potential_energy",
        ("potential energy", "gravitational energy", "pe"),
        {"mass": ("mass", "m"), "gravity": ("gravity", "g"), "height": ("height", "h")},
        unit="J",
        inverse_targets={
            "height": ("find height", "calculate height", "solve for height"),
            "mass": ("find mass", "calculate mass", "solve for mass"),
        },
    ),
    _RuleBinding(
        "physics.newton_force",
        ("newton force", "force", "f=ma", "f = ma"),
        {"mass": ("mass", "m"), "accel": ("acceleration", "accel", "a")},
        unit="N",
        inverse_targets={
            "accel": ("find acceleration", "calculate acceleration", "solve for acceleration", "what is acceleration"),
            "mass": ("find mass", "calculate mass", "solve for mass", "what is mass"),
        },
    ),
    _RuleBinding(
        "physics.ohms_voltage",
        ("voltage", "ohm", "ohms law", "v=ir", "v = ir"),
        {"current": ("current", "amps", "i"), "resistance": ("resistance", "ohms", "r")},
        unit="V",
        inverse_targets={
            "current": ("find current", "calculate current", "solve for current", "what is current"),
            "resistance": ("find resistance", "calculate resistance", "solve for resistance", "what is resistance"),
        },
    ),
    _RuleBinding(
        "physics.electrical_power",
        ("electrical power", "electric power", "wattage", "power"),
        {"voltage": ("voltage", "volts", "v"), "current": ("current", "amps", "i")},
        unit="W",
        inverse_targets={
            "current": ("find current", "calculate current", "solve for current"),
            "voltage": ("find voltage", "calculate voltage", "solve for voltage"),
        },
    ),
    _RuleBinding(
        "physics.density",
        ("density",),
        {"mass": ("mass", "m"), "volume": ("volume", "vol", "v")},
        unit="kg/m^3",
        inverse_targets={
            "mass": ("find mass", "calculate mass", "solve for mass"),
            "volume": ("find volume", "calculate volume", "solve for volume"),
        },
    ),
    _RuleBinding(
        "physics.wave_speed",
        ("wave speed", "wave velocity"),
        {"frequency": ("frequency", "freq", "f"), "wavelength": ("wavelength", "lambda")},
        unit="m/s",
        inverse_targets={
            "frequency": ("find frequency", "calculate frequency", "solve for frequency"),
            "wavelength": ("find wavelength", "calculate wavelength", "solve for wavelength"),
        },
    ),
    _RuleBinding(
        "eng.axial_stress",
        ("axial stress", "stress"),
        {"force": ("force", "load", "f"), "area": ("area", "a")},
        unit="Pa",
        inverse_targets={
            "force": ("find force", "calculate force", "solve for force"),
            "area": ("find area", "calculate area", "solve for area"),
        },
    ),
    _RuleBinding(
        "eng.safety_factor",
        ("safety factor", "factor of safety"),
        {"yield_strength": ("yield strength", "yield", "allowable"), "applied_stress": ("applied stress", "stress")},
        unit="ratio",
    ),
    _RuleBinding(
        "eng.beam_center_moment",
        ("bending moment", "beam moment", "center moment"),
        {"load": ("load", "force", "f"), "span": ("span", "length", "l")},
        unit="N*m",
    ),
    _RuleBinding(
        "bio.bmi",
        ("bmi", "body mass index"),
        {"mass_kg": ("mass", "weight", "kg"), "height_m": ("height", "h")},
        unit="kg/m^2",
        inverse_targets={
            "mass_kg": ("find mass", "find weight", "solve for mass"),
            "height_m": ("find height", "solve for height"),
        },
    ),
    _RuleBinding(
        "bio.michaelis_menten",
        ("michaelis", "enzyme rate", "reaction rate"),
        {"vmax": ("vmax",), "substrate": ("substrate", "s"), "km": ("km",)},
        unit="rate",
    ),
    _RuleBinding(
        "chem.molarity",
        ("molarity", "molar concentration"),
        {"moles": ("moles", "mol", "n"), "liters": ("liters", "volume", "l")},
        unit="M",
        inverse_targets={
            "moles": ("find moles", "calculate moles", "solve for moles"),
            "liters": ("find liters", "find volume", "solve for liters"),
        },
    ),
    _RuleBinding(
        "econ.compound_amount",
        ("compound interest", "compound amount"),
        {"principal": ("principal", "p"), "rate": ("rate", "r"), "periods": ("periods", "years", "n", "t")},
        unit="currency",
    ),
    _RuleBinding(
        "econ.simple_interest",
        ("simple interest",),
        {"principal": ("principal", "p"), "rate": ("rate", "r"), "time": ("time", "years", "t")},
        unit="currency",
    ),
    _RuleBinding(
        "econ.discount_price",
        ("discount price", "discounted price"),
        {"price": ("price", "cost"), "discount_rate": ("discount rate", "discount", "rate")},
        unit="currency",
    ),
    _RuleBinding(
        "math.quadratic_discriminant",
        ("discriminant", "quadratic discriminant"),
        {"a": ("a",), "b": ("b",), "c": ("c",)},
        unit="",
    ),
    _RuleBinding(
        "math.circle_area",
        ("circle area", "area of circle", "area of a circle"),
        {"r": ("radius", "r")},
        unit="m^2",
        inverse_targets={
            "r": ("find radius", "calculate radius", "solve for radius", "what is radius"),
        },
    ),
    _RuleBinding(
        "math.cylinder_volume",
        ("cylinder volume", "volume of cylinder", "volume of a cylinder"),
        {"r": ("radius", "r"), "h": ("height", "h")},
        unit="m^3",
    ),
    _RuleBinding(
        "math.triangle_area",
        ("triangle area", "area of triangle", "area of a triangle"),
        {"base": ("base", "b"), "height": ("height", "h")},
        unit="m^2",
    ),
)


class NaturalLanguageUOLRouter:
    """Deterministic NL -> UOL executor supporting forward, inverse, and composed chains."""

    def __init__(self, uol: UOL | None = None, *, include_inverses: bool = True) -> None:
        if uol is not None:
            self.uol = uol
        else:
            base = build_default_uol()
            if include_inverses:
                all_rules = tuple(base.rules.values()) + derive_all_inverse_rules(base.rules.values())
                self.uol = UOL(all_rules)
            else:
                self.uol = base

    def _run_rule(self, rule_id: str, inputs: Mapping[str, float]) -> float | int | bool:
        rule = self.uol.rules.get(rule_id)
        if rule is None:
            raise UOLError(f"rule not loaded: {rule_id}")
        program = Program(
            program_id=f"p:nl:{rule_id}",
            name=rule.name,
            inputs=rule.inputs,
            body=Call(rule_id, tuple(Var(p.name) for p in rule.inputs)),
            domain=rule.domains[0],
            rule_refs=(rule_id,),
        )
        return self.uol.execute(program, inputs)

    def _route_temperature(self, lower: str) -> NLRouteResult | None:
        # e.g. "convert 98.6 F to Kelvin", "100 celsius in fahrenheit", "300 K to F"
        m = re.search(
            rf"{_NUM}\s*(?:°|deg(?:rees?)?)?\s*(celsius|fahrenheit|kelvin|c|f|k)\b.*?\b(?:to|in|into)\s*(celsius|fahrenheit|kelvin|c|f|k)\b",
            lower,
        )
        if not m:
            return None
        val = float(m.group(1))
        src = m.group(2)[0]
        dst = m.group(3)[0]
        if src == dst:
            return NLRouteResult(True, (), {"value": val}, val, dst.upper(), f"Identity conversion: {val} {dst.upper()}")

        chain_map = {
            ("c", "f"): (("physics.celsius_to_fahrenheit", "c"),),
            ("f", "c"): (("physics.fahrenheit_to_celsius", "f"),),
            ("c", "k"): (("physics.celsius_to_kelvin", "c"),),
            ("k", "c"): (("physics.celsius_to_kelvin__solve_c", "result"),),
            ("f", "k"): (("physics.fahrenheit_to_celsius", "f"), ("physics.celsius_to_kelvin", "c")),
            ("k", "f"): (("physics.celsius_to_kelvin__solve_c", "result"), ("physics.celsius_to_fahrenheit", "c")),
        }
        steps = chain_map.get((src, dst))
        if not steps:
            return None
        current = val
        used: list[str] = []
        for rule_id, arg_name in steps:
            if self.uol.rules.get(rule_id) is None:
                return None
            current = float(self._run_rule(rule_id, {arg_name: current}))
            used.append(rule_id)
        unit_name = {"c": "°C", "f": "°F", "k": "K"}[dst]
        return NLRouteResult(
            matched=True,
            rule_ids=tuple(used),
            inputs={"value": val},
            output=current,
            unit=unit_name,
            explanation=f"Verified UOL chain {' -> '.join(used)}: {val} -> {current:.6g} {unit_name}",
        )

    def _route_composed_safety_factor(self, lower: str) -> NLRouteResult | None:
        if not any(k in lower for k in ("safety factor", "factor of safety")):
            return None
        y = _extract_param(lower, ("yield strength", "yield", "allowable"))
        f = _extract_param(lower, ("force", "load"))
        a = _extract_param(lower, ("area",))
        if y is not None and f is not None and a is not None:
            if self.uol.rules.get("eng.axial_stress") and self.uol.rules.get("eng.safety_factor"):
                stress = float(self._run_rule("eng.axial_stress", {"force": f, "area": a}))
                sf = float(self._run_rule("eng.safety_factor", {"yield_strength": y, "applied_stress": stress}))
                return NLRouteResult(
                    matched=True,
                    rule_ids=("eng.axial_stress", "eng.safety_factor"),
                    inputs={"yield_strength": y, "force": f, "area": a},
                    output=sf,
                    unit="ratio",
                    explanation=f"Composed UOL: axial_stress({f}, {a})={stress:.6g} -> safety_factor({y}, {stress:.6g})={sf:.6g}",
                )
        return None

    def route(self, prompt: str) -> NLRouteResult:
        """Parse plain text and execute matching verified UOL rule(s) if all inputs are present."""
        if not isinstance(prompt, str) or not prompt.strip():
            return NLRouteResult(False, explanation="empty prompt")
        lower = prompt.strip().casefold()

        # 1. Temperature conversion chains (including multi-step F->K and K->F)
        temp_hit = self._route_temperature(lower)
        if temp_hit is not None:
            return temp_hit

        # 2. Multi-step composed engineering safety factor from yield + force + area
        sf_hit = self._route_composed_safety_factor(lower)
        if sf_hit is not None:
            return sf_hit

        # 3. Forward and inverse rule bindings
        for binding in _BINDINGS:
            if not any(kw in lower for kw in binding.keywords):
                continue

            # Check if the user is asking for an INVERSE solve (e.g. "find velocity if kinetic energy is 180 and mass is 10")
            for target_var, phrases in binding.inverse_targets.items():
                if any(phrase in lower for phrase in phrases):
                    inv_rule_id = f"{binding.rule_id}__solve_{target_var}"
                    if self.uol.rules.get(inv_rule_id) is None:
                        continue
                    res_val = _extract_param(lower, binding.keywords + ("result",))
                    inv_inputs: dict[str, float] = {}
                    missing: list[str] = []
                    if res_val is None:
                        missing.append("result")
                    else:
                        inv_inputs["result"] = res_val
                    for pname, aliases in binding.param_aliases.items():
                        if pname == target_var:
                            continue
                        val = _extract_param(lower, aliases)
                        if val is None:
                            missing.append(pname)
                        else:
                            inv_inputs[pname] = val
                    if missing:
                        return NLRouteResult(
                            False,
                            rule_ids=(inv_rule_id,),
                            missing_inputs=tuple(missing),
                            explanation=f"Matched inverse UOL rule {inv_rule_id}, missing inputs: {', '.join(missing)}",
                        )
                    try:
                        out = self._run_rule(inv_rule_id, inv_inputs)
                    except (UOLError, UOLExecutionError) as exc:
                        return NLRouteResult(False, rule_ids=(inv_rule_id,), explanation=str(exc))
                    return NLRouteResult(
                        matched=True,
                        rule_ids=(inv_rule_id,),
                        inputs=inv_inputs,
                        output=out,
                        explanation=f"Verified inverse UOL rule {inv_rule_id}: {target_var} = {out:.6g}" if isinstance(out, float) else f"{target_var} = {out}",
                    )

            # Standard forward rule
            if self.uol.rules.get(binding.rule_id) is None:
                continue
            fwd_inputs: dict[str, float] = {}
            missing_fwd: list[str] = []
            for pname, aliases in binding.param_aliases.items():
                val = _extract_param(lower, aliases)
                if val is None:
                    # Default gravity = 9.81 only if explicitly omitted? No—never guess!
                    missing_fwd.append(pname)
                else:
                    fwd_inputs[pname] = val
            if missing_fwd:
                return NLRouteResult(
                    False,
                    rule_ids=(binding.rule_id,),
                    missing_inputs=tuple(missing_fwd),
                    explanation=f"Matched UOL rule {binding.rule_id}, missing inputs: {', '.join(missing_fwd)}",
                )
            try:
                out = self._run_rule(binding.rule_id, fwd_inputs)
            except (UOLError, UOLExecutionError) as exc:
                return NLRouteResult(False, rule_ids=(binding.rule_id,), explanation=str(exc))
            formatted = f"{out:.6g}" if isinstance(out, float) else str(out)
            unit_suffix = f" {binding.unit}" if binding.unit else ""
            return NLRouteResult(
                matched=True,
                rule_ids=(binding.rule_id,),
                inputs=fwd_inputs,
                output=out,
                unit=binding.unit,
                explanation=f"Verified UOL rule {binding.rule_id}: {formatted}{unit_suffix}",
            )

        return NLRouteResult(False, explanation="no matching UOL rule for prompt")
