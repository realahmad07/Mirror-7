"""Step 1: Natural Language -> UOL Auto-Router (Forward, Inverse & Multi-Step Composed).

Maps plain-text quantitative questions into verified UOL rule executions or
multi-step UOL compositions without guessing or hallucinating. If required
inputs are missing or no verified rule matches, it abstains cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from .code_repair_engine import CounterexampleCodeRepairEngine
from .compression_analogy_engine import AnalogicalTransferEngine
from .counterfactual_engine import CounterfactualWhatIfEngine
from .creative_conversation_engine import CreativeConversationEngine
from .formal_logic_engine import solve_formal_logic_prompt
from .native_realizer import NativeHybridRealizer
from .operational_language import (
    Call,
    Program,
    UOL,
    UOLError,
    UOLExecutionError,
    Var,
    VariableSpec,
)
from .semantic_frame_vm import SemanticStateVM
from .symbolic_algebra_solver import solve_algebra_prompt
from .tabular_data_analyst import ZeroParamDataAnalyst
from .uol_formula_teacher import compile_formula_to_uol, parse_teach_formula_prompt
from .uol_graph_planner import UOLGraphPlanner
from .uol_inverse_deriver import derive_all_inverse_rules
from .uol_rule_library import build_default_uol
from .unit_normalizer import normalize_to_si, unit_pattern_for_quantity
from .unstructured_prose_engine import UnstructuredProseEngine

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
    steps: tuple[dict[str, Any], ...] = ()

    def __post_init__(self):
        if self.inputs is None:
            object.__setattr__(self, "inputs", {})


def _extract_param(text: str, aliases: tuple[str, ...], *, quantity: str | None = None) -> float | None:
    """Extract a numeric value associated with any alias, converting attached units to SI."""
    unit_pat = unit_pattern_for_quantity(quantity)
    unit_group = rf"(?:\s*({unit_pat})(?![a-zA-Z0-9_]))?" if unit_pat else r"()"

    for alias in aliases:
        # Pattern 1: "alias [=|:|is|of] NUMBER [UNIT]"
        p1 = rf"\b{re.escape(alias)}\s*(?:=|:|\bis\b|\bof\b)?\s*{_NUM}{unit_group}"
        m = re.search(p1, text, flags=re.IGNORECASE)
        if m:
            raw_val = float(m.group(1))
            unit_tok = m.group(2) if m.lastindex and m.lastindex >= 2 else None
            val, _ = normalize_to_si(raw_val, unit_tok, quantity)
            return val
        # Pattern 2: "NUMBER [UNIT] alias" (e.g. "2000 g mass", "5 kN force", "200 cm2 area")
        if unit_pat:
            p2 = rf"{_NUM}\s*({unit_pat})\s+{re.escape(alias)}\b"
            m2 = re.search(p2, text, flags=re.IGNORECASE)
            if m2:
                val, _ = normalize_to_si(float(m2.group(1)), m2.group(2), quantity)
                return val
        p3 = rf"{_NUM}\s*(?:kg|g|m/s|m|s|n|pa|j|w|v|a|ohm|ohms|mol|l|%|k|c|f)?\s+{re.escape(alias)}\b"
        m3 = re.search(p3, text, flags=re.IGNORECASE)
        if m3:
            return float(m3.group(1))
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
        ("kinetic energy",),
        {"mass": ("mass", "m"), "velocity": ("velocity", "speed", "v")},
        unit="J",
        inverse_targets={
            "velocity": ("find velocity", "calculate velocity", "solve for velocity", "what is velocity", "find speed", "what is speed"),
            "mass": ("find mass", "calculate mass", "solve for mass", "what is mass"),
        },
    ),
    _RuleBinding(
        "physics.potential_energy",
        ("potential energy", "gravitational energy"),
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
        {"mass_kg": ("mass", "weight", "person", "kg"), "height_m": ("height", "tall", "h")},
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


_QUANTITY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("safety_factor", ("safety factor", "factor of safety")),
    ("electrical_power", ("electrical power", "electric power", "wattage", "power")),
    ("kinetic_energy", ("kinetic energy", "ke")),
    ("potential_energy", ("potential energy", "gravitational energy", "pe")),
    ("input_energy", ("input energy",)),
    ("yield_strength", ("yield strength", "allowable stress")),
    ("bending_moment", ("bending moment", "beam moment")),
    ("cylinder_volume", ("cylinder volume", "volume of cylinder", "volume of a cylinder")),
    ("circle_area", ("circle area", "area of circle", "area of a circle")),
    ("triangle_area", ("triangle area", "area of triangle", "area of a triangle")),
    ("wave_speed", ("wave speed", "wave velocity")),
    ("simple_interest", ("simple interest",)),
    ("discount_price", ("discount price", "discounted price")),
    ("discount_rate", ("discount rate",)),
    ("acceleration", ("acceleration", "accel")),
    ("momentum", ("momentum",)),
    ("efficiency", ("efficiency",)),
    ("velocity", ("velocity", "speed")),
    ("distance", ("distance",)),
    ("resistance", ("resistance", "ohms")),
    ("current", ("current", "amps")),
    ("voltage", ("voltage", "volts")),
    ("stress", ("applied stress", "axial stress", "stress")),
    ("force", ("force", "load")),
    ("mass", ("mass", "weight")),
    ("area", ("area",)),
    ("height", ("height", "tall")),
    ("radius", ("radius",)),
    ("volume", ("volume", "vol")),
    ("density", ("density",)),
    ("work", ("work done", "output energy", "work")),
    ("gravity", ("gravity",)),
    ("frequency", ("frequency", "freq")),
    ("wavelength", ("wavelength", "lambda")),
    ("celsius", ("celsius",)),
    ("fahrenheit", ("fahrenheit",)),
    ("kelvin", ("kelvin",)),
    ("moles", ("moles", "mol")),
    ("liters", ("liters",)),
    ("molarity", ("molarity",)),
    ("bmi", ("bmi", "body mass index")),
    ("span", ("span",)),
)


class NaturalLanguageUOLRouter:
    """Deterministic NL -> UOL executor supporting forward, inverse, and dynamic graph chains."""

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
        self.graph_planner = UOLGraphPlanner(self.uol)
        self.realizer = NativeHybridRealizer(default_style="step_by_step")
        self.semantic_vm = SemanticStateVM()
        self.analogy_engine = AnalogicalTransferEngine()
        self.prose_engine = UnstructuredProseEngine()
        self.counterfactual_engine = CounterfactualWhatIfEngine()
        self.code_repair_engine = CounterexampleCodeRepairEngine()
        self.data_analyst = ZeroParamDataAnalyst()
        self.creative_engine = CreativeConversationEngine()
        self._session_quantities: dict[str, dict[str, float]] = {}
        self._dynamic_quantity_aliases: list[tuple[str, tuple[str, ...]]] = []

    def _all_quantity_aliases(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(self._dynamic_quantity_aliases) + _QUANTITY_ALIASES

    def _register_quantity_alias(self, qty: str) -> None:
        clean = qty.strip().casefold()
        existing = {q for q, _ in self._all_quantity_aliases()}
        if clean not in existing:
            aliases = (clean, clean.replace("_", " ")) if "_" in clean else (clean,)
            self._dynamic_quantity_aliases.append((clean, aliases))

    def teach_formula(
        self,
        target_name: str,
        expr_str: str,
        *,
        domain: str = "taught",
        vault_path: str | None = None,
    ) -> NLRouteResult:
        """Compile a plain-text formula into a verified UOL rule + inverses and wire live into the graph."""
        taught = compile_formula_to_uol(target_name, expr_str, domain=domain, vault_path=vault_path)
        if not taught.ok or taught.forward_rule is None:
            return NLRouteResult(False, explanation=taught.message)

        fwd = taught.forward_rule
        in_map = {p.name: p.name.casefold() for p in fwd.inputs}
        self._register_quantity_alias(taught.target_quantity)
        for q in in_map.values():
            self._register_quantity_alias(q)

        # Register forward rule in both UOL and live GraphPlanner
        self.graph_planner.register_rule(
            fwd,
            input_quantities=in_map,
            output_quantity=taught.target_quantity,
        )
        # Register every round-trip verified inverse rule in live GraphPlanner too!
        for inv in taught.inverse_rules:
            target_port = inv.rule_id.split("__solve_")[-1]
            inv_map = {p.name: (taught.target_quantity if p.name == "result" else p.name.casefold()) for p in inv.inputs}
            self.graph_planner.register_rule(
                inv,
                input_quantities=inv_map,
                output_quantity=target_port.casefold(),
            )
        self.uol = self.graph_planner.uol
        return NLRouteResult(
            matched=True,
            rule_ids=tuple(r.rule_id for r in taught.all_rules),
            explanation=taught.message,
        )

    def get_session_quantities(self, session_id: str) -> dict[str, float]:
        return dict(self._session_quantities.get(session_id, {}))

    def clear_session(self, session_id: str) -> None:
        self._session_quantities.pop(session_id, None)

    def _extract_all_quantities(self, lower: str, *, exclude: str | None = None) -> dict[str, float]:
        found: dict[str, float] = {}
        for qty, aliases in self._all_quantity_aliases():
            if qty == exclude:
                continue
            val = _extract_param(lower, aliases, quantity=qty)
            if val is not None:
                found[qty] = val

        # Unit-inferred extraction for casual / natural phrasing without explicit variable names
        # e.g. "a 10 kg object moving at 6 m/s" or "10 kilo ki cheez 5 m/s par"
        unit_inferences = (
            ("mass", r"(\d+(?:\.\d+)?)\s*(kg|kilo|kilos|kilogram|kilograms|g|grams?|lbs?|pounds?)\b"),
            ("velocity", r"(\d+(?:\.\d+)?)\s*(m/s|mps|km/h|kph|kmh|mph)\b"),
            ("force", r"(\d+(?:\.\d+)?)\s*(kn|newtons?|lbf)\b"),
            ("area", r"(\d+(?:\.\d+)?)\s*(m\^2|m2|m²|cm\^2|cm2|cm²|mm2|ft2)\b"),
            ("current", r"(\d+(?:\.\d+)?)\s*(ma|amps?|amperes?)\b"),
            ("resistance", r"(\d+(?:\.\d+)?)\s*(kohms?|mohms?|ohms?|ω)\b"),
            ("voltage", r"(\d+(?:\.\d+)?)\s*(kv|mv|volts?)\b"),
            ("kinetic_energy", r"(\d+(?:\.\d+)?)\s*(kj|joules?)\b"),
            ("height", r"(\d+(?:\.\d+)?)\s*(cm|mm|m|meters?|ft|feet)\s+(?:tall|high|height)\b"),
        )
        for qty, pat in unit_inferences:
            if qty == exclude or qty in found:
                continue
            m = re.search(pat, lower, flags=re.IGNORECASE)
            if m:
                raw_num = float(m.group(1))
                tok = m.group(2).casefold()
                if tok in {"kilo", "kilos"}:
                    tok = "kg"
                val, _ = normalize_to_si(raw_num, tok, qty)
                found[qty] = val
        return found

    def _detect_target_quantity(self, lower: str) -> str | None:
        for qty, aliases in self._all_quantity_aliases():
            for alias in aliases:
                pat = (
                    rf"\b(?:find|calculate|compute|what\s+is(?:\s+the)?|what'?s(?:\s+the)?|"
                    rf"how\s+much(?:\s+is(?:\s+the)?)?|solve\s+for|get|determine)\s+(?:the\s+)?{re.escape(alias)}\b"
                )
                if re.search(pat, lower):
                    return qty
                # Suffix question forms (including Roman Urdu: "kinetic energy kitni?", "force nikalo", "stress batao")
                pat_suffix = rf"\b{re.escape(alias)}\s+(?:kitni|kitna|nikalo|batao|kya\s+hogi|kya\s+hai)\b"
                if re.search(pat_suffix, lower):
                    return qty
        return None

    def _route_dynamic_graph(self, lower: str, *, session_id: str | None = None) -> NLRouteResult | None:
        """Use UOLGraphPlanner to dynamically discover multi-step derivations (1-4 hops),
        optionally combining known quantities across multi-turn sessions."""
        target_qty = self._detect_target_quantity(lower)
        turn_known = self._extract_all_quantities(lower, exclude=target_qty)

        if session_id is not None:
            session_state = self._session_quantities.setdefault(session_id, {})
            # If user explicitly updates a base input, store it in session state
            session_state.update(turn_known)
            combined_known = dict(session_state)
            if target_qty is not None and target_qty in combined_known and turn_known:
                # Recompute target if new inputs were supplied in this turn
                combined_known.pop(target_qty, None)
        else:
            combined_known = turn_known

        # Pure statement turn (e.g., "mass is 10 kg and acceleration is 5") — remember for next turn!
        if target_qty is None:
            if session_id is not None and turn_known and not lower.endswith("?"):
                summary = ", ".join(f"{k}={v:g}" for k, v in sorted(turn_known.items()))
                return NLRouteResult(
                    matched=True,
                    rule_ids=("session.quantity_memory",),
                    inputs=turn_known,
                    output=None,
                    explanation=f"Stored session quantities: {summary}. Ask for any derived quantity next.",
                )
            return None

        if not combined_known:
            return None

        plan = self.graph_planner.plan_and_execute(combined_known, target_qty, max_hops=4)
        if not plan.solved or plan.output is None:
            return None

        # Persist derived target (and intermediate quantities) into session memory for follow-up turns!
        if session_id is not None:
            session_state = self._session_quantities.setdefault(session_id, {})
            for step in plan.steps:
                session_state[step.output_quantity] = step.output_value
            session_state[target_qty] = plan.output

        realized = self.realizer.realize_graph_plan(plan, style="step_by_step")
        steps_payload = tuple(
            {
                "step_index": s.step_index,
                "rule_id": s.rule_id,
                "rule_name": s.rule_name,
                "inputs": dict(s.inputs),
                "output_quantity": s.output_quantity,
                "output_value": s.output_value,
                "output_unit": s.output_unit,
            }
            for s in plan.steps
        )
        return NLRouteResult(
            matched=True,
            rule_ids=plan.rule_ids,
            inputs=combined_known,
            output=plan.output,
            unit=plan.unit,
            explanation=realized.text,
            steps=steps_payload,
        )

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
        y = _extract_param(lower, ("yield strength", "yield", "allowable"), quantity="yield_strength")
        f = _extract_param(lower, ("force", "load"), quantity="force")
        a = _extract_param(lower, ("area",), quantity="area")
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

    def route(self, prompt: str, *, session_id: str | None = None) -> NLRouteResult:
        """Parse plain text and execute matching verified UOL rule(s) if all inputs are present."""
        if not isinstance(prompt, str) or not prompt.strip():
            return NLRouteResult(False, explanation="empty prompt")
        lower = prompt.strip().casefold()

        if session_id is not None and re.search(r"\b(?:clear|reset)\s+(?:quantities|session|variables)\b", lower):
            self.clear_session(session_id)
            return NLRouteResult(True, rule_ids=("session.clear",), explanation="Cleared stored session quantities.")

        # 0. Live 1-shot formula teaching ("Learn formula: torque = force * lever_arm")
        taught_spec = parse_teach_formula_prompt(prompt)
        if taught_spec is not None:
            return self.teach_formula(taught_spec[0], taught_spec[1])

        # 0a. Live conversational blueprint distillation & feeding ("Feed: ... => ...", "Feed file: ...")
        if re.match(r"^\s*(?:feed|sikhao|distill|teach\s+reply|teach\s+dialogue|feed\s+file|feed\s+corpus)\s*[:\-]", prompt, flags=re.IGNORECASE):
            feed_hit = self.creative_engine.route_creative_or_dialogue(prompt, session_id=session_id or "default")
            if feed_hit.matched:
                return NLRouteResult(
                    matched=True,
                    rule_ids=(feed_hit.rule_id,),
                    explanation=feed_hit.response,
                )

        # 0b. Counterexample-Guided Autonomous AST Buggy-Code Repair
        repair_hit = self.code_repair_engine.repair_prompt(prompt)
        if repair_hit.matched:
            return NLRouteResult(
                matched=repair_hit.repaired,
                rule_ids=(f"code_repair.{repair_hit.func_name}",),
                explanation=repair_hit.explanation,
            )

        # 0c. Zero-Parameter Tabular, CSV & Time-Series Data Analyst
        data_hit = self.data_analyst.analyze_prompt(prompt)
        if data_hit.matched:
            return NLRouteResult(
                matched=True,
                rule_ids=("data_analyst.statistical_ols_engine",),
                output=data_hit.primary_output,
                explanation=data_hit.explanation,
            )

        # 0d. Symbolic Counterfactual & 'What-If' Simulation Engine
        cf_hit = self.counterfactual_engine.solve(prompt)
        if cf_hit.matched:
            return NLRouteResult(
                matched=True,
                rule_ids=(cf_hit.rule_id,),
                output=cf_hit.output,
                unit=cf_hit.unit,
                explanation=cf_hit.explanation,
            )

        # 0e. Parameter-Free Unstructured Prose Summarization, OpenIE Ingestion, Document QA & Comparative RST Discourse
        prose_hit = self.prose_engine.route_unstructured_prompt(prompt)
        if prose_hit is not None:
            return NLRouteResult(
                matched=bool(prose_hit["matched"]),
                rule_ids=tuple(prose_hit.get("rule_ids", ())),
                explanation=str(prose_hit.get("explanation", "")),
            )

        # 1. Temperature conversion chains (including multi-step F->K and K->F)
        temp_hit = self._route_temperature(lower)
        if temp_hit is not None:
            return temp_hit

        # 2. Multi-step composed engineering safety factor from yield + force + area
        sf_hit = self._route_composed_safety_factor(lower)
        if sf_hit is not None:
            return sf_hit

        # 3. Forward and inverse rule bindings (only when numeric inputs are present in the prompt)
        for binding in (_BINDINGS if re.search(r"\d", lower) else ()):
            if not any(re.search(r"\b" + re.escape(kw) + r"\b", lower) for kw in binding.keywords):
                continue

            # Check if the user is asking for an INVERSE solve (e.g. "find velocity if kinetic energy is 180 and mass is 10")
            for target_var, phrases in binding.inverse_targets.items():
                if any(phrase in lower for phrase in phrases):
                    inv_rule_id = f"{binding.rule_id}__solve_{target_var}"
                    if self.uol.rules.get(inv_rule_id) is None:
                        continue
                    res_qty = binding.rule_id.split(".")[-1]
                    res_val = _extract_param(lower, binding.keywords + ("result",), quantity=res_qty)
                    inv_inputs: dict[str, float] = {}
                    missing: list[str] = []
                    if res_val is None:
                        missing.append("result")
                    else:
                        inv_inputs["result"] = res_val
                    for pname, aliases in binding.param_aliases.items():
                        if pname == target_var:
                            continue
                        val = _extract_param(lower, aliases, quantity=pname)
                        if val is None:
                            missing.append(pname)
                        else:
                            inv_inputs[pname] = val
                    if missing:
                        graph_hit = self._route_dynamic_graph(lower, session_id=session_id)
                        if graph_hit is not None:
                            return graph_hit
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
                val = _extract_param(lower, aliases, quantity=pname)
                if val is None:
                    missing_fwd.append(pname)
                else:
                    fwd_inputs[pname] = val
            if missing_fwd:
                # Check if dynamic unit-graph planner (including session memory) can derive it!
                graph_hit = self._route_dynamic_graph(lower, session_id=session_id)
                if graph_hit is not None:
                    return graph_hit
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

        # 4. Zero-shot multi-step unit-graph planner & multi-turn session state fallback
        graph_hit = self._route_dynamic_graph(lower, session_id=session_id)
        if graph_hit is not None:
            return graph_hit

        # 5. Formal deductive logic, ordering chains & finite-domain constraint puzzles
        logic_hit = solve_formal_logic_prompt(prompt)
        if logic_hit.matched and logic_hit.proven is not None:
            return NLRouteResult(
                matched=True,
                rule_ids=(f"logic.{logic_hit.kind}",),
                output=logic_hit.answer if isinstance(logic_hit.answer, (bool, int, float)) else None,
                explanation=logic_hit.explanation,
            )

        # 6. Native symbolic algebra & linear system solver (linear, quadratic, N x N systems)
        alg_hit = solve_algebra_prompt(prompt)
        if alg_hit.solved and alg_hit.verified:
            first_val = next(iter(alg_hit.solutions.values()), None)
            out_scalar = first_val if isinstance(first_val, (int, float)) else None
            return NLRouteResult(
                matched=True,
                rule_ids=(f"algebra.{alg_hit.kind}",),
                output=out_scalar,
                explanation=alg_hit.explanation,
            )

        # 7. Parameter-Free Semantic Frame VM (free-form multi-clause word problems)
        vm_hit = self.semantic_vm.solve(prompt)
        if vm_hit.solved and vm_hit.output is not None:
            return NLRouteResult(
                matched=True,
                rule_ids=(f"semantic_vm.{vm_hit.frame_kind}",),
                output=vm_hit.output,
                explanation=vm_hit.explanation,
            )

        # 8. Parameter-Free Raw-DEFLATE Skeleton-NCD Analogical Program Transfer
        analogy_hit = self.analogy_engine.transfer_solve(prompt)
        if analogy_hit.matched and analogy_hit.output is not None:
            return NLRouteResult(
                matched=True,
                rule_ids=(analogy_hit.schema_id,),
                output=analogy_hit.output,
                explanation=analogy_hit.explanation,
            )

        # 9. Parameter-Free Creative Writing, Bilingual Dialogue & Unbounded Conversational Blueprints
        creative_hit = self.creative_engine.route_creative_or_dialogue(prompt, session_id=session_id or "default")
        if creative_hit.matched:
            return NLRouteResult(
                matched=True,
                rule_ids=(creative_hit.rule_id,),
                explanation=creative_hit.response,
            )

        return NLRouteResult(False, explanation="no matching UOL rule for prompt")
