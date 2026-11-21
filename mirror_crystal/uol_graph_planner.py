"""Dynamic Unit-Graph Multi-Step Planner for Mirror's Universal Operational Language.

Treats every verified UOL rule (forward + auto-derived inverse) as a typed
directed hyper-edge `(input_quantities -> output_quantity)` and uses bounded
BFS forward-chaining with minimal dependency backtracking to dynamically
discover and execute novel multi-step derivations (1 to N hops) with zero
hardcoded chains and zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .operational_language import (
    Call,
    OperationalRule,
    Program,
    RuleStatus,
    UOL,
    UOLError,
    UOLExecutionError,
    Var,
)
from .uol_inverse_deriver import derive_all_inverse_rules
from .uol_rule_library import build_default_uol


@dataclass(frozen=True)
class QuantityEdge:
    rule_id: str
    rule_name: str
    input_quantities: Mapping[str, str]   # {uol_port_name: canonical_quantity}
    output_quantity: str
    output_unit: str = ""


@dataclass(frozen=True)
class PlannedStep:
    step_index: int
    rule_id: str
    rule_name: str
    inputs: Mapping[str, float]           # {canonical_quantity: value}
    port_inputs: Mapping[str, float]      # {uol_port_name: value}
    output_quantity: str
    output_value: float
    output_unit: str


@dataclass(frozen=True)
class GraphPlanResult:
    solved: bool
    target_quantity: str
    output: float | None = None
    unit: str = ""
    steps: tuple[PlannedStep, ...] = ()
    rule_ids: tuple[str, ...] = ()
    known_inputs: Mapping[str, float] = None  # type: ignore[assignment]
    reason: str = ""

    def __post_init__(self):
        if self.known_inputs is None:
            object.__setattr__(self, "known_inputs", {})


# Canonical units for physical / mathematical quantities
QUANTITY_UNITS: dict[str, str] = {
    "mass": "kg",
    "acceleration": "m/s^2",
    "velocity": "m/s",
    "force": "N",
    "momentum": "kg*m/s",
    "kinetic_energy": "J",
    "potential_energy": "J",
    "energy": "J",
    "work": "J",
    "input_energy": "J",
    "distance": "m",
    "height": "m",
    "gravity": "m/s^2",
    "current": "A",
    "resistance": "Ohm",
    "voltage": "V",
    "electrical_power": "W",
    "volume": "m^3",
    "density": "kg/m^3",
    "frequency": "Hz",
    "wavelength": "m",
    "wave_speed": "m/s",
    "celsius": "°C",
    "fahrenheit": "°F",
    "kelvin": "K",
    "area": "m^2",
    "stress": "Pa",
    "yield_strength": "Pa",
    "safety_factor": "ratio",
    "pressure": "Pa",
    "span": "m",
    "bending_moment": "N*m",
    "efficiency": "ratio",
    "bmi": "kg/m^2",
    "moles": "mol",
    "liters": "L",
    "molarity": "M",
    "radius": "m",
    "circle_area": "m^2",
    "cylinder_volume": "m^3",
    "base": "m",
    "triangle_area": "m^2",
    "principal": "currency",
    "rate": "ratio",
    "time": "s",
    "simple_interest": "currency",
    "price": "currency",
    "discount_rate": "ratio",
    "discount_price": "currency",
}


# Base forward quantity signatures; inverse edges are derived automatically.
_BASE_SIGNATURES: tuple[tuple[str, dict[str, str], str], ...] = (
    # Physics
    ("physics.newton_force", {"mass": "mass", "accel": "acceleration"}, "force"),
    ("physics.momentum", {"mass": "mass", "velocity": "velocity"}, "momentum"),
    ("physics.kinetic_energy", {"mass": "mass", "velocity": "velocity"}, "kinetic_energy"),
    ("physics.potential_energy", {"mass": "mass", "gravity": "gravity", "height": "height"}, "potential_energy"),
    ("physics.work_done", {"force": "force", "distance": "distance"}, "work"),
    ("physics.ohms_voltage", {"current": "current", "resistance": "resistance"}, "voltage"),
    ("physics.electrical_power", {"voltage": "voltage", "current": "current"}, "electrical_power"),
    ("physics.density", {"mass": "mass", "volume": "volume"}, "density"),
    ("physics.wave_speed", {"frequency": "frequency", "wavelength": "wavelength"}, "wave_speed"),
    ("physics.celsius_to_fahrenheit", {"c": "celsius"}, "fahrenheit"),
    ("physics.fahrenheit_to_celsius", {"f": "fahrenheit"}, "celsius"),
    ("physics.celsius_to_kelvin", {"c": "celsius"}, "kelvin"),
    # Engineering
    ("eng.axial_stress", {"force": "force", "area": "area"}, "stress"),
    ("eng.safety_factor", {"yield_strength": "yield_strength", "applied_stress": "stress"}, "safety_factor"),
    ("eng.beam_center_moment", {"load": "force", "span": "span"}, "bending_moment"),
    ("eng.efficiency", {"output_energy": "work", "input_energy": "input_energy"}, "efficiency"),
    ("eng.pressure", {"force": "force", "area": "area"}, "pressure"),
    # Biology & Chemistry
    ("bio.bmi", {"mass_kg": "mass", "height_m": "height"}, "bmi"),
    ("chem.molarity", {"moles": "moles", "liters": "liters"}, "molarity"),
    # Geometry & Economics
    ("math.circle_area", {"r": "radius"}, "circle_area"),
    ("math.cylinder_volume", {"r": "radius", "h": "height"}, "cylinder_volume"),
    ("math.triangle_area", {"base": "base", "height": "height"}, "triangle_area"),
    ("econ.simple_interest", {"principal": "principal", "rate": "rate", "time": "time"}, "simple_interest"),
    ("econ.discount_price", {"price": "price", "discount_rate": "discount_rate"}, "discount_price"),
)


def _build_quantity_edges(uol: UOL) -> tuple[QuantityEdge, ...]:
    edges: list[QuantityEdge] = []
    seen_ids: set[str] = set()

    for rule_id, in_map, out_qty in _BASE_SIGNATURES:
        rule = uol.rules.get(rule_id)
        if rule is None or rule.status in {RuleStatus.UNKNOWN, RuleStatus.DISPUTED}:
            continue
        edges.append(QuantityEdge(
            rule_id=rule_id,
            rule_name=rule.name,
            input_quantities=dict(in_map),
            output_quantity=out_qty,
            output_unit=QUANTITY_UNITS.get(out_qty, ""),
        ))
        seen_ids.add(rule_id)

        # Automatically register all matching derived inverse edges!
        for port_name, target_qty in in_map.items():
            inv_id = f"{rule_id}__solve_{port_name}"
            inv_rule = uol.rules.get(inv_id)
            if inv_rule is None or inv_id in seen_ids:
                continue
            if inv_rule.status in {RuleStatus.UNKNOWN, RuleStatus.DISPUTED}:
                continue
            inv_inputs = {p: q for p, q in in_map.items() if p != port_name}
            inv_inputs["result"] = out_qty
            edges.append(QuantityEdge(
                rule_id=inv_id,
                rule_name=inv_rule.name,
                input_quantities=inv_inputs,
                output_quantity=target_qty,
                output_unit=QUANTITY_UNITS.get(target_qty, ""),
            ))
            seen_ids.add(inv_id)

    edges.sort(key=lambda e: (len(e.input_quantities), e.rule_id))
    return tuple(edges)


class UOLGraphPlanner:
    """Zero-shot multi-step quantity/unit hypergraph planner over verified UOL rules."""

    VERSION = "uol-graph-planner-v1"

    def __init__(self, uol: UOL | None = None, *, include_inverses: bool = True) -> None:
        if uol is not None:
            self.uol = uol
        else:
            base = build_default_uol()
            if include_inverses:
                self.uol = UOL(tuple(base.rules.values()) + derive_all_inverse_rules(base.rules.values()))
            else:
                self.uol = base
        self.edges = _build_quantity_edges(self.uol)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def register_rule(
        self,
        rule: OperationalRule,
        *,
        input_quantities: Mapping[str, str],
        output_quantity: str,
        output_unit: str = "",
    ) -> None:
        """Dynamically add a newly taught or synthesized verified UOL rule into the live graph."""
        existing = [r for r in self.uol.rules.values() if r.rule_id != rule.rule_id]
        existing.append(rule)
        self.uol = UOL(existing)
        new_edge = QuantityEdge(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            input_quantities={k: v.strip().casefold() for k, v in input_quantities.items()},
            output_quantity=output_quantity.strip().casefold(),
            output_unit=output_unit or QUANTITY_UNITS.get(output_quantity.strip().casefold(), ""),
        )
        updated = [e for e in self.edges if e.rule_id != rule.rule_id]
        updated.append(new_edge)
        updated.sort(key=lambda e: (len(e.input_quantities), e.rule_id))
        self.edges = tuple(updated)

    def _exec_rule(self, rule_id: str, port_inputs: Mapping[str, float]) -> float:
        rule = self.uol.rules.get(rule_id)
        if rule is None:
            raise UOLError(f"rule not found: {rule_id}")
        prog = Program(
            program_id=f"p:graph:{rule_id}",
            name=rule.name,
            inputs=rule.inputs,
            body=Call(rule_id, tuple(Var(p.name) for p in rule.inputs)),
            domain=rule.domains[0],
            rule_refs=(rule_id,),
        )
        out = self.uol.execute(prog, port_inputs)
        if isinstance(out, bool) or not isinstance(out, (int, float)) or not math.isfinite(float(out)):
            raise UOLExecutionError("non-finite numeric output")
        return float(out)

    def plan_and_execute(
        self,
        known: Mapping[str, float],
        target_quantity: str,
        *,
        max_hops: int = 5,
    ) -> GraphPlanResult:
        """Dynamically discover and execute the shortest verified UOL chain to `target_quantity`."""
        target = str(target_quantity).strip().casefold()
        values: dict[str, float] = {str(k).strip().casefold(): float(v) for k, v in known.items()}

        if target in values:
            return GraphPlanResult(
                solved=True,
                target_quantity=target,
                output=values[target],
                unit=QUANTITY_UNITS.get(target, ""),
                steps=(),
                rule_ids=(),
                known_inputs=dict(values),
                reason="target already provided in known inputs",
            )

        # Track how each derived quantity was produced
        produced_by: dict[str, tuple[QuantityEdge, dict[str, float], dict[str, float], float]] = {}

        for _hop in range(1, max(1, int(max_hops)) + 1):
            newly_derived: list[tuple[QuantityEdge, dict[str, float], dict[str, float], float]] = []
            for edge in self.edges:
                if edge.output_quantity in values:
                    continue
                if not all(qty in values for qty in edge.input_quantities.values()):
                    continue
                port_inputs = {port: values[qty] for port, qty in edge.input_quantities.items()}
                qty_inputs = {qty: values[qty] for qty in edge.input_quantities.values()}
                try:
                    out_val = self._exec_rule(edge.rule_id, port_inputs)
                except (UOLError, UOLExecutionError, ZeroDivisionError, ValueError, TypeError):
                    continue
                newly_derived.append((edge, qty_inputs, port_inputs, out_val))

            if not newly_derived:
                break

            for edge, qty_inputs, port_inputs, out_val in newly_derived:
                if edge.output_quantity not in values:
                    values[edge.output_quantity] = out_val
                    produced_by[edge.output_quantity] = (edge, qty_inputs, port_inputs, out_val)

            if target in values:
                break

        if target not in values:
            return GraphPlanResult(
                solved=False,
                target_quantity=target,
                known_inputs={str(k).strip().casefold(): float(v) for k, v in known.items()},
                reason=f"no verified UOL derivation path from {sorted(known.keys())} to '{target}' within {max_hops} hops",
            )

        # Backtrack ONLY the minimal dependency DAG needed to derive `target`
        ordered: list[tuple[QuantityEdge, dict[str, float], dict[str, float], float]] = []
        visited: set[str] = set()

        def collect(qty: str) -> None:
            if qty in visited or qty not in produced_by:
                return
            visited.add(qty)
            edge, qty_inputs, port_inputs, out_val = produced_by[qty]
            for dep_qty in edge.input_quantities.values():
                collect(dep_qty)
            ordered.append((edge, qty_inputs, port_inputs, out_val))

        collect(target)

        steps = tuple(
            PlannedStep(
                step_index=idx + 1,
                rule_id=edge.rule_id,
                rule_name=edge.rule_name,
                inputs=qty_inputs,
                port_inputs=port_inputs,
                output_quantity=edge.output_quantity,
                output_value=out_val,
                output_unit=edge.output_unit,
            )
            for idx, (edge, qty_inputs, port_inputs, out_val) in enumerate(ordered)
        )
        return GraphPlanResult(
            solved=True,
            target_quantity=target,
            output=values[target],
            unit=QUANTITY_UNITS.get(target, ""),
            steps=steps,
            rule_ids=tuple(s.rule_id for s in steps),
            known_inputs={str(k).strip().casefold(): float(v) for k, v in known.items()},
            reason=f"solved in {len(steps)} verified UOL step(s)",
        )
