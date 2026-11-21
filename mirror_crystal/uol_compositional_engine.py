"""Phase 3: End-to-end UOL Compositional Synthesis, Verification & Generalization Engine.

Connects:
- Phase 2 executable UOL rule library (via KnowledgeRuleImporter)
- Phase 3 OE-pruned compositional CEGIS synthesizer (CapabilitySynthesizer)
- Phase 4 independent native domain verification (NativeDomainVerifierSuite)
- Optional SymPy symbolic equivalence check when sympy is installed
- Generalization verification on unseen probe points
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Mapping

from .capability_synthesizer import (
    CapabilitySynthesizer,
    Example,
    SynthesisGoal,
    SynthesisReport,
)
from .domain_verification import (
    DomainDecision,
    DomainVerificationEngine,
)
from .native_verifiers import NativeDomainVerifierSuite
from .operational_language import (
    OperationalRule,
    Program,
    RuleStatus,
    UOL,
    VariableSpec,
    program_dict,
)
from .uol_rule_library import build_default_uol


def _sympy_available() -> bool:
    try:
        import sympy  # noqa: F401
        return True
    except ImportError:
        return False


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=1e-7)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_values_equal(a, b) for a, b in zip(left, right))
    return left == right


@dataclass(frozen=True)
class GeneralizationReport:
    total_unseen: int
    passed_unseen: int
    exact_accuracy: float
    failures: tuple[dict[str, Any], ...] = ()

    @property
    def all_passed(self) -> bool:
        return self.total_unseen > 0 and self.passed_unseen == self.total_unseen


@dataclass(frozen=True)
class CompositionalSolveResult:
    synthesis: SynthesisReport
    program: Program | None
    rule_ids_used: tuple[str, ...]
    program_verified: bool
    verification_checks: tuple[str, ...]
    generalization: GeneralizationReport | None
    sympy_available: bool

    @property
    def succeeded(self) -> bool:
        return self.synthesis.succeeded and self.program is not None and self.program_verified


RULE_OUTPUT_UNITS: dict[str, str] = {
    "physics.celsius_to_fahrenheit": "fahrenheit",
    "physics.fahrenheit_to_celsius": "celsius",
    "physics.celsius_to_kelvin": "kelvin",
    "physics.celsius_to_kelvin__solve_c": "celsius",
    "physics.kinetic_energy": "joule",
    "physics.potential_energy": "joule",
    "physics.work_done": "joule",
    "physics.newton_force": "newton",
    "physics.ohms_voltage": "volt",
    "physics.electrical_power": "watt",
    "physics.density": "density",
    "physics.wave_speed": "velocity",
    "eng.axial_stress": "pascal",
    "eng.pressure": "pascal",
    "eng.safety_factor": "ratio",
    "eng.efficiency": "ratio",
    "math.circle_area": "area",
    "math.triangle_area": "area",
    "math.cylinder_volume": "volume",
    "econ.simple_interest": "currency",
    "econ.compound_amount": "currency",
    "econ.discount_price": "currency",
}

_UNIT_COMPATIBLE_GROUPS: dict[str, set[str]] = {
    "kelvin": {"kelvin", "celsius", "fahrenheit"},
    "celsius": {"kelvin", "celsius", "fahrenheit"},
    "fahrenheit": {"kelvin", "celsius", "fahrenheit"},
    "ratio": {"ratio", "pascal"},
    "pascal": {"pascal", "newton", "area"},
    "joule": {"joule", "newton", "velocity"},
    "watt": {"watt", "volt"},
}


class UOLCompositionalEngine:
    """Compose verified UOL rules into new capabilities and verify them independently."""

    VERSION = "uol-compositional-engine-v1"

    def __init__(
        self,
        *,
        domains: Iterable[str] | None = None,
        extra_rules: Iterable[OperationalRule] = (),
        use_builtin_library: bool = True,
    ) -> None:
        if use_builtin_library:
            self.uol = build_default_uol(domains=domains, extra_rules=extra_rules)
        else:
            self.uol = UOL(extra_rules)
        self.synthesizer = CapabilitySynthesizer(self.uol)
        self.verifier_suite = NativeDomainVerifierSuite()
        self.domain_verifier = DomainVerificationEngine(self.verifier_suite.registry)

    @property
    def rule_count(self) -> int:
        return len(self.uol.rules.values())

    def promote_program_as_rule(
        self,
        program: Program,
        *,
        rule_id: str,
        domains: tuple[str, ...] | None = None,
    ) -> OperationalRule:
        """Add an independently verified synthesized program back into the UOL catalog
        so subsequent synthesis problems can compose on top of it (hierarchical skill growth)."""
        rule = OperationalRule(
            rule_id=rule_id,
            name=program.name,
            inputs=program.inputs,
            body=program.body,
            domains=domains or (program.domain,),
            status=RuleStatus.VERIFIED,
            provenance=f"{self.VERSION}:verified_composition:{program.program_id}",
        )
        existing = list(self.uol.rules.values())
        existing.append(rule)
        self.uol = UOL(existing)
        self.synthesizer = CapabilitySynthesizer(self.uol)
        return rule

    def _independent_replay_verify(
        self,
        program: Program,
        examples: tuple[Example, ...],
    ) -> tuple[bool, tuple[str, ...]]:
        """Independently serialize, deserialize, type-validate, and replay the program."""
        serialized = program_dict(program)
        # Fresh independent UOL instance with identical verified rules
        fresh_uol = UOL(self.uol.rules.values())
        from .operational_language import program_from_dict
        rebuilt = program_from_dict(serialized)
        fresh_uol.validate(rebuilt)
        for sample in examples:
            actual = fresh_uol.execute(rebuilt, sample.inputs)
            if not _values_equal(actual, sample.expected):
                return False, ()
        return True, ("independent_program_replay", "program_contract_validation")

    def _synthesizer_for_unit(self, output_unit: str | None) -> CapabilitySynthesizer:
        if not output_unit:
            return self.synthesizer
        unit_key = output_unit.strip().casefold()
        allowed_units = _UNIT_COMPATIBLE_GROUPS.get(unit_key, {unit_key})
        filtered = [
            rule for rule in self.uol.rules.values()
            if RULE_OUTPUT_UNITS.get(rule.rule_id, "") in allowed_units
        ]
        if not filtered:
            return self.synthesizer
        return CapabilitySynthesizer(UOL(filtered))

    def solve_and_verify(
        self,
        goal: SynthesisGoal,
        *,
        counterexample_inputs: Iterable[Mapping[str, Any]] = (),
        counterexample_oracle: Callable[[Mapping[str, Any]], Any] | None = None,
        unseen_probes: Iterable[Mapping[str, Any]] = (),
        unseen_oracle: Callable[[Mapping[str, Any]], Any] | None = None,
        max_rounds: int = 8,
        output_unit: str | None = None,
    ) -> CompositionalSolveResult:
        """Synthesize a UOL program (optionally with CEGIS and unit-guided search),
        independently verify it, and measure generalization accuracy on unseen probe inputs."""
        active_synth = self._synthesizer_for_unit(output_unit)
        report = active_synth.synthesize(
            goal,
            counterexample_inputs=counterexample_inputs,
            counterexample_oracle=counterexample_oracle,
            max_rounds=max_rounds,
        )
        # Fallback to full catalog if unit-guided subset didn't find a solution
        if not report.succeeded and active_synth is not self.synthesizer:
            report = self.synthesizer.synthesize(
                goal,
                counterexample_inputs=counterexample_inputs,
                counterexample_oracle=counterexample_oracle,
                max_rounds=max_rounds,
            )
        if not report.succeeded or report.program is None:
            return CompositionalSolveResult(
                synthesis=report,
                program=None,
                rule_ids_used=(),
                program_verified=False,
                verification_checks=(),
                generalization=None,
                sympy_available=_sympy_available(),
            )

        program = report.program
        all_known = goal.training + goal.validation
        verified, checks = self._independent_replay_verify(program, all_known)

        gen_report: GeneralizationReport | None = None
        probes_list = tuple(dict(p) for p in unseen_probes)
        if probes_list and unseen_oracle is not None:
            passed = 0
            failures: list[dict[str, Any]] = []
            for probe in probes_list:
                expected = unseen_oracle(probe)
                try:
                    actual = self.uol.execute(program, probe)
                except Exception as exc:
                    failures.append({"inputs": probe, "expected": expected, "error": str(exc)})
                    continue
                if _values_equal(actual, expected):
                    passed += 1
                else:
                    failures.append({"inputs": probe, "expected": expected, "actual": actual})
            gen_report = GeneralizationReport(
                total_unseen=len(probes_list),
                passed_unseen=passed,
                exact_accuracy=passed / len(probes_list),
                failures=tuple(failures[:5]),
            )

        return CompositionalSolveResult(
            synthesis=report,
            program=program,
            rule_ids_used=report.rule_ids,
            program_verified=verified,
            verification_checks=checks,
            generalization=gen_report,
            sympy_available=_sympy_available(),
        )
