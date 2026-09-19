from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

@dataclass(frozen=True)
class CapabilityGap:
    capability: str
    baseline: float
    target: float
    gap: float
    evidence_count: int

@dataclass(frozen=True)
class Hypothesis:
    capability: str
    parameter: str
    value: object
    rationale: str
    ordinal: int

class CapabilityGapDetector:
    """Detect bounded capability deficits from measured evidence without changing state."""
    def detect(self, capability: str, baseline: float, target: float, evidence: Sequence[bool]) -> CapabilityGap | None:
        if not capability:
            raise ValueError("capability must be non-empty")
        if not (0.0 <= float(baseline) <= 1.0):
            raise ValueError("baseline must be in [0,1]")
        if not (0.0 <= float(target) <= 1.0):
            raise ValueError("target must be in [0,1]")
        if not evidence or float(baseline) >= float(target):
            return None
        return CapabilityGap(capability, float(baseline), float(target), float(target) - float(baseline), len(evidence))

class HypothesisGenerator:
    """Generate deterministic, bounded parameter variants for an identified gap."""
    def generate(self, gap: CapabilityGap, parameter_space: Mapping[str, Iterable[object]], *, max_candidates: int = 8) -> List[Hypothesis]:
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if gap.gap <= 0 or not parameter_space:
            return []
        candidates: List[Hypothesis] = []
        ordinal = 0
        for parameter in sorted(parameter_space):
            for value in parameter_space[parameter]:
                ordinal += 1
                candidates.append(Hypothesis(
                    capability=gap.capability,
                    parameter=parameter,
                    value=value,
                    rationale=f"Target gap={gap.gap:.3f}; test bounded change to {parameter}={value!r}.",
                    ordinal=ordinal,
                ))
                if len(candidates) >= max_candidates:
                    return candidates
        return candidates
