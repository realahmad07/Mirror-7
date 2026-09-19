from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence

@dataclass(frozen=True)
class ExperimentPlan:
    sequence: tuple[str, ...]
    score: float
    disagreement: float
    uncertainty: float

class AutonomousExperimentSequencer:
    """Bounded beam search over action sequences using hypothesis disagreement."""
    def __init__(self, beam_width: int = 8, max_depth: int = 4, budget: int = 16):
        if beam_width < 1 or max_depth < 1 or budget < 1:
            raise ValueError("invalid bounds")
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.budget = budget
        self.used = 0

    def choose(self, hypotheses: Sequence[Mapping[str, Mapping[str, float]]], legal_actions: Sequence[str]) -> ExperimentPlan | None:
        actions = tuple(dict.fromkeys(legal_actions))
        if not actions or self.used >= self.budget or not hypotheses:
            return None
        frontier = [(0.0, (), 0.0)]
        best = None
        for depth in range(1, self.max_depth + 1):
            expanded = []
            for _, seq, cost in frontier:
                for action in actions:
                    p = seq + (action,)
                    disagreement = 0.0
                    uncertainty = 0.0
                    for h in hypotheses:
                        values = h.get(action, {})
                        uncertainty += sum(float(v) for v in values.values() if v is not None)
                    for i in range(len(hypotheses)):
                        for j in range(i + 1, len(hypotheses)):
                            ai = hypotheses[i].get(action, {})
                            aj = hypotheses[j].get(action, {})
                            for k in set(ai) | set(aj):
                                disagreement += abs(float(ai.get(k, 0)) - float(aj.get(k, 0)))
                    value = disagreement + 0.1 * uncertainty - 0.01 * depth
                    expanded.append((value, p, cost + 1))
            expanded.sort(key=lambda x: (-x[0], x[1]))
            if expanded:
                best = expanded[0] if best is None or expanded[0][0] > best[0] else best
            frontier = expanded[:self.beam_width]
        if best is None:
            return None
        self.used += 1
        seq = best[1]
        return ExperimentPlan(seq, best[0], max(0.0, best[0]), max(0.0, best[0] * 0.1))

    def reset_budget(self):
        self.used = 0
