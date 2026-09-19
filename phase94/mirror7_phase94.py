from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

from phase88 import TemporalAbstraction
from phase89 import HiddenStateInferer
from phase90 import OverlappingDelayedEffects
from phase91 import HypothesisRevision
from phase92 import AutonomousExperimentSequencer
from phase93 import StructuralTransfer

@dataclass(frozen=True)
class IntegratedSnapshot:
    event_count: int
    latent_states: int
    learned_effects: int
    hypotheses: int
    selected_experiment: tuple[str, ...] | None

class OpenEndedResearchLoop:
    """Phase 94: bounded integration of phases 88-93."""
    def __init__(self):
        self.temporal = TemporalAbstraction()
        self.hidden = HiddenStateInferer()
        self.effects = OverlappingDelayedEffects()
        self.revisions = HypothesisRevision()
        self.experiments = AutonomousExperimentSequencer()
        self.transfer = StructuralTransfer()

    def observe_stream(self, raw: Sequence[float], partial: Sequence[float | None]) -> None:
        self.temporal.discover(raw)
        self.hidden.ingest(partial)

    def learn_effect(self, action: str, lag: int, dimension: int, delta: float, repeats: int = 3) -> None:
        if repeats < 1:
            raise ValueError("repeats must be positive")
        for _ in range(repeats):
            self.effects.learn(action, lag, dimension, delta)

    def revise_hypothesis(self) -> int | None:
        sig = {f"{a}:{lag}:{d}": v for (a, lag, d), v in self.effects.signature().items()}
        if not sig:
            return None
        return self.revisions.observe(sig).hypothesis_id

    def choose_experiment(self, legal: Sequence[str]) -> tuple[str, ...] | None:
        hyps = []
        for h in self.revisions.hypotheses:
            mapping = {}
            for key, value in h.signature:
                action = key.split(":", 1)[0]
                mapping.setdefault(action, {})[key] = value
            hyps.append(mapping)
        plan = self.experiments.choose(hyps, legal)
        return None if plan is None else plan.sequence

    def transfer_signature(self, source: Sequence[float], target: Sequence[float]):
        return self.transfer.match(source, target)

    def snapshot(self, legal: Sequence[str]) -> IntegratedSnapshot:
        return IntegratedSnapshot(
            len(self.temporal.events),
            len(self.hidden.states),
            len(self.effects.signature()),
            len(self.revisions.hypotheses),
            self.choose_experiment(legal),
        )
