
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

@dataclass(frozen=True)
class Intervention:
    variable: str
    value: float

class CausalInterventionModel:
    """Separates observational association from intervention evidence."""

    def __init__(self):
        self.observations=[]
        self.interventions=defaultdict(list)

    def observe(self, sample: Mapping[str,float]) -> None:
        self.observations.append(dict(sample))

    def intervene(self, do: Mapping[str,float], outcome: Mapping[str,float]) -> None:
        self.interventions[tuple(sorted((k,float(v)) for k,v in do.items()))].append(dict(outcome))

    def effect(self, cause: str, outcome: str, level: float) -> Optional[float]:
        key=tuple(sorted(((cause,float(level)),)))
        rows=self.interventions.get(key,[])
        if not rows:
            return None
        vals=[float(r[outcome]) for r in rows if outcome in r]
        if not vals:
            return None
        return sum(vals)/len(vals)

    def direct_effect_supported(self, cause: str, outcome: str, level_a: float, level_b: float,
                                tolerance: float=1e-9) -> bool:
        ea=self.effect(cause,outcome,level_a)
        eb=self.effect(cause,outcome,level_b)
        if ea is None or eb is None:
            return False
        return abs(eb-ea)>tolerance

    def invariant_across_contexts(self, cause: str, outcome: str,
                                  levels: Iterable[float], tolerance: float=1e-9) -> bool:
        effects=[self.effect(cause,outcome,v) for v in levels]
        return all(e is not None for e in effects) and max(effects)-min(effects)<=tolerance
