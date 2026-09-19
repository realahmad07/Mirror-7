from dataclasses import dataclass

@dataclass(frozen=True)
class CurriculumItem:
    name: str
    uncertainty: float
    novelty: float
    relevance: float
    cost: int

class AdaptiveCurriculum:
    """Selects the next bounded learning task from uncertainty, novelty, relevance and cost."""
    def __init__(self, budget=10): self.budget=budget
    def select(self, items):
        eligible=[x for x in items if x.cost>0 and x.cost<=self.budget]
        if not eligible: return None
        def score(x): return .45*x.uncertainty+.35*x.novelty+.20*x.relevance
        return max(eligible,key=lambda x:(score(x),-x.cost))