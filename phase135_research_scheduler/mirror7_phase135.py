from dataclasses import dataclass
@dataclass(frozen=True)
class ResearchTask:
    name:str; uncertainty:float; information_gain:float; relevance:float; cost:float
class ResearchScheduler:
    def select(self,tasks,budget):
        eligible=[t for t in tasks if t.cost<=budget and t.cost>0]
        if not eligible: return None
        return max(eligible,key=lambda t:(t.uncertainty*t.information_gain*t.relevance)/t.cost)
