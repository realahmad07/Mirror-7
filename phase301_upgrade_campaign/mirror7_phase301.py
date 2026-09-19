from dataclasses import dataclass
from typing import Callable, List, Sequence

@dataclass(frozen=True)
class UpgradeCampaign:
    capability:str
    gap:float
    priority:float
    max_rounds:int=4
    max_candidates:int=8

@dataclass(frozen=True)
class CampaignReport:
    capability:str
    rounds:int
    start_score:float
    end_score:float
    promoted:bool
    stable:bool

class CampaignRunner:
    """Runs a bounded capability-specific improvement campaign through an injected upgrade engine."""
    def run(self,campaign:UpgradeCampaign,initial_score:float,upgrade:Callable[[int],tuple[float,bool]])->CampaignReport:
        if campaign.max_rounds<1 or campaign.max_candidates<1: raise ValueError("bounds must be positive")
        score=float(initial_score); promoted=False
        for n in range(1,campaign.max_rounds+1):
            new_score,changed=upgrade(n)
            if changed and new_score>score:
                score=float(new_score); promoted=True
            else:
                return CampaignReport(campaign.capability,n,float(initial_score),score,promoted,True)
        return CampaignReport(campaign.capability,campaign.max_rounds,float(initial_score),score,promoted,not promoted)
