from dataclasses import dataclass

@dataclass(frozen=True)
class MonitorReport:
    confidence: float
    calibrated: bool
    ask_context: bool
    reason: str

class SelfMonitor:
    """Tracks confidence against observed correctness and blocks overconfident action."""
    def __init__(self,window=32): self.window=window; self.history=[]
    def observe(self,confidence:float,correct:bool):
        c=max(0.0,min(1.0,float(confidence))); self.history.append((c,bool(correct))); self.history=self.history[-self.window:]
    def report(self,current_confidence:float):
        c=max(0.0,min(1.0,float(current_confidence)))
        if not self.history: return MonitorReport(c,False,c<.6,"insufficient calibration evidence")
        empirical=sum(int(ok) for _,ok in self.history)/len(self.history)
        gap=abs(c-empirical)
        calibrated=gap<=.2
        ask=(c>.8 and empirical<.6) or c<.5
        reason="calibrated" if calibrated else "confidence exceeds recent evidence"
        return MonitorReport(c,calibrated,ask,reason)
