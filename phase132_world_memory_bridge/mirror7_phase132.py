class WorldMemoryBridge:
    """Keeps persistent facts aligned with predictive world-model rules."""
    def __init__(self): self.facts={}; self.rules=[]
    def remember_fact(self,key,value,confidence=.8): self.facts[key]=(value,float(confidence))
    def absorb_rule(self,condition,action,effect,confidence=.8): self.rules.append((condition,action,effect,float(confidence)))
    def consistency(self,key,value):
        if key not in self.facts: return None
        old,_=self.facts[key]; return old==value
    def supported_rules(self,min_confidence=.0): return tuple(r for r in self.rules if r[3]>=min_confidence)
