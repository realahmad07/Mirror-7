from dataclasses import dataclass

@dataclass(frozen=True)
class SkillBinding:
    concept:str; skill:str; confidence:float

class ConceptSkillBridge:
    def __init__(self): self.bindings={}
    def bind(self,concept,skill,confidence,verified):
        if not verified or confidence<.7: return False
        self.bindings[concept]=SkillBinding(concept,skill,float(confidence)); return True
    def resolve(self,concept): return self.bindings.get(concept)
