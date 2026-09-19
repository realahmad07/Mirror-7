from dataclasses import dataclass
from typing import Any, Callable

@dataclass(frozen=True)
class Skill:
    name: str
    precondition: Callable[[Any], bool]
    action: Callable[[Any], Any]
    postcondition: Callable[[Any], bool]
    confidence: float

class SkillLibrary:
    """Stores only verified reusable skills and retrieves applicable ones."""
    def __init__(self, max_skills=128):
        self.max_skills=max_skills
        self.skills={}
    def promote(self, skill: Skill, verification_ok: bool):
        if not verification_ok or not (0.0 <= skill.confidence <= 1.0):
            return False
        self.skills[skill.name]=skill
        if len(self.skills)>self.max_skills:
            victim=min(self.skills.values(), key=lambda s:s.confidence)
            del self.skills[victim.name]
        return True
    def applicable(self, state):
        out=[]
        for s in self.skills.values():
            try:
                if s.precondition(state):
                    out.append(s)
            except Exception:
                pass
        return tuple(sorted(out,key=lambda s:s.confidence,reverse=True))
    def execute(self,name,state):
        s=self.skills.get(name)
        if not s: return False,None,"unknown skill"
        try:
            if not s.precondition(state): return False,None,"precondition failed"
            out=s.action(state)
            if not s.postcondition(out): return False,out,"postcondition failed"
            return True,out,"verified"
        except Exception as e:
            return False,None,f"skill error: {type(e).__name__}"