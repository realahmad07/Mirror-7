from typing import Optional

from phase263_raw_representation_discovery.mirror7_phase263 import StructureExtractor
from phase68_counterfactual_planning.mirror7_phase68 import Phase68Planner

class EndToEndAgent:
    """Connects raw structure discovery to evidence-derived planning."""
    def __init__(self):
        self.extractor=StructureExtractor(min_support=1)
        self.raw_history=[]

    def _vocab(self):
        return tuple(sorted(self.extractor.get_concepts(), key=lambda x:(len(x),x)))

    def _vector(self, raw: bytes):
        vocab=self._vocab(); parsed=self.extractor.parse(raw)
        return tuple(parsed.count(c) for c in vocab)

    class _ModelAdapter:
        def __init__(self, owner): self.owner=owner
        def predict_delta(self, action, state):
            matches=[]
            for before_raw,act,after_raw in self.owner.raw_history:
                if act!=action: continue
                before=self.owner._vector(before_raw); after=self.owner._vector(after_raw)
                if before==tuple(state):
                    delta=tuple(float(a-b) for b,a in zip(before,after))
                    if delta not in matches: matches.append(delta)
            if len(matches)==1: return matches[0]
            return None

    def observe(self, raw_state: bytes, action: str, next_raw: bytes):
        if not isinstance(raw_state,(bytes,bytearray)) or not isinstance(next_raw,(bytes,bytearray)): raise TypeError("raw states must be bytes")
        self.extractor.observe(bytes(raw_state)); self.extractor.observe(bytes(next_raw))
        self.raw_history.append((bytes(raw_state),action,bytes(next_raw)))

    def act(self, current_raw: bytes, goal_raw: bytes) -> Optional[str]:
        if not self.raw_history: return None
        current_raw=bytes(current_raw); goal_raw=bytes(goal_raw)
        vocab=self._vocab()
        known_goal=any(c in goal_raw for c in vocab)
        known_current=any(c in current_raw for c in vocab)
        if not known_goal or not known_current:
            return None
        current=self._vector(current_raw); goal=self._vector(goal_raw)
        planner=Phase68Planner(self._ModelAdapter(self),max_depth=6,beam_width=8)
        actions=[]
        for _,action,_ in self.raw_history:
            if action not in actions: actions.append(action)
        plan=planner.plan(current,actions,goal)
        return plan.sequence[0] if plan and plan.sequence and plan.score <= 0.25 else None
