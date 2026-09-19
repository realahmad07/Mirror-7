from collections import Counter
from typing import Dict, List, Optional, Tuple

class GroundedLanguageAgent:
    def __init__(self):
        self.word_to_action: Dict[str,Counter] = {}
        self.word_to_delta: Dict[str,Counter] = {}

    def learn_action_mapping(self,instruction:str,action:str):
        for word in instruction.lower().split():
            if word in {"please","the","to"}: continue
            self.word_to_action.setdefault(word,Counter())[action]+=1

    def learn_state_mapping(self,description:str,state_delta:Tuple[float,...]):
        key=tuple(float(x) for x in state_delta)
        for word in description.lower().split():
            if word in {"is","the","a"}: continue
            self.word_to_delta.setdefault(word,Counter())[key]+=1

    def translate_instruction(self,instruction:str)->Optional[str]:
        scores=Counter()
        for word in instruction.lower().split():
            for action,count in self.word_to_action.get(word,{}).items(): scores[action]+=count
        if not scores: return None
        ranked=scores.most_common()
        if len(ranked)>1 and ranked[0][1]==ranked[1][1]: return None
        return ranked[0][0]

    def interpret_state(self,description:str)->Tuple[float,...]:
        out=None
        for word in description.lower().split():
            evidence=self.word_to_delta.get(word)
            if not evidence: continue
            ranked=evidence.most_common()
            if len(ranked)>1 and ranked[0][1]==ranked[1][1]: continue
            delta=ranked[0][0]
            out=list(delta) if out is None else [x+y for x,y in zip(out,delta)]
        return tuple(out) if out is not None else (0.0,)

class AffordanceDiscoverer:
    def __init__(self): self.stats={}
    def observe_affordance(self,state:Tuple[float,...],action:str,success:bool):
        key=(tuple(state),action); ok,fail=self.stats.get(key,(0,0)); self.stats[key]=(ok+int(success),fail+int(not success))
    def get_valid_actions(self,state:Tuple[float,...])->List[str]:
        out=[]
        for (s,a),(ok,fail) in self.stats.items():
            if s==tuple(state) and ok>fail: out.append(a)
        return sorted(out)
