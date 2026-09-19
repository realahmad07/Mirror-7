
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

class TransferMemory:
    """Transfers relational action rules across domains by invariant transition signatures."""

    def __init__(self):
        self.rules=defaultdict(list)

    def observe(self, domain:str, state:Mapping[str,float], action:str, next_state:Mapping[str,float]):
        delta=tuple(sorted((k,round(float(next_state[k]-v),8)) for k,v in state.items() if k in next_state))
        signature=tuple(sorted(round(float(v),8) for _,v in delta))
        self.rules[(domain,action)].append((signature,delta))

    def infer_mapping(self, source_domain:str, target_domain:str) -> Dict[str,str]:
        src={}
        tgt={}
        for (domain,action),rows in self.rules.items():
            if domain==source_domain: src[action]=rows
            if domain==target_domain: tgt[action]=rows
        mapping={}
        for s_action,s_rows in src.items():
            s_sig={sig for sig,_ in s_rows}
            for t_action,t_rows in tgt.items():
                t_sig={sig for sig,_ in t_rows}
                if s_sig & t_sig:
                    mapping[s_action]=t_action
                    break
        return mapping

    def transfer_effect(self, source_domain:str, target_domain:str, source_action:str,
                        target_state:Mapping[str,float]) -> Optional[Dict[str,float]]:
        mapping=self.infer_mapping(source_domain,target_domain)
        target_action=mapping.get(source_action)
        if target_action is None:
            return None
        rows=self.rules.get((target_domain,target_action),[])
        if not rows:
            return None
        delta=rows[-1][1]
        return {k:float(target_state[k]+d) for k,d in delta if k in target_state}

    def interference_safe(self, domain:str, state:Mapping[str,float], action:str) -> bool:
        return (domain,action) in self.rules
