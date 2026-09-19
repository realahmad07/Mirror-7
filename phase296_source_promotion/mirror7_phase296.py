from dataclasses import dataclass
from hashlib import sha256
from typing import Tuple
from phase294_source_evaluator import SourceEvaluation

@dataclass(frozen=True)
class SourceRecord:
    version:int
    fingerprint:str
    source:str
    accepted:bool
    train:float
    held_out:float
    regression_ok:bool
    reason:str

class SourcePromotionRegistry:
    """Maintains the active source variant and an auditable promotion/rollback history."""
    def __init__(self,initial_source:str):
        self._source=initial_source; self._version=0; self._history:Tuple[SourceRecord,...]=()

    @staticmethod
    def fingerprint(source:str)->str:
        return sha256(source.encode("utf-8")).hexdigest()

    @property
    def source(self)->str: return self._source

    @property
    def history(self)->tuple[SourceRecord,...]: return self._history

    def promote(self,source:str,baseline:SourceEvaluation,candidate:SourceEvaluation)->SourceRecord:
        accepted=candidate.train>baseline.train and candidate.held_out>0 and candidate.held_out>=baseline.held_out and candidate.regression_ok
        if not accepted:
            return SourceRecord(self._version,self.fingerprint(self._source),self._source,False,candidate.train,candidate.held_out,candidate.regression_ok,"rejected by source gate")
        self._version+=1; self._source=source
        rec=SourceRecord(self._version,self.fingerprint(source),source,True,candidate.train,candidate.held_out,candidate.regression_ok,"promoted")
        self._history=self._history+(rec,)
        return rec

    def rollback(self,version:int|None=None)->SourceRecord:
        if not self._history:
            return SourceRecord(self._version,self.fingerprint(self._source),self._source,False,0,0,False,"no prior version")
        target=self._history[-1]
        if version is not None:
            matches=[r for r in self._history if r.version==version]
            if not matches:
                return SourceRecord(self._version,self.fingerprint(self._source),self._source,False,0,0,False,"unknown version")
            target=matches[0]
        previous=self._source
        self._source = previous if target.version==self._version else self._history[target.version-1].source
        self._version+=1
        rec=SourceRecord(self._version,self.fingerprint(self._source),self._source,True,0,0,True,"rollback")
        self._history=self._history+(rec,)
        return rec
