from dataclasses import dataclass
from hashlib import sha256
from phase294_source_evaluator import SourceEvaluation

@dataclass(frozen=True)
class SourceRecord:
    version:int
    fingerprint:str
    source:str
    previous_source:str
    accepted:bool
    train:float
    held_out:float
    regression_ok:bool
    reason:str

class SourcePromotionRegistry:
    """Maintains active source plus exact rollback provenance."""
    def __init__(self,initial_source:str):
        self._source=initial_source; self._version=0; self._history=()

    @staticmethod
    def fingerprint(source:str)->str:
        return sha256(source.encode("utf-8")).hexdigest()

    @property
    def source(self)->str: return self._source

    @property
    def history(self)->tuple[SourceRecord,...]: return tuple(self._history)

    def promote(self,source:str,baseline:SourceEvaluation,candidate:SourceEvaluation)->SourceRecord:
        accepted=candidate.train>baseline.train and candidate.held_out>0 and candidate.held_out>=baseline.held_out and candidate.regression_ok
        if not accepted:
            return SourceRecord(self._version,self.fingerprint(self._source),self._source,self._source,False,candidate.train,candidate.held_out,candidate.regression_ok,"rejected by source gate")
        previous=self._source
        self._version+=1
        self._source=source
        rec=SourceRecord(self._version,self.fingerprint(source),source,previous,True,candidate.train,candidate.held_out,candidate.regression_ok,"promoted")
        self._history=self._history+(rec,)
        return rec

    def rollback(self,version:int|None=None)->SourceRecord:
        promoted=[r for r in self._history if r.accepted and r.reason=="promoted"]
        if not promoted:
            return SourceRecord(self._version,self.fingerprint(self._source),self._source,self._source,False,0,0,False,"no promoted version")
        if version is None:
            target=promoted[-1]
        else:
            matches=[r for r in promoted if r.version==version]
            if not matches:
                return SourceRecord(self._version,self.fingerprint(self._source),self._source,self._source,False,0,0,False,"unknown version")
            target=matches[0]
        previous=self._source
        self._source=target.previous_source
        self._version+=1
        rec=SourceRecord(self._version,self.fingerprint(self._source),self._source,previous,True,0,0,True,f"rollback before version {target.version}")
        self._history=self._history+(rec,)
        return rec
