"""Phase 59: predictive concept learning.

Learns context-conditioned transitions over the hierarchical concept stream.
The model prefers the longest supported context, tracks ambiguity explicitly,
and abstains when evidence is insufficient instead of forcing a guess.
"""
from __future__ import annotations
from collections import defaultdict, Counter
from dataclasses import dataclass
from hashlib import sha256
from typing import Hashable, Sequence, Tuple

Token=Hashable

@dataclass(frozen=True)
class Prediction:
    value: Token | None
    support: int
    alternatives: int
    context_length: int
    confidence: float

class PredictiveConceptModel:
    def __init__(self,max_order:int=4,min_support:int=2):
        if max_order<1 or min_support<1:
            raise ValueError('invalid model bounds')
        self.max_order=max_order
        self.min_support=min_support
        self._counts: dict[Tuple[Token,...],Counter]=defaultdict(Counter)
        self._seen=0
        self._frozen=False
        self.signature=''

    def fit(self,episodes:Sequence[Sequence[Token]]):
        if not episodes or any(not ep for ep in episodes):
            raise ValueError('episodes cannot be empty')
        self._counts=defaultdict(Counter)
        self._seen=0
        for ep in episodes:
            for i in range(1,len(ep)):
                self._seen+=1
                for order in range(1,min(self.max_order,i)+1):
                    ctx=tuple(ep[i-order:i])
                    self._counts[ctx][ep[i]]+=1
        payload=repr(sorted(
            ((ctx,tuple(sorted(c.items(),key=repr))) for ctx,c in self._counts.items()),
            key=repr,
        )).encode('utf-8')
        self.signature=sha256(payload).hexdigest()
        self._frozen=True
        return self

    def _best(self,context:Sequence[Token]):
        for order in range(min(self.max_order,len(context)),0,-1):
            ctx=tuple(context[-order:])
            counts=self._counts.get(ctx)
            if counts and sum(counts.values())>=self.min_support:
                return ctx,counts
        return None

    def predict(self,context:Sequence[Token],*,min_confidence:float=0.75,require_unique:bool=True)->Prediction:
        if not self._frozen:
            raise RuntimeError('model is not fitted')
        found=self._best(context)
        if not found:
            return Prediction(None,0,0,0,0.0)
        ctx,counts=found
        total=sum(counts.values())
        best=counts.most_common()
        value,support=best[0]
        alternatives=sum(1 for _,n in best if n>0)
        confidence=support/total
        if (require_unique and alternatives>1 and support==best[1][1]) or confidence<min_confidence:
            return Prediction(None,support,alternatives,len(ctx),confidence)
        return Prediction(value,support,alternatives,len(ctx),confidence)

    def predict_sequence(self,prefix:Sequence[Token],steps:int)->Tuple[Token,...]:
        if steps<0:
            raise ValueError('steps cannot be negative')
        context=list(prefix)
        out=[]
        for _ in range(steps):
            p=self.predict(context)
            if p.value is None:
                break
            out.append(p.value)
            context.append(p.value)
        return tuple(out)

    def transition_entropy(self,context:Sequence[Token])->float:
        import math
        found=self._best(context)
        if not found:
            return float('inf')
        _,c=found
        total=sum(c.values())
        return -sum((n/total)*math.log2(n/total) for n in c.values())

    def known_contexts(self)->Tuple[Tuple[Token,...],...]:
        return tuple(sorted(self._counts,key=repr))
