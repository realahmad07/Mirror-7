"""Phases 221-230: bounded language grounding.

This is a symbolic, evidence-driven language layer. It learns word/phrase
associations from paired utterance-to-concept evidence and abstains on
ambiguous or unsupported commands. It is not a language model.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from hashlib import sha256
import re


@dataclass(frozen=True)
class Utterance:
    text:str
    concepts:tuple[str,...]=()


@dataclass(frozen=True)
class GroundedCommand:
    intent:str
    arguments:tuple[str,...]
    confidence:float
    evidence:tuple[str,...]


def _tokens(text:str)->tuple[str,...]:
    return tuple(re.findall(r"[a-z0-9_]+",text.lower()))

@dataclass
class LanguageGrounder:
    lexicon:dict[str,dict[str,float]]=field(default_factory=lambda:defaultdict(dict))
    intents:dict[str,dict[str,float]]=field(default_factory=lambda:defaultdict(dict))

    def learn(self,u:Utterance)->None:
        ts=_tokens(u.text)
        for t in ts:
            for c in u.concepts:
                self.lexicon[t][c]=self.lexicon[t].get(c,0.0)+1.0
        if u.concepts:
            intent=u.concepts[0]
            for t in ts: self.intents[intent][t]=self.intents[intent].get(t,0.0)+1.0

    def ground(self,text:str,allowed:tuple[str,...]|None=None)->GroundedCommand|None:
        ts=_tokens(text)
        scores=defaultdict(float)
        evidence=defaultdict(list)
        for t in ts:
            for c,s in self.lexicon.get(t,{}).items():
                scores[c]+=s; evidence[c].append(t)
        if allowed is not None:
            scores={c:s for c,s in scores.items() if c in allowed}
        if not scores: return None
        ranked=sorted(scores.items(),key=lambda x:(-x[1],x[0]))
        if len(ranked)>1 and ranked[0][1]==ranked[1][1]: return None
        top=ranked[0]
        conf=top[1]/max(1,len(ts))
        return GroundedCommand(top[0],tuple(ts),min(1.0,conf),tuple(sorted(evidence[top[0]])))

    def explain(self,cmd:GroundedCommand)->str:
        return f"{cmd.intent} <- {', '.join(cmd.evidence)}"

    def fingerprint(self,text:str)->str:
        return sha256(repr((_tokens(text),tuple(sorted(self.lexicon)))).encode()).hexdigest()

    def merge(self,other:"LanguageGrounder")->None:
        for t,vals in other.lexicon.items():
            for c,s in vals.items(): self.lexicon[t][c]=self.lexicon[t].get(c,0)+s
        for i,vals in other.intents.items():
            for t,s in vals.items(): self.intents[i][t]=self.intents[i].get(t,0)+s

    def reset(self)->None:
        self.lexicon.clear(); self.intents.clear()
