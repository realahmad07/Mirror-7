from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Iterable, Mapping, Sequence

_TOKEN_RE = re.compile(r"https?://\S+|[A-Za-z_][A-Za-z0-9_./:-]*|\d+(?:\.\d+)?|[^\s]+")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_QUOTED_RE = re.compile(r"""(["'])(.+?)\1""")
_STOP = {"a","an","the","to","of","for","in","on","at","with","and","or","is","are","was","were","be","been","being","this","that","it","me","my","you","your","please","can","could","would","should","do","does","did","i","we","they","he","she","them","about","from","as","by","into","than","then","also"}
_OPERATION_GROUPS = {
    "explain": {"explain","describe","define","meaning","what","why","how"},
    "compare": {"compare","comparison","difference","differences","versus","vs"},
    "debug": {"debug","fix","repair","error","broken","failing","failure"},
    "create": {"create","make","build","write","implement","generate","design"},
    "summarize": {"summarize","summary","shorten","condense"},
    "translate": {"translate","translation"},
    "calculate": {"calculate","compute","solve","evaluate"},
    "list": {"list","enumerate","give"},
    "analyze": {"analyze","analyse","analysis","inspect","audit","review"},
    "predict": {"predict","forecast","estimate"},
    "find": {"find","search","locate","look"},
}
_OPERATION_GROUPS_FLAT = {alias for aliases in _OPERATION_GROUPS.values() for alias in aliases}
_OUTPUT_BY_OPERATION = {"explain":"explanation","compare":"comparison","debug":"diagnosis_or_fix","create":"artifact_or_implementation","summarize":"summary","translate":"translation","calculate":"computed_result","list":"enumeration","analyze":"analysis","predict":"prediction","find":"retrieval"}

@dataclass(frozen=True)
class SemanticState:
    entities: tuple[str, ...] = ()
    operations: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    context: tuple[str, ...] = ()
    desired_output: str | None = None
    evidence: tuple[str, ...] = ()
    def as_dict(self) -> dict:
        return {"entities":list(self.entities),"operations":list(self.operations),"constraints":list(self.constraints),"context":list(self.context),"desired_output":self.desired_output,"evidence":list(self.evidence)}

@dataclass
class SchemaStats:
    operation_counts: Counter[str] = field(default_factory=Counter)
    operation_cooccurrence: Counter[tuple[str,str]] = field(default_factory=Counter)
    entity_counts: Counter[str] = field(default_factory=Counter)
    pattern_counts: Counter[str] = field(default_factory=Counter)

class SemanticStateInducer:
    VERSION = 1
    def __init__(self, *, min_support: int = 2):
        if min_support < 1: raise ValueError("min_support must be positive")
        self.min_support = min_support
        self.stats = SchemaStats()
        self._history: list[SemanticState] = []

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return _TOKEN_RE.findall(text)
    @staticmethod
    def _norm(token: str) -> str:
        return token.strip(".,!?;:()[]{}").lower()

    def _operation_candidates(self, tokens: Sequence[str]) -> tuple[str,...]:
        found=[]
        for token in tokens:
            word=self._norm(token)
            for canonical, aliases in _OPERATION_GROUPS.items():
                if word in aliases:
                    found.append(canonical); break
        return tuple(dict.fromkeys(found))

    def _entity_candidates(self, text: str, tokens: Sequence[str]) -> tuple[str,...]:
        entities=[]
        for match in _QUOTED_RE.finditer(text):
            value=match.group(2).strip()
            if value: entities.append(value)
        for token in tokens:
            raw=token.strip(".,!?;:()[]{}")
            if ("://" in raw or "/" in raw or "_" in raw or "." in raw or any(ch.isdigit() for ch in raw)) and len(raw)>1:
                entities.append(raw)
        current=[]
        for token in tokens:
            raw=token.strip(".,!?;:()[]{}")
            if raw and raw.lower() not in _OPERATION_GROUPS_FLAT and raw[0].isupper() and raw[1:].islower(): current.append(raw)
            elif current and raw.isdigit(): current.append(raw)
            else:
                if current: entities.append(" ".join(current)); current=[]
        if current: entities.append(" ".join(current))
        for token in tokens:
            word=self._norm(token)
            if len(word)>=4 and word not in _STOP and word not in {alias for aliases in _OPERATION_GROUPS.values() for alias in aliases} and word.isalpha(): entities.append(word)
        return tuple(dict.fromkeys(entities))

    @staticmethod
    def _constraints(text: str) -> tuple[str,...]:
        lower=text.lower(); out=[]
        if _NUMBER_RE.search(text): out.append("numeric")
        if any(x in lower for x in ("must","required","require","need to")): out.append("requirement")
        if any(x in lower for x in ("under","below","less than","at most")): out.append("upper_bound")
        if any(x in lower for x in ("over","above","more than","at least")): out.append("lower_bound")
        if "?" in text: out.append("question")
        if any(x in lower for x in ("python","code","script","program")): out.append("code")
        if any(x in lower for x in ("json","yaml","csv","table")): out.append("structured_output")
        return tuple(dict.fromkeys(out))

    @staticmethod
    def _context(text: str) -> tuple[str,...]:
        lower=text.lower(); out=[]
        if re.search(r"\b(this|that|it|they|them|he|she|same|previous|above|below)\b", lower): out.append("anaphoric_reference")
        if any(x in lower for x in ("again","continue","also","then","next")): out.append("carryover")
        return tuple(dict.fromkeys(out))

    def discover(self, text: str, *, previous: SemanticState|None=None) -> SemanticState:
        if not isinstance(text,str) or not text.strip(): raise ValueError("text must be a non-empty string")
        tokens=self._tokens(text)
        operations=self._operation_candidates(tokens)
        entities=self._entity_candidates(text,tokens)
        constraints=self._constraints(text)
        context=list(self._context(text))
        if previous is not None and ("anaphoric_reference" in context or "carryover" in context):
            if previous.entities: context.append("inherits_entity")
            if previous.operations: context.append("inherits_operation")
        desired=_OUTPUT_BY_OPERATION.get(operations[0]) if operations else None
        evidence_items=[*(f"operation:{x}" for x in operations), *(f"entity:{x}" for x in entities[:8]), *(f"constraint:{x}" for x in constraints)]
        step_match = re.search(r"\b(?:under|below|less than|at most)\s+(\d+)\s+(?:steps?|actions?)\b", text, re.I)
        if step_match:
            evidence_items.append(f"max_steps:{int(step_match.group(1))}")
        evidence=tuple(dict.fromkeys(evidence_items))
        state=SemanticState(entities,operations,constraints,tuple(dict.fromkeys(context)),desired,evidence)
        self._history.append(state)
        for op in operations: self.stats.operation_counts[op]+=1
        for entity in entities: self.stats.entity_counts[entity]+=1
        pattern="|".join((str(bool(operations)),str(bool(constraints)),str(bool(context)),str(bool(entities))))
        self.stats.pattern_counts[pattern]+=1
        for left,right in zip(operations,operations[1:]): self.stats.operation_cooccurrence[(left,right)]+=1
        return state

    def discover_batch(self, texts: Iterable[str]) -> list[SemanticState]:
        return [self.discover(text) for text in texts]

    def induced_schema(self) -> Mapping[str,tuple[str,...]]:
        return {
            "operations":tuple(sorted(op for op,c in self.stats.operation_counts.items() if c>=self.min_support)),
            "entities":tuple(sorted(e for e,c in self.stats.entity_counts.items() if c>=self.min_support)),
            "patterns":tuple(sorted(p for p,c in self.stats.pattern_counts.items() if c>=self.min_support)),
        }

    def state_signature(self,state:SemanticState)->tuple:
        return (tuple(sorted(state.operations)),tuple(sorted(state.constraints)),state.desired_output,tuple(sorted(state.context)))

    def contrast(self,left:SemanticState,right:SemanticState)->dict[str,tuple[tuple[str,...],tuple[str,...]]]:
        return {"entities":(left.entities,right.entities),"operations":(left.operations,right.operations),"constraints":(left.constraints,right.constraints),"context":(left.context,right.context)}
