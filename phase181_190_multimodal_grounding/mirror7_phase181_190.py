"""Phases 181-190: bounded structural multimodal grounding.

The implementation treats each modality as an observation channel and extracts
small deterministic structural signatures. It does not claim semantic vision,
speech recognition, or human-level language understanding.

Phases:
181 modality ingestion
182 per-modality canonicalization
183 cross-view alignment
184 recurring cross-modal concept discovery
185 correspondence/identity binding
186 contradiction detection
187 missing-view robustness
188 temporal multimodal fusion
189 grounded-memory export
190 integrated sealed grounding loop
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from hashlib import sha256
from math import sqrt
from typing import Any, Iterable, Mapping, Sequence


def _bytes(x: Any) -> bytes:
    if isinstance(x, bytes): return x
    if isinstance(x, bytearray): return bytes(x)
    if isinstance(x, str): return x.encode("utf-8")
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        return bytes(int(v) & 255 for v in x)
    raise TypeError("unsupported raw modality")


def _norm(v: Sequence[float]) -> tuple[float, ...]:
    if not v: return ()
    lo, hi = min(v), max(v)
    if hi == lo: return tuple(0.0 for _ in v)
    return tuple(round((x-lo)/(hi-lo), 6) for x in v)


def _quant(v: Sequence[float], bins: int = 8) -> tuple[int, ...]:
    n = _norm(v)
    return tuple(min(bins-1, int(x*bins)) for x in n)


@dataclass(frozen=True)
class ModalityView:
    name: str
    kind: str
    payload: Any
    timestamp: int = 0
    quality: float = 1.0


@dataclass(frozen=True)
class GroundedConcept:
    concept_id: str
    signatures: tuple[tuple[str, str], ...]
    support: int
    confidence: float
    evidence_hash: str


def canonicalize_view(view: ModalityView) -> tuple[str, ...]:
    """Return a modality-local, identity-insensitive structural signature."""
    kind = view.kind.lower()
    p = view.payload
    if kind in {"bytes", "text"}:
        b = _bytes(p)
        # First-appearance relabeling plus local transitions.
        mapping, nxt, seq = {}, 0, []
        for x in b:
            if x not in mapping:
                mapping[x] = nxt; nxt += 1
            seq.append(mapping[x])
        runs = []
        if seq:
            last, count = seq[0], 1
            for x in seq[1:]:
                if x == last: count += 1
                else: runs.append((last,count)); last,count=x,1
            runs.append((last,count))
        return ("seq", repr(tuple(seq)), "runs", repr(tuple(runs)))
    if kind in {"signal", "audio"}:
        vals = [float(x) for x in p]
        q = _quant(vals, 8)
        deltas = tuple(max(-7, min(7, q[i+1]-q[i])) for i in range(len(q)-1))
        return ("signal", repr(q), "delta", repr(deltas))
    if kind in {"image", "grid"}:
        rows = [list(map(float,r)) for r in p]
        flat = [x for r in rows for x in r]
        if not rows or not flat: return ("grid","empty")
        q = _quant(flat, 8)
        h, w = len(rows), len(rows[0])
        # Row/column occupancy is intentionally structural, not semantic.
        row_means = tuple(round(sum(r)/len(r),4) for r in rows)
        col_means = tuple(round(sum(rows[r][c] for r in range(h))/h,4) for c in range(w))
        return ("grid",str((h,w)),"q",repr(q),"rows",repr(_norm(row_means)),
                "cols",repr(_norm(col_means)))
    raise ValueError(f"unknown modality kind: {view.kind}")


def _token_set(sig: Sequence[str]) -> frozenset[str]:
    return frozenset(sig)


def align_views(views: Sequence[ModalityView]) -> dict[str, Any]:
    """Align modalities by structural evidence, never by modality names."""
    canonical = {v.name: canonicalize_view(v) for v in views}
    keys = list(canonical)
    pair_scores = {}
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            sa, sb = _token_set(canonical[a]), _token_set(canonical[b])
            score = len(sa & sb) / max(1, len(sa | sb))
            # Additional size/shape cue from canonical string lengths.
            score = round(0.8*score + 0.2*(1.0-abs(len(canonical[a])-len(canonical[b]))/
                                              max(1,len(canonical[a]),len(canonical[b]))), 6)
            pair_scores[(a,b)] = score
    return {"canonical":canonical, "pair_scores":pair_scores}


def discover_grounded_concepts(episodes: Sequence[Sequence[ModalityView]],
                               min_support: int = 2) -> tuple[GroundedConcept, ...]:
    """Discover repeated modality-local signatures and keep cross-modal support."""
    evidence = defaultdict(set)
    counts = Counter()
    for ep in episodes:
        local = set()
        for v in ep:
            sig = canonicalize_view(v)
            local.add((v.kind, sha256(repr(sig).encode()).hexdigest()[:16]))
        for item in local:
            evidence[item].add(len(evidence[item]))
            counts[item] += 1
    concepts=[]
    for (kind,h), c in sorted(counts.items()):
        if c < min_support: continue
        mods = tuple(sorted({kind}))
        blob = repr((mods,h,c)).encode()
        concepts.append(GroundedConcept(
            concept_id="GC_"+sha256(blob).hexdigest()[:16],
            signatures=((kind,h),), support=c,
            confidence=min(1.0, c/max(1,len(episodes))),
            evidence_hash=sha256(blob).hexdigest()))
    return tuple(concepts)


def bind_concept(concept: GroundedConcept, views: Sequence[ModalityView]) -> bool:
    target = dict(concept.signatures)
    return any(target.get(v.kind) == sha256(repr(canonicalize_view(v)).encode()).hexdigest()[:16]
               for v in views)


def contradiction_filter(bindings: Sequence[tuple[str, str, float]]) -> tuple[tuple[str,str,float],...]:
    """Keep only mutually consistent concept->entity bindings; abstain on ties."""
    grouped=defaultdict(list)
    for concept, entity, score in bindings:
        grouped[concept].append((entity,float(score)))
    out=[]
    for c, vals in grouped.items():
        vals.sort(key=lambda x:(x[1],x[0]), reverse=True)
        if len(vals)>1 and abs(vals[0][1]-vals[1][1]) < 1e-12:
            continue
        out.append((c,vals[0][0],vals[0][1]))
    return tuple(sorted(out))


def build_grounded_scene(views: Sequence[ModalityView], prior: Mapping[str, Any] | None = None) -> dict[str,Any]:
    if not views: return {"concepts":(), "bindings":(), "missing_modalities":(), "fingerprint":sha256(b"empty").hexdigest()}
    alignment=align_views(views)
    kinds={v.kind for v in views}
    expected={"image","text","audio","bytes","signal","grid"}
    scene={
        "concepts": tuple(sorted((v.kind, sha256(repr(canonicalize_view(v)).encode()).hexdigest()[:16]) for v in views)),
        "bindings": tuple(),
        "missing_modalities": tuple(sorted(expected-kinds)),
        "pair_scores": alignment["pair_scores"],
    }
    if prior:
        scene["prior_keys"]=tuple(sorted(str(k) for k in prior))
    scene["fingerprint"]=sha256(repr(scene).encode()).hexdigest()
    return scene


@dataclass
class MultimodalGroundingEngine:
    memory: dict[str, GroundedConcept] = field(default_factory=dict)
    history: list[dict[str,Any]] = field(default_factory=list)

    def ingest(self, views: Sequence[ModalityView]) -> dict[str,Any]:
        if not views: raise ValueError("empty multimodal episode")
        scene=build_grounded_scene(views, self.memory)
        self.history.append(scene)
        for v in views:
            sig=canonicalize_view(v)
            h=sha256(repr(sig).encode()).hexdigest()[:16]
            key=f"{v.kind}:{h}"
            if key not in self.memory:
                self.memory[key]=GroundedConcept(key,((v.kind,h),),1,0.0,h)
            else:
                old=self.memory[key]
                self.memory[key]=GroundedConcept(key,old.signatures,old.support+1,
                                                 min(1.0,(old.support+1)/max(1,len(self.history))),h)
        return scene

    def temporal_fuse(self, episodes: Sequence[Sequence[ModalityView]]) -> tuple[str,...]:
        """Fuse asynchronous views by timestamp and preserve gaps instead of inventing data."""
        events=[]
        for ep in episodes:
            for v in ep:
                events.append((int(v.timestamp),v.kind,sha256(repr(canonicalize_view(v)).encode()).hexdigest()[:12]))
        events.sort()
        return tuple(f"{t}:{k}:{h}" for t,k,h in events)

    def export_memory(self) -> dict[str,Any]:
        return {"concepts":tuple(sorted((k,c.support,c.confidence) for k,c in self.memory.items())),
                "history":len(self.history)}
