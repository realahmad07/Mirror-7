"""Organ 2: Quotient E-Graph Resonator & Twin-E-Graph World Engine
================================================================
Unifies Stages 2, 5A, 7, 11 (algebraic core), and 13-L4 into a single pure-CPU,
zero-float-matmul symbolic-hyperdimensional engine:

1. D = 10,240-bit (160 blocks x 64-bit words) Sparse Block Code (SBC) / BSC Algebra:
   - Deterministic blake2b seeding, block-cyclic binding, bit-sliced vertical bundling.
2. 0-Drift Fixed-Point Lee-Arc Spatial Semantic Pointers (LeeArcSSP — Stage 5A):
   - Width-W=8 circular arcs inside 160x64-bit blocks with exact modular wavevectors
     k_{b,j} in Z_64 (ZERO floor-carry drift over arbitrary translation steps).
3. K-Hop 1-WL Relational Role Analogical Mapper (Stage 5A):
   - Weisfeiler-Leman structural isomorphism across completely disjoint entity vocabularies
     (e.g., Solar System -> Rutherford Atom) with automatic relation transfer.
4. Deferred-Rebuild E-Graph Equality Saturation + QE-HDC Projector (Stage 2):
   - Union-find e-classes, hashcons congruence closure, declarative rewrite rules,
     and Quotient E-Graph Hypervector invariance: equivalent expressions map to the
     exact same 10,240-bit vector with Hamming distance d_H == 0.
5. Bit-Sliced Stochastic Resonator with uint8 Multiset Histogram Deflation (Stages 2 & 6):
   - Iterative unbinding factorization + exact uint8 histogram deflation
     A_{t+1}[i] = max(0, A_t[i] - E*[i]) to peel composites cleanly from superpositions.
6. Poole-Specificity Tropical JTMS & Copy-on-Write Twin-E-Graph do-Calculus (Stages 7 & 13-L4):
   - Defeasible reasoning with specificity defeat (penguin defeats bird->flies).
   - Pearl do(X=x) intervention on Copy-on-Write world forks.
   - Dual-Mode World Gate: literal M_truth (factual Q&A) vs fictional M_canon (narrative).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# 1. Unified D = 10,240-bit (160 x 64-bit) Hypervector & Block Algebra
# ---------------------------------------------------------------------------

D_UNIFIED: int = 10_240
NUM_BLOCKS: int = 160
BLOCK_WIDTH: int = 64
BLOCK_MASK: int = (1 << BLOCK_WIDTH) - 1
UNIFIED_MASK: int = (1 << D_UNIFIED) - 1
UNIFIED_ODD_MASK: int = int("AA" * (D_UNIFIED // 8), 16)

_HV10240_CACHE: dict[bytes, int] = {}


def hv10240_of(key: bytes | str) -> int:
    """Deterministic pseudo-random 10,240-bit hypervector via chained blake2b."""
    raw_key = key.encode("utf-8") if isinstance(key, str) else bytes(key)
    cached = _HV10240_CACHE.get(raw_key)
    if cached is not None:
        return cached
    chunks: list[bytes] = []
    # 10,240 bits = 1,280 bytes = 20 x 64-byte blake2b digests
    for idx in range(20):
        chunks.append(hashlib.blake2b(raw_key + bytes([idx]), digest_size=64).digest())
    val = int.from_bytes(b"".join(chunks), "little") & UNIFIED_MASK
    _HV10240_CACHE[raw_key] = val
    return val


def rot64(word: int, shift: int) -> int:
    """Exact circular bit-rotation of a 64-bit unsigned integer mod 64 (0 drift)."""
    s = shift & 63
    w = word & BLOCK_MASK
    if s == 0:
        return w
    return ((w << s) | (w >> (64 - s))) & BLOCK_MASK


def permute10240(hv: int, steps: int = 1) -> int:
    """Cyclic bit permutation of a 10,240-bit hypervector."""
    s = steps % D_UNIFIED
    if s == 0:
        return hv & UNIFIED_MASK
    return ((hv << s) | (hv >> (D_UNIFIED - s))) & UNIFIED_MASK


def bind10240(a: int, b: int) -> int:
    """Self-inverse XOR binding in 10,240-bit BSC space."""
    return (a ^ b) & UNIFIED_MASK


def hamming_distance10240(a: int, b: int) -> int:
    """Exact integer Hamming distance in [0, 10240]."""
    return ((a ^ b) & UNIFIED_MASK).bit_count()


def hamming_sim10240(a: int, b: int) -> int:
    """Exact signed integer similarity in [-10240, +10240]."""
    return D_UNIFIED - 2 * hamming_distance10240(a, b)


def sim_q16_10240(a: int, b: int) -> int:
    """Signed Q16.16 fixed-point cosine-equivalent similarity in [-65536, +65536]."""
    return (hamming_sim10240(a, b) * 65536) // D_UNIFIED


class BitSlicedBundle10240:
    """Bit-parallel vertical ripple-carry counter across all 10,240 bits simultaneously."""

    def __init__(self, bits: int = 12):
        self.K = max(4, int(bits))
        self.planes: list[int] = [0] * self.K
        self.total_weight: int = 0

    def add(self, hv: int, weight: int = 1) -> None:
        if weight <= 0:
            return
        v = hv & UNIFIED_MASK
        self.total_weight += weight
        planes = self.planes
        K = self.K
        bit_pos = 0
        w = weight
        while w:
            if w & 1:
                carry = v
                for k in range(bit_pos, K):
                    new_carry = planes[k] & carry
                    planes[k] ^= carry
                    carry = new_carry
                    if not carry:
                        break
            w >>= 1
            bit_pos += 1

    def binary(self) -> int:
        """Majority vote across all 10,240 bits in K bitwise passes (0 floats)."""
        if self.total_weight == 0:
            return 0
        thresh = (self.total_weight // 2) + 1
        gt = 0
        eq = UNIFIED_MASK
        planes = self.planes
        for k in range(self.K - 1, -1, -1):
            pk = planes[k]
            if (thresh >> k) & 1:
                eq &= pk
            else:
                gt |= eq & pk
        if self.total_weight % 2 == 0:
            gt |= eq & UNIFIED_ODD_MASK
        return gt & UNIFIED_MASK


def bundle10240(vectors: Sequence[int], weights: Sequence[int] | None = None) -> int:
    bd = BitSlicedBundle10240()
    if weights is None:
        for v in vectors:
            bd.add(v, 1)
    else:
        for v, w in zip(vectors, weights):
            bd.add(v, w)
    return bd.binary()


# ---------------------------------------------------------------------------
# 2. 0-Drift Fixed-Point Lee-Arc Spatial Semantic Pointers (Stage 5A)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseBlockVector:
    """160 blocks x 64-bit words = 10,240 bits with exact per-block circular shifts."""

    blocks: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.blocks) != NUM_BLOCKS:
            raise ValueError(f"Expected {NUM_BLOCKS} blocks, got {len(self.blocks)}")

    def to_int(self) -> int:
        out = 0
        for idx, word in enumerate(self.blocks):
            out |= (word & BLOCK_MASK) << (idx * BLOCK_WIDTH)
        return out

    @classmethod
    def from_int(cls, val: int) -> SparseBlockVector:
        v = val & UNIFIED_MASK
        return cls(tuple((v >> (idx * BLOCK_WIDTH)) & BLOCK_MASK for idx in range(NUM_BLOCKS)))

    def shift_by_wavevector(self, wavevector: Sequence[int], step: int = 1) -> SparseBlockVector:
        """Exact modular rotation per block: zero carry drift for any integer step."""
        return SparseBlockVector(
            tuple(rot64(w, (k * step) & 63) for w, k in zip(self.blocks, wavevector))
        )

    def xor_bind(self, other: SparseBlockVector) -> SparseBlockVector:
        return SparseBlockVector(tuple((a ^ b) & BLOCK_MASK for a, b in zip(self.blocks, other.blocks)))

    def and_overlap(self, other: SparseBlockVector) -> int:
        """Total bit intersection count across all 160 blocks."""
        return sum((a & b).bit_count() for a, b in zip(self.blocks, other.blocks))

    def hamming_distance(self, other: SparseBlockVector) -> int:
        return sum((a ^ b).bit_count() for a, b in zip(self.blocks, other.blocks))

    def active_bits(self) -> int:
        return sum(w.bit_count() for w in self.blocks)


class LeeArcSSP:
    """0-Drift Fixed-Point Lee-Arc Spatial Semantic Pointers for 2D/3D Intuitive Physics.

    Each of the 160 64-bit blocks holds a contiguous circular arc of width W=8 bits
    (12.5% density -> 1,280 active bits in 10,240-bit space). Translation along axis j
    by integer delta rotates block b by (k_{b,j} * delta) mod 64 using exact modular
    integer arithmetic, so:
        translate(translate(p, dx1, dy1), dx2, dy2) == translate(p, dx1+dx2, dy1+dy2)
    with EXACT Hamming distance 0 even after 100,000 sequential Newtonian steps!
    """

    ARC_WIDTH: int = 8
    BASE_ARC: int = (1 << 8) - 1  # 0xFF

    def __init__(self, axes: int = 2, seed: str = "mirror7_lee_arc_ssp_v1"):
        self.axes = max(1, int(axes))
        self.seed = seed
        # Deterministic odd/coprime-heavy modular wavevectors k_{b, j} in Z_64
        self.wavevectors: list[tuple[int, ...]] = []
        self.origin_phases: tuple[int, ...] = self._gen_phases(f"{seed}:origin")
        for axis in range(self.axes):
            self.wavevectors.append(self._gen_wavevector(f"{seed}:axis:{axis}"))
        self.origin: SparseBlockVector = SparseBlockVector(
            tuple(rot64(self.BASE_ARC, ph) for ph in self.origin_phases)
        )

    @staticmethod
    def _gen_phases(tag: str) -> tuple[int, ...]:
        buf = b""
        idx = 0
        while len(buf) < NUM_BLOCKS:
            buf += hashlib.blake2b(f"{tag}:{idx}".encode("utf-8"), digest_size=64).digest()
            idx += 1
        return tuple(buf[b] & 63 for b in range(NUM_BLOCKS))

    @staticmethod
    def _gen_wavevector(tag: str) -> tuple[int, ...]:
        buf = b""
        idx = 0
        while len(buf) < NUM_BLOCKS:
            buf += hashlib.blake2b(f"{tag}:{idx}".encode("utf-8"), digest_size=64).digest()
            idx += 1
        # Choose non-zero wave frequencies in {1..63}, preferring odd numbers (coprime to 64)
        # for maximal spatial resolution across Z_64 without aliasing collapse.
        freqs: list[int] = []
        for b in range(NUM_BLOCKS):
            raw = (buf[b] % 31) * 2 + 1  # odd in {1, 3, ..., 61}
            if b % 5 == 0:
                raw = ((buf[b] % 15) + 1) * 2  # mix some even frequencies for multiscale locality
            freqs.append(raw & 63)
        return tuple(freqs)

    def encode_point(self, coords: Sequence[int]) -> SparseBlockVector:
        """Encode discrete fixed-point coordinates (x, y, ...) into a Lee-Arc SSP."""
        if len(coords) != self.axes:
            raise ValueError(f"Expected {self.axes} coordinates, got {len(coords)}")
        blocks = list(self.origin.blocks)
        for axis, c in enumerate(coords):
            wv = self.wavevectors[axis]
            step = int(c)
            blocks = [rot64(w, (k * step) & 63) for w, k in zip(blocks, wv)]
        return SparseBlockVector(tuple(blocks))

    def translate(self, vec: SparseBlockVector, deltas: Sequence[int]) -> SparseBlockVector:
        """Exact 0-drift Newtonian translation by integer deltas."""
        if len(deltas) != self.axes:
            raise ValueError(f"Expected {self.axes} deltas, got {len(deltas)}")
        blocks = list(vec.blocks)
        for axis, d in enumerate(deltas):
            wv = self.wavevectors[axis]
            step = int(d)
            blocks = [rot64(w, (k * step) & 63) for w, k in zip(blocks, wv)]
        return SparseBlockVector(tuple(blocks))

    def overlap_q16(self, a: SparseBlockVector, b: SparseBlockVector) -> int:
        """Normalized Q16.16 spatial kernel overlap in [0, 65536]."""
        max_bits = NUM_BLOCKS * self.ARC_WIDTH  # 1,280
        return (a.and_overlap(b) * 65536) // max_bits

    def bind_object_at(self, object_id: str, coords: Sequence[int]) -> int:
        """Bind an object's 10,240-bit identity HV with its Lee-Arc spatial pointer."""
        obj_hv = hv10240_of(f"ssp_obj:{object_id}")
        pos_hv = self.encode_point(coords).to_int()
        return bind10240(obj_hv, pos_hv)

    def simulate_ballistic_trajectory(
        self,
        pos0: tuple[int, int],
        vel: tuple[int, int],
        gravity: tuple[int, int],
        steps: int,
    ) -> list[tuple[tuple[int, int], SparseBlockVector]]:
        """Run exact integer Verlet/Euler ballistic steps with 0-drift Lee-Arc pointers."""
        if self.axes != 2:
            raise ValueError("simulate_ballistic_trajectory requires axes=2")
        x, y = int(pos0[0]), int(pos0[1])
        vx, vy = int(vel[0]), int(vel[1])
        gx, gy = int(gravity[0]), int(gravity[1])
        vec = self.encode_point((x, y))
        trace: list[tuple[tuple[int, int], SparseBlockVector]] = [((x, y), vec)]
        for _ in range(max(0, int(steps))):
            vx += gx
            vy += gy
            x += vx
            y += vy
            vec = self.translate(vec, (vx, vy))
            trace.append(((x, y), vec))
        return trace

    def check_collision(
        self,
        a_pos: SparseBlockVector,
        b_pos: SparseBlockVector,
        threshold_q16: int = 45_000,
    ) -> tuple[bool, int]:
        """Return (collides, overlap_q16) based on Lee-Arc kernel overlap."""
        ov = self.overlap_q16(a_pos, b_pos)
        return (ov >= threshold_q16, ov)


# ---------------------------------------------------------------------------
# 3. K-Hop 1-WL Relational Role Analogical Mapper (Stage 5A)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalogicalTransferResult:
    source_domain: str
    target_domain: str
    entity_mapping: dict[str, str]
    transferred_triples: list[tuple[str, str, str]]
    structural_match_q16: int
    explanation: str


class WeisfeilerLemanAnalogicalMapper:
    """K-Hop 1-WL Structural Graph Alignment for Cross-Domain Analogical Transfer.

    Computes topology-only 1-WL color refinements over directed labeled graphs
    (ignoring surface entity names!), aligns structurally equivalent nodes between
    a base domain and a target domain, and transfers unstated relational inferences.
    """

    def __init__(self, hops: int = 3):
        self.hops = max(1, int(hops))

    @staticmethod
    def _entities_of(triples: Sequence[tuple[str, str, str]]) -> list[str]:
        seen: list[str] = []
        s_set: set[str] = set()
        for s, _r, o in triples:
            if s not in s_set:
                s_set.add(s)
                seen.append(s)
            if o not in s_set:
                s_set.add(o)
                seen.append(o)
        return seen

    def compute_wl_signatures(
        self,
        triples: Sequence[tuple[str, str, str]],
        *,
        use_relation_names: bool = True,
    ) -> dict[str, tuple[int, dict[str, int]]]:
        """Return per-entity (10240-bit WL color HV, structural role counts) purely from topology."""
        entities = self._entities_of(triples)
        out_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
        in_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for s, r, o in triples:
            rel_tag = r.casefold() if use_relation_names else "REL"
            out_edges[s].append((rel_tag, o))
            in_edges[o].append((rel_tag, s))

        # Initial structural color: (out_degree, in_degree, sorted relation profile)
        colors: dict[str, str] = {}
        role_features: dict[str, dict[str, int]] = {}
        for e in entities:
            out_rels = sorted(r for r, _ in out_edges[e])
            in_rels = sorted(r for r, _ in in_edges[e])
            feats: dict[str, int] = {
                "out_deg": len(out_rels),
                "in_deg": len(in_rels),
            }
            for r in out_rels:
                feats[f"out:{r}"] = feats.get(f"out:{r}", 0) + 1
            for r in in_rels:
                feats[f"in:{r}"] = feats.get(f"in:{r}", 0) + 1
            role_features[e] = feats
            colors[e] = f"d0:{len(out_rels)}:{len(in_rels)}:{','.join(out_rels)}|{','.join(in_rels)}"

        # K-hop 1-WL color refinement
        for step in range(1, self.hops + 1):
            next_colors: dict[str, str] = {}
            for e in entities:
                out_sig = sorted(f"{r}->{colors[nbr]}" for r, nbr in out_edges[e])
                in_sig = sorted(f"{r}<-{colors[nbr]}" for r, nbr in in_edges[e])
                raw = f"h{step}:{colors[e]}|O:{';'.join(out_sig)}|I:{';'.join(in_sig)}"
                digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()
                next_colors[e] = digest
            colors = next_colors

        # Convert final 1-WL color + structural role counts into 10,240-bit hypervectors
        result: dict[str, tuple[int, dict[str, int]]] = {}
        for e in entities:
            bd = BitSlicedBundle10240()
            bd.add(hv10240_of(f"wl_color:{colors[e]}"), 4)
            for fk, fv in sorted(role_features[e].items()):
                bd.add(hv10240_of(f"wl_feat:{fk}={fv}"), 2)
            result[e] = (bd.binary(), role_features[e])
        return result

    def align_and_transfer(
        self,
        source_domain: str,
        source_triples: Sequence[tuple[str, str, str]],
        target_domain: str,
        target_triples: Sequence[tuple[str, str, str]],
        *,
        known_anchor_map: Mapping[str, str] | None = None,
    ) -> AnalogicalTransferResult:
        """Align source entities -> target entities by 1-WL structure & transfer missing relations."""
        src_entities = self._entities_of(source_triples)
        tgt_entities = self._entities_of(target_triples)
        if not src_entities or not tgt_entities:
            return AnalogicalTransferResult(
                source_domain, target_domain, {}, [], 0, "Empty source or target relational graph."
            )

        # Build induced structural signatures on the shared relation skeleton (or full topology)
        src_rels = {r.casefold() for _, r, _ in source_triples}
        tgt_rels = {r.casefold() for _, r, _ in target_triples}
        shared_rels = src_rels & tgt_rels
        use_names = bool(shared_rels)

        # Restrict source to shared relations for alignment if target only has a subset of relations
        align_src_triples = (
            [t for t in source_triples if t[1].casefold() in shared_rels]
            if shared_rels and len(shared_rels) < len(src_rels)
            else list(source_triples)
        )
        src_sigs = self.compute_wl_signatures(align_src_triples or source_triples, use_relation_names=use_names)
        tgt_sigs = self.compute_wl_signatures(target_triples, use_relation_names=use_names)

        mapping: dict[str, str] = {}
        used_targets: set[str] = set()
        if known_anchor_map:
            for s, t in known_anchor_map.items():
                if s in src_entities and t in tgt_entities:
                    mapping[s] = t
                    used_targets.add(t)

        # Greedy highest-similarity bijective matching using integer 10,240-bit WL similarity + role overlap
        scored_pairs: list[tuple[int, str, str]] = []
        for s in src_entities:
            if s in mapping:
                continue
            s_hv, s_feats = src_sigs.get(s, (0, {}))
            for t in tgt_entities:
                if t in used_targets:
                    continue
                t_hv, t_feats = tgt_sigs.get(t, (0, {}))
                hv_score = sim_q16_10240(s_hv, t_hv) if (s_hv and t_hv) else 0
                feat_matches = sum(
                    2000 for fk, fv in s_feats.items() if t_feats.get(fk) == fv
                )
                deg_penalty = 1500 * (
                    abs(s_feats.get("out_deg", 0) - t_feats.get("out_deg", 0))
                    + abs(s_feats.get("in_deg", 0) - t_feats.get("in_deg", 0))
                )
                scored_pairs.append((hv_score + feat_matches - deg_penalty, s, t))

        scored_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        matched_scores: list[int] = []
        for score, s, t in scored_pairs:
            if s not in mapping and t not in used_targets:
                mapping[s] = t
                used_targets.add(t)
                matched_scores.append(max(0, min(65536, score)))

        existing_tgt = {(s.casefold(), r.casefold(), o.casefold()) for s, r, o in target_triples}
        transferred: list[tuple[str, str, str]] = []
        for s, r, o in source_triples:
            if s in mapping and o in mapping:
                cand = (mapping[s], r, mapping[o])
                key = (cand[0].casefold(), cand[1].casefold(), cand[2].casefold())
                if key not in existing_tgt:
                    transferred.append(cand)
                    existing_tgt.add(key)

        avg_q16 = sum(matched_scores) // max(1, len(matched_scores)) if matched_scores else 65536
        map_str = ", ".join(f"{k}->{v}" for k, v in sorted(mapping.items()))
        return AnalogicalTransferResult(
            source_domain=source_domain,
            target_domain=target_domain,
            entity_mapping=mapping,
            transferred_triples=transferred,
            structural_match_q16=avg_q16,
            explanation=(
                f"1-WL analogical transfer ({source_domain} -> {target_domain}): "
                f"mapping [{map_str}] transferred {len(transferred)} unseen relation(s)."
            ),
        )


# ---------------------------------------------------------------------------
# 4. Deferred-Rebuild E-Graph Equality Saturation + QE-HDC (Stage 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ENode:
    op: str
    children: tuple[int, ...] = ()


@dataclass
class EClass:
    class_id: int
    nodes: set[ENode] = field(default_factory=set)
    parents: set[tuple[ENode, int]] = field(default_factory=set)


@dataclass(frozen=True)
class RewriteRule:
    """Declarative S-expression rewrite rule: pattern -> replacement.

    Variables start with '?' (e.g., ('add', '?x', '?y') -> ('add', '?y', '?x')).
    """

    name: str
    lhs: Any
    rhs: Any


class QuotientEGraph:
    """Deferred-Rebuild E-Graph (egg architecture) with QE-HDC projection.

    Satisfies the Quotient E-Graph Hypervector (QE-HDC) invariant:
        e1 ==_E e2  ==>  project_eclass_hv(find(e1)) == project_eclass_hv(find(e2))
    so algebraically or semantically equivalent AST/logic/NL expressions produce
    bit-identical 10,240-bit hypervectors (Hamming distance d_H == 0).
    """

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._classes: dict[int, EClass] = {}
        self._hashcons: dict[ENode, int] = {}
        self._worklist: list[int] = []
        self._next_id: int = 0
        self._qehdc_cache: dict[int, int] = {}

    def find(self, eclass_id: int) -> int:
        cid = eclass_id
        while self._parent[cid] != cid:
            self._parent[cid] = self._parent[self._parent[cid]]
            cid = self._parent[cid]
        return cid

    def _canonicalize(self, node: ENode) -> ENode:
        return ENode(node.op, tuple(self.find(c) for c in node.children))

    def add_node(self, node: ENode) -> int:
        canon = self._canonicalize(node)
        existing = self._hashcons.get(canon)
        if existing is not None:
            return self.find(existing)
        cid = self._next_id
        self._next_id += 1
        self._parent[cid] = cid
        ec = EClass(class_id=cid, nodes={canon})
        self._classes[cid] = ec
        self._hashcons[canon] = cid
        for ch in canon.children:
            self._classes[self.find(ch)].parents.add((canon, cid))
        self._qehdc_cache.clear()
        return cid

    def add_expr(self, expr: Any) -> int:
        """Intern a nested S-expression tuple e.g. ('add', 'x', ('mul', 'y', '1'))."""
        if isinstance(expr, tuple) and expr:
            op = str(expr[0])
            child_ids = tuple(self.add_expr(arg) for arg in expr[1:])
            return self.add_node(ENode(op, child_ids))
        return self.add_node(ENode(str(expr), ()))

    def union(self, id1: int, id2: int) -> int:
        r1, r2 = self.find(id1), self.find(id2)
        if r1 == r2:
            return r1
        # Union by smaller ID for deterministic canonical representatives
        if r1 > r2:
            r1, r2 = r2, r1
        self._parent[r2] = r1
        c1, c2 = self._classes[r1], self._classes[r2]
        c1.nodes.update(c2.nodes)
        c1.parents.update(c2.parents)
        del self._classes[r2]
        self._worklist.append(r1)
        self._qehdc_cache.clear()
        return r1

    def rebuild(self) -> int:
        """Deferred congruence-closure rebuild loop until fixpoint."""
        merges = 0
        while self._worklist:
            todo = sorted({self.find(cid) for cid in self._worklist})
            self._worklist.clear()
            for cid in todo:
                if cid not in self._classes:
                    continue
                ec = self._classes[cid]
                new_parents: dict[ENode, int] = {}
                for pnode, pclass in list(ec.parents):
                    old_key = pnode
                    self._hashcons.pop(old_key, None)
                    canon_p = self._canonicalize(pnode)
                    canon_pc = self.find(pclass)
                    if canon_p in new_parents:
                        prev_pc = new_parents[canon_p]
                        if self.find(prev_pc) != canon_pc:
                            self.union(prev_pc, canon_pc)
                            merges += 1
                    existing_hc = self._hashcons.get(canon_p)
                    if existing_hc is not None and self.find(existing_hc) != self.find(canon_pc):
                        self.union(existing_hc, canon_pc)
                        merges += 1
                    root_pc = self.find(canon_pc)
                    self._hashcons[canon_p] = root_pc
                    new_parents[canon_p] = root_pc
                    if root_pc in self._classes:
                        self._classes[root_pc].nodes.add(canon_p)
                if cid in self._classes:
                    self._classes[cid].parents = {(pn, self.find(pc)) for pn, pc in new_parents.items()}
                    self._classes[cid].nodes = {self._canonicalize(n) for n in self._classes[cid].nodes}
        self._qehdc_cache.clear()
        return merges

    def _ematch_pattern(self, pattern: Any, eclass_id: int, subst: dict[str, int]) -> list[dict[str, int]]:
        cid = self.find(eclass_id)
        if isinstance(pattern, str) and pattern.startswith("?"):
            bound = subst.get(pattern)
            if bound is None:
                ns = dict(subst)
                ns[pattern] = cid
                return [ns]
            return [subst] if self.find(bound) == cid else []
        if isinstance(pattern, tuple) and pattern:
            op = str(pattern[0])
            arg_pats = pattern[1:]
        else:
            op = str(pattern)
            arg_pats = ()
        results: list[dict[str, int]] = []
        ec = self._classes.get(cid)
        if ec is None:
            return results
        for node in list(ec.nodes):
            if node.op != op or len(node.children) != len(arg_pats):
                continue
            cur_substs = [subst]
            for pat_arg, child_id in zip(arg_pats, node.children):
                next_substs: list[dict[str, int]] = []
                for st in cur_substs:
                    next_substs.extend(self._ematch_pattern(pat_arg, child_id, st))
                cur_substs = next_substs
                if not cur_substs:
                    break
            results.extend(cur_substs)
        return results

    def _instantiate(self, pattern: Any, subst: Mapping[str, int]) -> int:
        if isinstance(pattern, str) and pattern.startswith("?"):
            return self.find(subst[pattern])
        if isinstance(pattern, tuple) and pattern:
            op = str(pattern[0])
            child_ids = tuple(self._instantiate(arg, subst) for arg in pattern[1:])
            return self.add_node(ENode(op, child_ids))
        return self.add_node(ENode(str(pattern), ()))

    def saturate(
        self,
        rules: Sequence[RewriteRule],
        *,
        max_iters: int = 8,
        max_nodes: int = 2000,
    ) -> dict[str, int]:
        """Run equality saturation until fixpoint or budget cap."""
        iters = 0
        total_applications = 0
        for it in range(1, max_iters + 1):
            iters = it
            matches: list[tuple[int, RewriteRule, dict[str, int]]] = []
            for cid in list(self._classes.keys()):
                for rule in rules:
                    for subst in self._ematch_pattern(rule.lhs, cid, {}):
                        matches.append((cid, rule, subst))
            changed = False
            for cid, rule, subst in matches:
                if len(self._hashcons) >= max_nodes:
                    break
                before_nodes = len(self._hashcons)
                rhs_id = self._instantiate(rule.rhs, subst)
                if self.find(cid) != self.find(rhs_id):
                    self.union(cid, rhs_id)
                    changed = True
                    total_applications += 1
                elif len(self._hashcons) != before_nodes:
                    changed = True
            merges = self.rebuild()
            if not changed and merges == 0:
                break
        return {
            "iterations": iters,
            "applications": total_applications,
            "eclasses": len(self._classes),
            "enodes": len(self._hashcons),
        }

    def equivalent(self, id1: int, id2: int) -> bool:
        return self.find(id1) == self.find(id2)

    def extract_best(self, eclass_id: int) -> tuple[int, Any]:
        """Extract minimum-AST-size representative expression from an e-class."""
        costs: dict[int, tuple[int, Any]] = {}
        # Bellman-Ford fixed-point cost relaxation handles cycles cleanly
        for _ in range(len(self._classes) + 2):
            changed = False
            for cid, ec in self._classes.items():
                for node in ec.nodes:
                    if not node.children:
                        cand = (1, node.op)
                    elif all(self.find(ch) in costs for ch in node.children):
                        c_sum = 1 + sum(costs[self.find(ch)][0] for ch in node.children)
                        expr = (node.op,) + tuple(costs[self.find(ch)][1] for ch in node.children)
                        cand = (c_sum, expr)
                    else:
                        continue
                    prev = costs.get(cid)
                    if prev is None or (cand[0], repr(cand[1])) < (prev[0], repr(prev[1])):
                        costs[cid] = cand
                        changed = True
            if not changed:
                break
        return costs[self.find(eclass_id)]

    def project_eclass_hv(self, eclass_id: int) -> int:
        """QE-HDC projection: canonical 10,240-bit HV invariant across the entire e-class.

        Equivalent expressions e1, e2 in the same e-class share find(e1) == find(e2),
        and their QE-HDC vector is computed from the canonical extracted normal form,
        guaranteeing d_H(QE_HDC(e1), QE_HDC(e2)) == 0.
        """
        root = self.find(eclass_id)
        cached = self._qehdc_cache.get(root)
        if cached is not None:
            return cached
        _cost, best_expr = self.extract_best(root)
        hv = self._encode_canon_expr_hv(best_expr)
        self._qehdc_cache[root] = hv
        return hv

    def _encode_canon_expr_hv(self, expr: Any) -> int:
        if isinstance(expr, tuple) and expr:
            op_hv = hv10240_of(f"qehdc_op:{expr[0]}")
            acc = op_hv
            for idx, arg in enumerate(expr[1:], start=1):
                arg_hv = self._encode_canon_expr_hv(arg)
                acc = bind10240(acc, permute10240(arg_hv, idx * 17))
            return acc
        return hv10240_of(f"qehdc_atom:{expr}")


DEFAULT_ALGEBRA_REWRITE_RULES: tuple[RewriteRule, ...] = (
    RewriteRule("add_comm", ("add", "?x", "?y"), ("add", "?y", "?x")),
    RewriteRule("mul_comm", ("mul", "?x", "?y"), ("mul", "?y", "?x")),
    RewriteRule("add_zero_r", ("add", "?x", "0"), "?x"),
    RewriteRule("mul_one_r", ("mul", "?x", "1"), "?x"),
    RewriteRule("mul_zero_r", ("mul", "?x", "0"), "0"),
    RewriteRule("mul_two_to_add", ("mul", "?x", "2"), ("add", "?x", "?x")),
    RewriteRule("add_self_to_mul_two", ("add", "?x", "?x"), ("mul", "?x", "2")),
    RewriteRule("double_neg", ("not", ("not", "?p")), "?p"),
)


# ---------------------------------------------------------------------------
# 5. Bit-Sliced Stochastic Resonator + uint8 Multiset Deflation (Stages 2 & 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResonatorFactorization:
    converged: bool
    factors: tuple[str, ...]
    iterations: int
    similarity_q16: int


class BitSlicedStochasticResonator:
    """Factorizes bound hypervectors S = F_1 ^ F_2 ^ ... ^ F_m & superpositions via uint8 deflation.

    Uses 100% integer bitwise XOR unbinding + POPCNT cleanup over codebooks, stochastic
    random restart when trapped in local minima, and exact uint8 multiset histogram
    deflation A_{t+1}[i] = max(0, A_t[i] - E*[i]) to peel multiple composites cleanly.
    """

    def __init__(self, codebooks: Mapping[str, Mapping[str, int]], *, seed: int = 7):
        self.factor_names: tuple[str, ...] = tuple(codebooks.keys())
        self.codebooks: dict[str, dict[str, int]] = {
            fname: {k: v & UNIFIED_MASK for k, v in cb.items()}
            for fname, cb in codebooks.items()
        }
        self._rng = random.Random(seed)

    def _cleanup_best(self, probe: int, codebook: Mapping[str, int], *, top_stochastic: int = 1) -> tuple[str, int, int]:
        scored = [(sim_q16_10240(probe, hv), name, hv) for name, hv in codebook.items()]
        scored.sort(key=lambda x: (-x[0], x[1]))
        if top_stochastic > 1 and len(scored) >= top_stochastic and scored[0][0] < 12_000:
            choice = self._rng.choice(scored[:top_stochastic])
            return choice[1], choice[2], choice[0]
        best = scored[0]
        return best[1], best[2], best[0]

    def factorize(
        self,
        composite_hv: int,
        *,
        max_iters: int = 40,
        restarts: int = 4,
        accept_q16: int = 32_000,
    ) -> ResonatorFactorization:
        """Recover (name_1, ..., name_m) such that XOR(hv(name_i)) ~= composite_hv."""
        target = composite_hv & UNIFIED_MASK
        n_factors = len(self.factor_names)
        if n_factors == 0:
            return ResonatorFactorization(False, (), 0, 0)

        best_overall: ResonatorFactorization | None = None
        for attempt in range(max(1, restarts)):
            # Initialize each factor guess as the bundled superposition of its codebook
            # (or random codebook element on restart to escape symmetric saddle points)
            est_hvs: list[int] = []
            est_names: list[str] = []
            for fname in self.factor_names:
                cb = self.codebooks[fname]
                if attempt == 0:
                    est_hvs.append(bundle10240(list(cb.values())))
                    est_names.append(next(iter(cb.keys())))
                else:
                    rk = self._rng.choice(list(cb.keys()))
                    est_names.append(rk)
                    est_hvs.append(cb[rk])

            for it in range(1, max_iters + 1):
                changed = False
                for idx, fname in enumerate(self.factor_names):
                    # Unbind all other factor estimates from composite target
                    probe = target
                    for jdx in range(n_factors):
                        if jdx != idx:
                            probe ^= est_hvs[jdx]
                    top_k = 2 if (it % 7 == 0 and attempt > 0) else 1
                    new_name, new_hv, _score = self._cleanup_best(
                        probe, self.codebooks[fname], top_stochastic=top_k
                    )
                    if new_name != est_names[idx]:
                        est_names[idx] = new_name
                        est_hvs[idx] = new_hv
                        changed = True

                reconstructed = 0
                for h in est_hvs:
                    reconstructed ^= h
                sim = sim_q16_10240(reconstructed, target)
                cand = ResonatorFactorization(
                    converged=(sim >= accept_q16),
                    factors=tuple(est_names),
                    iterations=it,
                    similarity_q16=sim,
                )
                if best_overall is None or cand.similarity_q16 > best_overall.similarity_q16:
                    best_overall = cand
                if sim >= accept_q16:
                    return cand
                if not changed and it >= 3:
                    break

        return best_overall or ResonatorFactorization(False, (), max_iters, 0)

    @staticmethod
    def build_uint8_histogram(vectors: Sequence[int]) -> bytearray:
        """Exact per-bit uint8 multiset histogram A_0[i] = sum_k bit_i(E_k), clamped to 255."""
        hist = bytearray(D_UNIFIED)
        for vec in vectors:
            v = vec & UNIFIED_MASK
            while v:
                lsb = v & -v
                bit_idx = lsb.bit_length() - 1
                if hist[bit_idx] < 255:
                    hist[bit_idx] += 1
                v ^= lsb
        return hist

    @staticmethod
    def histogram_to_hv(hist: bytearray, count: int) -> int:
        """Binarize a uint8 multiset histogram into a 10,240-bit hypervector."""
        if count <= 0:
            return 0
        thresh = (count // 2) + 1
        out = 0
        tie = (count % 2 == 0)
        for i, c in enumerate(hist):
            if c >= thresh or (tie and c == count // 2 and (i & 1)):
                out |= 1 << i
        return out

    @staticmethod
    def deflate_uint8_histogram(hist: bytearray, extracted_hv: int) -> bytearray:
        """Exact uint8 multiset histogram deflation: A_{t+1}[i] = max(0, A_t[i] - E*[i])."""
        out = bytearray(hist)
        v = extracted_hv & UNIFIED_MASK
        while v:
            lsb = v & -v
            bit_idx = lsb.bit_length() - 1
            if out[bit_idx] > 0:
                out[bit_idx] -= 1
            v ^= lsb
        return out

    def peel_superposition(
        self,
        composites: Sequence[int],
        *,
        max_items: int | None = None,
    ) -> list[ResonatorFactorization]:
        """Peel multiple superposed composites one-by-one using uint8 histogram deflation."""
        count = len(composites)
        limit = count if max_items is None else min(count, max_items)
        hist = self.build_uint8_histogram(composites)
        recovered: list[ResonatorFactorization] = []
        remaining = count
        for _ in range(limit):
            if remaining <= 0:
                break
            cur_hv = self.histogram_to_hv(hist, remaining)
            res = self.factorize(cur_hv, accept_q16=10_000 if remaining > 1 else 32_000)
            if not res.factors:
                break
            recovered.append(res)
            # Reconstruct exact extracted composite E* and deflate uint8 histogram
            e_star = 0
            for fname, sym in zip(self.factor_names, res.factors):
                e_star ^= self.codebooks[fname][sym]
            hist = self.deflate_uint8_histogram(hist, e_star)
            remaining -= 1
        return recovered


# ---------------------------------------------------------------------------
# 6. Poole-Specificity Tropical JTMS & Twin-E-Graph do-Calculus (Stages 7 & 13-L4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DefeasibleRule:
    rule_id: str
    antecedents: tuple[str, ...]
    consequent: str
    negated: bool = False
    is_strict: bool = False
    tropical_cost: int = 10  # Lower cost = higher specificity / priority in min-plus semiring
    specificity_rank: int = 0  # Higher rank = strictly more specific (Poole specificity)


@dataclass(frozen=True)
class BeliefJustification:
    proposition: str
    truth_value: bool
    winning_rule_id: str
    tropical_cost: int
    defeated_rules: tuple[str, ...]


class TropicalJTMS:
    """Poole-Specificity Defeasible Justification-Based Truth Maintenance System.

    Computes specificity ordering over taxonomy chains (e.g., penguin -> bird):
    if A entails B, a rule anchored on A is strictly more specific than a rule
    anchored on B, so `penguin -> not flies` deterministically defeats `bird -> flies`.
    """

    def __init__(self) -> None:
        self.facts: dict[str, bool] = {}
        self.rules: dict[str, DefeasibleRule] = {}
        self._isa: dict[str, set[str]] = defaultdict(set)  # subtype -> set of supertypes

    def clone(self) -> TropicalJTMS:
        cp = TropicalJTMS()
        cp.facts = dict(self.facts)
        cp.rules = dict(self.rules)
        cp._isa = defaultdict(set, {k: set(v) for k, v in self._isa.items()})
        return cp

    def assert_fact(self, prop: str, value: bool = True) -> None:
        self.facts[prop.strip().casefold()] = bool(value)

    def retract_fact(self, prop: str) -> None:
        self.facts.pop(prop.strip().casefold(), None)

    def add_isa(self, subtype: str, supertype: str) -> None:
        sub, sup = subtype.strip().casefold(), supertype.strip().casefold()
        self._isa[sub].add(sup)
        self.add_rule(
            DefeasibleRule(
                rule_id=f"isa:{sub}->{sup}",
                antecedents=(sub,),
                consequent=sup,
                negated=False,
                is_strict=True,
                tropical_cost=1,
                specificity_rank=1,
            )
        )

    def add_rule(self, rule: DefeasibleRule) -> None:
        norm = DefeasibleRule(
            rule_id=rule.rule_id,
            antecedents=tuple(a.strip().casefold() for a in rule.antecedents),
            consequent=rule.consequent.strip().casefold(),
            negated=bool(rule.negated),
            is_strict=bool(rule.is_strict),
            tropical_cost=int(rule.tropical_cost),
            specificity_rank=int(rule.specificity_rank),
        )
        self.rules[norm.rule_id] = norm

    def _ancestors(self, term: str) -> set[str]:
        seen: set[str] = set()
        stack = [term]
        while stack:
            cur = stack.pop()
            for sup in self._isa.get(cur, ()):
                if sup not in seen:
                    seen.add(sup)
                    stack.append(sup)
        return seen

    def _rule_specificity(self, rule: DefeasibleRule) -> tuple[int, int, int]:
        """Higher tuple wins: (is_strict, poole_depth + explicit_rank + len(antecedents), -tropical_cost)."""
        depth = 0
        for ant in rule.antecedents:
            depth += len(self._ancestors(ant))
        return (
            1 if rule.is_strict else 0,
            depth + rule.specificity_rank + len(rule.antecedents),
            -rule.tropical_cost,
        )

    def derive_all(self) -> dict[str, BeliefJustification]:
        """Fixed-point Tropical-JTMS belief closure with Poole specificity resolution."""
        beliefs: dict[str, BeliefJustification] = {}
        for prop, val in self.facts.items():
            beliefs[prop] = BeliefJustification(prop, val, "explicit_fact", 0, ())

        for _ in range(len(self.rules) + 5):
            candidates: dict[str, list[tuple[tuple[int, int, int], DefeasibleRule, int]]] = defaultdict(list)
            for rule in self.rules.values():
                if all(beliefs.get(a) is not None and beliefs[a].truth_value for a in rule.antecedents):
                    ant_cost = sum(beliefs[a].tropical_cost for a in rule.antecedents)
                    spec = self._rule_specificity(rule)
                    candidates[rule.consequent].append((spec, rule, ant_cost + rule.tropical_cost))

            changed = False
            for prop, cands in candidates.items():
                if prop in self.facts:
                    continue  # Explicit observed/intervened facts are immutable anchors
                cands.sort(key=lambda item: (item[0], -item[2], item[1].rule_id), reverse=True)
                best_spec, best_rule, best_cost = cands[0]
                defeated = tuple(r.rule_id for _s, r, _c in cands[1:] if r.negated != best_rule.negated)
                new_j = BeliefJustification(
                    proposition=prop,
                    truth_value=(not best_rule.negated),
                    winning_rule_id=best_rule.rule_id,
                    tropical_cost=best_cost,
                    defeated_rules=defeated,
                )
                if beliefs.get(prop) != new_j:
                    beliefs[prop] = new_j
                    changed = True
            if not changed:
                break
        return beliefs

    def query(self, prop: str) -> BeliefJustification | None:
        return self.derive_all().get(prop.strip().casefold())


@dataclass(frozen=True)
class DoInterventionReport:
    intervention: dict[str, bool]
    baseline_beliefs: dict[str, bool]
    counterfactual_beliefs: dict[str, bool]
    changed_propositions: dict[str, tuple[bool | None, bool]]
    severed_rules: tuple[str, ...]
    explanation: str


class TwinEGraphDoCalculus:
    """Copy-on-Write Twin-E-Graph Causal Intervention Engine + Dual-Mode Gate (M_truth vs M_canon).

    - Pearl's do(X = x): creates a Copy-on-Write twin fork, severs ALL incoming causal rules
      whose consequent is X (graph mutilation), clamps X = x, and re-runs Tropical-JTMS closure.
    - Dual-Mode Gate:
      * mode="truth" (M_truth): validates candidate propositions against verified real-world facts.
      * mode="canon" (M_canon): permits figurative metaphor & fictional assertions while rejecting
        direct contradictions against established story canon (e.g. a dead character speaking).
    """

    def __init__(self) -> None:
        self.truth_jtms = TropicalJTMS()
        self.canon_jtms = TropicalJTMS()
        self.egraph = QuotientEGraph()

    def add_causal_rule(
        self,
        rule_id: str,
        antecedents: Sequence[str],
        consequent: str,
        *,
        negated: bool = False,
        is_strict: bool = False,
        specificity_rank: int = 0,
        world: str = "truth",
    ) -> None:
        target = self.canon_jtms if world == "canon" else self.truth_jtms
        target.add_rule(
            DefeasibleRule(
                rule_id=rule_id,
                antecedents=tuple(antecedents),
                consequent=consequent,
                negated=negated,
                is_strict=is_strict,
                specificity_rank=specificity_rank,
            )
        )

    def assert_world_fact(self, prop: str, value: bool = True, *, world: str = "truth") -> None:
        target = self.canon_jtms if world == "canon" else self.truth_jtms
        target.assert_fact(prop, value)

    def do_intervene(
        self,
        interventions: Mapping[str, bool],
        *,
        world: str = "truth",
    ) -> DoInterventionReport:
        """Execute Pearl do(X=x) on a Copy-on-Write twin fork without mutating the base world."""
        base = self.canon_jtms if world == "canon" else self.truth_jtms
        baseline_all = base.derive_all()
        baseline_map = {k: v.truth_value for k, v in baseline_all.items()}

        # Copy-on-Write twin fork with incoming causal parents severed for every intervened variable
        twin = base.clone()
        norm_interventions = {k.strip().casefold(): bool(v) for k, v in interventions.items()}
        severed: list[str] = []
        for rid, rule in list(twin.rules.items()):
            if rule.consequent in norm_interventions:
                del twin.rules[rid]
                severed.append(rid)
        for var, val in norm_interventions.items():
            twin.assert_fact(var, val)

        cf_all = twin.derive_all()
        cf_map = {k: v.truth_value for k, v in cf_all.items()}

        changed: dict[str, tuple[bool | None, bool]] = {}
        for prop, cf_val in sorted(cf_map.items()):
            base_val = baseline_map.get(prop)
            if base_val != cf_val:
                changed[prop] = (base_val, cf_val)

        do_str = ", ".join(f"do({k}={v})" for k, v in sorted(norm_interventions.items()))
        delta_str = ", ".join(f"{k}: {old}->{new}" for k, (old, new) in changed.items()) or "no downstream change"
        return DoInterventionReport(
            intervention=norm_interventions,
            baseline_beliefs=baseline_map,
            counterfactual_beliefs=cf_map,
            changed_propositions=changed,
            severed_rules=tuple(sorted(severed)),
            explanation=f"Twin-E-Graph {do_str} (severed {len(severed)} incoming rule(s)): {delta_str}.",
        )

    def verify_claim_gate(
        self,
        proposition: str,
        claimed_value: bool = True,
        *,
        mode: str = "truth",
        is_figurative: bool = False,
    ) -> tuple[bool, str]:
        """Dual-Mode World Mask: M_truth (strict factual) vs M_canon (fictional/metaphor-safe)."""
        prop = proposition.strip().casefold()
        if mode == "canon":
            if is_figurative:
                return True, "M_canon=1: figurative/metaphorical expression permitted in fictional mode."
            canon_belief = self.canon_jtms.query(prop)
            if canon_belief is not None and canon_belief.truth_value != bool(claimed_value):
                return (
                    False,
                    f"M_canon=0: contradicts established story canon '{prop}={canon_belief.truth_value}' "
                    f"(via {canon_belief.winning_rule_id}).",
                )
            return True, "M_canon=1: consistent with story canon."

        # Literal factual truth mode (M_truth)
        if is_figurative:
            return False, "M_truth=0: literal truth mode rejects ungrounded figurative claim."
        truth_belief = self.truth_jtms.query(prop)
        if truth_belief is None:
            return False, f"M_truth=0: proposition '{prop}' unverified in literal world model."
        if truth_belief.truth_value != bool(claimed_value):
            return (
                False,
                f"M_truth=0: contradicts verified truth '{prop}={truth_belief.truth_value}' "
                f"(via {truth_belief.winning_rule_id}).",
            )
        return True, f"M_truth=1: verified '{prop}={claimed_value}' via {truth_belief.winning_rule_id}."


__all__ = [
    "D_UNIFIED",
    "NUM_BLOCKS",
    "BLOCK_WIDTH",
    "hv10240_of",
    "rot64",
    "permute10240",
    "bind10240",
    "hamming_distance10240",
    "hamming_sim10240",
    "sim_q16_10240",
    "BitSlicedBundle10240",
    "bundle10240",
    "SparseBlockVector",
    "LeeArcSSP",
    "AnalogicalTransferResult",
    "WeisfeilerLemanAnalogicalMapper",
    "ENode",
    "EClass",
    "RewriteRule",
    "QuotientEGraph",
    "DEFAULT_ALGEBRA_REWRITE_RULES",
    "ResonatorFactorization",
    "BitSlicedStochasticResonator",
    "DefeasibleRule",
    "BeliefJustification",
    "TropicalJTMS",
    "DoInterventionReport",
    "TwinEGraphDoCalculus",
]
