"""Stage S5 RDC Mechanism 4: Behavioral-Equivalence Consolidation & Cross-Domain Concept Abstraction.

STRICT ISOLATION (Stage S5):
- Contains ZERO hardcoded domain action, predicate, attribute, or entity-type literals.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- Intra-domain consolidation strictly gates merges on:
  (1) exact behavioral equivalence across an independent probe set,
  (2) zero regression failures and zero collateral changes on full regression replay,
  (3) strict MDL J improvement (``J_after < J_before`` and ``bits_after < bits_before``),
  (4) reversible retirement in ``OperatorVersionStore`` (NEVER deletes history).
- Cross-domain concept extraction (Kill Test 4, PDF pp. 8, 13, 17) discovers reversible
  parametric relational templates purely by alpha-renaming structural isomorphism across
  two unrelated verified domains, verifies strict bit reduction in BOTH domains, and
  transfers structural inductive bias to a held-out 3rd domain in a few-shot regime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from research.cpu_learning.itl.delta_extractor import (
    CorpusStatistics,
    ExtractedEpisodeObservation,
)
from research.cpu_learning.itl.operator_card import (
    AttributeGuard,
    ConditionalRoleFactGuard,
    IndirectRoleFactGuard,
    LiftedAttributeEdit,
    OperatorCard,
    RoleInequalityGuard,
)
from research.cpu_learning.itl.operator_inducer import _is_attr_functionally_entailed
from research.cpu_learning.itl.role_unifier import (
    extract_direct_lifted_facts,
    extract_entity_literal_facts,
    extract_one_hop_links,
    extract_role_attribute_map,
    intersect_lifted_deltas,
    lift_sparse_delta,
)
from research.cpu_learning.itl.version_store import (
    OperatorVersionStore,
    VersionStoreFrozenError,
)
from research.cpu_learning.rdc.mdl_judge import (
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    MDLExample,
    compute_library_compressed_bits,
    evaluate_library_mdl,
)
from research.cpu_learning.simulator.state_graph import (
    StateGraph,
    canonical_json_bytes,
    compressed_bit_length,
    sha256_hex,
)

RDC_CONSOLIDATION_SCHEMA_VERSION: str = "s5-rdc-consolidation-v1"


# ---------------------------------------------------------------------------
# 1. Intra-Domain Behavioral-Equivalence Safe Consolidation & Rollback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EquivalenceCheckReport:
    """Result of checking exact behavioral equivalence between two verified OperatorCards."""

    card_a_id: str
    card_b_id: str
    probes_checked: int
    divergences: int
    equivalent: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_a_id": self.card_a_id,
            "card_b_id": self.card_b_id,
            "probes_checked": self.probes_checked,
            "divergences": self.divergences,
            "equivalent": self.equivalent,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ConsolidationResult:
    """Auditable decision record for an intra-domain consolidation attempt."""

    canonical_action_id: str
    duplicate_action_id: str
    accepted: bool
    reason: str
    equivalence_report: EquivalenceCheckReport
    j_before: float
    j_after: float
    library_bits_before: int
    library_bits_after: int
    regression_failures_after: int
    retired_version: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_action_id": self.canonical_action_id,
            "duplicate_action_id": self.duplicate_action_id,
            "accepted": self.accepted,
            "reason": self.reason,
            "equivalence_report": self.equivalence_report.to_dict(),
            "j_before": round(self.j_before, 4),
            "j_after": round(self.j_after, 4),
            "library_bits_before": self.library_bits_before,
            "library_bits_after": self.library_bits_after,
            "regression_failures_after": self.regression_failures_after,
            "retired_version": self.retired_version,
        }


def check_behavioral_equivalence(
    card_a: OperatorCard,
    card_b: OperatorCard,
    probes: Sequence[tuple[StateGraph, Mapping[str, str]]],
) -> EquivalenceCheckReport:
    """Verify whether ``card_a`` and ``card_b`` behave identically across every probe."""
    if not probes:
        return EquivalenceCheckReport(
            card_a_id=card_a.card_id,
            card_b_id=card_b.card_id,
            probes_checked=0,
            divergences=1,
            equivalent=False,
            reason="rejected_empty_probe_set",
        )
    if card_a.role_variables != card_b.role_variables:
        return EquivalenceCheckReport(
            card_a_id=card_a.card_id,
            card_b_id=card_b.card_id,
            probes_checked=0,
            divergences=1,
            equivalent=False,
            reason="role_signature_mismatch",
        )

    divergences = 0
    for state, roles in probes:
        res_a = card_a.apply(state, roles)
        res_b = card_b.apply(state, roles)
        if (
            res_a.applied != res_b.applied
            or res_a.abstained != res_b.abstained
            or res_a.after_state != res_b.after_state
            or res_a.delta != res_b.delta
        ):
            divergences += 1

    eq = divergences == 0
    return EquivalenceCheckReport(
        card_a_id=card_a.card_id,
        card_b_id=card_b.card_id,
        probes_checked=len(probes),
        divergences=divergences,
        equivalent=eq,
        reason="behaviorally_equivalent" if eq else "behavioral_divergence_detected",
    )


# ---------------------------------------------------------------------------
# 2. Cross-Domain Concept Abstraction & Few-Shot Transfer (Kill Test 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossDomainConcept:
    """Reversible parametric relational operator template abstracted across >= 2 unrelated domains.

    All role names (``$R*``), predicate names (``$P*``), entity types (``$T*``), attribute
    names (``$A*``), literal values (``$V*``), and attribute values (``$AV*``) are pure
    domain-agnostic placeholders.
    """

    concept_id: str
    role_slots: tuple[str, ...]
    type_slots: tuple[tuple[str, tuple[str, ...]], ...]
    required_facts: tuple[tuple[str, str, str], ...]
    conditional_fact_guards: tuple[tuple[str, str, str, str], ...]
    indirect_fact_guards: tuple[tuple[str, str, str, str, str], ...]
    attribute_guards: tuple[tuple[str, str, str], ...]
    inequality_guards: tuple[tuple[str, str], ...]
    add_facts: tuple[tuple[str, str, str], ...]
    delete_facts: tuple[tuple[str, str, str], ...]
    attribute_edits: tuple[tuple[str, str, str], ...]
    source_domains: tuple[str, ...] = ()

    def template_fingerprint(self) -> str:
        return sha256_hex(canonical_json_bytes(self.to_structural_dict()))

    def to_structural_dict(self) -> dict[str, Any]:
        return {
            "role_slots": list(self.role_slots),
            "type_slots": {r: list(ts) for r, ts in self.type_slots},
            "required_facts": [list(f) for f in self.required_facts],
            "conditional_fact_guards": [list(g) for g in self.conditional_fact_guards],
            "indirect_fact_guards": [list(g) for g in self.indirect_fact_guards],
            "attribute_guards": [list(g) for g in self.attribute_guards],
            "inequality_guards": [list(g) for g in self.inequality_guards],
            "add_facts": [list(f) for f in self.add_facts],
            "delete_facts": [list(f) for f in self.delete_facts],
            "attribute_edits": [list(e) for e in self.attribute_edits],
        }

    def to_executable_dict(self) -> dict[str, Any]:
        out = self.to_structural_dict()
        out["concept_id"] = self.concept_id
        return out

    def to_dict(self) -> dict[str, Any]:
        out = self.to_executable_dict()
        out["source_domains"] = list(self.source_domains)
        return out

    def instantiate(
        self,
        instantiation: ConceptInstantiation,
        *,
        version: int = 1,
        support_positive: int = 0,
        support_negative: int = 0,
        confidence: float = 1.0,
    ) -> OperatorCard:
        """Losslessly reconstruct an executable ``OperatorCard`` from this concept + binding."""
        if instantiation.concept_id != self.concept_id:
            raise ValueError(
                f"Instantiation concept_id {instantiation.concept_id!r} != {self.concept_id!r}"
            )
        rmap = {f"$R{i}": v for i, v in enumerate(instantiation.roles)}
        pmap = {f"$P{i}": v for i, v in enumerate(instantiation.preds)}
        tmap = {f"$T{i}": v for i, v in enumerate(instantiation.types)}
        amap = {f"$A{i}": v for i, v in enumerate(instantiation.attrs)}
        vmap = {f"$V{i}": v for i, v in enumerate(instantiation.lit_vals)}
        avmap = {f"$AV{i}": v for i, v in enumerate(instantiation.attr_vals)}

        def _rterm(tok: str) -> str:
            if tok.startswith("$R"):
                return f"?{rmap[tok]}"
            if tok.startswith("$V"):
                return vmap[tok]
            return tok

        role_vars = tuple(sorted(rmap[r] for r in self.role_slots))
        req_types = tuple(
            sorted((rmap[r], frozenset(tmap[t] for t in ts)) for r, ts in self.type_slots)
        )
        req_facts = tuple(
            sorted((_rterm(s), pmap[p], _rterm(o)) for s, p, o in self.required_facts)
        )
        cond_guards = tuple(
            ConditionalRoleFactGuard(
                role=rmap[r],
                when_entity_type=tmap[t],
                predicate=pmap[p],
                value=vmap[v],
            )
            for r, t, p, v in self.conditional_fact_guards
        )
        ind_guards = tuple(
            IndirectRoleFactGuard(
                role=rmap[r],
                via_predicate=pmap[vp],
                when_neighbor_type=tmap[nt],
                predicate=pmap[p],
                value=vmap[v],
            )
            for r, vp, nt, p, v in self.indirect_fact_guards
        )
        attr_guards = tuple(
            AttributeGuard(role=rmap[r], attribute=amap[a], expected_value=avmap[av])
            for r, a, av in self.attribute_guards
        )
        ineq_guards = tuple(
            RoleInequalityGuard(role_a=rmap[ra], role_b=rmap[rb])
            for ra, rb in self.inequality_guards
        )
        add_facts = tuple(sorted((_rterm(s), pmap[p], _rterm(o)) for s, p, o in self.add_facts))
        del_facts = tuple(sorted((_rterm(s), pmap[p], _rterm(o)) for s, p, o in self.delete_facts))
        attr_edits = tuple(
            LiftedAttributeEdit(role=rmap[r], attribute=amap[a], new_value=avmap[av])
            for r, a, av in self.attribute_edits
        )

        return OperatorCard(
            card_id=f"op_{instantiation.action_id}_v{version}",
            action_id=instantiation.action_id,
            role_variables=role_vars,
            required_types=req_types,
            required_facts=req_facts,
            conditional_fact_guards=cond_guards,
            indirect_fact_guards=ind_guards,
            attribute_guards=attr_guards,
            inequality_guards=ineq_guards,
            add_facts=add_facts,
            delete_facts=del_facts,
            attribute_edits=attr_edits,
            confidence=confidence,
            support_positive=support_positive,
            support_negative=support_negative,
            version=version,
        )


@dataclass(frozen=True)
class ConceptInstantiation:
    """Compact lossless binding table instantiating a ``CrossDomainConcept`` in a concrete domain."""

    concept_id: str
    action_id: str
    roles: tuple[str, ...]
    preds: tuple[str, ...]
    types: tuple[str, ...]
    attrs: tuple[str, ...]
    lit_vals: tuple[str, ...]
    attr_vals: tuple[Any, ...]

    def to_compact_list(self) -> list[Any]:
        return [
            self.concept_id,
            self.action_id,
            list(self.roles),
            list(self.preds),
            list(self.types),
            list(self.attrs),
            list(self.lit_vals),
            list(self.attr_vals),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "action_id": self.action_id,
            "roles": list(self.roles),
            "preds": list(self.preds),
            "types": list(self.types),
            "attrs": list(self.attrs),
            "lit_vals": list(self.lit_vals),
            "attr_vals": list(self.attr_vals),
        }


def canonicalize_operator_to_concept(
    card: OperatorCard,
    *,
    concept_id_prefix: str = "concept",
) -> tuple[CrossDomainConcept, ConceptInstantiation]:
    """Alpha-rename a concrete ``OperatorCard`` into a canonical structural template + binding.

    Anchors role/predicate/type numbering strictly on structural effect & precondition
    topology (zero domain strings) so isomorphic operators from completely unrelated
    domains map to the exact same ``template_fingerprint()``.
    """
    role_map: dict[str, str] = {}
    role_list: list[str] = []
    pred_map: dict[str, str] = {}
    pred_list: list[str] = []
    type_map: dict[str, str] = {}
    type_list: list[str] = []
    attr_map: dict[str, str] = {}
    attr_list: list[str] = []
    val_map: dict[str, str] = {}
    val_list: list[str] = []
    av_keys: list[tuple[str, Any]] = []
    av_map: dict[tuple[str, Any], str] = {}
    av_list: list[Any] = []

    def _r(name: str) -> str:
        if name not in role_map:
            role_map[name] = f"$R{len(role_list)}"
            role_list.append(name)
        return role_map[name]

    def _p(name: str) -> str:
        if name not in pred_map:
            pred_map[name] = f"$P{len(pred_list)}"
            pred_list.append(name)
        return pred_map[name]

    def _t(name: str) -> str:
        if name not in type_map:
            type_map[name] = f"$T{len(type_list)}"
            type_list.append(name)
        return type_map[name]

    def _a(name: str) -> str:
        if name not in attr_map:
            attr_map[name] = f"$A{len(attr_list)}"
            attr_list.append(name)
        return attr_map[name]

    def _v(lit: str) -> str:
        if lit not in val_map:
            val_map[lit] = f"$V{len(val_list)}"
            val_list.append(lit)
        return val_map[lit]

    def _av(val: Any) -> str:
        key = (type(val).__name__, val)
        if key not in av_map:
            av_map[key] = f"$AV{len(av_list)}"
            av_keys.append(key)
            av_list.append(val)
        return av_map[key]

    # 1. Anchor primary roles & predicate on effects (delete_facts -> add_facts -> attribute_edits)
    is_binary_transfer = (
        len(card.delete_facts) == 1
        and len(card.add_facts) == 1
        and card.delete_facts[0][1] == card.add_facts[0][1]
        and card.delete_facts[0][0] == card.add_facts[0][0]
        and card.delete_facts[0][0].startswith("?")
        and card.delete_facts[0][2].startswith("?")
        and card.add_facts[0][2].startswith("?")
        and card.delete_facts[0][2] != card.add_facts[0][2]
    )
    if is_binary_transfer:
        ds, dp, do = card.delete_facts[0]
        _as, _ap, ao = card.add_facts[0]
        _r(ds[1:])  # $R0: transferred item
        _r(do[1:])  # $R1: source / giver
        _r(ao[1:])  # $R2: destination / recipient
        _p(dp)      # $P0: transferred relation
    else:
        for s, p, o in card.delete_facts:
            if s.startswith("?"):
                _r(s[1:])
            _p(p)
            if o.startswith("?"):
                _r(o[1:])
            else:
                _v(o)
        for s, p, o in card.add_facts:
            if s.startswith("?"):
                _r(s[1:])
            _p(p)
            if o.startswith("?"):
                _r(o[1:])
            else:
                _v(o)
        for edit in card.attribute_edits:
            _r(edit.role)
            _a(edit.attribute)
            _av(edit.new_value)

    # 2. Role-to-role required facts (discovers authorization / actor role deterministically)
    rel_req_facts = [f for f in card.required_facts if f[0].startswith("?") and f[2].startswith("?")]
    rel_req_facts.sort(
        key=lambda f: (
            0 if f[1] in pred_map else 1,
            role_map.get(f[0][1:], "ZZ"),
            role_map.get(f[2][1:], "ZZ"),
            f[1],
        )
    )
    for s, p, o in rel_req_facts:
        _r(s[1:])
        _p(p)
        _r(o[1:])

    # 3. Direct literal required facts (ordered by role slot)
    lit_req_facts = [f for f in card.required_facts if f[0].startswith("?") and not f[2].startswith("?")]
    lit_req_facts.sort(key=lambda f: (role_map.get(f[0][1:], "ZZ"), f[1], f[2]))
    for s, p, o in lit_req_facts:
        _r(s[1:])
        _p(p)
        _v(o)

    # 4. Any remaining role variables
    for rv in card.role_variables:
        _r(rv)

    # 5. Entity types: order canonically by (guarded_subtype_flag, roles_using_type)
    guarded_types = {g.when_entity_type for g in card.conditional_fact_guards} | {
        g.when_neighbor_type for g in card.indirect_fact_guards
    }
    allowed_map = card.allowed_types_map()
    all_types = sorted(
        {t for ts in allowed_map.values() for t in ts} | guarded_types,
        key=lambda t: (
            1 if t in guarded_types else 0,
            tuple(sorted(role_map[r] for r, ts in allowed_map.items() if t in ts)),
            t,
        ),
    )
    for et in all_types:
        _t(et)

    # 6. Attribute guards (ordered by role slot)
    for ag in sorted(card.attribute_guards, key=lambda g: (role_map[g.role], g.attribute)):
        _a(ag.attribute)
        _av(ag.expected_value)

    # 7. Conditional guards: group by (when_entity_type, predicate, value); prioritize values already in val_map!
    cond_pairs = sorted(
        {(g.when_entity_type, g.predicate, g.value) for g in card.conditional_fact_guards},
        key=lambda x: (
            type_map[x[0]],
            0 if x[2] in val_map else 1,
            val_map.get(x[2], "ZZ"),
            x[1],
        ),
    )
    for _wt, pred, val in cond_pairs:
        _p(pred)
        _v(val)

    # 8. Indirect 1-hop guards
    ind_ordered = sorted(
        card.indirect_fact_guards,
        key=lambda g: (
            role_map[g.role],
            type_map[g.when_neighbor_type],
            0 if g.value in val_map else 1,
            val_map.get(g.value, "ZZ"),
            g.via_predicate,
            g.predicate,
        ),
    )
    for ig in ind_ordered:
        _p(ig.via_predicate)
        _t(ig.when_neighbor_type)
        _p(ig.predicate)
        _v(ig.value)

    def _abs_term(tok: str) -> str:
        if tok.startswith("?"):
            return role_map[tok[1:]]
        return val_map[tok]

    abs_roles = tuple(sorted(role_map.values()))
    abs_types = tuple(
        sorted((role_map[r], tuple(sorted(type_map[t] for t in ts))) for r, ts in card.required_types)
    )
    abs_req = tuple(sorted((_abs_term(s), pred_map[p], _abs_term(o)) for s, p, o in card.required_facts))
    abs_cond = tuple(
        sorted(
            (role_map[g.role], type_map[g.when_entity_type], pred_map[g.predicate], val_map[g.value])
            for g in card.conditional_fact_guards
        )
    )
    abs_ind = tuple(
        sorted(
            (
                role_map[g.role],
                pred_map[g.via_predicate],
                type_map[g.when_neighbor_type],
                pred_map[g.predicate],
                val_map[g.value],
            )
            for g in card.indirect_fact_guards
        )
    )
    abs_attr = tuple(
        sorted(
            (role_map[g.role], attr_map[g.attribute], av_map[(type(g.expected_value).__name__, g.expected_value)])
            for g in card.attribute_guards
        )
    )
    abs_ineq = tuple(
        sorted((role_map[g.role_a], role_map[g.role_b]) for g in card.inequality_guards)
    )
    abs_add = tuple(sorted((_abs_term(s), pred_map[p], _abs_term(o)) for s, p, o in card.add_facts))
    abs_del = tuple(sorted((_abs_term(s), pred_map[p], _abs_term(o)) for s, p, o in card.delete_facts))
    abs_edits = tuple(
        sorted(
            (role_map[e.role], attr_map[e.attribute], av_map[(type(e.new_value).__name__, e.new_value)])
            for e in card.attribute_edits
        )
    )

    provisional = CrossDomainConcept(
        concept_id=concept_id_prefix,
        role_slots=abs_roles,
        type_slots=abs_types,
        required_facts=abs_req,
        conditional_fact_guards=abs_cond,
        indirect_fact_guards=abs_ind,
        attribute_guards=abs_attr,
        inequality_guards=abs_ineq,
        add_facts=abs_add,
        delete_facts=abs_del,
        attribute_edits=abs_edits,
    )
    fp = provisional.template_fingerprint()[:10]
    cid = f"{concept_id_prefix}_{fp}"
    concept = CrossDomainConcept(
        concept_id=cid,
        role_slots=provisional.role_slots,
        type_slots=provisional.type_slots,
        required_facts=provisional.required_facts,
        conditional_fact_guards=provisional.conditional_fact_guards,
        indirect_fact_guards=provisional.indirect_fact_guards,
        attribute_guards=provisional.attribute_guards,
        inequality_guards=provisional.inequality_guards,
        add_facts=provisional.add_facts,
        delete_facts=provisional.delete_facts,
        attribute_edits=provisional.attribute_edits,
    )
    inst = ConceptInstantiation(
        concept_id=cid,
        action_id=card.action_id,
        roles=tuple(role_list),
        preds=tuple(pred_list),
        types=tuple(type_list),
        attrs=tuple(attr_list),
        lit_vals=tuple(val_list),
        attr_vals=tuple(av_list),
    )
    return (concept, inst)


@dataclass(frozen=True)
class CrossDomainAbstractionReport:
    """Verified cross-domain concept extraction & two-domain compression audit."""

    concepts: tuple[CrossDomainConcept, ...]
    domain1_instantiations: tuple[ConceptInstantiation, ...]
    domain2_instantiations: tuple[ConceptInstantiation, ...]
    domain1_bits_without_concept: int
    domain1_bits_with_concept: int
    domain2_bits_without_concept: int
    domain2_bits_with_concept: int
    joint_bits_without_concepts: int
    joint_bits_with_concepts: int
    reversibility_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_ids": [c.concept_id for c in self.concepts],
            "concepts": [c.to_dict() for c in self.concepts],
            "domain1_bits_without_concept": self.domain1_bits_without_concept,
            "domain1_bits_with_concept": self.domain1_bits_with_concept,
            "domain1_bits_saved": self.domain1_bits_without_concept - self.domain1_bits_with_concept,
            "domain2_bits_without_concept": self.domain2_bits_without_concept,
            "domain2_bits_with_concept": self.domain2_bits_with_concept,
            "domain2_bits_saved": self.domain2_bits_without_concept - self.domain2_bits_with_concept,
            "joint_bits_without_concepts": self.joint_bits_without_concepts,
            "joint_bits_with_concepts": self.joint_bits_with_concepts,
            "joint_bits_saved": self.joint_bits_without_concepts - self.joint_bits_with_concepts,
            "reversibility_verified": self.reversibility_verified,
        }


def extract_cross_domain_concepts(
    domain1_name: str,
    domain1_cards: Mapping[str, OperatorCard],
    domain2_name: str,
    domain2_cards: Mapping[str, OperatorCard],
) -> CrossDomainAbstractionReport:
    """Extract promoted ``CrossDomainConcept``s shared across two unrelated verified domains.

    Promotes ONLY templates supported by BOTH domains AND strictly lossless on reconstruction,
    and reports both per-domain incremental bit savings and unamortized joint MDL savings.
    """
    d1_canon: dict[str, list[tuple[CrossDomainConcept, ConceptInstantiation, OperatorCard]]] = {}
    for aid, card in sorted(domain1_cards.items()):
        c, inst = canonicalize_operator_to_concept(card, concept_id_prefix="xdom")
        d1_canon.setdefault(c.template_fingerprint(), []).append((c, inst, card))

    d2_canon: dict[str, list[tuple[CrossDomainConcept, ConceptInstantiation, OperatorCard]]] = {}
    for aid, card in sorted(domain2_cards.items()):
        c, inst = canonicalize_operator_to_concept(card, concept_id_prefix="xdom")
        d2_canon.setdefault(c.template_fingerprint(), []).append((c, inst, card))

    shared_fps = sorted(set(d1_canon.keys()) & set(d2_canon.keys()))
    promoted_concepts: list[CrossDomainConcept] = []
    d1_insts: list[ConceptInstantiation] = []
    d2_insts: list[ConceptInstantiation] = []
    reversibility_ok = True

    d1_covered_aids: set[str] = set()
    d2_covered_aids: set[str] = set()

    for fp in shared_fps:
        c1_rep = d1_canon[fp][0][0]
        promoted = CrossDomainConcept(
            concept_id=c1_rep.concept_id,
            role_slots=c1_rep.role_slots,
            type_slots=c1_rep.type_slots,
            required_facts=c1_rep.required_facts,
            conditional_fact_guards=c1_rep.conditional_fact_guards,
            indirect_fact_guards=c1_rep.indirect_fact_guards,
            attribute_guards=c1_rep.attribute_guards,
            inequality_guards=c1_rep.inequality_guards,
            add_facts=c1_rep.add_facts,
            delete_facts=c1_rep.delete_facts,
            attribute_edits=c1_rep.attribute_edits,
            source_domains=(domain1_name, domain2_name),
        )
        # Verify exact lossless reconstruction for EVERY matching card in BOTH domains!
        all_ok = True
        for _c, inst1, orig1 in d1_canon[fp]:
            if promoted.instantiate(inst1).to_executable_dict() != orig1.to_executable_dict():
                all_ok = False
        for _c, inst2, orig2 in d2_canon[fp]:
            if promoted.instantiate(inst2).to_executable_dict() != orig2.to_executable_dict():
                all_ok = False
        if not all_ok:
            reversibility_ok = False
            continue
        promoted_concepts.append(promoted)
        for _c, inst1, orig1 in d1_canon[fp]:
            d1_insts.append(inst1)
            d1_covered_aids.add(orig1.action_id)
        for _c, inst2, orig2 in d2_canon[fp]:
            d2_insts.append(inst2)
            d2_covered_aids.add(orig2.action_id)

    def _raw_domain_bits(cards: Mapping[str, OperatorCard]) -> int:
        return compressed_bit_length(
            canonical_json_bytes([c.to_executable_dict() for _k, c in sorted(cards.items())])
        )

    def _encoded_domain_bits(
        cards: Mapping[str, OperatorCard],
        insts: Sequence[ConceptInstantiation],
        covered_aids: set[str],
    ) -> int:
        residual = [
            c.to_executable_dict()
            for aid, c in sorted(cards.items())
            if aid not in covered_aids
        ]
        payload = {
            "inst": [i.to_compact_list() for i in insts],
            "res": residual,
        }
        return compressed_bit_length(canonical_json_bytes(payload))

    d1_raw = _raw_domain_bits(domain1_cards)
    d1_with = _encoded_domain_bits(domain1_cards, d1_insts, d1_covered_aids)
    d2_raw = _raw_domain_bits(domain2_cards)
    d2_with = _encoded_domain_bits(domain2_cards, d2_insts, d2_covered_aids)

    joint_raw = compressed_bit_length(
        canonical_json_bytes(
            {
                "d1": [c.to_executable_dict() for _k, c in sorted(domain1_cards.items())],
                "d2": [c.to_executable_dict() for _k, c in sorted(domain2_cards.items())],
            }
        )
    )
    joint_with = compressed_bit_length(
        canonical_json_bytes(
            {
                "concepts": [c.to_executable_dict() for c in promoted_concepts],
                "d1_inst": [i.to_compact_list() for i in d1_insts],
                "d1_res": [
                    c.to_executable_dict()
                    for aid, c in sorted(domain1_cards.items())
                    if aid not in d1_covered_aids
                ],
                "d2_inst": [i.to_compact_list() for i in d2_insts],
                "d2_res": [
                    c.to_executable_dict()
                    for aid, c in sorted(domain2_cards.items())
                    if aid not in d2_covered_aids
                ],
            }
        )
    )

    return CrossDomainAbstractionReport(
        concepts=tuple(promoted_concepts),
        domain1_instantiations=tuple(d1_insts),
        domain2_instantiations=tuple(d2_insts),
        domain1_bits_without_concept=d1_raw,
        domain1_bits_with_concept=d1_with,
        domain2_bits_without_concept=d2_raw,
        domain2_bits_with_concept=d2_with,
        joint_bits_without_concepts=joint_raw,
        joint_bits_with_concepts=joint_with,
        reversibility_verified=bool(reversibility_ok and len(promoted_concepts) >= 1),
    )


def transfer_concepts_to_few_shot_domain(
    concepts: Sequence[CrossDomainConcept],
    grouped_pos_obs: Mapping[str, Sequence[ExtractedEpisodeObservation]],
    corpus_stats: CorpusStatistics,
) -> tuple[dict[str, OperatorCard], dict[str, ConceptInstantiation], int, int]:
    """Transfer promoted ``CrossDomainConcept`` priors onto a brand-new 3rd domain from few-shot positives.

    Returns ``(transferred_cards, instantiations, bits_without_concepts, bits_with_concepts)``.
    100% domain-agnostic: matches each new action's lifted positive delta skeleton against
    promoted ``CrossDomainConcept`` templates and transfers structural role-type symmetry,
    selective precondition scope, subtype guard symmetry, 1-hop enclosure guards, inequality
    guards, and capability attribute guards directly from the positive demonstrations.
    """
    transferred: dict[str, OperatorCard] = {}
    instantiations: dict[str, ConceptInstantiation] = {}

    for aid, pos_obs in sorted(grouped_pos_obs.items()):
        if not pos_obs:
            continue
        common_roles = sorted(set.intersection(*(set(ep.roles.keys()) for ep in pos_obs)))
        lifted_deltas = [
            lift_sparse_delta(ep.sparse_delta, ep.roles, ep.before_frame, ep.after_frame)
            for ep in pos_obs
        ]
        unified = intersect_lifted_deltas(lifted_deltas)
        if unified.is_empty:
            continue

        # Identify binary transfer skeleton: del (?item, p_rel, ?from) & add (?item, p_rel, ?to)
        del_list = sorted(unified.delete_facts)
        add_list = sorted(unified.add_facts)
        if not (
            len(del_list) == 1
            and len(add_list) == 1
            and del_list[0][1] == add_list[0][1]
            and del_list[0][0] == add_list[0][0]
            and del_list[0][0].startswith("?")
            and del_list[0][2].startswith("?")
            and add_list[0][2].startswith("?")
        ):
            continue

        r_item = del_list[0][0][1:]
        r_from = del_list[0][2][1:]
        r_to = add_list[0][2][1:]
        p_rel = del_list[0][1]

        # Find matching CrossDomainConcept by role count & binary transfer effect skeleton
        matched: CrossDomainConcept | None = None
        for concept in concepts:
            if len(concept.role_slots) != len(common_roles):
                continue
            if concept.delete_facts == (("$R0", "$P0", "$R1"),) and concept.add_facts == (
                ("$R0", "$P0", "$R2"),
            ):
                matched = concept
                break
        if matched is None:
            continue

        # Map concrete roles to $R0..$Rk
        abs_to_role: dict[str, str] = {"$R0": r_item, "$R1": r_from, "$R2": r_to}
        remaining_roles = [r for r in common_roles if r not in abs_to_role.values()]
        for slot in matched.role_slots:
            if slot not in abs_to_role and remaining_roles:
                abs_to_role[slot] = remaining_roles.pop(0)
        role_to_abs = {v: k for k, v in abs_to_role.items()}

        # Observed types per concrete role
        obs_types: dict[str, set[str]] = {}
        for r in common_roles:
            obs_types[r] = {
                t
                for ep in pos_obs
                for t in (ep.before_frame.entity_type_of(ep.roles[r]),)
                if t is not None and t != "unknown"
            }

        # 1. Transfer role-type sharing symmetry from concept.type_slots!
        #    If two concept slots share the identical abstract type tuple (e.g. $R1 & $R2),
        #    unify their observed concrete types across few-shot demos!
        slots_by_abs_ts: dict[tuple[str, ...], list[str]] = {}
        for slot, abs_ts in matched.type_slots:
            slots_by_abs_ts.setdefault(abs_ts, []).append(slot)
        unified_types: dict[str, frozenset[str]] = {}
        for abs_ts, group_slots in slots_by_abs_ts.items():
            merged: set[str] = set()
            for sl in group_slots:
                merged.update(obs_types.get(abs_to_role[sl], set()))
            for sl in group_slots:
                unified_types[abs_to_role[sl]] = frozenset(merged)
        for r in common_roles:
            if r not in unified_types:
                unified_types[r] = frozenset(obs_types.get(r, set()))

        # 2. Direct lifted facts observed across all positive demos
        direct_sets = [extract_direct_lifted_facts(ep.before_frame, ep.roles) for ep in pos_obs]
        pos_common_direct = frozenset.intersection(*direct_sets)

        # Keep ALL role-to-role required facts, PLUS direct literal preconditions ONLY on roles
        # that carry a direct literal precondition in the abstract concept!
        concept_lit_roles = {
            abs_to_role[s]
            for s, _p, o in matched.required_facts
            if s.startswith("$R") and o.startswith("$V")
        }
        kept_req_facts: set[tuple[str, str, str]] = set()
        for s, p, o in pos_common_direct:
            if o.startswith("?"):
                kept_req_facts.add((s, p, o))
            elif s.startswith("?") and s[1:] in concept_lit_roles:
                kept_req_facts.add((s, p, o))

        # 3. Transfer role inequality guards
        ineq_guards: list[RoleInequalityGuard] = []
        for ra_s, rb_s in matched.inequality_guards:
            ra, rb = abs_to_role[ra_s], abs_to_role[rb_s]
            if all(ep.roles[ra] != ep.roles[rb] for ep in pos_obs):
                ineq_guards.append(RoleInequalityGuard(role_a=ra, role_b=rb))

        # 4. Transfer capability attribute guards on roles guarded in the concept
        attr_guards: list[AttributeGuard] = []
        pos_attr_maps = [extract_role_attribute_map(ep.before_frame, ep.roles) for ep in pos_obs]
        concept_attr_roles = {abs_to_role[r] for r, _a, _av in matched.attribute_guards}
        for rname in sorted(concept_attr_roles):
            common_ks: set[str] | None = None
            for amap in pos_attr_maps:
                ks = set(amap.get(rname, {}).keys())
                common_ks = ks if common_ks is None else (common_ks & ks)
            if not common_ks:
                continue
            r_types = unified_types[rname]
            role_lits = frozenset(
                (p, o)
                for s, p, o in kept_req_facts
                if s == f"?{rname}" and not o.startswith("?")
            )
            for ak in sorted(common_ks):
                vals = {amap[rname][ak] for amap in pos_attr_maps}
                if len(vals) != 1:
                    continue
                exp_v = next(iter(vals))
                if len(corpus_stats.attr_domain_for_types(r_types, ak)) < 2:
                    continue
                if _is_attr_functionally_entailed(r_types, ak, exp_v, role_lits, corpus_stats):
                    continue
                if not corpus_stats.is_attr_edited_on_types(r_types, ak):
                    if corpus_stats.other_varying_pos_role_slots(
                        r_types, ak, exclude_action=aid, exclude_role=rname
                    ):
                        continue
                attr_guards.append(AttributeGuard(role=rname, attribute=ak, expected_value=exp_v))

        # 5. Transfer conditional subtype guard symmetry across multi-type roles!
        cond_guards: list[ConditionalRoleFactGuard] = []
        if matched.conditional_fact_guards:
            cond_roles = sorted({abs_to_role[r] for r, _t, _p, _v in matched.conditional_fact_guards})
            # Pool observations across all symmetric cond_roles to discover subtype-specific invariants
            # even if one role (e.g. origin) never saw the enclosure subtype in the 2 positive demos!
            shared_subtypes: set[str] = set()
            for rname in cond_roles:
                shared_subtypes.update(unified_types[rname])
            preds_by_sub: dict[str, set[str]] = {}
            invs_by_sub: dict[str, set[tuple[str, str]]] = {}
            for sub_t in sorted(shared_subtypes):
                lit_sets: list[frozenset[tuple[str, str]]] = []
                for ep in pos_obs:
                    for rname in cond_roles:
                        eid = ep.roles[rname]
                        if ep.before_frame.entity_type_of(eid) == sub_t:
                            lit_sets.append(extract_entity_literal_facts(ep.before_frame, eid))
                if not lit_sets:
                    continue
                inter = set(lit_sets[0])
                for ls in lit_sets[1:]:
                    inter &= ls
                invs_by_sub[sub_t] = inter
                preds_by_sub[sub_t] = {p for ls in lit_sets for p, _v in ls}

            for sub_t, inv_fs in sorted(invs_by_sub.items()):
                other_ps: set[str] | None = None
                for ot, ops in preds_by_sub.items():
                    if ot == sub_t:
                        continue
                    other_ps = set(ops) if other_ps is None else (other_ps & ops)
                shared_others = other_ps or set()
                for pred, val in sorted(inv_fs):
                    if pred in shared_others:
                        continue
                    for rname in cond_roles:
                        cond_guards.append(
                            ConditionalRoleFactGuard(
                                role=rname,
                                when_entity_type=sub_t,
                                predicate=pred,
                                value=val,
                            )
                        )

        # 6. Transfer 1-hop indirect neighbor guards from positive 1-hop links (no negative demos needed!)
        ind_guards: list[IndirectRoleFactGuard] = []
        if matched.indirect_fact_guards:
            ind_roles = sorted({abs_to_role[r] for r, _vp, _nt, _p, _v in matched.indirect_fact_guards})
            pos_links_by_key: dict[tuple[str, str], list[tuple[ExtractedEpisodeObservation, str, str]]] = {}
            for pep in pos_obs:
                for link in extract_one_hop_links(pep.before_frame, pep.roles):
                    if link.role in ind_roles:
                        pos_links_by_key.setdefault((link.role, link.via_predicate), []).append(
                            (pep, link.neighbor_id, link.neighbor_type)
                        )
            for (rname, via_pred), entries in sorted(pos_links_by_key.items()):
                for ntype in sorted({nt for _p, _nid, nt in entries}):
                    same_t = [(pep, nid) for pep, nid, nt in entries if nt == ntype]
                    lsets = [extract_entity_literal_facts(pep.before_frame, nid) for pep, nid in same_t]
                    if not lsets:
                        continue
                    pos_invs = set(lsets[0])
                    for ls in lsets[1:]:
                        pos_invs &= ls
                    other_ps = {
                        p
                        for pep, nid, nt in entries
                        if nt != ntype
                        for p, _v in extract_entity_literal_facts(pep.before_frame, nid)
                    }
                    # Only guard neighbor subtypes carrying subtype-specific predicates!
                    sub_specific = [(p, v) for p, v in sorted(pos_invs) if p not in other_ps]
                    for pred, val in sub_specific:
                        ind_guards.append(
                            IndirectRoleFactGuard(
                                role=rname,
                                via_predicate=via_pred,
                                when_neighbor_type=ntype,
                                predicate=pred,
                                value=val,
                            )
                        )

        card = OperatorCard(
            card_id=f"op_{aid}_v1",
            action_id=aid,
            role_variables=tuple(common_roles),
            required_types=tuple((r, unified_types[r]) for r in common_roles),
            required_facts=tuple(sorted(kept_req_facts)),
            conditional_fact_guards=tuple(cond_guards),
            indirect_fact_guards=tuple(ind_guards),
            attribute_guards=tuple(attr_guards),
            inequality_guards=tuple(ineq_guards),
            add_facts=tuple(sorted(unified.add_facts)),
            delete_facts=tuple(sorted(unified.delete_facts)),
            attribute_edits=tuple(sorted(unified.attribute_edits, key=lambda e: (e.role, e.attribute))),
            confidence=1.0,
            support_positive=len(pos_obs),
            support_negative=0,
            version=1,
        )

        # Verify 100% exact positive-demo replay before accepting transferred card
        pos_ok = all(
            card.apply(
                ep.before_frame.to_observed_state_graph(),
                ep.roles,
                mask=ep.before_frame.observation_mask,
            ).delta
            == ep.sparse_delta
            for ep in pos_obs
        )
        if pos_ok:
            transferred[aid] = card
            _c_chk, inst = canonicalize_operator_to_concept(card, concept_id_prefix="xdom")
            if _c_chk.template_fingerprint() == matched.template_fingerprint():
                instantiations[aid] = ConceptInstantiation(
                    concept_id=matched.concept_id,
                    action_id=inst.action_id,
                    roles=inst.roles,
                    preds=inst.preds,
                    types=inst.types,
                    attrs=inst.attrs,
                    lit_vals=inst.lit_vals,
                    attr_vals=inst.attr_vals,
                )

    bits_without = (
        0
        if not transferred
        else compressed_bit_length(
            canonical_json_bytes([c.to_executable_dict() for _k, c in sorted(transferred.items())])
        )
    )
    bits_with = (
        0
        if not transferred
        else compressed_bit_length(
            canonical_json_bytes(
                {
                    "inst": [instantiations[k].to_compact_list() for k in sorted(instantiations)],
                    "res": [
                        c.to_executable_dict()
                        for aid, c in sorted(transferred.items())
                        if aid not in instantiations
                    ],
                }
            )
        )
    )
    return (transferred, instantiations, bits_without, bits_with)


class RDCConsolidator:
    """Mechanism 4: Behavioral-Equivalence Library Consolidation & Cross-Domain Compression."""

    def __init__(
        self,
        *,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
    ) -> None:
        self.beta = float(beta)
        self.gamma = float(gamma)

    def attempt_consolidate_pair(
        self,
        library: dict[str, OperatorCard],
        alias_map: dict[str, str],
        version_store: OperatorVersionStore,
        canonical_action_id: str,
        duplicate_action_id: str,
        equivalence_probes: Sequence[tuple[StateGraph, Mapping[str, str]]],
        regression_examples: Sequence[MDLExample],
    ) -> ConsolidationResult:
        """Merge ``duplicate_action_id`` into ``canonical_action_id`` ONLY if behaviorally equivalent,
        zero regression failures, and ``J_after < J_before``; retires reversibly in ``version_store``."""
        if version_store.is_frozen:
            raise VersionStoreFrozenError(
                "RDCConsolidator cannot consolidate against a frozen OperatorVersionStore"
            )
        if canonical_action_id not in library or duplicate_action_id not in library:
            raise KeyError(
                f"Both {canonical_action_id!r} and {duplicate_action_id!r} must be active in library"
            )
        if canonical_action_id == duplicate_action_id:
            raise ValueError("canonical_action_id and duplicate_action_id must be distinct")

        card_canon = library[canonical_action_id]
        card_dup = library[duplicate_action_id]

        before_mdl = evaluate_library_mdl(
            library,
            regression_examples,
            alias_map=alias_map,
            beta=self.beta,
            gamma=self.gamma,
        )

        # Gate 1: Exact behavioral equivalence across independent probes
        eq_rep = check_behavioral_equivalence(card_canon, card_dup, equivalence_probes)
        if not eq_rep.equivalent:
            return ConsolidationResult(
                canonical_action_id=canonical_action_id,
                duplicate_action_id=duplicate_action_id,
                accepted=False,
                reason="rejected_behavioral_divergence",
                equivalence_report=eq_rep,
                j_before=before_mdl.j_score,
                j_after=before_mdl.j_score,
                library_bits_before=before_mdl.library_bits,
                library_bits_after=before_mdl.library_bits,
                regression_failures_after=before_mdl.verified_failures,
                retired_version=None,
            )

        # Gate 2: Full regression replay + strict MDL J & library bit reduction
        trial_lib = {k: v for k, v in library.items() if k != duplicate_action_id}
        trial_aliases = {
            k: (canonical_action_id if v == duplicate_action_id else v)
            for k, v in alias_map.items()
        }
        trial_aliases[duplicate_action_id] = canonical_action_id
        after_mdl = evaluate_library_mdl(
            trial_lib,
            regression_examples,
            alias_map=trial_aliases,
            beta=self.beta,
            gamma=self.gamma,
        )

        if (
            after_mdl.verified_failures > 0
            or after_mdl.collateral_changes > 0
            or after_mdl.j_score >= before_mdl.j_score
            or after_mdl.library_bits >= before_mdl.library_bits
        ):
            return ConsolidationResult(
                canonical_action_id=canonical_action_id,
                duplicate_action_id=duplicate_action_id,
                accepted=False,
                reason="rejected_mdl_or_regression_gate",
                equivalence_report=eq_rep,
                j_before=before_mdl.j_score,
                j_after=after_mdl.j_score,
                library_bits_before=before_mdl.library_bits,
                library_bits_after=after_mdl.library_bits,
                regression_failures_after=after_mdl.verified_failures,
                retired_version=None,
            )

        # Gate 3: Reversible retirement in OperatorVersionStore (NEVER delete!)
        retired_rec = version_store.retire_active(
            duplicate_action_id,
            reason=f"consolidated_into:{canonical_action_id}",
        )
        del library[duplicate_action_id]
        for k, v in list(alias_map.items()):
            if v == duplicate_action_id:
                alias_map[k] = canonical_action_id
        alias_map[duplicate_action_id] = canonical_action_id

        return ConsolidationResult(
            canonical_action_id=canonical_action_id,
            duplicate_action_id=duplicate_action_id,
            accepted=True,
            reason="consolidated_behaviorally_equivalent_mdl_improved",
            equivalence_report=eq_rep,
            j_before=before_mdl.j_score,
            j_after=after_mdl.j_score,
            library_bits_before=before_mdl.library_bits,
            library_bits_after=after_mdl.library_bits,
            regression_failures_after=after_mdl.verified_failures,
            retired_version=retired_rec.card.version,
        )

    def rollback_consolidation(
        self,
        library: dict[str, OperatorCard],
        alias_map: dict[str, str],
        version_store: OperatorVersionStore,
        action_id: str,
        target_version: int,
        *,
        reason: str = "rollback_consolidation",
    ) -> OperatorCard:
        """Reversibly restore a retired consolidated operator from ``OperatorVersionStore``."""
        restored = version_store.rollback(action_id, target_version, reason=reason)
        alias_map.pop(action_id, None)
        library[action_id] = restored.card
        return restored.card

    def extract_cross_domain_abstractions(
        self,
        domain1_name: str,
        domain1_cards: Mapping[str, OperatorCard],
        domain2_name: str,
        domain2_cards: Mapping[str, OperatorCard],
    ) -> CrossDomainAbstractionReport:
        return extract_cross_domain_concepts(
            domain1_name, domain1_cards, domain2_name, domain2_cards
        )
