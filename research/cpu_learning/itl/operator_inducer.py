"""Domain-agnostic Stage S2 ITL Operator Inducer and BaseS0Solver learner.

STRICT ISOLATION (PDF pp. 6-7, 10-11, 13, 15-17):
- Contains ZERO hardcoded action names, ZERO hardcoded predicate names, ZERO hardcoded
  entity-type names inside the induction algorithm, and NEVER imports ``hidden_actions``,
  ``oracle``, or ``generators``.
- Ingests training demonstrations strictly through S1's append-only ``EpisodeLedger``
  with SHA-256 chain integrity verification, induces versioned ``OperatorCard``s purely
  from observed before/action/after sparse deltas and contrast episodes, and executes
  exclusively from the compact ``OperatorCard`` library at test time (zero episode-store
  lookup during inference).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from research.cpu_learning.evaluation.baselines import BaseS0Solver
from research.cpu_learning.evaluation.metrics import BitBudgetAudit, TaskPrediction
from research.cpu_learning.frames.frame_parser import TypedRoleFrameParser
from research.cpu_learning.frames.state_frame import TypedStateFrame
from research.cpu_learning.itl.delta_extractor import (
    CorpusStatistics,
    ExtractedEpisodeObservation,
    collect_corpus_statistics,
    extract_observations_from_ledger,
    partition_positive_negative,
)
from research.cpu_learning.itl.operator_card import (
    DEFAULT_WORLD_TYPE_TAXONOMY,
    AttributeGuard,
    ConditionalRoleFactGuard,
    IndirectRoleFactGuard,
    OperatorCard,
    RoleInequalityGuard,
    TypeTaxonomy,
)
from research.cpu_learning.itl.role_unifier import (
    extract_direct_lifted_facts,
    extract_entity_literal_facts,
    extract_one_hop_links,
    extract_role_attribute_map,
    intersect_lifted_deltas,
    lift_sparse_delta,
)
from research.cpu_learning.ledger.episode_ledger import EpisodeLedger
from research.cpu_learning.simulator.state_graph import (
    EvaluationTaskPublic,
    StateGraph,
    TrainingEpisode,
    canonical_json_bytes,
    compressed_bit_length,
)


def _infer_inequality_guards(
    role_variables: Sequence[str],
    allowed_types: Mapping[str, frozenset[str]],
    add_facts: frozenset[tuple[str, str, str]],
    delete_facts: frozenset[tuple[str, str, str]],
    pos_obs: Sequence[ExtractedEpisodeObservation],
    neg_obs: Sequence[ExtractedEpisodeObservation],
) -> tuple[RoleInequalityGuard, ...]:
    """Infer non-identity guards (?r1 != ?r2) between type-compatible roles.

    Two purely structural triggers:
    1. Opposite effect roles: ``delete_facts`` and ``add_facts`` share a predicate and
       differ only in ``?r1`` vs ``?r2`` (so ``?r1 == ?r2`` collapses the effect into a
       self-canceling no-op), AND ``r1 != r2`` across all positive demonstrations.
    2. Direct negative contrast: a negative demonstration violates ``r1 != r2`` while
       every positive demonstration satisfies ``r1 != r2`` and types overlap.
    """
    guards: set[RoleInequalityGuard] = set()
    role_set = set(role_variables)

    # Trigger 1: Opposite roles in delete_facts vs add_facts
    for ds, dp, do in delete_facts:
        for as_, ap, ao in add_facts:
            if dp != ap:
                continue
            if ds == as_ and do != ao and do.startswith("?") and ao.startswith("?"):
                r1, r2 = do[1:], ao[1:]
                if (
                    r1 in role_set
                    and r2 in role_set
                    and (allowed_types[r1] & allowed_types[r2])
                    and all(ep.roles[r1] != ep.roles[r2] for ep in pos_obs)
                ):
                    guards.add(RoleInequalityGuard(role_a=r1, role_b=r2))
            elif do == ao and ds != as_ and ds.startswith("?") and as_.startswith("?"):
                r1, r2 = ds[1:], as_[1:]
                if (
                    r1 in role_set
                    and r2 in role_set
                    and (allowed_types[r1] & allowed_types[r2])
                    and all(ep.roles[r1] != ep.roles[r2] for ep in pos_obs)
                ):
                    guards.add(RoleInequalityGuard(role_a=r1, role_b=r2))

    # Trigger 2: Negative demonstration directly exhibiting role equality
    sorted_roles = sorted(role_variables)
    for i in range(len(sorted_roles)):
        for j in range(i + 1, len(sorted_roles)):
            r1, r2 = sorted_roles[i], sorted_roles[j]
            if not (allowed_types[r1] & allowed_types[r2]):
                continue
            if not all(ep.roles[r1] != ep.roles[r2] for ep in pos_obs):
                continue
            if any(ep.roles.get(r1) == ep.roles.get(r2) for ep in neg_obs):
                if not any(
                    (g.role_a == r1 and g.role_b == r2) or (g.role_a == r2 and g.role_b == r1)
                    for g in guards
                ):
                    guards.add(RoleInequalityGuard(role_a=r1, role_b=r2))

    return tuple(sorted(guards, key=lambda g: (g.role_a, g.role_b)))


def _is_attr_functionally_entailed(
    entity_types: frozenset[str],
    attr_name: str,
    expected_val: Any,
    role_literal_preconds: frozenset[tuple[str, str]],
    corpus_stats: CorpusStatistics,
) -> bool:
    """MDL redundancy test: return True iff every corpus entity of ``entity_types``
    satisfying ``role_literal_preconds`` already has ``attr_name == expected_val``
    (zero conditional entropy H(attr | required_facts) == 0)."""
    if not role_literal_preconds:
        return False
    matching_vals: set[Any] = set()
    matched_count = 0
    for et in entity_types:
        for ef, lit_facts in corpus_stats.entity_instances_by_type.get(et, ()):
            if role_literal_preconds.issubset(lit_facts):
                matched_count += 1
                matching_vals.add(ef.get_attr(attr_name))
    return matched_count > 0 and matching_vals == {expected_val}


def _infer_attribute_guards(
    action_id: str,
    role_variables: Sequence[str],
    allowed_types: Mapping[str, frozenset[str]],
    direct_facts: frozenset[tuple[str, str, str]],
    pre_attr_card: OperatorCard,
    pos_obs: Sequence[ExtractedEpisodeObservation],
    neg_obs: Sequence[ExtractedEpisodeObservation],
    corpus_stats: CorpusStatistics,
) -> tuple[AttributeGuard, ...]:
    """Induce selective entity attribute guards via invariance, negative contrast, MDL
    functional-entailment pruning, and cross-operator decorative-attribute filtering."""
    guards: list[AttributeGuard] = []
    pos_attr_maps = [extract_role_attribute_map(ep.before_frame, ep.roles) for ep in pos_obs]

    # Only negatives unexplained by types, direct/conditional facts, and inequality guards
    # provide independent causal contrast evidence for an entity attribute.
    unexplained_negs = [
        nep
        for nep in neg_obs
        if pre_attr_card.check_preconditions(
            nep.before_frame.to_observed_state_graph(),
            nep.roles,
            mask=nep.before_frame.observation_mask,
        )[0]
    ]

    for rname in sorted(role_variables):
        # Attribute keys present on this role across ALL positive demonstrations
        common_keys: set[str] | None = None
        for amap in pos_attr_maps:
            keys = set(amap.get(rname, {}).keys())
            common_keys = keys if common_keys is None else (common_keys & keys)
        if not common_keys:
            continue

        r_types = allowed_types[rname]
        role_lit_preconds = frozenset(
            (pred, obj)
            for subj, pred, obj in direct_facts
            if subj == f"?{rname}" and not obj.startswith("?")
        )

        for attr_name in sorted(common_keys):
            pos_vals = {amap[rname][attr_name] for amap in pos_attr_maps}
            if len(pos_vals) != 1:
                continue  # Varies across positive demos -> decorative
            expected_val = next(iter(pos_vals))

            # Direct independent contrast on an otherwise unexplained negative demo
            directly_contrasted_in_neg = False
            for nep in unexplained_negs:
                neid = nep.roles.get(rname)
                if neid is None:
                    continue
                nef = nep.before_frame.get_entity(neid)
                if nef is not None and nef.attributes_observed:
                    if nef.get_attr(attr_name) != expected_val:
                        directly_contrasted_in_neg = True
                        break

            if not directly_contrasted_in_neg:
                # 1. Must vary across corpus instances of this role's entity type(s)
                corpus_vals = corpus_stats.attr_domain_for_types(r_types, attr_name)
                if len(corpus_vals) < 2:
                    continue  # Constant background attribute across corpus (zero entropy)
                # 2. Prune derived summary attributes 100% functionally entailed by fact preconditions
                if _is_attr_functionally_entailed(
                    r_types, attr_name, expected_val, role_lit_preconds, corpus_stats
                ):
                    continue
                # 3. Cross-operator decorative filter: an unedited attribute that varies freely
                #    across other positive role slots of the same entity type(s) in the ledger
                #    is empirically non-causal (prevents small-sample coincidence overfitting).
                if not corpus_stats.is_attr_edited_on_types(r_types, attr_name):
                    other_varying = corpus_stats.other_varying_pos_role_slots(
                        r_types,
                        attr_name,
                        exclude_action=action_id,
                        exclude_role=rname,
                    )
                    if other_varying:
                        continue

            guards.append(
                AttributeGuard(role=rname, attribute=attr_name, expected_value=expected_val)
            )

    return tuple(guards)


def _infer_conditional_role_fact_guards(
    role_variables: Sequence[str],
    allowed_types: Mapping[str, frozenset[str]],
    direct_facts: frozenset[tuple[str, str, str]],
    pos_obs: Sequence[ExtractedEpisodeObservation],
) -> tuple[ConditionalRoleFactGuard, ...]:
    """For any multi-type role, induce subtype-specific invariant literal facts."""
    guards: list[ConditionalRoleFactGuard] = []
    for rname in sorted(role_variables):
        r_types = allowed_types[rname]
        if len(r_types) <= 1:
            continue

        uncond_preds = {pred for subj, pred, _obj in direct_facts if subj == f"?{rname}"}

        # Predicates observed on each subtype in positive demos
        preds_by_subtype: dict[str, set[str]] = {}
        invariants_by_subtype: dict[str, set[tuple[str, str]]] = {}

        for subtype in sorted(r_types):
            sub_eps = [
                ep
                for ep in pos_obs
                if ep.before_frame.entity_type_of(ep.roles[rname]) == subtype
            ]
            if not sub_eps:
                continue
            lit_sets = [extract_entity_literal_facts(ep.before_frame, ep.roles[rname]) for ep in sub_eps]
            inter = set(lit_sets[0])
            for ls in lit_sets[1:]:
                inter &= ls
            invariants_by_subtype[subtype] = inter
            preds_by_subtype[subtype] = {p for ls in lit_sets for p, _v in ls}

        for subtype, inv_facts in sorted(invariants_by_subtype.items()):
            other_common_preds: set[str] | None = None
            for other_t, other_ps in preds_by_subtype.items():
                if other_t == subtype:
                    continue
                other_common_preds = (
                    set(other_ps) if other_common_preds is None else (other_common_preds & other_ps)
                )
            shared_by_others = other_common_preds or set()
            for pred, val in sorted(inv_facts):
                if pred in uncond_preds or pred in shared_by_others:
                    continue
                guards.append(
                    ConditionalRoleFactGuard(
                        role=rname,
                        when_entity_type=subtype,
                        predicate=pred,
                        value=val,
                    )
                )

    return tuple(guards)


def _infer_indirect_contrast_guards(
    provisional_card: OperatorCard,
    pos_obs: Sequence[ExtractedEpisodeObservation],
    neg_obs: Sequence[ExtractedEpisodeObservation],
    corpus_stats: CorpusStatistics,
) -> tuple[IndirectRoleFactGuard, ...]:
    """Induce 1-hop relational guards triggered strictly by unexplained zero-delta negatives.

    When a negative demonstration satisfies all direct/conditional/attribute/inequality
    preconditions on ``provisional_card``, inspect 1-hop entity links from every role
    entity and discover subtype-specific neighbor invariants that separate positive
    demonstrations from the unexplained negative(s).
    """
    unexplained_negs: list[ExtractedEpisodeObservation] = []
    for nep in neg_obs:
        ok, _reason, _unk = provisional_card.check_preconditions(
            nep.before_frame.to_observed_state_graph(),
            nep.roles,
            mask=nep.before_frame.observation_mask,
        )
        if ok:
            unexplained_negs.append(nep)

    if not unexplained_negs:
        return ()

    # Pre-index positive 1-hop links by (role, via_predicate)
    pos_links_by_key: dict[tuple[str, str], list[tuple[ExtractedEpisodeObservation, str, str]]] = {}
    for pep in pos_obs:
        for link in extract_one_hop_links(pep.before_frame, pep.roles):
            pos_links_by_key.setdefault((link.role, link.via_predicate), []).append(
                (pep, link.neighbor_id, link.neighbor_type)
            )

    discovered: set[IndirectRoleFactGuard] = set()

    for nep in unexplained_negs:
        for nlink in extract_one_hop_links(nep.before_frame, nep.roles):
            key = (nlink.role, nlink.via_predicate)
            all_pos_for_key = pos_links_by_key.get(key, [])
            if not all_pos_for_key:
                continue

            same_type_pos = [
                (pep, z_id) for pep, z_id, z_t in all_pos_for_key if z_t == nlink.neighbor_type
            ]
            if not same_type_pos:
                continue

            # Intersect literal facts on positive neighbors of the same type
            pos_lit_sets = [
                extract_entity_literal_facts(pep.before_frame, z_id) for pep, z_id in same_type_pos
            ]
            pos_invariants = set(pos_lit_sets[0])
            for ls in pos_lit_sets[1:]:
                pos_invariants &= ls

            # Identify predicates shared with OTHER neighbor types along the same relation
            other_type_preds: set[str] = set()
            for pep, z_id, z_t in all_pos_for_key:
                if z_t != nlink.neighbor_type:
                    for pred, _val in extract_entity_literal_facts(pep.before_frame, z_id):
                        other_type_preds.add(pred)

            candidate_invariants = {
                (p, v) for p, v in pos_invariants if p not in other_type_preds
            } or pos_invariants

            neg_lit_facts = extract_entity_literal_facts(nep.before_frame, nlink.neighbor_id)
            violated = [pv for pv in candidate_invariants if pv not in neg_lit_facts]
            if not violated:
                continue  # This 1-hop link does NOT explain the negative demonstration

            # Link explains nep: retain violated predicates AND subtype-specific varying invariants
            for pred, val in sorted(candidate_invariants):
                if (pred, val) in violated or len(
                    corpus_stats.literal_fact_domain_for_type(nlink.neighbor_type, pred)
                ) >= 2:
                    discovered.add(
                        IndirectRoleFactGuard(
                            role=nlink.role,
                            via_predicate=nlink.via_predicate,
                            when_neighbor_type=nlink.neighbor_type,
                            predicate=pred,
                            value=val,
                        )
                    )

    return tuple(
        sorted(
            discovered,
            key=lambda g: (g.role, g.via_predicate, g.when_neighbor_type, g.predicate, g.value),
        )
    )


def induce_operator_card(
    action_id: str,
    observations: Sequence[ExtractedEpisodeObservation],
    corpus_stats: CorpusStatistics,
    *,
    use_negative_contrasts: bool = True,
    version: int = 1,
    parent_version: int | None = None,
) -> OperatorCard | None:
    """Induce a single executable, versioned ``OperatorCard`` purely from public demonstrations."""
    pos_obs, neg_obs = partition_positive_negative(observations)
    if not pos_obs:
        return None

    # 1. Common role variables across positive demonstrations
    role_sets = [set(ep.roles.keys()) for ep in pos_obs]
    common_roles = sorted(set.intersection(*role_sets))
    if not common_roles:
        return None

    # 2. Lift and anti-unify observed sparse deltas across positive demonstrations
    lifted_deltas = [
        lift_sparse_delta(ep.sparse_delta, ep.roles, ep.before_frame, ep.after_frame)
        for ep in pos_obs
    ]
    unified_delta = intersect_lifted_deltas(lifted_deltas)
    if unified_delta.is_empty:
        return None

    # 3. Allowed entity types per role
    allowed_types: dict[str, frozenset[str]] = {}
    for rname in common_roles:
        types = {
            t
            for ep in pos_obs
            for t in (ep.before_frame.entity_type_of(ep.roles[rname]),)
            if t is not None and t != "unknown"
        }
        allowed_types[rname] = frozenset(types)

    # 4. Direct lifted fact preconditions (intersection across all positive demonstrations)
    direct_fact_sets = [extract_direct_lifted_facts(ep.before_frame, ep.roles) for ep in pos_obs]
    direct_facts = frozenset.intersection(*direct_fact_sets) if direct_fact_sets else frozenset()

    # 5. Type-conditional role fact guards (for multi-type roles)
    cond_guards = _infer_conditional_role_fact_guards(
        common_roles, allowed_types, direct_facts, pos_obs
    )

    # 6. Role inequality guards (?r1 != ?r2)
    ineq_guards = _infer_inequality_guards(
        common_roles,
        allowed_types,
        unified_delta.add_facts,
        unified_delta.delete_facts,
        pos_obs,
        neg_obs if use_negative_contrasts else (),
    )

    all_ids = tuple(ep.episode_id for ep in observations)
    all_hashes = tuple(ep.record_sha256 for ep in observations)

    pre_attr_card = OperatorCard(
        card_id=f"op_{action_id}_v{version}",
        action_id=action_id,
        role_variables=tuple(common_roles),
        required_types=tuple((r, allowed_types[r]) for r in common_roles),
        required_facts=tuple(sorted(direct_facts)),
        conditional_fact_guards=cond_guards,
        indirect_fact_guards=(),
        attribute_guards=(),
        inequality_guards=ineq_guards,
        add_facts=tuple(sorted(unified_delta.add_facts)),
        delete_facts=tuple(sorted(unified_delta.delete_facts)),
        attribute_edits=tuple(sorted(unified_delta.attribute_edits, key=lambda e: (e.role, e.attribute))),
        confidence=1.0,
        support_positive=len(pos_obs),
        support_negative=len(neg_obs),
        version=version,
        parent_version=parent_version,
        provenance_episode_ids=all_ids,
        provenance_record_hashes=all_hashes,
    )

    # 7. Role attribute guards with MDL functional-entailment + cross-operator decorative pruning
    attr_guards = _infer_attribute_guards(
        action_id,
        common_roles,
        allowed_types,
        direct_facts,
        pre_attr_card,
        pos_obs,
        neg_obs if use_negative_contrasts else (),
        corpus_stats,
    )

    provisional = OperatorCard(
        card_id=pre_attr_card.card_id,
        action_id=pre_attr_card.action_id,
        role_variables=pre_attr_card.role_variables,
        required_types=pre_attr_card.required_types,
        required_facts=pre_attr_card.required_facts,
        conditional_fact_guards=pre_attr_card.conditional_fact_guards,
        indirect_fact_guards=(),
        attribute_guards=attr_guards,
        inequality_guards=pre_attr_card.inequality_guards,
        add_facts=pre_attr_card.add_facts,
        delete_facts=pre_attr_card.delete_facts,
        attribute_edits=pre_attr_card.attribute_edits,
        confidence=1.0,
        support_positive=len(pos_obs),
        support_negative=len(neg_obs),
        version=version,
        parent_version=parent_version,
        provenance_episode_ids=all_ids,
        provenance_record_hashes=all_hashes,
    )

    # 8. Indirect 1-hop contrast guards triggered strictly by unexplained negatives
    if use_negative_contrasts and neg_obs:
        indirect_guards = _infer_indirect_contrast_guards(
            provisional, pos_obs, neg_obs, corpus_stats
        )
    else:
        indirect_guards = ()

    final_card = OperatorCard(
        card_id=provisional.card_id,
        action_id=provisional.action_id,
        role_variables=provisional.role_variables,
        required_types=provisional.required_types,
        required_facts=provisional.required_facts,
        conditional_fact_guards=provisional.conditional_fact_guards,
        indirect_fact_guards=indirect_guards,
        attribute_guards=provisional.attribute_guards,
        inequality_guards=provisional.inequality_guards,
        add_facts=provisional.add_facts,
        delete_facts=provisional.delete_facts,
        attribute_edits=provisional.attribute_edits,
        confidence=provisional.confidence,
        support_positive=provisional.support_positive,
        support_negative=provisional.support_negative,
        version=provisional.version,
        parent_version=provisional.parent_version,
        provenance_episode_ids=provisional.provenance_episode_ids,
        provenance_record_hashes=provisional.provenance_record_hashes,
    )

    # Empirical training consistency score (confidence in [0, 1])
    total_checked = len(pos_obs) + len(neg_obs)
    matched = 0
    for ep in pos_obs:
        res = final_card.apply(
            ep.before_frame.to_observed_state_graph(),
            ep.roles,
            mask=ep.before_frame.observation_mask,
        )
        if res.applied and res.delta == ep.sparse_delta:
            matched += 1
    for ep in neg_obs:
        res = final_card.apply(
            ep.before_frame.to_observed_state_graph(),
            ep.roles,
            mask=ep.before_frame.observation_mask,
        )
        if not res.applied and not res.abstained:
            matched += 1
    conf = matched / float(total_checked) if total_checked > 0 else 1.0

    return OperatorCard(
        card_id=final_card.card_id,
        action_id=final_card.action_id,
        role_variables=final_card.role_variables,
        required_types=final_card.required_types,
        required_facts=final_card.required_facts,
        conditional_fact_guards=final_card.conditional_fact_guards,
        indirect_fact_guards=final_card.indirect_fact_guards,
        attribute_guards=final_card.attribute_guards,
        inequality_guards=final_card.inequality_guards,
        add_facts=final_card.add_facts,
        delete_facts=final_card.delete_facts,
        attribute_edits=final_card.attribute_edits,
        confidence=conf,
        support_positive=final_card.support_positive,
        support_negative=final_card.support_negative,
        version=final_card.version,
        parent_version=final_card.parent_version,
        provenance_episode_ids=final_card.provenance_episode_ids,
        provenance_record_hashes=final_card.provenance_record_hashes,
    )


def induce_operator_library_from_ledger(
    ledger: EpisodeLedger,
    *,
    split_id: str | None = None,
    seed: int | None = None,
    use_negative_contrasts: bool = True,
    caller_tag: str = "itl_operator_inducer",
) -> dict[str, OperatorCard]:
    """Induce a full dictionary of ``action_id -> OperatorCard`` strictly from an S1 EpisodeLedger."""
    grouped = extract_observations_from_ledger(
        ledger, split_id=split_id, seed=seed, caller_tag=caller_tag
    )
    corpus_stats = collect_corpus_statistics(grouped)
    library: dict[str, OperatorCard] = {}
    for action_id in sorted(grouped):
        card = induce_operator_card(
            action_id,
            grouped[action_id],
            corpus_stats,
            use_negative_contrasts=use_negative_contrasts,
        )
        if card is not None:
            library[action_id] = card
    return library


class ITLOperatorLearner(BaseS0Solver):
    """Stage S2 CPU-native Inductive Transition Learner (ITL).

    - Ingests training episodes into S1's append-only SQLite ``EpisodeLedger`` and verifies
      SHA-256 chain integrity before inducing reusable ``OperatorCard``s.
    - Executes unseen multi-step compositions strictly via induced ``OperatorCard``s with
      ZERO test-time episode-store retrieval (``memory_accesses == 0``).
    - Supports three controlled modes:
      1. ``itl_operator_learner`` (primary S2): typed action IDs + full contrast induction.
      2. ``itl_operator_learner_no_contrast`` (ablation): disables unexplained negative contrast.
      3. ``itl_operator_learner_raw_utf8_e2e`` (S1+S2 end-to-end): ignores typed test action
         roles and parses raw UTF-8 instruction bytes through S1's ``TypedRoleFrameParser``.
    """

    category = "itl_candidate_learner"

    def __init__(
        self,
        seed: int = 0,
        *,
        use_negative_contrasts: bool = True,
        raw_utf8_parse_mode: bool = False,
        taxonomy: TypeTaxonomy | None = DEFAULT_WORLD_TYPE_TAXONOMY,
        abstain_on_unknown_guard: bool = True,
    ) -> None:
        super().__init__(seed=seed)
        self.use_negative_contrasts = bool(use_negative_contrasts)
        self.raw_utf8_parse_mode = bool(raw_utf8_parse_mode)
        self.taxonomy = taxonomy if taxonomy is not None else TypeTaxonomy()
        self.abstain_on_unknown_guard = bool(abstain_on_unknown_guard)
        self._parser = TypedRoleFrameParser(use_state_types=True) if self.raw_utf8_parse_mode else None
        self._operators: dict[str, OperatorCard] = {}
        self._ledger: EpisodeLedger | None = None
        self._ledger_chain_head: str = ""

        if self.raw_utf8_parse_mode:
            self.name = "itl_operator_learner_raw_utf8_e2e"
            self.category = "s1_s2_end_to_end_learner"
        elif not self.use_negative_contrasts:
            self.name = "itl_operator_learner_no_contrast"
            self.category = "itl_ablation"
        else:
            self.name = "itl_operator_learner"
            self.category = "itl_candidate_learner"

    @property
    def operators(self) -> dict[str, OperatorCard]:
        return dict(self._operators)

    @property
    def ledger_chain_head(self) -> str:
        return self._ledger_chain_head

    def freeze(self) -> None:
        super().freeze()
        if self._ledger is not None:
            self._ledger.freeze()

    def fit_from_ledger(
        self,
        ledger: EpisodeLedger,
        *,
        split_id: str | None = None,
        seed: int | None = None,
    ) -> dict[str, OperatorCard]:
        """Induce ``OperatorCard``s directly from an existing S1 ``EpisodeLedger``."""
        self._check_not_frozen()
        integrity = ledger.verify_integrity(strict=True)
        self._ledger_chain_head = integrity.chain_head_sha256
        self._operators = induce_operator_library_from_ledger(
            ledger,
            split_id=split_id,
            seed=seed,
            use_negative_contrasts=self.use_negative_contrasts,
            caller_tag=f"{self.name}:fit",
        )
        return self.operators

    def fit(self, episodes: Sequence[TrainingEpisode]) -> None:
        """Ingest public TrainingEpisodes into an S1 EpisodeLedger and induce OperatorCards."""
        self._check_not_frozen()
        if self._ledger is not None:
            self._ledger.close()
        self._ledger = EpisodeLedger(":memory:")
        for ep in episodes:
            self._ledger.ingest_training_episode(ep, use_typed_action=True)
        self.fit_from_ledger(self._ledger, split_id=None)
        # Reset ledger audit log after training so any accidental test-time ledger read is caught
        self._ledger.reset_audit_log()

    def predict(self, task: EvaluationTaskPublic) -> TaskPrediction:
        """Execute induced OperatorCards sequentially across all steps (zero test-time retrieval)."""
        cur = task.initial_state.copy()
        step_preds: list[StateGraph | None] = []
        step_reasons: list[str] = []

        for step_idx, act in enumerate(task.actions):
            if self.raw_utf8_parse_mode:
                assert self._parser is not None
                cur_frame = TypedStateFrame.from_state_graph(cur, mask=task.observation_mask)
                parsed = self._parser.parse(act.raw_instruction_bytes, cur_frame)
                unamb = parsed.unambiguous_frame
                if unamb is None:
                    return TaskPrediction(
                        predicted_final_state=None,
                        abstained=True,
                        retrieval_hits=0,
                        memory_accesses=0,
                        step_predictions=tuple(step_preds),
                        metadata={
                            "abstain_step": step_idx,
                            "reason": "raw_utf8_parse_ambiguous_or_abstained",
                            "parser_abstention": parsed.abstention_reason,
                        },
                    )
                action_id = unamb.action_id
                roles = unamb.role_dict()
            else:
                action_id = act.action_id
                roles = act.role_dict()

            card = self._operators.get(action_id)
            if card is None:
                return TaskPrediction(
                    predicted_final_state=None,
                    abstained=True,
                    retrieval_hits=0,
                    memory_accesses=0,
                    step_predictions=tuple(step_preds),
                    metadata={"abstain_step": step_idx, "reason": f"unknown_operator:{action_id}"},
                )

            res = card.apply(
                cur,
                roles,
                mask=task.observation_mask,
                abstain_on_unknown=self.abstain_on_unknown_guard,
            )
            if res.abstained or res.after_state is None:
                return TaskPrediction(
                    predicted_final_state=None,
                    abstained=True,
                    retrieval_hits=0,
                    memory_accesses=0,
                    step_predictions=tuple(step_preds),
                    metadata={"abstain_step": step_idx, "reason": res.reason},
                )
            cur = res.after_state
            step_preds.append(cur)
            step_reasons.append(res.reason)

        test_ledger_reads = (
            self._ledger.audit_summary()["total_accesses"] if self._ledger is not None else 0
        )
        return TaskPrediction(
            predicted_final_state=cur,
            abstained=False,
            retrieval_hits=0,
            memory_accesses=test_ledger_reads,
            step_predictions=tuple(step_preds),
            metadata={"step_reasons": step_reasons},
        )

    def recovered_operator_schemas(self) -> dict[str, dict[str, Any]]:
        """Export induced OperatorCards in evaluator-compatible schema format."""
        return {
            aid: card.to_schema_dict(taxonomy=self.taxonomy)
            for aid, card in sorted(self._operators.items())
        }

    def bit_budget(self, raw_training_bits: int, compressed_training_bits: int) -> BitBudgetAudit:
        """Report exact MDL footprint of the induced OperatorCard library (zero episode store at inference)."""
        if not self._operators:
            return BitBudgetAudit(
                raw_training_data_bits=raw_training_bits,
                compressed_training_data_bits=compressed_training_bits,
                episode_store_bits=0,
                active_bundle_bits=0,
                operator_library_bits=0,
                compressed_model_mdl_bits=0,
                compression_ratio_vs_raw=0.0,
            )
        exec_payload = canonical_json_bytes(
            {aid: card.to_executable_dict() for aid, card in sorted(self._operators.items())}
        )
        op_raw_bits = len(exec_payload) * 8
        mdl_bits = compressed_bit_length(exec_payload)
        return BitBudgetAudit(
            raw_training_data_bits=raw_training_bits,
            compressed_training_data_bits=compressed_training_bits,
            episode_store_bits=0,
            active_bundle_bits=0,
            operator_library_bits=op_raw_bits,
            compressed_model_mdl_bits=mdl_bits,
            compression_ratio_vs_raw=mdl_bits / max(1, raw_training_bits),
        )
