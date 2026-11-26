"""Stage S4 CPU-native 10,240-bit BSC hypervector encoders for SHARD proposal generation.

STRICT ISOLATION (Stage S4):
- Reuses Mirror 7's pure-CPU 10,240-bit Binary Spatter Code primitives from
  ``mirror_crystal.quotient_egraph_resonator`` (ZERO GPU, ZERO backprop, ZERO floats).
- Contains ZERO hardcoded action, predicate, attribute, or entity-type literals.
- NEVER imports ``hidden_actions``, ``oracle``, or ``generators``.
- Aligns ``encode_sparse_delta_hv`` and ``encode_operator_card_hv`` into the same
  entity-ID-invariant 10,240-bit structural subspace so observed transitions and goal
  residuals rank verified ``OperatorCard``s directly via POPCNT Hamming similarity.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from mirror_crystal.quotient_egraph_resonator import (
    D_UNIFIED,
    UNIFIED_MASK,
    UNIFIED_ODD_MASK,
    BitSlicedBundle10240,
    bind10240,
    hamming_sim10240,
    hv10240_of,
    permute10240,
)
from research.cpu_learning.itl.operator_card import OperatorCard, canonical_value_str
from research.cpu_learning.simulator.state_graph import SparseDelta, StateGraph

SHARD_ENCODER_SCHEMA_VERSION: str = "s4-shard-bsc10240-v1"

# Coprime role-shift strides in [1, D_UNIFIED) for orthogonal structural subspaces
SHIFT_ADD_FACT: int = 3
SHIFT_DEL_FACT: int = 11
SHIFT_ATTR_EDIT: int = 23
SHIFT_PRED_ADD: int = 37
SHIFT_PRED_DEL: int = 53
SHIFT_ATTR_KEY: int = 71
SHIFT_STATE_ENTITY: int = 89
SHIFT_STATE_FACT: int = 107
SHIFT_ROLE_SLOT: int = 131
SHIFT_DELTA_SLOT: int = 157
DEPTH_PERM_STRIDE: int = 17

NOOP_DELTA_HV: int = hv10240_of("shard:delta:noop_v1")
EMPTY_STATE_HV: int = hv10240_of("shard:state:empty_v1")
EMPTY_ROLES_HV: int = hv10240_of("shard:roles:empty_v1")
EMPTY_SEQ_HV: int = hv10240_of("shard:seq:empty_v1")

_HV_KIND_ADD = hv10240_of("shard:kind:add_fact")
_HV_KIND_DEL = hv10240_of("shard:kind:del_fact")
_HV_KIND_EDIT = hv10240_of("shard:kind:attr_edit")
_HV_OBJ_ENTITY = hv10240_of("shard:obj:ENTITY_REF")
_HV_SUBJ_TAG = hv10240_of("shard:pos:subj")
_HV_OBJ_TAG = hv10240_of("shard:pos:obj")
_HV_STATE_FACT_TAG = hv10240_of("shard:state:fact")
_HV_STATE_ATTR_TAG = hv10240_of("shard:state:attr")
_HV_STATE_TYPE_TAG = hv10240_of("shard:state:etype")


def _majority_bundle10240(
    weighted_hvs: Sequence[tuple[int, int]],
    *,
    empty_hv: int,
    bits: int = 10,
) -> int:
    """Exact majority vote via Mirror 7's ``BitSlicedBundle10240`` (handles N=1 and even weights).

    ``BitSlicedBundle10240.binary()`` tests strict ``count > (W // 2) + 1`` for odd ``W``.
    Scaling each input weight by 2 and adding a +1 bias on ``UNIFIED_MASK`` (plus a
    balanced weight-2 tie-breaker on ``UNIFIED_ODD_MASK`` when total weight is even)
    yields exact majority ``sum(w_i * bit_i) > total_w / 2`` for every ``total_w >= 1``.
    """
    pos_items = [(hv, w) for hv, w in weighted_hvs if w > 0]
    if not pos_items:
        return empty_hv
    if len(pos_items) == 1:
        return pos_items[0][0] & UNIFIED_MASK

    total_w = sum(w for _hv, w in pos_items)
    needed_bits = max(int(bits), (total_w * 2 + 3).bit_length())
    bd = BitSlicedBundle10240(bits=needed_bits)
    for hv, w in pos_items:
        bd.add(hv, w * 2)
    if total_w % 2 == 0:
        bd.add(UNIFIED_ODD_MASK, 2)
    bd.add(UNIFIED_MASK, 1)
    return bd.binary()


def _fact_effect_atoms(
    is_add: bool,
    pred: str,
    subj_types: Sequence[str],
    obj_types: Sequence[str] | None,
    obj_literal: str | None,
) -> list[tuple[int, int]]:
    """Build weighted 10,240-bit atoms for an added or deleted relational fact."""
    kind_hv = _HV_KIND_ADD if is_add else _HV_KIND_DEL
    shift_fact = SHIFT_ADD_FACT if is_add else SHIFT_DEL_FACT
    shift_pred = SHIFT_PRED_ADD if is_add else SHIFT_PRED_DEL
    pred_hv = hv10240_of(f"shard:pred:{pred}")

    atoms: list[tuple[int, int]] = []
    # 1. Predicate-polarity anchor (strong alignment even across multi-step residuals)
    atoms.append((permute10240(bind10240(kind_hv, pred_hv), shift_pred), 3))

    if obj_literal is not None:
        lit_hv = hv10240_of(f"shard:lit:{obj_literal}")
        # Predicate + literal polarity anchor
        atoms.append((permute10240(bind10240(pred_hv, lit_hv), shift_fact), 3))
        for st in subj_types:
            st_hv = hv10240_of(f"shard:etype:{st}")
            full_atom = bind10240(
                bind10240(kind_hv, pred_hv),
                bind10240(permute10240(st_hv, 1), permute10240(lit_hv, 2)),
            )
            atoms.append((permute10240(full_atom, shift_fact), 2))
    else:
        # Entity-valued relational fact: coarse ENTITY anchor + specific typed atoms
        for st in subj_types:
            st_hv = hv10240_of(f"shard:etype:{st}")
            coarse = bind10240(
                bind10240(kind_hv, pred_hv),
                bind10240(permute10240(st_hv, 1), _HV_OBJ_ENTITY),
            )
            atoms.append((permute10240(coarse, shift_fact), 3))
            if obj_types:
                for ot in obj_types:
                    ot_hv = hv10240_of(f"shard:etype:{ot}")
                    spec = bind10240(
                        bind10240(kind_hv, pred_hv),
                        bind10240(permute10240(st_hv, 1), permute10240(ot_hv, 2)),
                    )
                    atoms.append((permute10240(spec, shift_fact), 1))
    return atoms


def _attr_edit_atoms(
    attr_name: str,
    entity_types: Sequence[str],
    new_val_str: str,
) -> list[tuple[int, int]]:
    """Build weighted 10,240-bit atoms for an entity attribute mutation."""
    attr_hv = hv10240_of(f"shard:attr:{attr_name}")
    val_hv = hv10240_of(f"shard:val:{new_val_str}")
    key_atom = permute10240(bind10240(attr_hv, val_hv), SHIFT_ATTR_KEY)
    atoms: list[tuple[int, int]] = [(key_atom, 3)]
    for et in entity_types:
        et_hv = hv10240_of(f"shard:etype:{et}")
        spec = bind10240(
            _HV_KIND_EDIT,
            bind10240(permute10240(et_hv, 1), bind10240(attr_hv, permute10240(val_hv, 2))),
        )
        atoms.append((permute10240(spec, SHIFT_ATTR_EDIT), 2))
    return atoms


def encode_typed_state_hv(state: StateGraph) -> int:
    """Encode a relational ``StateGraph`` into a structural 10,240-bit BSC hypervector."""
    if not state.entities and not state.facts:
        return EMPTY_STATE_HV
    ent_map = state.entity_map()
    atoms: list[tuple[int, int]] = []

    for _eid, ent in state.entities:
        et_hv = hv10240_of(f"shard:etype:{ent.entity_type}")
        atoms.append((permute10240(bind10240(_HV_STATE_TYPE_TAG, et_hv), SHIFT_STATE_ENTITY), 1))
        for ak, av in ent.attributes:
            ak_hv = hv10240_of(f"shard:attr:{ak}")
            av_hv = hv10240_of(f"shard:val:{canonical_value_str(av)}")
            atom = bind10240(
                _HV_STATE_ATTR_TAG,
                bind10240(permute10240(et_hv, 1), bind10240(ak_hv, permute10240(av_hv, 2))),
            )
            atoms.append((permute10240(atom, SHIFT_STATE_ENTITY), 1))

    for subj, pred, obj in state.facts:
        s_ent = ent_map.get(subj)
        s_type = s_ent.entity_type if s_ent is not None else "unknown"
        o_ent = ent_map.get(obj)
        st_hv = hv10240_of(f"shard:etype:{s_type}")
        pred_hv = hv10240_of(f"shard:pred:{pred}")
        if o_ent is not None:
            o_hv = hv10240_of(f"shard:etype:{o_ent.entity_type}")
        else:
            o_hv = hv10240_of(f"shard:lit:{obj}")
        fact_atom = bind10240(
            _HV_STATE_FACT_TAG,
            bind10240(pred_hv, bind10240(permute10240(st_hv, 1), permute10240(o_hv, 2))),
        )
        atoms.append((permute10240(fact_atom, SHIFT_STATE_FACT), 1))

    return _majority_bundle10240(atoms, empty_hv=EMPTY_STATE_HV, bits=12)


def encode_sparse_delta_hv(delta: SparseDelta, before_state: StateGraph) -> int:
    """Encode a concrete ``SparseDelta`` into the entity-ID-invariant 10,240-bit effect subspace.

    Aligned directly with ``encode_operator_card_hv`` so single-step transitions and
    multi-step goal residuals match relevant verified ``OperatorCard``s via POPCNT Hamming
    similarity even on 100% unseen entity vocabularies.
    """
    if delta.is_noop:
        return NOOP_DELTA_HV
    ent_map = before_state.entity_map()
    atoms: list[tuple[int, int]] = []

    for subj, pred, obj in delta.added_facts:
        s_ent = ent_map.get(subj)
        s_types = (s_ent.entity_type,) if s_ent is not None else ("unknown",)
        o_ent = ent_map.get(obj)
        if o_ent is not None:
            atoms.extend(_fact_effect_atoms(True, pred, s_types, (o_ent.entity_type,), None))
        else:
            atoms.extend(_fact_effect_atoms(True, pred, s_types, None, obj))

    for subj, pred, obj in delta.deleted_facts:
        s_ent = ent_map.get(subj)
        s_types = (s_ent.entity_type,) if s_ent is not None else ("unknown",)
        o_ent = ent_map.get(obj)
        if o_ent is not None:
            atoms.extend(_fact_effect_atoms(False, pred, s_types, (o_ent.entity_type,), None))
        else:
            atoms.extend(_fact_effect_atoms(False, pred, s_types, None, obj))

    for eid, attr_name, _old_v, new_v in delta.attribute_edits:
        ent = ent_map.get(eid)
        e_types = (ent.entity_type,) if ent is not None else ("unknown",)
        atoms.extend(_attr_edit_atoms(attr_name, e_types, canonical_value_str(new_v)))

    return _majority_bundle10240(atoms, empty_hv=NOOP_DELTA_HV, bits=10)


def encode_operator_card_hv(card: OperatorCard) -> int:
    """Encode an induced ``OperatorCard``'s lifted effects & role types into the shared 10,240-bit subspace."""
    if not card.add_facts and not card.delete_facts and not card.attribute_edits:
        return NOOP_DELTA_HV
    allowed_map = card.allowed_types_map()
    atoms: list[tuple[int, int]] = []

    def _types_for_term(term: str) -> tuple[str, ...]:
        if term.startswith("?"):
            ts = sorted(allowed_map.get(term[1:], frozenset()))
            return tuple(ts) if ts else ("unknown",)
        return ("unknown",)

    for s_t, pred, o_t in card.add_facts:
        s_types = _types_for_term(s_t)
        if o_t.startswith("?"):
            atoms.extend(_fact_effect_atoms(True, pred, s_types, _types_for_term(o_t), None))
        else:
            atoms.extend(_fact_effect_atoms(True, pred, s_types, None, o_t))

    for s_t, pred, o_t in card.delete_facts:
        s_types = _types_for_term(s_t)
        if o_t.startswith("?"):
            atoms.extend(_fact_effect_atoms(False, pred, s_types, _types_for_term(o_t), None))
        else:
            atoms.extend(_fact_effect_atoms(False, pred, s_types, None, o_t))

    for edit in card.attribute_edits:
        e_types = tuple(sorted(allowed_map.get(edit.role, frozenset()))) or ("unknown",)
        atoms.extend(_attr_edit_atoms(edit.attribute, e_types, canonical_value_str(edit.new_value)))

    return _majority_bundle10240(atoms, empty_hv=NOOP_DELTA_HV, bits=10)


def entity_symbol_hv(entity_id: str) -> int:
    """Canonical 10,240-bit symbol HV for a concrete world entity ID."""
    return hv10240_of(f"shard:ent_id:{entity_id}")


def role_key_hv(role_name: str) -> int:
    """Canonical 10,240-bit role-slot key HV for binding/unbinding role assignments."""
    return permute10240(hv10240_of(f"shard:role_slot:{role_name}"), SHIFT_ROLE_SLOT)


def effect_slot_key_hv(slot_kind: str, predicate_or_attr: str) -> int:
    """Canonical 10,240-bit key HV for a structural effect position (e.g., add_subj, del_obj, edit_ent)."""
    return permute10240(
        hv10240_of(f"shard:eff_slot:{slot_kind}:{predicate_or_attr}"),
        SHIFT_DELTA_SLOT,
    )


def encode_role_bindings_hv(roles: Mapping[str, str], before_state: StateGraph) -> int:
    """Encode concrete ``role -> entity_id`` bindings into a single 10,240-bit superposition."""
    if not roles:
        return EMPTY_ROLES_HV
    atoms: list[tuple[int, int]] = []
    for rname, eid in sorted(roles.items()):
        atoms.append((bind10240(role_key_hv(rname), entity_symbol_hv(eid)), 2))
        ent = before_state.get_entity(eid)
        if ent is not None:
            atoms.append(
                (
                    bind10240(
                        permute10240(role_key_hv(rname), 1),
                        hv10240_of(f"shard:etype:{ent.entity_type}"),
                    ),
                    1,
                )
            )
    return _majority_bundle10240(atoms, empty_hv=EMPTY_ROLES_HV, bits=8)


def encode_delta_entity_slots_hv(delta: SparseDelta, before_state: StateGraph) -> int:
    """Bind every entity appearing in ``delta`` to its structural effect-slot key in 10,240-bit BSC space."""
    if delta.is_noop:
        return NOOP_DELTA_HV
    ent_map = before_state.entity_map()
    atoms: list[tuple[int, int]] = []

    for subj, pred, obj in delta.added_facts:
        if subj in ent_map:
            atoms.append((bind10240(effect_slot_key_hv("add_subj", pred), entity_symbol_hv(subj)), 1))
        if obj in ent_map:
            atoms.append((bind10240(effect_slot_key_hv("add_obj", pred), entity_symbol_hv(obj)), 1))

    for subj, pred, obj in delta.deleted_facts:
        if subj in ent_map:
            atoms.append((bind10240(effect_slot_key_hv("del_subj", pred), entity_symbol_hv(subj)), 1))
        if obj in ent_map:
            atoms.append((bind10240(effect_slot_key_hv("del_obj", pred), entity_symbol_hv(obj)), 1))

    for eid, attr_name, _old_v, _new_v in delta.attribute_edits:
        if eid in ent_map:
            atoms.append((bind10240(effect_slot_key_hv("edit_ent", attr_name), entity_symbol_hv(eid)), 1))

    return _majority_bundle10240(atoms, empty_hv=NOOP_DELTA_HV, bits=8)


def unbind_role_slot_hv(bound_hv: int, pred_role_hv: int) -> int:
    """Self-inverse XOR unbinding in 10,240-bit BSC space to recover a bound slot filler."""
    return bind10240(bound_hv, pred_role_hv)


def cleanup_entity_ranking(
    unbound_hv: int,
    candidate_entity_ids: Sequence[str],
) -> list[tuple[str, int]]:
    """Rank candidate entity IDs by exact 10,240-bit POPCNT Hamming similarity to ``unbound_hv``."""
    scored = [
        (eid, hamming_sim10240(unbound_hv, entity_symbol_hv(eid)))
        for eid in candidate_entity_ids
    ]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored


def encode_sequence_hv(
    step_hvs: Sequence[int],
    *,
    use_role_permutation: bool = True,
) -> int:
    """Encode an ordered sequence of 10,240-bit step HVs for bind-depth geometry probing.

    - ``use_role_permutation=True``: binds each step ``k`` with an orthogonal position key
      AND a coprime cyclic bit permutation ``permute10240(hv, (k + 1) * DEPTH_PERM_STRIDE)``
      so arbitrary depth slots (and repeated symbols) unbind cleanly through depth 6+.
    - ``use_role_permutation=False``: naive commutative superposition (control baseline
      demonstrating order/depth collapse without role-specific permutations).
    """
    if not step_hvs:
        return EMPTY_SEQ_HV
    atoms: list[tuple[int, int]] = []
    for idx, hv in enumerate(step_hvs):
        if use_role_permutation:
            pos_hv = hv10240_of(f"shard:seq_pos:{idx}")
            shifted = permute10240(hv, (idx + 1) * DEPTH_PERM_STRIDE)
            atoms.append((bind10240(pos_hv, shifted), 1))
        else:
            atoms.append((hv & UNIFIED_MASK, 1))
    return _majority_bundle10240(atoms, empty_hv=EMPTY_SEQ_HV, bits=8)


def unbind_step_from_sequence(
    seq_hv: int,
    depth_idx: int,
    codebook: Mapping[str, int],
    *,
    use_role_permutation: bool = True,
) -> tuple[str, int]:
    """Unbind the symbol at ``depth_idx`` from ``seq_hv`` and clean up against ``codebook``."""
    if not codebook:
        raise ValueError("codebook must be non-empty for clean-up memory unbinding")
    if use_role_permutation:
        pos_hv = hv10240_of(f"shard:seq_pos:{depth_idx}")
        unshifted = bind10240(seq_hv, pos_hv)
        probe = permute10240(unshifted, -((depth_idx + 1) * DEPTH_PERM_STRIDE))
    else:
        probe = seq_hv & UNIFIED_MASK

    best_label = ""
    best_sim = -D_UNIFIED - 1
    for label in sorted(codebook):
        sim = hamming_sim10240(probe, codebook[label])
        if sim > best_sim:
            best_sim = sim
            best_label = label
    return (best_label, best_sim)
