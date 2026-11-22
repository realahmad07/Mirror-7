"""Deterministic generator-level train and held-out split builders for S0.

All held-out splits are constructed by generator (never random row splits) with
100% disjoint entity/color vocabularies across Train, Held-Out, and Frozen Temporal
splits. Hidden ground-truth labels are computed here via evaluator-only
``hidden_actions`` and stored separately in ``EvaluationTaskHiddenLabel``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .hidden_actions import HIDDEN_ACTION_NAMES, apply_hidden_action, apply_hidden_sequence
from .state_graph import (
    ActionInvocation,
    Entity,
    EvaluationCase,
    EvaluationTaskHiddenLabel,
    EvaluationTaskPublic,
    Fact,
    ObservationMask,
    StateGraph,
    TrainingEpisode,
    canonical_json_bytes,
    sha256_hex,
)

GENERATOR_VERSION: str = "s0-relworld-v1.0"
TEMPORAL_GENERATOR_VERSION: str = "s0-temporal-v2.0"
DEFAULT_SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)
DEFAULT_DEMOS_PER_ACTION: int = 6
FIXED_TIMESTAMP: str = "2026-11-22T00:00:00Z"


@dataclass(frozen=True)
class EntityPool:
    """Disjoint lexical pool for a generator family."""

    pool_id: str
    agents: tuple[str, ...]
    objects: tuple[str, ...]
    surfaces: tuple[str, ...]
    containers: tuple[str, ...]
    colors: tuple[str, ...]

    def all_entity_roots(self) -> frozenset[str]:
        return frozenset(self.agents + self.objects + self.surfaces + self.containers)


TRAIN_POOL = EntityPool(
    pool_id="train_pool_v1",
    agents=("alice", "bob", "clara", "david"),
    objects=("cup", "book", "coin", "lamp", "orb", "bell"),
    surfaces=("table", "shelf", "desk", "counter"),
    containers=("box", "chest", "crate", "drawer"),
    colors=("red", "blue", "green", "yellow"),
)

HELDOUT_POOL = EntityPool(
    pool_id="heldout_pool_v1",
    agents=("zara", "kieran", "nadia", "tariq", "orion", "selene"),
    objects=("chalice", "prism", "scroll", "lantern", "vial", "compass", "relic", "crystal"),
    surfaces=("pedestal", "altar", "ledge", "workbench", "console"),
    containers=("vault", "casket", "locker", "satchel", "coffer"),
    colors=("amber", "violet", "crimson", "silver", "obsidian", "teal"),
)

TEMPORAL_POOL = EntityPool(
    pool_id="temporal_pool_v2",
    agents=("lyra", "marcus", "farah", "devon", "soren", "amara"),
    objects=("astrolabe", "amulet", "flask", "tablet", "locket", "medallion"),
    surfaces=("plinth", "hearth", "mantel", "dais", "podium"),
    containers=("strongbox", "reliquary", "trunk", "pouch", "cabinet"),
    colors=("indigo", "copper", "ivory", "jade", "scarlet", "bronze"),
)

UNSEEN_EVAL_SPLITS: tuple[str, ...] = (
    "heldout_unseen_names",
    "heldout_role_swaps",
    "heldout_guard_contrasts",
    "heldout_repeated_idempotent",
    "heldout_order_reversal",
    "heldout_comp_depth2",
    "heldout_comp_depth4",
    "heldout_comp_depth6",
    "heldout_temporal_frozen",
)

CANARY_SPLITS: tuple[str, ...] = ("canary_exact_memorization",)


def verify_pool_disjointness() -> dict[str, Any]:
    """Assert and certify 100% lexical disjointness across train, heldout, and temporal pools."""
    pools = (TRAIN_POOL, HELDOUT_POOL, TEMPORAL_POOL)
    for i in range(len(pools)):
        for j in range(i + 1, len(pools)):
            e_overlap = pools[i].all_entity_roots() & pools[j].all_entity_roots()
            c_overlap = frozenset(pools[i].colors) & frozenset(pools[j].colors)
            if e_overlap or c_overlap:
                raise AssertionError(
                    f"Pool overlap between {pools[i].pool_id} and {pools[j].pool_id}: "
                    f"entities={sorted(e_overlap)}, colors={sorted(c_overlap)}"
                )
    return {
        "disjoint_verified": True,
        "train_entity_roots": sorted(TRAIN_POOL.all_entity_roots()),
        "heldout_entity_roots": sorted(HELDOUT_POOL.all_entity_roots()),
        "temporal_entity_roots": sorted(TEMPORAL_POOL.all_entity_roots()),
        "train_colors": list(TRAIN_POOL.colors),
        "heldout_colors": list(HELDOUT_POOL.colors),
        "temporal_colors": list(TEMPORAL_POOL.colors),
    }


def _find_span(raw_bytes: bytes, token: str, start_from: int = 0) -> tuple[int, int]:
    needle = token.encode("utf-8")
    idx = raw_bytes.find(needle, start_from)
    if idx < 0:
        idx = raw_bytes.find(needle)
    if idx < 0:
        raise ValueError(f"Token {token!r} not found in raw instruction {raw_bytes!r}")
    return (idx, idx + len(needle))


def build_action_invocation(
    action_id: str,
    roles: Mapping[str, str],
    *,
    template_variant: int = 0,
) -> ActionInvocation:
    """Create a typed ActionInvocation with raw UTF-8 instruction bytes and verified byte spans."""
    if action_id == "move":
        a, o, s, d = roles["actor"], roles["object"], roles["source"], roles["destination"]
        templates = (
            f"{a} move {o} from {s} to {d}",
            f"Have {a} move {o} out of {s} onto {d}",
        )
        text = templates[template_variant % len(templates)]
        verb = "move"
    elif action_id == "give":
        a, o, r = roles["actor"], roles["object"], roles["recipient"]
        templates = (
            f"{a} give {o} to {r}",
            f"Have {a} give {o} over to {r}",
        )
        text = templates[template_variant % len(templates)]
        verb = "give"
    elif action_id == "open":
        a, t = roles["actor"], roles["target"]
        templates = (
            f"{a} open {t}",
            f"Have {a} open the {t}",
        )
        text = templates[template_variant % len(templates)]
        verb = "open"
    elif action_id == "lock":
        a, t = roles["actor"], roles["target"]
        templates = (
            f"{a} lock {t}",
            f"Have {a} lock the {t}",
        )
        text = templates[template_variant % len(templates)]
        verb = "lock"
    else:
        raise ValueError(f"Unsupported action_id: {action_id!r}")

    raw = text.encode("utf-8")
    spans: list[tuple[str, int, int]] = []
    act_s, act_e = _find_span(raw, verb)
    spans.append(("action", act_s, act_e))
    cursor = 0
    role_order = {
        "move": ("actor", "object", "source", "destination"),
        "give": ("actor", "object", "recipient"),
        "open": ("actor", "target"),
        "lock": ("actor", "target"),
    }[action_id]
    for rk in role_order:
        val = roles[rk]
        s_idx, e_idx = _find_span(raw, val, cursor)
        spans.append((rk, s_idx, e_idx))
        cursor = e_idx

    return ActionInvocation(
        action_id=action_id,
        roles=tuple(sorted((k, v) for k, v in roles.items())),
        raw_instruction_bytes=raw,
        byte_spans=tuple(spans),
    )


def _sample_ids(
    pool: EntityPool,
    rng: random.Random,
    *,
    n_agents: int,
    n_objs: int,
    n_surfs: int,
    n_conts: int,
) -> dict[str, list[str]]:
    suffix = rng.randint(1, 9)
    agents = [f"{name}_{suffix}" for name in rng.sample(pool.agents, n_agents)]
    objs = [f"{name}_{suffix}" for name in rng.sample(pool.objects, n_objs)]
    surfs = [f"{name}_{suffix}" for name in rng.sample(pool.surfaces, n_surfs)]
    conts = [f"{name}_{suffix}" for name in rng.sample(pool.containers, n_conts)]
    return {"agents": agents, "objects": objs, "surfaces": surfs, "containers": conts}


def _make_base_world(
    pool: EntityPool,
    rng: random.Random,
    *,
    n_agents: int = 3,
    n_objs: int = 3,
    n_surfs: int = 3,
    n_conts: int = 3,
    container_states: Sequence[tuple[bool, bool, bool]] | None = None,
    surface_accessible: Sequence[bool] | None = None,
    obj_portable: Sequence[bool] | None = None,
) -> tuple[StateGraph, dict[str, list[str]]]:
    """Build a deterministic relational world with active entities plus unrelated distractors."""
    ids = _sample_ids(
        pool, rng, n_agents=n_agents, n_objs=n_objs, n_surfs=n_surfs, n_conts=n_conts
    )
    entities: dict[str, Entity] = {}
    facts: set[Fact] = set()

    for idx, aid in enumerate(ids["agents"]):
        entities[aid] = Entity(
            entity_id=aid,
            entity_type="agent",
            attributes=(("role_badge", pool.colors[idx % len(pool.colors)]), ("status", "active")),
        )

    for idx, sid in enumerate(ids["surfaces"]):
        acc = True if surface_accessible is None or idx >= len(surface_accessible) else surface_accessible[idx]
        entities[sid] = Entity(
            entity_id=sid,
            entity_type="surface",
            attributes=(("color", rng.choice(pool.colors)),),
        )
        facts.add((sid, "accessible", "true" if acc else "false"))

    for idx, cid in enumerate(ids["containers"]):
        if container_states is not None and idx < len(container_states):
            is_open, is_locked, acc = container_states[idx]
        else:
            is_open, is_locked, acc = (False, False, True)
        status = "locked" if is_locked else ("open" if is_open else "closed")
        entities[cid] = Entity(
            entity_id=cid,
            entity_type="container",
            attributes=(("color", rng.choice(pool.colors)), ("status", status)),
        )
        facts.add((cid, "is_open", "true" if is_open else "false"))
        facts.add((cid, "is_locked", "true" if is_locked else "false"))
        facts.add((cid, "accessible", "true" if acc else "false"))

    for idx, oid in enumerate(ids["objects"]):
        portable = True if obj_portable is None or idx >= len(obj_portable) else obj_portable[idx]
        entities[oid] = Entity(
            entity_id=oid,
            entity_type="object",
            attributes=(("color", rng.choice(pool.colors)), ("portable", portable)),
        )
        loc = ids["surfaces"][idx % len(ids["surfaces"])]
        owner = ids["agents"][idx % len(ids["agents"])]
        facts.add((oid, "located_on", loc))
        facts.add((oid, "owned_by", owner))

    return StateGraph.create(entities, facts), ids


def _make_case(
    *,
    task_id: str,
    split_id: str,
    generator_id: str,
    generator_version: str,
    seed: int,
    initial_state: StateGraph,
    actions: Sequence[ActionInvocation],
    metadata: Mapping[str, Any] | None = None,
) -> EvaluationCase:
    mask = ObservationMask(fully_observed=True)
    final_s, inter_s, deltas, applied, reasons = apply_hidden_sequence(
        initial_state, actions, mask=mask
    )
    public = EvaluationTaskPublic(
        task_id=task_id,
        split_id=split_id,
        generator_id=generator_id,
        generator_version=generator_version,
        seed=seed,
        depth=len(actions),
        initial_state=initial_state,
        actions=tuple(actions),
        observation_mask=mask,
        metadata=tuple(sorted((str(k), v) for k, v in (metadata or {}).items())),
    )
    hidden = EvaluationTaskHiddenLabel(
        task_id=task_id,
        gold_final_state=final_s,
        gold_intermediate_states=inter_s,
        gold_step_deltas=deltas,
        gold_step_applied=applied,
        gold_block_reasons=reasons,
    )
    return EvaluationCase(public=public, hidden=hidden)


def generate_training_episodes(
    seed: int,
    demos_per_action: int = DEFAULT_DEMOS_PER_ACTION,
) -> list[TrainingEpisode]:
    """Generate 3-8 training demonstrations per hidden action using TRAIN_POOL only."""
    if not (3 <= demos_per_action <= 8):
        raise ValueError("demos_per_action must be in [3, 8] per S0 specification")

    rng = random.Random(seed * 10_007 + 17)
    episodes: list[TrainingEpisode] = []
    mask = ObservationMask(fully_observed=True)

    for action_id in HIDDEN_ACTION_NAMES:
        n_pos = max(2, demos_per_action - 2)

        for idx in range(demos_per_action):
            is_positive = idx < n_pos
            if action_id == "move":
                if is_positive:
                    c_open = idx >= 2
                    world, ids = _make_base_world(
                        TRAIN_POOL,
                        rng,
                        container_states=((c_open, False, True), (False, False, True), (False, True, True)),
                    )
                    actor = ids["agents"][0]
                    obj = ids["objects"][0]
                    src = ids["surfaces"][0]
                    dst = ids["containers"][0] if c_open else ids["surfaces"][1]
                    if idx == 3:
                        world = world.with_changes(
                            delete_facts=((obj, "located_on", src),),
                            add_facts=((obj, "located_on", ids["containers"][0]),),
                        )
                        src, dst = ids["containers"][0], ids["surfaces"][1]
                    roles = {"actor": actor, "object": obj, "source": src, "destination": dst}
                else:
                    neg_i = idx - n_pos
                    if neg_i == 0:
                        world, ids = _make_base_world(
                            TRAIN_POOL, rng, surface_accessible=(True, False, True)
                        )
                        roles = {
                            "actor": ids["agents"][0],
                            "object": ids["objects"][0],
                            "source": ids["surfaces"][0],
                            "destination": ids["surfaces"][1],
                        }
                    else:
                        world, ids = _make_base_world(
                            TRAIN_POOL, rng, container_states=((False, False, True), (False, False, True), (False, False, True))
                        )
                        roles = {
                            "actor": ids["agents"][0],
                            "object": ids["objects"][0],
                            "source": ids["surfaces"][0],
                            "destination": ids["containers"][0],
                        }

            elif action_id == "give":
                if is_positive:
                    world, ids = _make_base_world(TRAIN_POOL, rng)
                    roles = {
                        "actor": ids["agents"][0],
                        "object": ids["objects"][0],
                        "recipient": ids["agents"][1],
                    }
                else:
                    neg_i = idx - n_pos
                    world, ids = _make_base_world(TRAIN_POOL, rng)
                    if neg_i == 0:
                        roles = {
                            "actor": ids["agents"][1],
                            "object": ids["objects"][0],
                            "recipient": ids["agents"][2],
                        }
                    else:
                        world = world.with_changes(
                            delete_facts=((ids["objects"][0], "located_on", ids["surfaces"][0]),),
                            add_facts=((ids["objects"][0], "located_on", ids["containers"][0]),),
                        )
                        roles = {
                            "actor": ids["agents"][0],
                            "object": ids["objects"][0],
                            "recipient": ids["agents"][1],
                        }

            elif action_id == "open":
                if is_positive:
                    world, ids = _make_base_world(
                        TRAIN_POOL, rng, container_states=((False, False, True), (False, True, True), (True, False, True))
                    )
                    roles = {"actor": ids["agents"][0], "target": ids["containers"][0]}
                else:
                    neg_i = idx - n_pos
                    if neg_i == 0:
                        world, ids = _make_base_world(
                            TRAIN_POOL, rng, container_states=((False, True, True), (False, False, True), (True, False, True))
                        )
                    else:
                        world, ids = _make_base_world(
                            TRAIN_POOL, rng, container_states=((True, False, True), (False, False, True), (False, True, True))
                        )
                    roles = {"actor": ids["agents"][0], "target": ids["containers"][0]}

            else:  # lock
                if is_positive:
                    world, ids = _make_base_world(
                        TRAIN_POOL, rng, container_states=((False, False, True), (True, False, True), (False, True, True))
                    )
                    roles = {"actor": ids["agents"][0], "target": ids["containers"][0]}
                else:
                    neg_i = idx - n_pos
                    if neg_i == 0:
                        world, ids = _make_base_world(
                            TRAIN_POOL, rng, container_states=((True, False, True), (False, False, True), (False, True, True))
                        )
                    else:
                        world, ids = _make_base_world(
                            TRAIN_POOL, rng, container_states=((False, True, True), (False, False, True), (True, False, True))
                        )
                    roles = {"actor": ids["agents"][0], "target": ids["containers"][0]}

            act = build_action_invocation(action_id, roles, template_variant=0)
            res = apply_hidden_action(world, act, mask=mask)
            if is_positive and not res.applied:
                raise AssertionError(f"Training positive demo unexpectedly blocked: {res.reason}")
            if (not is_positive) and res.applied:
                raise AssertionError(f"Training negative demo unexpectedly applied: {action_id} {idx}")

            ep = TrainingEpisode(
                episode_id=f"train_s{seed}_{action_id}_{idx:02d}",
                before_state=world,
                action=act,
                observation_mask=mask,
                after_state=res.after_state,
                sparse_delta=res.delta,
                context=(("demo_index", idx), ("expected_transition_kind", "effect" if is_positive else "blocked")),
                observation_source="hidden_relational_simulator_v1",
                split_id="train",
                generator_id="gen_train_v1",
                generator_version=GENERATOR_VERSION,
                seed=seed,
                timestamp=FIXED_TIMESTAMP,
            )
            episodes.append(ep)

    return episodes

def _gen_unseen_names(seed: int) -> list[EvaluationCase]:
    rng = random.Random(seed * 31 + 101)
    cases: list[EvaluationCase] = []
    for i in range(20):
        action_id = HIDDEN_ACTION_NAMES[i % 4]
        world, ids = _make_base_world(
            HELDOUT_POOL,
            rng,
            container_states=((action_id == "move" and i % 8 >= 4, False, True), (False, False, True), (False, True, True)),
        )
        if action_id == "move":
            dst = ids["containers"][0] if (i % 8 >= 4) else ids["surfaces"][1]
            roles = {
                "actor": ids["agents"][0],
                "object": ids["objects"][0],
                "source": ids["surfaces"][0],
                "destination": dst,
            }
        elif action_id == "give":
            roles = {
                "actor": ids["agents"][0],
                "object": ids["objects"][0],
                "recipient": ids["agents"][1],
            }
        else:
            roles = {"actor": ids["agents"][0], "target": ids["containers"][0]}
        act = build_action_invocation(action_id, roles, template_variant=i % 2)
        case = _make_case(
            task_id=f"unseen_names_s{seed}_{i:02d}",
            split_id="heldout_unseen_names",
            generator_id="gen_unseen_names_v1",
            generator_version=GENERATOR_VERSION,
            seed=seed,
            initial_state=world,
            actions=(act,),
            metadata={"action_id": action_id},
        )
        assert case.hidden.gold_step_applied[0] is True
        cases.append(case)
    return cases


def _gen_role_swaps(seed: int) -> list[EvaluationCase]:
    rng = random.Random(seed * 37 + 202)
    cases: list[EvaluationCase] = []
    for pair_idx in range(8):
        if pair_idx % 2 == 0:
            world_a, ids = _make_base_world(HELDOUT_POOL, rng)
            actor, obj = ids["agents"][0], ids["objects"][0]
            s0, s1 = ids["surfaces"][0], ids["surfaces"][1]
            act_fwd = build_action_invocation(
                "move",
                {"actor": actor, "object": obj, "source": s0, "destination": s1},
                template_variant=1,
            )
            world_b = world_a.with_changes(
                delete_facts=((obj, "located_on", s0),),
                add_facts=((obj, "located_on", s1),),
            )
            act_rev = build_action_invocation(
                "move",
                {"actor": actor, "object": obj, "source": s1, "destination": s0},
                template_variant=1,
            )
            c_a = _make_case(
                task_id=f"role_swap_s{seed}_{pair_idx:02d}_a",
                split_id="heldout_role_swaps",
                generator_id="gen_role_swaps_v1",
                generator_version=GENERATOR_VERSION,
                seed=seed,
                initial_state=world_a,
                actions=(act_fwd,),
                metadata={"swap_pair": pair_idx, "direction": "forward", "action_id": "move"},
            )
            c_b = _make_case(
                task_id=f"role_swap_s{seed}_{pair_idx:02d}_b",
                split_id="heldout_role_swaps",
                generator_id="gen_role_swaps_v1",
                generator_version=GENERATOR_VERSION,
                seed=seed,
                initial_state=world_b,
                actions=(act_rev,),
                metadata={"swap_pair": pair_idx, "direction": "swapped", "action_id": "move"},
            )
        else:
            world_a, ids = _make_base_world(HELDOUT_POOL, rng)
            a0, a1, obj = ids["agents"][0], ids["agents"][1], ids["objects"][0]
            act_fwd = build_action_invocation(
                "give", {"actor": a0, "object": obj, "recipient": a1}, template_variant=1
            )
            world_b = world_a.with_changes(
                delete_facts=((obj, "owned_by", a0),),
                add_facts=((obj, "owned_by", a1),),
            )
            act_rev = build_action_invocation(
                "give", {"actor": a1, "object": obj, "recipient": a0}, template_variant=1
            )
            c_a = _make_case(
                task_id=f"role_swap_s{seed}_{pair_idx:02d}_a",
                split_id="heldout_role_swaps",
                generator_id="gen_role_swaps_v1",
                generator_version=GENERATOR_VERSION,
                seed=seed,
                initial_state=world_a,
                actions=(act_fwd,),
                metadata={"swap_pair": pair_idx, "direction": "forward", "action_id": "give"},
            )
            c_b = _make_case(
                task_id=f"role_swap_s{seed}_{pair_idx:02d}_b",
                split_id="heldout_role_swaps",
                generator_id="gen_role_swaps_v1",
                generator_version=GENERATOR_VERSION,
                seed=seed,
                initial_state=world_b,
                actions=(act_rev,),
                metadata={"swap_pair": pair_idx, "direction": "swapped", "action_id": "give"},
            )
        assert c_a.hidden.gold_step_applied[0] and c_b.hidden.gold_step_applied[0]
        cases.extend((c_a, c_b))
    return cases


def _gen_guard_contrasts(seed: int) -> list[EvaluationCase]:
    """10 minimal pairs (20 tasks): 10 guard-blocked cases + 10 minimal-change valid counterparts."""
    rng = random.Random(seed * 41 + 303)
    cases: list[EvaluationCase] = []
    guard_specs = (
        "move_non_portable",
        "move_inaccessible_dst",
        "move_closed_dst_container",
        "move_locked_dst_container",
        "move_wrong_owner",
        "give_wrong_owner",
        "give_inside_closed_container",
        "open_locked_container",
        "lock_open_container",
        "lock_inaccessible_container",
    )
    for idx, kind in enumerate(guard_specs):
        if kind == "move_non_portable":
            w_blk, ids = _make_base_world(HELDOUT_POOL, rng, obj_portable=(False, True, True))
            w_ok = w_blk.with_changes(attribute_edits=((ids["objects"][0], "portable", True),))
            act = build_action_invocation(
                "move",
                {"actor": ids["agents"][0], "object": ids["objects"][0], "source": ids["surfaces"][0], "destination": ids["surfaces"][1]},
                template_variant=1,
            )
        elif kind == "move_inaccessible_dst":
            w_blk, ids = _make_base_world(HELDOUT_POOL, rng, surface_accessible=(True, False, True))
            w_ok = w_blk.with_changes(
                delete_facts=((ids["surfaces"][1], "accessible", "false"),),
                add_facts=((ids["surfaces"][1], "accessible", "true"),),
            )
            act = build_action_invocation(
                "move",
                {"actor": ids["agents"][0], "object": ids["objects"][0], "source": ids["surfaces"][0], "destination": ids["surfaces"][1]},
                template_variant=1,
            )
        elif kind == "move_closed_dst_container":
            w_blk, ids = _make_base_world(HELDOUT_POOL, rng, container_states=((False, False, True),))
            w_ok = w_blk.with_changes(
                delete_facts=((ids["containers"][0], "is_open", "false"),),
                add_facts=((ids["containers"][0], "is_open", "true"),),
                attribute_edits=((ids["containers"][0], "status", "open"),),
            )
            act = build_action_invocation(
                "move",
                {"actor": ids["agents"][0], "object": ids["objects"][0], "source": ids["surfaces"][0], "destination": ids["containers"][0]},
                template_variant=1,
            )
        elif kind == "move_locked_dst_container":
            w_blk, ids = _make_base_world(HELDOUT_POOL, rng, container_states=((False, True, True),))
            w_ok = w_blk.with_changes(
                delete_facts=((ids["containers"][0], "is_locked", "true"), (ids["containers"][0], "is_open", "false")),
                add_facts=((ids["containers"][0], "is_locked", "false"), (ids["containers"][0], "is_open", "true")),
                attribute_edits=((ids["containers"][0], "status", "open"),),
            )
            act = build_action_invocation(
                "move",
                {"actor": ids["agents"][0], "object": ids["objects"][0], "source": ids["surfaces"][0], "destination": ids["containers"][0]},
                template_variant=1,
            )
        elif kind == "move_wrong_owner":
            w_blk, ids = _make_base_world(HELDOUT_POOL, rng)
            w_ok = w_blk.with_changes(
                delete_facts=((ids["objects"][0], "owned_by", ids["agents"][0]),),
                add_facts=((ids["objects"][0], "owned_by", ids["agents"][1]),),
            )
            act = build_action_invocation(
                "move",
                {"actor": ids["agents"][1], "object": ids["objects"][0], "source": ids["surfaces"][0], "destination": ids["surfaces"][1]},
                template_variant=1,
            )
        elif kind == "give_wrong_owner":
            w_blk, ids = _make_base_world(HELDOUT_POOL, rng)
            w_ok = w_blk.with_changes(
                delete_facts=((ids["objects"][0], "owned_by", ids["agents"][0]),),
                add_facts=((ids["objects"][0], "owned_by", ids["agents"][1]),),
            )
            act = build_action_invocation(
                "give",
                {"actor": ids["agents"][1], "object": ids["objects"][0], "recipient": ids["agents"][2]},
                template_variant=1,
            )
        elif kind == "give_inside_closed_container":
            w_blk, ids = _make_base_world(HELDOUT_POOL, rng, container_states=((False, False, True),))
            w_blk = w_blk.with_changes(
                delete_facts=((ids["objects"][0], "located_on", ids["surfaces"][0]),),
                add_facts=((ids["objects"][0], "located_on", ids["containers"][0]),),
            )
            w_ok = w_blk.with_changes(
                delete_facts=((ids["containers"][0], "is_open", "false"),),
                add_facts=((ids["containers"][0], "is_open", "true"),),
                attribute_edits=((ids["containers"][0], "status", "open"),),
            )
            act = build_action_invocation(
                "give",
                {"actor": ids["agents"][0], "object": ids["objects"][0], "recipient": ids["agents"][1]},
                template_variant=1,
            )
        elif kind == "open_locked_container":
            w_blk, ids = _make_base_world(HELDOUT_POOL, rng, container_states=((False, True, True),))
            w_ok = w_blk.with_changes(
                delete_facts=((ids["containers"][0], "is_locked", "true"),),
                add_facts=((ids["containers"][0], "is_locked", "false"),),
                attribute_edits=((ids["containers"][0], "status", "closed"),),
            )
            act = build_action_invocation(
                "open", {"actor": ids["agents"][0], "target": ids["containers"][0]}, template_variant=1
            )
        elif kind == "lock_open_container":
            w_blk, ids = _make_base_world(HELDOUT_POOL, rng, container_states=((True, False, True),))
            w_ok = w_blk.with_changes(
                delete_facts=((ids["containers"][0], "is_open", "true"),),
                add_facts=((ids["containers"][0], "is_open", "false"),),
                attribute_edits=((ids["containers"][0], "status", "closed"),),
            )
            act = build_action_invocation(
                "lock", {"actor": ids["agents"][0], "target": ids["containers"][0]}, template_variant=1
            )
        else:
            w_blk, ids = _make_base_world(HELDOUT_POOL, rng, container_states=((False, False, False),))
            w_ok = w_blk.with_changes(
                delete_facts=((ids["containers"][0], "accessible", "false"),),
                add_facts=((ids["containers"][0], "accessible", "true"),),
            )
            act = build_action_invocation(
                "lock", {"actor": ids["agents"][0], "target": ids["containers"][0]}, template_variant=1
            )

        c_blk = _make_case(
            task_id=f"guard_s{seed}_{idx:02d}_blocked",
            split_id="heldout_guard_contrasts",
            generator_id="gen_guard_contrasts_v1",
            generator_version=GENERATOR_VERSION,
            seed=seed,
            initial_state=w_blk,
            actions=(act,),
            metadata={"contrast_kind": kind, "expected_applied": False},
        )
        c_ok = _make_case(
            task_id=f"guard_s{seed}_{idx:02d}_valid",
            split_id="heldout_guard_contrasts",
            generator_id="gen_guard_contrasts_v1",
            generator_version=GENERATOR_VERSION,
            seed=seed,
            initial_state=w_ok,
            actions=(act,),
            metadata={"contrast_kind": kind, "expected_applied": True},
        )
        assert c_blk.hidden.gold_step_applied[0] is False
        assert c_ok.hidden.gold_step_applied[0] is True
        cases.extend((c_blk, c_ok))
    return cases


def _gen_repeated_idempotent(seed: int) -> list[EvaluationCase]:
    rng = random.Random(seed * 43 + 404)
    cases: list[EvaluationCase] = []
    for i in range(16):
        action_id = HIDDEN_ACTION_NAMES[i % 4]
        world, ids = _make_base_world(HELDOUT_POOL, rng)
        if action_id == "move":
            roles = {
                "actor": ids["agents"][0],
                "object": ids["objects"][0],
                "source": ids["surfaces"][0],
                "destination": ids["surfaces"][1],
            }
        elif action_id == "give":
            roles = {
                "actor": ids["agents"][0],
                "object": ids["objects"][0],
                "recipient": ids["agents"][1],
            }
        else:
            roles = {"actor": ids["agents"][0], "target": ids["containers"][0]}
        act = build_action_invocation(action_id, roles, template_variant=i % 2)
        case = _make_case(
            task_id=f"repeated_s{seed}_{i:02d}",
            split_id="heldout_repeated_idempotent",
            generator_id="gen_repeated_idempotent_v1",
            generator_version=GENERATOR_VERSION,
            seed=seed,
            initial_state=world,
            actions=(act, act),
            metadata={"action_id": action_id},
        )
        assert case.hidden.gold_step_applied == (True, False)
        cases.append(case)
    return cases


def _gen_order_reversal(seed: int) -> list[EvaluationCase]:
    """8 guaranteed non-commuting pairs (16 tasks): (a, b) vs (b, a) with distinct gold states."""
    rng = random.Random(seed * 47 + 505)
    cases: list[EvaluationCase] = []
    pair_types = ("open_move", "give_move", "open_lock", "move_move") * 2
    for idx, ptype in enumerate(pair_types):
        world, ids = _make_base_world(HELDOUT_POOL, rng)
        a0, a1 = ids["agents"][0], ids["agents"][1]
        obj = ids["objects"][0]
        s0, s1, s2 = ids["surfaces"][0], ids["surfaces"][1], ids["surfaces"][2]
        c0 = ids["containers"][0]

        if ptype == "open_move":
            act_a = build_action_invocation("open", {"actor": a0, "target": c0}, template_variant=0)
            act_b = build_action_invocation(
                "move", {"actor": a0, "object": obj, "source": s0, "destination": c0}, template_variant=1
            )
        elif ptype == "give_move":
            act_a = build_action_invocation("give", {"actor": a0, "object": obj, "recipient": a1}, template_variant=0)
            act_b = build_action_invocation(
                "move", {"actor": a1, "object": obj, "source": s0, "destination": s1}, template_variant=1
            )
        elif ptype == "open_lock":
            act_a = build_action_invocation("open", {"actor": a0, "target": c0}, template_variant=0)
            act_b = build_action_invocation("lock", {"actor": a0, "target": c0}, template_variant=1)
        else:
            act_a = build_action_invocation(
                "move", {"actor": a0, "object": obj, "source": s0, "destination": s1}, template_variant=0
            )
            act_b = build_action_invocation(
                "move", {"actor": a0, "object": obj, "source": s1, "destination": s2}, template_variant=1
            )

        c_ab = _make_case(
            task_id=f"order_s{seed}_{idx:02d}_ab",
            split_id="heldout_order_reversal",
            generator_id="gen_order_reversal_v1",
            generator_version=GENERATOR_VERSION,
            seed=seed,
            initial_state=world,
            actions=(act_a, act_b),
            metadata={"order_pair": idx, "pair_type": ptype, "order": "ab"},
        )
        c_ba = _make_case(
            task_id=f"order_s{seed}_{idx:02d}_ba",
            split_id="heldout_order_reversal",
            generator_id="gen_order_reversal_v1",
            generator_version=GENERATOR_VERSION,
            seed=seed,
            initial_state=world,
            actions=(act_b, act_a),
            metadata={"order_pair": idx, "pair_type": ptype, "order": "ba"},
        )
        assert c_ab.hidden.gold_final_state != c_ba.hidden.gold_final_state, (
            f"Order pair {ptype} unexpectedly commuted!"
        )
        cases.extend((c_ab, c_ba))
    return cases

def _gen_compositions(
    seed: int,
    *,
    depth: int,
    count: int,
    split_id: str,
    generator_id: str,
    pool: EntityPool = HELDOUT_POOL,
    generator_version: str = GENERATOR_VERSION,
    rng_offset: int = 0,
) -> list[EvaluationCase]:
    """Generate depth-stratified valid causal compositions (depths 2, 4, or 6)."""
    rng = random.Random(seed * 53 + depth * 997 + rng_offset)
    cases: list[EvaluationCase] = []

    for i in range(count):
        world, ids = _make_base_world(
            pool,
            rng,
            n_agents=3,
            n_objs=4,
            n_surfs=3,
            n_conts=3,
            container_states=((False, False, True), (False, False, True), (False, False, True)),
        )
        a0, a1, a2 = ids["agents"]
        o0, o1 = ids["objects"][0], ids["objects"][1]
        s0, s1, s2 = ids["surfaces"]
        c0, c1, c2 = ids["containers"]
        v = i % 2

        if depth == 2:
            pattern = i % 4
            if pattern == 0:
                acts = (
                    build_action_invocation("open", {"actor": a0, "target": c0}, template_variant=v),
                    build_action_invocation("move", {"actor": a0, "object": o0, "source": s0, "destination": c0}, template_variant=1 - v),
                )
            elif pattern == 1:
                acts = (
                    build_action_invocation("give", {"actor": a0, "object": o0, "recipient": a1}, template_variant=v),
                    build_action_invocation("move", {"actor": a1, "object": o0, "source": s0, "destination": s1}, template_variant=1 - v),
                )
            elif pattern == 2:
                acts = (
                    build_action_invocation("move", {"actor": a0, "object": o0, "source": s0, "destination": s1}, template_variant=v),
                    build_action_invocation("give", {"actor": a0, "object": o0, "recipient": a2}, template_variant=1 - v),
                )
            else:
                acts = (
                    build_action_invocation("open", {"actor": a1, "target": c0}, template_variant=v),
                    build_action_invocation("lock", {"actor": a0, "target": c1}, template_variant=1 - v),
                )

        elif depth == 4:
            pattern = i % 4
            if pattern == 0:
                acts = (
                    build_action_invocation("open", {"actor": a0, "target": c0}, template_variant=v),
                    build_action_invocation("move", {"actor": a0, "object": o0, "source": s0, "destination": c0}, template_variant=1 - v),
                    build_action_invocation("give", {"actor": a0, "object": o0, "recipient": a1}, template_variant=v),
                    build_action_invocation("move", {"actor": a1, "object": o0, "source": c0, "destination": s2}, template_variant=1 - v),
                )
            elif pattern == 1:
                acts = (
                    build_action_invocation("give", {"actor": a0, "object": o0, "recipient": a1}, template_variant=v),
                    build_action_invocation("open", {"actor": a1, "target": c0}, template_variant=1 - v),
                    build_action_invocation("move", {"actor": a1, "object": o0, "source": s0, "destination": c0}, template_variant=v),
                    build_action_invocation("lock", {"actor": a2, "target": c1}, template_variant=1 - v),
                )
            elif pattern == 2:
                acts = (
                    build_action_invocation("move", {"actor": a0, "object": o0, "source": s0, "destination": s1}, template_variant=v),
                    build_action_invocation("give", {"actor": a0, "object": o0, "recipient": a2}, template_variant=1 - v),
                    build_action_invocation("move", {"actor": a2, "object": o0, "source": s1, "destination": s2}, template_variant=v),
                    build_action_invocation("give", {"actor": a2, "object": o0, "recipient": a1}, template_variant=1 - v),
                )
            else:
                acts = (
                    build_action_invocation("open", {"actor": a0, "target": c0}, template_variant=v),
                    build_action_invocation("open", {"actor": a1, "target": c1}, template_variant=1 - v),
                    build_action_invocation("move", {"actor": a0, "object": o0, "source": s0, "destination": c0}, template_variant=v),
                    build_action_invocation("move", {"actor": a1, "object": o1, "source": s1, "destination": c1}, template_variant=1 - v),
                )

        elif depth == 6:
            pattern = i % 3
            if pattern == 0:
                acts = (
                    build_action_invocation("open", {"actor": a0, "target": c0}, template_variant=v),
                    build_action_invocation("open", {"actor": a1, "target": c1}, template_variant=1 - v),
                    build_action_invocation("move", {"actor": a0, "object": o0, "source": s0, "destination": c0}, template_variant=v),
                    build_action_invocation("give", {"actor": a0, "object": o0, "recipient": a1}, template_variant=1 - v),
                    build_action_invocation("move", {"actor": a1, "object": o0, "source": c0, "destination": c1}, template_variant=v),
                    build_action_invocation("lock", {"actor": a2, "target": c2}, template_variant=1 - v),
                )
            elif pattern == 1:
                acts = (
                    build_action_invocation("give", {"actor": a0, "object": o0, "recipient": a1}, template_variant=v),
                    build_action_invocation("open", {"actor": a1, "target": c0}, template_variant=1 - v),
                    build_action_invocation("move", {"actor": a1, "object": o0, "source": s0, "destination": c0}, template_variant=v),
                    build_action_invocation("give", {"actor": a1, "object": o0, "recipient": a2}, template_variant=1 - v),
                    build_action_invocation("move", {"actor": a2, "object": o0, "source": c0, "destination": s2}, template_variant=v),
                    build_action_invocation("lock", {"actor": a0, "target": c1}, template_variant=1 - v),
                )
            else:
                acts = (
                    build_action_invocation("move", {"actor": a0, "object": o0, "source": s0, "destination": s2}, template_variant=v),
                    build_action_invocation("open", {"actor": a1, "target": c0}, template_variant=1 - v),
                    build_action_invocation("move", {"actor": a1, "object": o1, "source": s1, "destination": c0}, template_variant=v),
                    build_action_invocation("give", {"actor": a1, "object": o1, "recipient": a0}, template_variant=1 - v),
                    build_action_invocation("move", {"actor": a0, "object": o1, "source": c0, "destination": s1}, template_variant=v),
                    build_action_invocation("lock", {"actor": a2, "target": c2}, template_variant=1 - v),
                )
        else:
            raise ValueError(f"Unsupported composition depth: {depth}")

        case = _make_case(
            task_id=f"{split_id}_s{seed}_{i:02d}",
            split_id=split_id,
            generator_id=generator_id,
            generator_version=generator_version,
            seed=seed,
            initial_state=world,
            actions=acts,
            metadata={"depth": depth, "pattern": i % (3 if depth == 6 else 4)},
        )
        assert all(case.hidden.gold_step_applied), (
            f"Composition case {case.public.task_id} had blocked step: {case.hidden.gold_block_reasons}"
        )
        cases.append(case)
    return cases


def _gen_temporal_frozen(seed: int) -> list[EvaluationCase]:
    """Frozen temporal holdout generated from disjoint TEMPORAL_POOL & version tag."""
    derived_seed = seed + 100_000
    part_d2 = _gen_compositions(
        derived_seed,
        depth=2,
        count=10,
        split_id="heldout_temporal_frozen",
        generator_id="gen_temporal_frozen_v2",
        pool=TEMPORAL_POOL,
        generator_version=TEMPORAL_GENERATOR_VERSION,
        rng_offset=777,
    )
    part_d4 = _gen_compositions(
        derived_seed,
        depth=4,
        count=10,
        split_id="heldout_temporal_frozen",
        generator_id="gen_temporal_frozen_v2",
        pool=TEMPORAL_POOL,
        generator_version=TEMPORAL_GENERATOR_VERSION,
        rng_offset=888,
    )
    out: list[EvaluationCase] = []
    for idx, c in enumerate(part_d2 + part_d4):
        pub = EvaluationTaskPublic(
            task_id=f"temporal_s{seed}_{idx:02d}",
            split_id=c.public.split_id,
            generator_id=c.public.generator_id,
            generator_version=c.public.generator_version,
            seed=c.public.seed,
            depth=c.public.depth,
            initial_state=c.public.initial_state,
            actions=c.public.actions,
            observation_mask=c.public.observation_mask,
            metadata=c.public.metadata + (("base_seed", seed),),
        )
        hid = EvaluationTaskHiddenLabel(
            task_id=pub.task_id,
            gold_final_state=c.hidden.gold_final_state,
            gold_intermediate_states=c.hidden.gold_intermediate_states,
            gold_step_deltas=c.hidden.gold_step_deltas,
            gold_step_applied=c.hidden.gold_step_applied,
            gold_block_reasons=c.hidden.gold_block_reasons,
        )
        out.append(EvaluationCase(public=pub, hidden=hid))
    return out


def _gen_retrieval_canary(seed: int, training_episodes: Sequence[TrainingEpisode]) -> list[EvaluationCase]:
    """Exact training copies used ONLY to audit that retrieval hits are flagged & excluded."""
    by_action: dict[str, list[TrainingEpisode]] = {a: [] for a in HIDDEN_ACTION_NAMES}
    for ep in training_episodes:
        by_action[ep.action.action_id].append(ep)

    cases: list[EvaluationCase] = []
    idx = 0
    for action_id in HIDDEN_ACTION_NAMES:
        for ep in by_action[action_id][:2]:
            case = _make_case(
                task_id=f"canary_exact_s{seed}_{idx:02d}",
                split_id="canary_exact_memorization",
                generator_id="gen_retrieval_canary_v1",
                generator_version=GENERATOR_VERSION,
                seed=seed,
                initial_state=ep.before_state,
                actions=(ep.action,),
                metadata={"source_episode_id": ep.episode_id, "is_retrieval_canary": True},
            )
            cases.append(case)
            idx += 1
    return cases


def generate_heldout_suite(
    seed: int,
    training_episodes: Sequence[TrainingEpisode],
) -> dict[str, list[EvaluationCase]]:
    """Generate all 9 generator-level unseen splits + 1 retrieval-audit canary split."""
    return {
        "heldout_unseen_names": _gen_unseen_names(seed),
        "heldout_role_swaps": _gen_role_swaps(seed),
        "heldout_guard_contrasts": _gen_guard_contrasts(seed),
        "heldout_repeated_idempotent": _gen_repeated_idempotent(seed),
        "heldout_order_reversal": _gen_order_reversal(seed),
        "heldout_comp_depth2": _gen_compositions(
            seed, depth=2, count=20, split_id="heldout_comp_depth2", generator_id="gen_comp_depth2_v1"
        ),
        "heldout_comp_depth4": _gen_compositions(
            seed, depth=4, count=20, split_id="heldout_comp_depth4", generator_id="gen_comp_depth4_v1"
        ),
        "heldout_comp_depth6": _gen_compositions(
            seed, depth=6, count=20, split_id="heldout_comp_depth6", generator_id="gen_comp_depth6_v1"
        ),
        "heldout_temporal_frozen": _gen_temporal_frozen(seed),
        "canary_exact_memorization": _gen_retrieval_canary(seed, training_episodes),
    }


def build_split_manifests(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    demos_per_action: int = DEFAULT_DEMOS_PER_ACTION,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build reproducible cryptographic manifests for train and held-out splits."""
    disjoint_cert = verify_pool_disjointness()
    train_by_seed: dict[str, Any] = {}
    heldout_by_seed: dict[str, Any] = {}

    for seed in seeds:
        train_eps = generate_training_episodes(seed, demos_per_action=demos_per_action)
        train_payload = [ep.to_dict() for ep in train_eps]
        train_bytes = canonical_json_bytes(train_payload)
        train_by_seed[str(seed)] = {
            "seed": seed,
            "episode_count": len(train_eps),
            "demos_per_action": demos_per_action,
            "actions": list(HIDDEN_ACTION_NAMES),
            "raw_bytes": len(train_bytes),
            "sha256": sha256_hex(train_bytes),
        }

        suite = generate_heldout_suite(seed, train_eps)
        split_digests: dict[str, Any] = {}
        total_unseen = 0
        for split_name, cases in suite.items():
            pub_bytes = canonical_json_bytes([c.public.to_dict() for c in cases])
            hid_bytes = canonical_json_bytes([c.hidden.to_dict() for c in cases])
            if split_name in UNSEEN_EVAL_SPLITS:
                total_unseen += len(cases)
            split_digests[split_name] = {
                "task_count": len(cases),
                "depths": sorted({c.public.depth for c in cases}),
                "generator_id": cases[0].public.generator_id if cases else "",
                "generator_version": cases[0].public.generator_version if cases else "",
                "public_sha256": sha256_hex(pub_bytes),
                "hidden_labels_sha256": sha256_hex(hid_bytes),
                "excluded_from_learning_score": split_name in CANARY_SPLITS,
            }
        heldout_by_seed[str(seed)] = {
            "seed": seed,
            "total_unseen_tasks": total_unseen,
            "total_canary_tasks": sum(len(suite[s]) for s in CANARY_SPLITS),
            "splits": split_digests,
        }

    train_manifest = {
        "schema_version": "s0-manifest-v1",
        "generator_version": GENERATOR_VERSION,
        "seeds": list(seeds),
        "demos_per_action": demos_per_action,
        "pool_disjointness": disjoint_cert,
        "per_seed": train_by_seed,
    }
    heldout_manifest = {
        "schema_version": "s0-manifest-v1",
        "generator_version": GENERATOR_VERSION,
        "temporal_generator_version": TEMPORAL_GENERATOR_VERSION,
        "seeds": list(seeds),
        "unseen_eval_splits": list(UNSEEN_EVAL_SPLITS),
        "canary_splits": list(CANARY_SPLITS),
        "pool_disjointness": disjoint_cert,
        "per_seed": heldout_by_seed,
    }
    return train_manifest, heldout_manifest
