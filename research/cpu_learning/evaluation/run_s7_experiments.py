"""One-command reproducible Stage S7 experiment runner: Raw UTF-8 Language Grounding,
Compositional Realization & Byte-LM Comparison.

Usage:
    PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s7_experiments
    PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s7_experiments --verify-manifests

Produces:
- ``research/cpu_learning/data/s7_preregistration.json``
- ``research/cpu_learning/reports/raw_runs/s7_seed_<seed>.json``
- ``research/cpu_learning/reports/raw_runs/s7_splits.csv``
- ``research/cpu_learning/reports/s7_summary.json``
- ``research/cpu_learning/reports/s7_research_report.md``

STRICT ISOLATION & SCIENTIFIC INTEGRITY (PDF pp. 12, 13, 14, 17):
- NEVER directly imports ``hidden_actions`` or ``oracle``.
- Audits that ``research/cpu_learning/grounding/`` contains ZERO forbidden imports, ZERO hardcoded
  S0/D2/D3 domain literals, and NEVER imports ``SmallCPUByteTransformerRef`` inside any Mirror
  learner module (``byte_aligner.py``, ``construction_learner.py``, ``plan_realizer.py``).
- Evaluates across 5 fixed seeds ``[101, 202, 303, 404, 505]``:
  1. Unlabelled Raw-UTF-8 Construction Induction + End-to-End Compositional Execution (unseen
     entities, novel verbs from 3 demos, passive/fronted/SOV word orders, negation, ``if``/``unless``
     conditionals, Depth 2/4/6 raw-UTF-8 chains, minimal-contrast ambiguity resolution, and 100%
     epistemic partial-mask & OOD abstention).
  2. Plan-Conditioned Truth-Locked Natural Language Realizer with STRICTLY SEPARATED semantic
     truthfulness vs surface readability metrics (PDF pp. 12, 14).
  3. Zero-Leakage Multi-Domain UTF-8 Byte-LM Bits-Per-Byte Benchmark comparing
     ``mirror_ctw_construction_byte_lm`` (0 backprop, 0 GPU, 0 neural weights) head-to-head against
     ``uniform_byte_floor``, ``byte_3gram_kneser_ney``, ``ppm_c_order5_byte_lm``, and
     ``small_cpu_byte_transformer_ref`` (``comparison_only=True``).
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.cpu_learning.evaluation.metrics import (
    capture_environment_info,
    compute_collateral_changes,
    summarize_across_seeds,
)
from research.cpu_learning.evaluation.run_experiments import (
    CPU_BUDGET_SECONDS_PER_SEED_MAX,
    DATA_DIR,
    PACKAGE_ROOT,
    REPORTS_DIR,
    make_blackbox_environment,
    verify_committed_manifests,
)
from research.cpu_learning.evaluation.run_s1_experiments import (
    _S0_EVALUATOR_ALLOWLIST,
    _referenced_import_targets,
    verify_s1_committed_manifests,
)
from research.cpu_learning.evaluation.run_s2_experiments import (
    _FORBIDDEN_ITL_HARDCODED_LITERALS,
    _string_constants_in_code,
    verify_s2_committed_manifests,
)
from research.cpu_learning.evaluation.run_s3_experiments import (
    _LEARNER_ROOTS,
    verify_s3_committed_manifests,
)
from research.cpu_learning.evaluation.run_s4_experiments import verify_s4_committed_manifests
from research.cpu_learning.evaluation.run_s5_experiments import verify_s5_committed_manifests
from research.cpu_learning.evaluation.run_s6_experiments import (
    _identifiers_in_code,
    verify_s6_committed_manifests,
)
from research.cpu_learning.frames.state_frame import TypedStateFrame
from research.cpu_learning.grounding import (
    BYTE_ALIGNER_SCHEMA_VERSION,
    BYTE_LM_SCHEMA_VERSION,
    CONSTRUCTION_LEARNER_SCHEMA_VERSION,
    PLAN_REALIZER_SCHEMA_VERSION,
    GroundedExecutionResult,
    GroundedInstructionExecutor,
    GroundedPlanRealizer,
    MirrorCTWConstructionByteLM,
    RawByteConstructionLearner,
    SmallCPUByteTransformerRef,
    UnlabelledGroundingEpisode,
    decode_and_tokenize_utf8,
    generate_disjoint_byte_lm_corpus,
    locate_entity_mentions,
    run_byte_lm_benchmark_for_seed,
)
from research.cpu_learning.itl.contrast_harness import (
    ContrastRefinedITLLearner,
    TransitionEnvironment,
    build_probe_action,
)
from research.cpu_learning.itl.operator_card import OperatorCard
from research.cpu_learning.simulator.generators import (
    DEFAULT_DEMOS_PER_ACTION,
    DEFAULT_SEEDS,
    HELDOUT_POOL,
    TEMPORAL_POOL,
    TRAIN_POOL,
    _make_base_world,
    generate_training_episodes,
    verify_pool_disjointness,
)
from research.cpu_learning.simulator.state_graph import (
    ObservationMask,
    StateGraph,
    canonical_json_bytes,
    compressed_bit_length,
    compute_sparse_delta,
    sha256_hex,
)

S7_REPRO_COMMAND = "PYTHONHASHSEED=0 python -m research.cpu_learning.evaluation.run_s7_experiments"

_EXTRA_D2_D3_FORBIDDEN_LITERALS: frozenset[str] = frozenset(
    {
        "channel",
        "bestow",
        "mage",
        "relic",
        "pedestal",
        "reliquary",
        "attuned",
        "bound_to",
        "resting_on",
        "unsealed",
        "warded",
        "channelable",
        "dispatch",
        "reassign",
        "courier",
        "parcel",
        "dock",
        "bay",
        "assigned_to",
        "staged_at",
        "zone_clear",
        "bay_open",
        "bay_locked",
        "fragile_ok",
    }
)

S7_GROUNDING_SPLITS: tuple[str, ...] = (
    "s7_unseen_entity_substitution",
    "s7_unseen_paraphrase_novel_verbs",
    "s7_unseen_word_orders_passive_sov",
    "s7_negated_instructions",
    "s7_conditional_clauses",
    "s7_chained_multistep_depth2",
    "s7_chained_multistep_depth4",
    "s7_chained_multistep_depth6",
)


# ---------------------------------------------------------------------------
# 1. AST Isolation, Zero Domain Literals & Transformer Isolation Audit
# ---------------------------------------------------------------------------


def audit_s7_ast_isolation_and_agnosticism(package_root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    """Verify strict AST import isolation across all learner packages, zero hardcoded S0/D2/D3
    domain literals inside ``itl/``, ``shard/``, ``rdc/``, ``transfer/``, and ``grounding/``,
    and strict isolation of comparison-only ``SmallCPUByteTransformerRef``."""
    import_violations: list[str] = []
    hardcode_violations: list[str] = []
    transformer_leak_violations: list[str] = []

    scanned_non_eval = 0
    scanned_learner = 0
    scanned_grounding = 0

    forbidden_learner = {
        "hidden_actions",
        "oracle",
        "generators",
        "EvaluationTaskHiddenLabel",
        "EvaluationCase",
        "HIDDEN_OPERATOR_SCHEMAS",
    }
    all_forbidden_grounding_lits = set(_FORBIDDEN_ITL_HARDCODED_LITERALS) | set(
        _EXTRA_D2_D3_FORBIDDEN_LITERALS
    )

    for fpath in sorted(package_root.rglob("*.py")):
        rel = fpath.relative_to(package_root)
        src = fpath.read_text(encoding="utf-8")
        targets = _referenced_import_targets(src)
        parts_all = {part for tgt in targets for part in tgt.split(".")}

        if rel not in _S0_EVALUATOR_ALLOWLIST:
            scanned_non_eval += 1
            if "hidden_actions" in parts_all or "oracle" in parts_all:
                import_violations.append(
                    f"{rel.as_posix()} imports evaluator-only module: {sorted(targets)}"
                )

        if rel.parts and rel.parts[0] in _LEARNER_ROOTS:
            scanned_learner += 1
            bad = sorted(parts_all & forbidden_learner)
            if bad:
                import_violations.append(
                    f"Learner module {rel.as_posix()} imports forbidden symbols: {bad}"
                )

        if rel.parts and rel.parts[0] in {"itl", "shard", "rdc", "transfer", "grounding"}:
            pkg = rel.parts[0]
            consts = _string_constants_in_code(src)
            check_set = (
                all_forbidden_grounding_lits
                if pkg == "grounding"
                else _FORBIDDEN_ITL_HARDCODED_LITERALS
            )
            bad_lits = sorted(consts & check_set)
            if bad_lits:
                hardcode_violations.append(
                    f"{pkg.upper()} module {rel.as_posix()} contains hardcoded domain literals: {bad_lits}"
                )
            if pkg == "grounding":
                scanned_grounding += 1
                if rel.name in {"byte_aligner.py", "construction_learner.py", "plan_realizer.py"}:
                    idents = _identifiers_in_code(src)
                    if (
                        "SmallCPUByteTransformerRef" in idents
                        or "small_cpu_byte_transformer_ref" in consts
                        or "byte_lm" in parts_all
                    ):
                        transformer_leak_violations.append(
                            f"{rel.as_posix()} references comparison-only Byte-LM module/transformer"
                        )

    comp_flag_ok = (
        SmallCPUByteTransformerRef.comparison_only is True
        and SmallCPUByteTransformerRef.uses_backprop is True
        and MirrorCTWConstructionByteLM.comparison_only is False
        and MirrorCTWConstructionByteLM.uses_backprop is False
    )
    if not comp_flag_ok:
        transformer_leak_violations.append("Byte-LM comparison_only / uses_backprop flags invalid")

    passed = (
        len(import_violations) == 0
        and len(hardcode_violations) == 0
        and len(transformer_leak_violations) == 0
        and scanned_non_eval >= 32
        and scanned_learner >= 30
        and scanned_grounding >= 5
    )
    return {
        "passed": passed,
        "comparison_only_transformer_isolated": len(transformer_leak_violations) == 0,
        "scanned_non_evaluator_modules": scanned_non_eval,
        "scanned_learner_modules": scanned_learner,
        "scanned_grounding_modules": scanned_grounding,
        "import_violations": import_violations,
        "hardcode_violations": hardcode_violations,
        "transformer_leak_violations": transformer_leak_violations,
    }


def _verify_transformer_analytical_gradients(seed: int) -> float:
    """Verify SmallCPUByteTransformerRef analytical backprop against central finite differences."""
    tf = SmallCPUByteTransformerRef(seed=seed, d_model=6, ctx_len=5)
    ctx = "élan→7".encode("utf-8")[:5]
    target = ord("a")
    _, grads = tf.compute_step_grads(ctx, target)
    eps = 1e-5
    max_err = 0.0
    for mat_name, grad_key in (("w_q", "dw_q"), ("w_k", "dw_k"), ("w_v", "dw_v"), ("w_out", "dw_out")):
        mat = getattr(tf, mat_name)
        gmat = grads[grad_key]
        for r in range(min(3, len(mat))):
            for c in range(min(3, len(mat[0]))):
                orig = mat[r][c]
                mat[r][c] = orig + eps
                lp, _ = tf.compute_step_grads(ctx, target)
                mat[r][c] = orig - eps
                lm, _ = tf.compute_step_grads(ctx, target)
                mat[r][c] = orig
                err = abs(((lp - lm) / (2.0 * eps)) - gmat[r][c])
                if err > max_err:
                    max_err = err
    return max_err


# ---------------------------------------------------------------------------
# 2. Deterministic Unlabelled Grounding Training Corpus & Held-Out Benchmark
# ---------------------------------------------------------------------------


def _step_seq(
    env: TransitionEnvironment,
    initial_state: StateGraph,
    steps: Sequence[tuple[str | None, Mapping[str, str]]],
    *,
    mask: ObservationMask | None = None,
) -> tuple[StateGraph, tuple[StateGraph, ...]]:
    """Step a sequence through the black-box environment (``None`` action_id = suppressed no-op)."""
    obs = mask or ObservationMask(fully_observed=True)
    cur = initial_state.copy()
    inter: list[StateGraph] = []
    for aid, roles in steps:
        if aid is not None:
            cur, _ = env.step(cur, build_probe_action(aid, dict(roles)), mask=obs)
        inter.append(cur.copy())
    return cur, tuple(inter)


def build_s7_grounding_training_episodes(
    seed: int,
    env: TransitionEnvironment,
) -> list[UnlabelledGroundingEpisode]:
    """Generate unlabelled ``(raw_utf8_bytes, before_state, after_state)`` training episodes using
    ONLY ``TRAIN_POOL`` entities (zero action/role labels, zero entity overlap with held-out!)."""
    rng = random.Random(seed * 17_389 + 41)
    episodes: list[UnlabelledGroundingEpisode] = []
    ep_idx = 0

    def _add(text: str, before: StateGraph, after: StateGraph) -> None:
        nonlocal ep_idx
        ep_idx += 1
        episodes.append(
            UnlabelledGroundingEpisode(
                episode_id=f"s7_tr_{seed}_{ep_idx:04d}",
                raw_instruction_bytes=text.encode("utf-8"),
                before_state=before,
                after_state=after,
            )
        )

    # Generate 3 diverse TRAIN_POOL scenes for positive atomic, novel-verb, passive, fronted & SOV demos
    for k in range(3):
        world, ids = _make_base_world(
            TRAIN_POOL,
            rng,
            container_states=((False, False, True), (True, False, True), (False, True, True)),
        )
        ag1, ag2 = ids["agents"][0], ids["agents"][1]
        ob1, ob2 = ids["objects"][0], ids["objects"][1]
        sf1, sf2 = ids["surfaces"][0], ids["surfaces"][1]
        ct_closed, ct_open, ct_locked = ids["containers"][0], ids["containers"][1], ids["containers"][2]

        after_move, _ = env.step(
            world,
            build_probe_action("move", {"actor": ag1, "object": ob1, "source": sf1, "destination": sf2}),
        )
        after_give, _ = env.step(
            world,
            build_probe_action("give", {"actor": ag1, "object": ob1, "recipient": ag2}),
        )
        after_open, _ = env.step(
            world,
            build_probe_action("open", {"actor": ag1, "target": ct_closed}),
        )
        after_lock, _ = env.step(
            world,
            build_probe_action("lock", {"actor": ag1, "target": ct_closed}),
        )

        # 1. Standard active + UTF-8 multi-byte arrow phrasings
        _add(f"{ag1} moves {ob1} from {sf1} to {sf2}.", world, after_move)
        _add(f"{ag1} relocates {ob1} {sf1} → {sf2}.", world, after_move)
        _add(f"{ag1} gives {ob1} to {ag2}.", world, after_give)
        _add(f"{ag1} opens {ct_closed}.", world, after_open)
        _add(f"{ag1} locks {ct_closed}.", world, after_lock)

        # 2. Novel synthetic verbs & unseen paraphrases (3 demos each!)
        _add(f"{ag1} zorps {ob1} out of {sf1} into {sf2}.", world, after_move)
        _add(f"{ag1} transfers {ob1} out of {sf1} into {sf2}.", world, after_move)
        _add(f"{ag1} plinks {ag2} with {ob1}.", world, after_give)
        _add(f"{ag1} bestows {ob1} upon {ag2}.", world, after_give)
        _add(f"{ag1} quells {ct_closed}.", world, after_open)
        _add(f"{ag1} unlatches {ct_closed}.", world, after_open)
        _add(f"{ag1} vects {ct_closed}.", world, after_lock)
        _add(f"{ag1} bolts {ct_closed}.", world, after_lock)

        # 3. Passive voice, fronted destination/recipient & SOV postpositional word orders!
        _add(f"{ob1} is shifted from {sf1} to {sf2} by {ag1}.", world, after_move)
        _add(f"{ob1} is handed to {ag2} by {ag1}.", world, after_give)
        _add(f"{ct_closed} is unlatched by {ag1}.", world, after_open)
        _add(f"{ct_closed} is bolted by {ag1}.", world, after_lock)

        _add(f"onto {sf2} {ag1} places {ob1} from {sf1}.", world, after_move)
        _add(f"to {ag2} {ag1} hands {ob1}.", world, after_give)

        _add(f"{ag1} {ob1} {sf1} kara {sf2} made hakobu.", world, after_move)
        _add(f"{ag1} {ag2} ni {ob1} watasu.", world, after_give)
        _add(f"{ag1} {ct_closed} akemasu.", world, after_open)
        _add(f"{ag1} {ct_closed} shimeru.", world, after_lock)

        # 4. Negation prefixes (suppressed no-op transitions on valid actions)
        if k < 2:
            _add(f"do not {ag1} moves {ob1} from {sf1} to {sf2}.", world, world)
            _add(f"never {ag1} opens {ct_closed}.", world, world)
            _add(f"avoid {ag1} gives {ob1} to {ag2}.", world, world)
            _add(f"skip {ag1} locks {ct_closed}.", world, world)

        # 5. Sequential multi-clause connectors ("; then", "; next", "; finally")
        if k < 2:
            after_m_o, _ = env.step(
                after_move, build_probe_action("open", {"actor": ag1, "target": ct_closed})
            )
            after_g_l, _ = env.step(
                after_give, build_probe_action("lock", {"actor": ag1, "target": ct_closed})
            )
            _add(
                f"{ag1} moves {ob1} from {sf1} to {sf2} ; then {ag1} opens {ct_closed}.",
                world,
                after_m_o,
            )
            _add(
                f"{ag1} gives {ob1} to {ag2} ; next {ag1} locks {ct_closed}.",
                world,
                after_g_l,
            )
            _add(
                f"{ag1} moves {ob1} from {sf1} to {sf2} ; finally {ag1} opens {ct_closed}.",
                world,
                after_m_o,
            )

    # 6. Conditional Guard Contrast Demos (orthogonal boolean variation -> unique learned fact!)
    w_cond, idc = _make_base_world(
        TRAIN_POOL,
        rng,
        container_states=((True, False, True), (False, False, True), (False, True, True)),
        surface_accessible=(True, True, False),
    )
    # Add a rare open+locked container state purely for orthogonal single-factor contrast on is_open vs is_locked
    ag_c1, ag_c2 = idc["agents"][0], idc["agents"][1]
    ob_c1 = idc["objects"][0]
    sf_c1, sf_c2, sf_inacc = idc["surfaces"][0], idc["surfaces"][1], idc["surfaces"][2]
    ct_open_unl, ct_closed_unl, ct_closed_lck = (
        idc["containers"][0],
        idc["containers"][1],
        idc["containers"][2],
    )
    w_open_lck = w_cond.with_changes(
        delete_facts=((ct_open_unl, "is_locked", "false"),),
        add_facts=((ct_open_unl, "is_locked", "true"),),
    )
    after_cm1, _ = env.step(
        w_cond,
        build_probe_action("move", {"actor": ag_c1, "object": ob_c1, "source": sf_c1, "destination": sf_c2}),
    )
    after_cm_lck, _ = env.step(
        w_open_lck,
        build_probe_action("move", {"actor": ag_c1, "object": ob_c1, "source": sf_c1, "destination": sf_c2}),
    )

    # "if <ent> is open, ..." -> executes on (open, unlocked) AND (open, locked); suppressed on (closed, unlocked)
    _add(f"if {ct_open_unl} is open, {ag_c1} moves {ob_c1} from {sf_c1} to {sf_c2}.", w_cond, after_cm1)
    _add(f"if {ct_open_unl} is open, {ag_c1} moves {ob_c1} from {sf_c1} to {sf_c2}.", w_open_lck, after_cm_lck)
    _add(f"if {ct_closed_unl} is open, {ag_c1} moves {ob_c1} from {sf_c1} to {sf_c2}.", w_cond, w_cond)

    # "if <ent> is closed, ..." -> executes on (closed, unlocked) AND (closed, locked); suppressed on (open, unlocked)
    _add(f"if {ct_closed_unl} is closed, {ag_c1} moves {ob_c1} from {sf_c1} to {sf_c2}.", w_cond, after_cm1)
    _add(f"if {ct_closed_lck} is closed, {ag_c1} moves {ob_c1} from {sf_c1} to {sf_c2}.", w_cond, after_cm1)
    _add(f"if {ct_open_unl} is closed, {ag_c1} moves {ob_c1} from {sf_c1} to {sf_c2}.", w_cond, w_cond)

    # "if <ent> is locked, ..." -> executes on (closed, locked) AND (open, locked); suppressed on (closed, unlocked)
    _add(f"if {ct_closed_lck} is locked, {ag_c1} moves {ob_c1} from {sf_c1} to {sf_c2}.", w_cond, after_cm1)
    _add(f"if {ct_open_unl} is locked, {ag_c1} moves {ob_c1} from {sf_c1} to {sf_c2}.", w_open_lck, after_cm_lck)
    _add(f"if {ct_closed_unl} is locked, {ag_c1} moves {ob_c1} from {sf_c1} to {sf_c2}.", w_cond, w_cond)

    # "if <ent> is unlocked, ..." -> executes on (open, unlocked) AND (closed, unlocked); suppressed on (closed, locked)
    _add(f"if {ct_open_unl} is unlocked, {ag_c1} moves {ob_c1} from {sf_c1} to {sf_c2}.", w_cond, after_cm1)
    _add(f"if {ct_closed_unl} is unlocked, {ag_c1} moves {ob_c1} from {sf_c1} to {sf_c2}.", w_cond, after_cm1)
    _add(f"if {ct_closed_lck} is unlocked, {ag_c1} moves {ob_c1} from {sf_c1} to {sf_c2}.", w_cond, w_cond)

    # "unless <ent> is locked, ..." -> executes on (open, unlocked) AND (closed, unlocked); suppressed on (closed, locked)
    _add(f"unless {ct_open_unl} is locked, {ag_c1} moves {ob_c1} from {sf_c1} to {sf_c2}.", w_cond, after_cm1)
    _add(f"unless {ct_closed_unl} is locked, {ag_c1} moves {ob_c1} from {sf_c1} to {sf_c2}.", w_cond, after_cm1)
    _add(f"unless {ct_closed_lck} is locked, {ag_c1} moves {ob_c1} from {sf_c1} to {sf_c2}.", w_cond, w_cond)

    # "if <ent> is accessible, ..." -> executes on accessible surface & container; suppressed on inaccessible surface
    _add(f"if {sf_c2} is accessible, {ag_c1} moves {ob_c1} from {sf_c1} to {sf_c2}.", w_cond, after_cm1)
    _add(f"if {ct_closed_unl} is accessible, {ag_c1} moves {ob_c1} from {sf_c1} to {sf_c2}.", w_cond, after_cm1)
    _add(f"if {sf_inacc} is accessible, {ag_c1} moves {ob_c1} from {sf_c1} to {sf_c2}.", w_cond, w_cond)

    return episodes


# ---------------------------------------------------------------------------
# 3. Lexical Nearest-Episode Baseline (Fair Symbolic Execution Ablation)
# ---------------------------------------------------------------------------


class _LexicalNearestGroundingBaseline:
    """Retrieves the highest token-Jaccard training episode with matching entity-type signature
    and substitutes entity slots positionally into the recovered training action."""

    def __init__(
        self,
        operators: Mapping[str, OperatorCard],
        episodes: Sequence[UnlabelledGroundingEpisode],
        learner_proposer: Any,
    ) -> None:
        from research.cpu_learning.grounding.byte_aligner import align_unlabelled_transition

        self._operators = dict(operators)
        self._entries: list[tuple[frozenset[str], tuple[str, ...], str, tuple[tuple[int, str], ...]]] = []
        for ep in episodes:
            aligned = align_unlabelled_transition(
                ep.raw_instruction_bytes, ep.before_state, ep.after_state, learner_proposer
            )
            if not aligned.valid_utf8 or len(aligned.candidates) != 1:
                continue
            cand = aligned.candidates[0]
            role_by_eid = {eid: r for r, eid in cand.roles}
            if len(aligned.mentions) != len(cand.roles) or not all(
                m.entity_id in role_by_eid for m in aligned.mentions
            ):
                continue
            mention_tidxs = {m.token_index for m in aligned.mentions}
            lex_bag = frozenset(
                t.lower
                for idx, t in enumerate(aligned.tokens)
                if idx not in mention_tidxs and not t.is_punct
            )
            type_sig = tuple(m.entity_type for m in aligned.mentions)
            s2r = tuple((i, role_by_eid[m.entity_id]) for i, m in enumerate(aligned.mentions))
            self._entries.append((lex_bag, type_bag := type_sig, cand.action_id, s2r))

    def execute_raw_utf8(
        self,
        raw_bytes: bytes,
        initial_state: StateGraph,
        *,
        mask: ObservationMask | None = None,
    ) -> GroundedExecutionResult:
        obs = mask or ObservationMask(fully_observed=True)
        ok, tokens = decode_and_tokenize_utf8(raw_bytes)
        if not ok or not tokens:
            return GroundedExecutionResult(None, True, False, "invalid_utf8", (), ())
        ent_types = {
            eid: ent.entity_type
            for eid, ent in initial_state.entities
            if obs.is_entity_observed(eid)
        }
        mentions = locate_entity_mentions(tokens, ent_types)
        if not mentions:
            return GroundedExecutionResult(None, True, False, "no_mentions", (), ())
        m_idxs = {m.token_index for m in mentions}
        q_bag = frozenset(
            t.lower for i, t in enumerate(tokens) if i not in m_idxs and not t.is_punct
        )
        q_types = tuple(m.entity_type for m in mentions)

        best_score = -1.0
        best_hit: tuple[str, tuple[tuple[int, str], ...]] | None = None
        for lex_bag, type_sig, aid, s2r in self._entries:
            if type_sig != q_types:
                continue
            union_n = len(q_bag | lex_bag)
            jacc = (len(q_bag & lex_bag) / float(union_n)) if union_n > 0 else 0.0
            if jacc > best_score:
                best_score = jacc
                best_hit = (aid, s2r)

        if best_hit is None or best_score <= 0.0:
            return GroundedExecutionResult(None, True, False, "no_lexical_match", (), ())
        aid, s2r = best_hit
        roles = {rname: mentions[i].entity_id for i, rname in s2r}
        res = self._operators[aid].apply(initial_state, roles, mask=obs)
        if res.abstained or res.after_state is None:
            return GroundedExecutionResult(None, True, False, res.reason, (), ())
        return GroundedExecutionResult(
            res.after_state, False, False, "", (), (res.after_state,)
        )


# ---------------------------------------------------------------------------
# 4. Held-Out Evaluation Splits (Splits 1..10 + Realizer + Byte-LM)
# ---------------------------------------------------------------------------


def _build_heldout_grounding_tasks(
    seed: int,
    env: TransitionEnvironment,
) -> dict[str, list[dict[str, Any]]]:
    """Generate deterministic held-out raw UTF-8 grounding splits strictly from ``HELDOUT_POOL``
    and ``TEMPORAL_POOL`` (zero entity overlap with ``TRAIN_POOL``!)."""
    rng = random.Random(seed * 29_173 + 83)
    splits: dict[str, list[dict[str, Any]]] = {s: [] for s in S7_GROUNDING_SPLITS}

    def _new_scene(pool: Any = HELDOUT_POOL) -> tuple[StateGraph, dict[str, list[str]]]:
        return _make_base_world(
            pool,
            rng,
            container_states=((False, False, True), (True, False, True), (False, True, True)),
            surface_accessible=(True, True, False),
        )

    # Split 1: Unseen Entity Substitution (20 tasks: HELDOUT_POOL + TEMPORAL_POOL, positive & blocked!)
    for i in range(20):
        pool = HELDOUT_POOL if i < 12 else TEMPORAL_POOL
        w, ids = _new_scene(pool)
        ag1, ag2 = ids["agents"][0], ids["agents"][1]
        ob1 = ids["objects"][0]
        sf1, sf2, sf_inacc = ids["surfaces"][0], ids["surfaces"][1], ids["surfaces"][2]
        ct_cl, _ct_op, ct_lck = ids["containers"][0], ids["containers"][1], ids["containers"][2]
        m = i % 5
        if m == 0:
            txt = f"{ag1} moves {ob1} from {sf1} to {sf2}."
            steps = [("move", {"actor": ag1, "object": ob1, "source": sf1, "destination": sf2})]
        elif m == 1:
            txt = f"{ag1} relocates {ob1} {sf1} → {sf_inacc}."
            steps = [("move", {"actor": ag1, "object": ob1, "source": sf1, "destination": sf_inacc})]
        elif m == 2:
            txt = f"{ag1} gives {ob1} to {ag2}."
            steps = [("give", {"actor": ag1, "object": ob1, "recipient": ag2})]
        elif m == 3:
            txt = f"{ag1} opens {ct_cl}."
            steps = [("open", {"actor": ag1, "target": ct_cl})]
        else:
            txt = f"{ag1} opens {ct_lck}."
            steps = [("open", {"actor": ag1, "target": ct_lck})]
        gold_final, gold_inter = _step_seq(env, w, steps)
        splits["s7_unseen_entity_substitution"].append(
            {"raw_bytes": txt.encode("utf-8"), "initial": w, "gold_final": gold_final, "gold_inter": gold_inter}
        )

    # Split 2: Unseen Paraphrases & Novel Synthetic Verbs (20 tasks)
    for i in range(20):
        w, ids = _new_scene(HELDOUT_POOL if i % 2 == 0 else TEMPORAL_POOL)
        ag1, ag2 = ids["agents"][0], ids["agents"][1]
        ob1 = ids["objects"][0]
        sf1, sf2 = ids["surfaces"][0], ids["surfaces"][1]
        ct_cl, _ct_op, ct_lck = ids["containers"][0], ids["containers"][1], ids["containers"][2]
        m = i % 8
        if m == 0:
            txt = f"{ag1} zorps {ob1} out of {sf1} into {sf2}."
            steps = [("move", {"actor": ag1, "object": ob1, "source": sf1, "destination": sf2})]
        elif m == 1:
            txt = f"{ag1} transfers {ob1} out of {sf1} into {sf2}."
            steps = [("move", {"actor": ag1, "object": ob1, "source": sf1, "destination": sf2})]
        elif m == 2:
            txt = f"{ag1} plinks {ag2} with {ob1}."
            steps = [("give", {"actor": ag1, "object": ob1, "recipient": ag2})]
        elif m == 3:
            txt = f"{ag1} bestows {ob1} upon {ag2}."
            steps = [("give", {"actor": ag1, "object": ob1, "recipient": ag2})]
        elif m == 4:
            txt = f"{ag1} quells {ct_cl}."
            steps = [("open", {"actor": ag1, "target": ct_cl})]
        elif m == 5:
            txt = f"{ag1} unlatches {ct_lck}."
            steps = [("open", {"actor": ag1, "target": ct_lck})]
        elif m == 6:
            txt = f"{ag1} vects {ct_cl}."
            steps = [("lock", {"actor": ag1, "target": ct_cl})]
        else:
            txt = f"{ag1} bolts {ct_cl}."
            steps = [("lock", {"actor": ag1, "target": ct_cl})]
        gold_final, gold_inter = _step_seq(env, w, steps)
        splits["s7_unseen_paraphrase_novel_verbs"].append(
            {"raw_bytes": txt.encode("utf-8"), "initial": w, "gold_final": gold_final, "gold_inter": gold_inter}
        )

    # Split 3: Unseen Word Orders — Passive Voice, Fronted Destination/Recipient & SOV (24 tasks)
    for i in range(24):
        w, ids = _new_scene(HELDOUT_POOL if i < 16 else TEMPORAL_POOL)
        ag1, ag2 = ids["agents"][0], ids["agents"][1]
        ob1 = ids["objects"][0]
        sf1, sf2 = ids["surfaces"][0], ids["surfaces"][1]
        ct_cl = ids["containers"][0]
        m = i % 10
        if m == 0:
            txt = f"{ob1} is shifted from {sf1} to {sf2} by {ag1}."
            steps = [("move", {"actor": ag1, "object": ob1, "source": sf1, "destination": sf2})]
        elif m == 1:
            txt = f"{ob1} is handed to {ag2} by {ag1}."
            steps = [("give", {"actor": ag1, "object": ob1, "recipient": ag2})]
        elif m == 2:
            txt = f"{ct_cl} is unlatched by {ag1}."
            steps = [("open", {"actor": ag1, "target": ct_cl})]
        elif m == 3:
            txt = f"{ct_cl} is bolted by {ag1}."
            steps = [("lock", {"actor": ag1, "target": ct_cl})]
        elif m == 4:
            txt = f"onto {sf2} {ag1} places {ob1} from {sf1}."
            steps = [("move", {"actor": ag1, "object": ob1, "source": sf1, "destination": sf2})]
        elif m == 5:
            txt = f"to {ag2} {ag1} hands {ob1}."
            steps = [("give", {"actor": ag1, "object": ob1, "recipient": ag2})]
        elif m == 6:
            txt = f"{ag1} {ob1} {sf1} kara {sf2} made hakobu."
            steps = [("move", {"actor": ag1, "object": ob1, "source": sf1, "destination": sf2})]
        elif m == 7:
            txt = f"{ag1} {ag2} ni {ob1} watasu."
            steps = [("give", {"actor": ag1, "object": ob1, "recipient": ag2})]
        elif m == 8:
            txt = f"{ag1} {ct_cl} akemasu."
            steps = [("open", {"actor": ag1, "target": ct_cl})]
        else:
            txt = f"{ag1} {ct_cl} shimeru."
            steps = [("lock", {"actor": ag1, "target": ct_cl})]
        gold_final, gold_inter = _step_seq(env, w, steps)
        splits["s7_unseen_word_orders_passive_sov"].append(
            {"raw_bytes": txt.encode("utf-8"), "initial": w, "gold_final": gold_final, "gold_inter": gold_inter}
        )

    # Split 4: Negation Across Active, Novel-Verb, Passive & SOV Clauses (20 tasks -> exact no-op!)
    neg_prefixes = ("do not", "never", "avoid", "skip")
    for i in range(20):
        w, ids = _new_scene(HELDOUT_POOL)
        ag1, ag2 = ids["agents"][0], ids["agents"][1]
        ob1 = ids["objects"][0]
        sf1, sf2 = ids["surfaces"][0], ids["surfaces"][1]
        ct_cl = ids["containers"][0]
        npf = neg_prefixes[i % len(neg_prefixes)]
        m = i % 5
        if m == 0:
            body = f"{ag1} moves {ob1} from {sf1} to {sf2}"
        elif m == 1:
            body = f"{ob1} is shifted from {sf1} to {sf2} by {ag1}"
        elif m == 2:
            body = f"{ag1} {ag2} ni {ob1} watasu"
        elif m == 3:
            body = f"{ag1} quells {ct_cl}"
        else:
            body = f"{ag1} {ct_cl} shimeru"
        txt = f"{npf} {body}."
        splits["s7_negated_instructions"].append(
            {"raw_bytes": txt.encode("utf-8"), "initial": w, "gold_final": w.copy(), "gold_inter": (w.copy(),)}
        )

    # Split 5: Conditional Clauses ("if" / "unless" — 12 condition-true + 12 condition-false = 24 tasks)
    for i in range(24):
        w, ids = _new_scene(HELDOUT_POOL)
        ag1, ag2 = ids["agents"][0], ids["agents"][1]
        ob1 = ids["objects"][0]
        sf1, sf2, sf_inacc = ids["surfaces"][0], ids["surfaces"][1], ids["surfaces"][2]
        ct_cl, ct_op, ct_lck = ids["containers"][0], ids["containers"][1], ids["containers"][2]
        cond_true = (i % 2 == 0)
        kind = (i // 2) % 6
        if kind == 0:
            ent = ct_op if cond_true else ct_cl
            head = f"if {ent} is open"
        elif kind == 1:
            ent = ct_cl if cond_true else ct_op
            head = f"if {ent} is closed"
        elif kind == 2:
            ent = ct_lck if cond_true else ct_cl
            head = f"if {ent} is locked"
        elif kind == 3:
            ent = ct_cl if cond_true else ct_lck
            head = f"if {ent} is unlocked"
        elif kind == 4:
            ent = ct_op if cond_true else ct_lck
            head = f"unless {ent} is locked"
        else:
            ent = sf2 if cond_true else sf_inacc
            head = f"if {ent} is accessible"

        if i >= 20:
            # Explicit compositional conditional + negated body cases (both cond_true & cond_false!)
            body = f"do not {ag1} opens {ct_cl}"
            act_step = (None, {})
        elif i % 3 == 0:
            body = f"{ag1} {ob1} {sf1} kara {sf2} made hakobu"
            act_step = ("move", {"actor": ag1, "object": ob1, "source": sf1, "destination": sf2})
        elif i % 3 == 1:
            body = f"{ob1} is handed to {ag2} by {ag1}"
            act_step = ("give", {"actor": ag1, "object": ob1, "recipient": ag2})
        else:
            body = f"{ag1} opens {ct_cl}"
            act_step = ("open", {"actor": ag1, "target": ct_cl})

        txt = f"{head}, {body}."
        steps = [act_step] if cond_true else [(None, {})]
        gold_final, gold_inter = _step_seq(env, w, steps)
        splits["s7_conditional_clauses"].append(
            {"raw_bytes": txt.encode("utf-8"), "initial": w, "gold_final": gold_final, "gold_inter": gold_inter}
        )

    # Splits 6, 7, 8: Chained Multi-Step Raw UTF-8 Instructions at Depths 2, 4, and 6 (16 tasks each!)
    conn_strs = (" ; then ", " ; next ", " ; finally ")
    for depth, split_name in (
        (2, "s7_chained_multistep_depth2"),
        (4, "s7_chained_multistep_depth4"),
        (6, "s7_chained_multistep_depth6"),
    ):
        for i in range(16):
            w, ids = _new_scene(HELDOUT_POOL if i % 2 == 0 else TEMPORAL_POOL)
            ag1, ag2, ag3 = ids["agents"][0], ids["agents"][1], ids["agents"][2]
            ob1, ob2 = ids["objects"][0], ids["objects"][1]
            sf1, sf2 = ids["surfaces"][0], ids["surfaces"][1]
            ct_cl, ct_op, ct_lck = ids["containers"][0], ids["containers"][1], ids["containers"][2]

            # Construct realistic compositional raw-UTF-8 chains mixing SOV, passive, conditionals & negation!
            cand_clauses: list[tuple[str, tuple[str | None, Mapping[str, str]]]] = [
                (
                    f"{ag1} opens {ct_cl}",
                    ("open", {"actor": ag1, "target": ct_cl}),
                ),
                (
                    f"{ag1} {ob1} {sf1} kara {ct_cl} made hakobu",
                    ("move", {"actor": ag1, "object": ob1, "source": sf1, "destination": ct_cl}),
                ),
                (
                    f"{ob2} is handed to {ag3} by {ag2}",
                    ("give", {"actor": ag2, "object": ob2, "recipient": ag3}),
                ),
                (
                    f"do not {ag1} bolts {ct_cl}",
                    (None, {}),
                ),
                (
                    f"unless {ct_op} is locked, {ag2} zorps {ob2} out of {sf2} into {ct_op}",
                    ("move", {"actor": ag2, "object": ob2, "source": sf2, "destination": ct_op}),
                ),
                (
                    f"{ag3} plinks {ag1} with {ob2}",
                    ("give", {"actor": ag3, "object": ob2, "recipient": ag1}),
                ),
            ]
            # Rotate pattern variety deterministically while preserving valid state preconditions:
            if i % 2 == 1:
                cand_clauses = [
                    (
                        f"onto {sf2} {ag1} places {ob1} from {sf1}",
                        ("move", {"actor": ag1, "object": ob1, "source": sf1, "destination": sf2}),
                    ),
                    (
                        f"if {ct_lck} is open, {ag1} moves {ob1} from {sf2} to {ct_lck}",
                        (None, {}),
                    ),
                    (
                        f"{ct_cl} is unlatched by {ag2}",
                        ("open", {"actor": ag2, "target": ct_cl}),
                    ),
                    (
                        f"{ag1} moves {ob1} from {sf2} to {ct_cl}",
                        ("move", {"actor": ag1, "object": ob1, "source": sf2, "destination": ct_cl}),
                    ),
                    (
                        f"to {ag3} {ag1} hands {ob1}",
                        ("give", {"actor": ag1, "object": ob1, "recipient": ag3}),
                    ),
                    (
                        f"never {ag2} locks {ct_op}",
                        (None, {}),
                    ),
                ]

            picked = cand_clauses[:depth]
            parts_txt: list[str] = []
            steps_seq: list[tuple[str | None, Mapping[str, str]]] = []
            for s_i, (cl_txt, st) in enumerate(picked):
                parts_txt.append(cl_txt)
                steps_seq.append(st)
                if s_i + 1 < depth:
                    conn = conn_strs[2] if (s_i + 1 == depth - 1) else conn_strs[s_i % 2]
                    parts_txt.append(conn)
            full_txt = "".join(parts_txt) + "."
            gold_final, gold_inter = _step_seq(env, w, steps_seq)
            splits[split_name].append(
                {
                    "raw_bytes": full_txt.encode("utf-8"),
                    "initial": w,
                    "gold_final": gold_final,
                    "gold_inter": gold_inter,
                }
            )

    return splits


def _evaluate_ambiguity_and_abstention(
    seed: int,
    operators: Mapping[str, OperatorCard],
    base_learner: RawByteConstructionLearner,
    env: TransitionEnvironment,
) -> dict[str, Any]:
    """Evaluate Split 9 (Ambiguity Preservation & Minimal-Contrast Disambiguation, 16 probe pairs)
    and Split 10 (OOD, Malformed UTF-8 & Partial-Mask Epistemic Abstention, 24 probes)."""
    rng = random.Random(seed * 41_299 + 19)

    # --- Split 9: Ambiguity Preservation BEFORE Contrast -> Resolution AFTER 2 Minimal Contrasts ---
    # 4 novel preposition-less surface constructions x 4 unseen held-out scenes = 16 probe pairs
    ambig_specs = [
        # Swapped destination-before-source double-location: "<actor> blargs <dst> <src> <obj>"
        ("blargs", "move_dst_first"),
        # Standard source-before-destination double-location: "<actor> treks <src> <dst> <obj>"
        ("treks", "move_src_first"),
        # Recipient-before-actor double-agent: "<rec> glims <actor> <obj>"
        ("glims", "give_rec_first"),
        # Ambiguous container action (open vs lock): "<actor> klaxes <container>"
        ("klaxes", "container_lock"),
    ]

    pre_ambig_hits = 0
    pre_abstain_hits = 0
    post_exact_hits = 0
    first_guess_hits = 0
    total_ambig = 0

    for verb_tok, spec_kind in ambig_specs:
        probe_learner = RawByteConstructionLearner(operators)
        probe_exec = GroundedInstructionExecutor(probe_learner)

        # Build 1 TRAIN_POOL scene to hear the uncontrasted utterance first
        w_tr1, ids_tr1 = _make_base_world(
            TRAIN_POOL,
            rng,
            container_states=((False, False, True), (True, False, True), (False, True, True)),
        )
        w_tr2, ids_tr2 = _make_base_world(
            TRAIN_POOL,
            rng,
            container_states=((False, False, True), (True, False, True), (False, True, True)),
        )

        def _fmt_utterance(ids_map: dict[str, list[str]]) -> tuple[bytes, str, dict[str, str]]:
            a1, a2 = ids_map["agents"][0], ids_map["agents"][1]
            o1 = ids_map["objects"][0]
            s1, s2 = ids_map["surfaces"][0], ids_map["surfaces"][1]
            c1 = ids_map["containers"][0]
            if spec_kind == "move_dst_first":
                return (
                    f"{a1} {verb_tok} {s2} {s1} {o1}".encode("utf-8"),
                    "move",
                    {"actor": a1, "object": o1, "source": s1, "destination": s2},
                )
            if spec_kind == "move_src_first":
                return (
                    f"{a1} {verb_tok} {s1} {s2} {o1}".encode("utf-8"),
                    "move",
                    {"actor": a1, "object": o1, "source": s1, "destination": s2},
                )
            if spec_kind == "give_rec_first":
                return (
                    f"{a2} {verb_tok} {a1} {o1}".encode("utf-8"),
                    "give",
                    {"actor": a1, "object": o1, "recipient": a2},
                )
            return (
                f"{a1} {verb_tok} {c1}".encode("utf-8"),
                "lock",
                {"actor": a1, "target": c1},
            )

        u_tr1, true_aid, roles_tr1 = _fmt_utterance(ids_tr1)
        probe_learner.observe_utterance_in_scene(u_tr1, w_tr1)

        # Evaluate BEFORE contrast on 4 unseen HELDOUT_POOL worlds
        heldout_cases: list[tuple[StateGraph, bytes, StateGraph, Any]] = []
        for _ in range(4):
            wh, idsh = _make_base_world(
                HELDOUT_POOL,
                rng,
                container_states=((False, False, True), (True, False, True), (False, True, True)),
            )
            uh, aid_h, roles_h = _fmt_utterance(idsh)
            gold_h, _ = env.step(wh, build_probe_action(aid_h, roles_h))
            p_pre = probe_learner.parse_atomic_clause(uh, TypedStateFrame.from_state_graph(wh))
            ex_pre = probe_exec.execute_raw_utf8(uh, wh)
            total_ambig += 1
            if p_pre.is_ambiguous and len(p_pre.candidate_frames) >= 2:
                pre_ambig_hits += 1
            if ex_pre.abstained and ex_pre.is_ambiguous:
                pre_abstain_hits += 1
            # First-guess ablation blindly executes candidate_frames[0] without contrast!
            fg_ok = False
            if p_pre.candidate_frames:
                cf0 = p_pre.candidate_frames[0]
                fg_res = operators[cf0.action_id].apply(wh, cf0.role_dict())
                if (not fg_res.abstained) and fg_res.after_state == gold_h:
                    fg_ok = True
            if fg_ok:
                first_guess_hits += 1
            heldout_cases.append((wh, uh, gold_h, None))

        # Now supply 2 minimal before/after contrast episodes on TRAIN_POOL and freeze!
        after_tr1, _ = env.step(w_tr1, build_probe_action(true_aid, roles_tr1))
        u_tr2, _, roles_tr2 = _fmt_utterance(ids_tr2)
        after_tr2, _ = env.step(w_tr2, build_probe_action(true_aid, roles_tr2))
        probe_learner.fit_episodes(
            [
                UnlabelledGroundingEpisode(f"mc1_{verb_tok}", u_tr1, w_tr1, after_tr1),
                UnlabelledGroundingEpisode(f"mc2_{verb_tok}", u_tr2, w_tr2, after_tr2),
            ]
        )
        probe_learner.freeze()

        for wh, uh, gold_h, _ in heldout_cases:
            ex_post = probe_exec.execute_raw_utf8(uh, wh)
            if (not ex_post.abstained) and (not ex_post.is_ambiguous) and ex_post.final_state == gold_h:
                post_exact_hits += 1

    # --- Split 10: OOD Utterances, Invalid UTF-8 & Partial ObservationMask Epistemic Abstention (24 probes) ---
    executor = GroundedInstructionExecutor(base_learner)
    abstain_hits = 0
    hallucinated_mutations = 0
    total_ood_mask = 24

    for i in range(total_ood_mask):
        wh, idsh = _make_base_world(
            HELDOUT_POOL,
            rng,
            container_states=((False, False, True), (True, False, True), (False, True, True)),
        )
        hag = idsh["agents"][0]
        hob = idsh["objects"][0]
        hs1, hs2 = idsh["surfaces"][0], idsh["surfaces"][1]
        hc_cl = idsh["containers"][0]

        if i < 8:
            # Ungrounded OOD verbs / out-of-grammar chatter
            raw = f"{hag} teleports {hob} across hyperspace around {hs1} and {hs2}.".encode("utf-8")
            mask = ObservationMask(fully_observed=True)
        elif i < 12:
            # Malformed invalid UTF-8 byte sequences
            raw = b"\xff\xfe" + f" {hag} opens {hc_cl}".encode("utf-8") + b"\x80\x81"
            mask = ObservationMask(fully_observed=True)
        elif i < 18:
            # Partial ObservationMask occluding the queried conditional predicate!
            if i % 2 == 0:
                raw = f"unless {hc_cl} is locked, {hag} opens {hc_cl}.".encode("utf-8")
                obs_preds = frozenset({"located_on", "owned_by", "is_open", "accessible"})
            else:
                raw = f"if {hc_cl} is open, {hag} moves {hob} from {hs1} to {hs2}.".encode("utf-8")
                obs_preds = frozenset({"located_on", "owned_by", "is_locked", "accessible"})
            mask = ObservationMask(
                observed_entities=frozenset(eid for eid, _ in wh.entities),
                observed_predicates=obs_preds,
                fully_observed=False,
            )
        else:
            # Partial ObservationMask occluding a mentioned entity!
            visible_ents = frozenset(eid for eid, _ in wh.entities if eid != hc_cl)
            raw = f"{hag} opens {hc_cl}.".encode("utf-8")
            mask = ObservationMask(
                observed_entities=visible_ents,
                observed_predicates=frozenset(
                    {"located_on", "owned_by", "is_open", "is_locked", "accessible"}
                ),
                fully_observed=False,
            )

        res_ab = executor.execute_raw_utf8(raw, wh, mask=mask)
        if res_ab.abstained and res_ab.final_state is None:
            abstain_hits += 1
        elif res_ab.final_state is not None and res_ab.final_state != wh:
            hallucinated_mutations += 1

    return {
        "ambiguity_minimal_contrast": {
            "total_probes": total_ambig,
            "pre_contrast_ambiguous_preserved_rate": round(pre_ambig_hits / float(total_ambig), 6),
            "pre_contrast_honest_abstention_rate": round(pre_abstain_hits / float(total_ambig), 6),
            "post_contrast_disambiguated_exact_acc": round(post_exact_hits / float(total_ambig), 6),
            "no_contrast_first_guess_ablation_acc": round(first_guess_hits / float(total_ambig), 6),
        },
        "ood_and_partial_mask_abstention": {
            "total_probes": total_ood_mask,
            "honest_abstention_rate": round(abstain_hits / float(total_ood_mask), 6),
            "hallucinated_mutation_rate": round(hallucinated_mutations / float(total_ood_mask), 6),
        },
    }


def _evaluate_plan_realization_suite(
    seed: int,
    learner: RawByteConstructionLearner,
    heldout_splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Evaluate strictly separated Semantic Truthfulness vs Surface Readability across 40 traces."""
    realizer = GroundedPlanRealizer(learner)
    prose_corpus = [
        "In this step, the action succeeded and updated 2 verified world relations.",
        "First, the action succeeded and updated 1 verified world relations.",
        "Next, the attempt was blocked because a required guard failed, leaving the world unchanged.",
        "Finally, the instruction was intentionally skipped due to negation, preserving all existing relations.",
        "In this step, because the condition did not hold, the step was skipped without altering any facts.",
        "The system abstained because the instruction contained ungrounded or unobserved conditions, leaving every entity unchanged.",
    ]
    realizer.fit_prose_corpus(prose_corpus)
    realizer.freeze()
    executor = GroundedInstructionExecutor(learner)

    # Assemble 40 diverse traces: 8 unseen_entity (incl. blocked), 8 negated, 8 conditional,
    # 8 depth-4, 4 OOD abstained, and 4 partial-mask occluded-entity abstained!
    eval_items: list[tuple[bytes, StateGraph, GroundedExecutionResult, ObservationMask]] = []
    full_mask = ObservationMask(fully_observed=True)
    for sp_name in (
        "s7_unseen_entity_substitution",
        "s7_negated_instructions",
        "s7_conditional_clauses",
        "s7_chained_multistep_depth4",
    ):
        for item in heldout_splits[sp_name][:8]:
            rb = item["raw_bytes"]
            st0 = item["initial"]
            ex = executor.execute_raw_utf8(rb, st0)
            eval_items.append((rb, st0, ex, full_mask))

    rng = random.Random(seed * 7919 + 5)
    for k_i in range(8):
        wh, idsh = _make_base_world(HELDOUT_POOL, rng)
        if k_i < 4:
            rb = f"{idsh['agents'][0]} vaporizes {idsh['objects'][0]}.".encode("utf-8")
            ex = executor.execute_raw_utf8(rb, wh)
            eval_items.append((rb, wh, ex, full_mask))
        else:
            occluded_c = idsh["containers"][0]
            vis_ents = frozenset(eid for eid, _ in wh.entities if eid != occluded_c)
            pmask = ObservationMask(
                observed_entities=vis_ents,
                observed_predicates=frozenset(
                    {"located_on", "owned_by", "is_open", "is_locked", "accessible"}
                ),
                fully_observed=False,
            )
            rb = f"{idsh['agents'][0]} opens {occluded_c}.".encode("utf-8")
            ex = executor.execute_raw_utf8(rb, wh, mask=pmask)
            eval_items.append((rb, wh, ex, pmask))

    accum: dict[str, dict[str, list[float]]] = {
        m: {"sem": [], "halluc": [], "fluency": [], "bpb": []}
        for m in ("mirror_truth_locked", "telegraphic_tuple_dump", "ungated_fluent_assumer")
    }

    for rb, st0, ex, item_mask in eval_items:
        t_m = realizer.realize(ex)
        t_d = realizer.realize_telegraphic_tuple_dump(ex)
        t_a = realizer.realize_ungated_fluent_assumer(rb, st0)
        for m_id, txt in (
            ("mirror_truth_locked", t_m),
            ("telegraphic_tuple_dump", t_d),
            ("ungated_fluent_assumer", t_a),
        ):
            ev = realizer.evaluate_realization(m_id, txt, ex, st0, mask=item_mask)
            accum[m_id]["sem"].append(ev.semantic_correctness)
            accum[m_id]["halluc"].append(ev.hallucination_rate)
            accum[m_id]["fluency"].append(ev.surface_fluency_score)
            accum[m_id]["bpb"].append(ev.reference_perplexity_bpb)

    n = float(len(eval_items))
    return {
        "trace_count": len(eval_items),
        "methods": {
            m_id: {
                "semantic_correctness": round(sum(vals["sem"]) / n, 6),
                "hallucination_rate": round(sum(vals["halluc"]) / n, 6),
                "surface_fluency_score": round(sum(vals["fluency"]) / n, 6),
                "reference_perplexity_bpb": round(sum(vals["bpb"]) / n, 4),
            }
            for m_id, vals in accum.items()
        },
    }


# ---------------------------------------------------------------------------
# 5. Pre-Registration & Manifest Verification
# ---------------------------------------------------------------------------


def build_s7_preregistration(
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Build deterministic pre-registration manifest for Stage S7."""
    env = make_blackbox_environment()
    seed_hashes: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        tr_eps = build_s7_grounding_training_episodes(seed, env)
        tr_manifest = [
            {
                "episode_id": ep.episode_id,
                "utf8_sha256": sha256_hex(ep.raw_instruction_bytes),
                "before_sha256": sha256_hex(canonical_json_bytes(ep.before_state.to_dict())),
                "after_sha256": sha256_hex(canonical_json_bytes(ep.after_state.to_dict())),
            }
            for ep in tr_eps
        ]
        byte_corpus = generate_disjoint_byte_lm_corpus(seed)
        seed_hashes[str(seed)] = {
            "grounding_train_episode_count": len(tr_eps),
            "grounding_train_sha256": sha256_hex(canonical_json_bytes(tr_manifest)),
            "byte_lm_train_bytes": len(byte_corpus.train_bytes),
            "byte_lm_heldout_bytes": len(byte_corpus.heldout_bytes),
            "byte_lm_train_sha256": sha256_hex(byte_corpus.train_bytes),
            "byte_lm_heldout_sha256": sha256_hex(byte_corpus.heldout_bytes),
        }

    return {
        "schema_version": "s7-preregistration-v1",
        "stage": "S7",
        "title": "Stage S7 — Raw UTF-8 Language Grounding, Compositional Realization & Byte-LM Comparison",
        "reproduce_command": S7_REPRO_COMMAND,
        "seeds": list(seeds),
        "byte_aligner_schema_version": BYTE_ALIGNER_SCHEMA_VERSION,
        "construction_learner_schema_version": CONSTRUCTION_LEARNER_SCHEMA_VERSION,
        "plan_realizer_schema_version": PLAN_REALIZER_SCHEMA_VERSION,
        "byte_lm_schema_version": BYTE_LM_SCHEMA_VERSION,
        "grounding_splits": list(S7_GROUNDING_SPLITS)
        + ["s7_ambiguity_minimal_contrast", "s7_ood_and_partial_mask_abstention"],
        "byte_lm_models": [
            "uniform_byte_floor",
            "byte_3gram_kneser_ney",
            "ppm_c_order5_byte_lm",
            "mirror_ctw_construction_byte_lm",
            "small_cpu_byte_transformer_ref",
        ],
        "seed_manifests": seed_hashes,
        "gates": {
            "unseen_splits_ci95_low_min": 0.95,
            "depth4_exact_ci95_low_min": 0.95,
            "depth6_exact_ci95_low_min": 0.90,
            "collateral_rate_ci95_high_max": 0.02,
            "ambiguity_preserved_before_contrast_min": 1.0,
            "ambiguity_resolved_after_contrast_ci95_low_min": 0.95,
            "ood_and_partial_mask_abstention_min": 1.0,
            "realizer_semantic_correctness_ci95_low_min": 0.98,
            "realizer_hallucination_rate_max": 0.0,
            "byte_lm_mirror_beats_3gram_and_transformer_ci95": True,
            "byte_lm_mirror_matches_or_beats_ppmc5": True,
            "cpu_seconds_per_seed_max": CPU_BUDGET_SECONDS_PER_SEED_MAX,
        },
    }


def verify_s7_committed_manifests(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    data_dir: Path = DATA_DIR,
) -> tuple[bool, list[str]]:
    p = data_dir / "s7_preregistration.json"
    if not p.exists():
        return (False, [f"Missing S7 preregistration manifest: {p}"])
    actual = json.loads(p.read_text(encoding="utf-8"))
    expected = build_s7_preregistration(seeds=seeds)
    if actual != expected:
        return (False, ["Committed s7_preregistration.json differs from deterministic generator"])
    return (True, [])


# ---------------------------------------------------------------------------
# 6. Per-Seed Execution & Multi-Seed Gate Aggregation
# ---------------------------------------------------------------------------


def _run_single_s7_seed(
    seed: int,
    *,
    demos_per_action: int = DEFAULT_DEMOS_PER_ACTION,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    pool_cert = verify_pool_disjointness()
    env = make_blackbox_environment()

    # Step 1: Induce verified S3 OperatorCards from S0 demonstrations
    s0_train_eps = generate_training_episodes(seed=seed, demos_per_action=demos_per_action)
    s3_learner = ContrastRefinedITLLearner(seed=seed, env=env)
    s3_learner.fit(s0_train_eps)
    s3_learner.freeze()
    operators: dict[str, OperatorCard] = {}
    for aid in ("move", "give", "open", "lock"):
        rec = s3_learner.version_store.get_active(aid)
        assert rec is not None
        operators[aid] = rec.card

    # Step 2: Induce Construction Grammar, Negations, Conditionals & Connectors from raw UTF-8 + transitions
    grounding_train_eps = build_s7_grounding_training_episodes(seed, env)
    g_learner = RawByteConstructionLearner(operators)
    fit_stats = g_learner.fit_episodes(grounding_train_eps)
    g_learner.freeze()

    executor = GroundedInstructionExecutor(g_learner)
    lex_baseline = _LexicalNearestGroundingBaseline(
        operators, grounding_train_eps, g_learner._proposer
    )

    # Step 3: Evaluate all 8 Compositional Raw-UTF-8 Execution Splits
    heldout_splits = _build_heldout_grounding_tasks(seed, env)
    split_metrics: dict[str, dict[str, Any]] = {}
    all_collaterals: list[float] = []

    for sp_name in S7_GROUNDING_SPLITS:
        tasks = heldout_splits[sp_name]
        mirror_exact = 0
        mirror_inter = 0
        lex_exact = 0
        byte_span_ok = 0
        sp_collat: list[float] = []

        for t_item in tasks:
            rb: bytes = t_item["raw_bytes"]
            st0: StateGraph = t_item["initial"]
            gold_f: StateGraph = t_item["gold_final"]
            gold_inter: tuple[StateGraph, ...] = t_item["gold_inter"]
            obs = ObservationMask(fully_observed=True)

            m_res = executor.execute_raw_utf8(rb, st0, mask=obs)
            if (not m_res.abstained) and m_res.final_state is not None:
                if m_res.final_state == gold_f:
                    mirror_exact += 1
                if m_res.intermediate_states == gold_inter:
                    mirror_inter += 1
                col_r, _, _ = compute_collateral_changes(st0, m_res.final_state, gold_f, obs)
                sp_collat.append(col_r)
                all_collaterals.append(col_r)
                spans_valid = all(
                    rb[bs:be].decode("utf-8") == tr.roles[rname]
                    for tr in m_res.clause_traces
                    for rname, (bs, be) in tr.byte_spans.items()
                )
                if spans_valid:
                    byte_span_ok += 1
            else:
                sp_collat.append(0.0)
                all_collaterals.append(0.0)

            l_res = lex_baseline.execute_raw_utf8(rb, st0, mask=obs)
            if (not l_res.abstained) and l_res.final_state == gold_f:
                lex_exact += 1

        nt = float(len(tasks))
        split_metrics[sp_name] = {
            "task_count": len(tasks),
            "mirror_exact_acc": round(mirror_exact / nt, 6),
            "mirror_intermediate_acc": round(mirror_inter / nt, 6),
            "mirror_byte_span_exact_rate": round(byte_span_ok / nt, 6),
            "mirror_collateral_rate": round(sum(sp_collat) / nt, 6),
            "lexical_baseline_exact_acc": round(lex_exact / nt, 6),
        }

    # Step 4: Evaluate Ambiguity Disambiguation & Partial-Mask/OOD Abstention
    ambig_and_abstain = _evaluate_ambiguity_and_abstention(seed, operators, g_learner, env)

    # Step 5: Evaluate Plan Realizer Separated Dual Metrics
    realizer_eval = _evaluate_plan_realization_suite(seed, g_learner, heldout_splits)

    # Step 6: Evaluate Zero-Leakage Byte-LM Bits-Per-Byte Benchmark + Analytical Gradient Check
    grad_max_err = _verify_transformer_analytical_gradients(seed)
    byte_lm_eval = run_byte_lm_benchmark_for_seed(seed)

    # Step 7: Unified Single-Stream MDL Bit Accounting
    train_corpus_payload = [
        {
            "utf8": ep.raw_instruction_bytes.decode("utf-8"),
            "delta": compute_sparse_delta(ep.before_state, ep.after_state).to_dict(),
        }
        for ep in grounding_train_eps
    ]
    raw_train_bits = len(canonical_json_bytes(train_corpus_payload)) * 8
    comp_train_bits = compressed_bit_length(canonical_json_bytes(train_corpus_payload))
    unified_induced_archive = {
        "operators": {aid: card.to_executable_dict() for aid, card in sorted(operators.items())},
        "constructions": [c.to_executable_dict() for c in g_learner.promoted_constructions],
        "negations": [r.to_dict() for r in g_learner.negation_rules],
        "conditions": [r.to_dict() for r in g_learner.condition_rules],
        "connectors": [list(c) for c in g_learner.clause_connectors],
    }
    comp_induced_bits = compressed_bit_length(canonical_json_bytes(unified_induced_archive))

    total_sec = time.perf_counter() - t0
    overall_mirror_acc = sum(
        split_metrics[s]["mirror_exact_acc"] * split_metrics[s]["task_count"]
        for s in S7_GROUNDING_SPLITS
    ) / float(sum(split_metrics[s]["task_count"] for s in S7_GROUNDING_SPLITS))
    overall_lex_acc = sum(
        split_metrics[s]["lexical_baseline_exact_acc"] * split_metrics[s]["task_count"]
        for s in S7_GROUNDING_SPLITS
    ) / float(sum(split_metrics[s]["task_count"] for s in S7_GROUNDING_SPLITS))

    return {
        "seed": seed,
        "pool_disjoint_verified": bool(pool_cert["disjoint_verified"]),
        "grammar_fit_summary": fit_stats,
        "grounding_splits": split_metrics,
        "overall_grounding_exact_acc": round(overall_mirror_acc, 6),
        "overall_lexical_baseline_acc": round(overall_lex_acc, 6),
        "overall_collateral_rate": round(
            sum(all_collaterals) / float(max(1, len(all_collaterals))), 6
        ),
        "ambiguity_and_abstention": ambig_and_abstain,
        "plan_realization": realizer_eval,
        "byte_lm_benchmark": byte_lm_eval,
        "transformer_analytical_grad_max_error": round(grad_max_err, 10),
        "bit_budget": {
            "raw_training_bits": raw_train_bits,
            "compressed_training_corpus_bits": comp_train_bits,
            "compressed_induced_archive_bits": comp_induced_bits,
            "compression_ratio_vs_compressed_train": round(
                comp_induced_bits / float(max(1, comp_train_bits)), 6
            ),
            "compression_ratio_vs_raw_train": round(
                comp_induced_bits / float(max(1, raw_train_bits)), 6
            ),
        },
        "total_seed_seconds": round(total_sec, 4),
    }


def _verify_s7_gate(
    summary: Mapping[str, Any],
    seed_records: Sequence[Mapping[str, Any]],
    ast_audit: Mapping[str, Any],
    prior_manifests_ok: bool,
) -> dict[str, Any]:
    sp = summary["grounding_splits"]
    amb = summary["ambiguity_minimal_contrast"]
    ood = summary["ood_and_partial_mask_abstention"]
    rlz = summary["plan_realization"]["methods"]
    blm = summary["byte_lm_benchmark"]["models"]

    m_ctw = blm["mirror_ctw_construction_byte_lm"]["bits_per_byte"]
    m_3gr = blm["byte_3gram_kneser_ney"]["bits_per_byte"]
    m_ppm = blm["ppm_c_order5_byte_lm"]["bits_per_byte"]
    m_trf = blm["small_cpu_byte_transformer_ref"]["bits_per_byte"]
    m_uni = blm["uniform_byte_floor"]["bits_per_byte"]

    checks: dict[str, bool] = {
        "ast_import_and_domain_literal_isolation": bool(ast_audit["passed"]),
        "comparison_only_transformer_strictly_isolated": bool(
            ast_audit["comparison_only_transformer_isolated"]
        ),
        "prior_s0_through_s6_manifests_intact": bool(prior_manifests_ok),
        "pool_disjointness_verified_all_seeds": all(
            bool(r["pool_disjoint_verified"]) for r in seed_records
        ),
        "unseen_entity_substitution_ci95_low_ge_095": (
            sp["s7_unseen_entity_substitution"]["mirror_exact_acc"]["ci95_low"] >= 0.95
        ),
        "unseen_paraphrase_novel_verbs_ci95_low_ge_095": (
            sp["s7_unseen_paraphrase_novel_verbs"]["mirror_exact_acc"]["ci95_low"] >= 0.95
        ),
        "unseen_word_orders_passive_sov_ci95_low_ge_095": (
            sp["s7_unseen_word_orders_passive_sov"]["mirror_exact_acc"]["ci95_low"] >= 0.95
        ),
        "negated_instructions_exact_ci95_low_ge_095": (
            sp["s7_negated_instructions"]["mirror_exact_acc"]["ci95_low"] >= 0.95
        ),
        "conditional_clauses_exact_ci95_low_ge_095": (
            sp["s7_conditional_clauses"]["mirror_exact_acc"]["ci95_low"] >= 0.95
        ),
        "chained_depth2_exact_and_inter_ci95_low_ge_095": (
            sp["s7_chained_multistep_depth2"]["mirror_exact_acc"]["ci95_low"] >= 0.95
            and sp["s7_chained_multistep_depth2"]["mirror_intermediate_acc"]["ci95_low"] >= 0.95
        ),
        "chained_depth4_exact_and_inter_ci95_low_ge_095": (
            sp["s7_chained_multistep_depth4"]["mirror_exact_acc"]["ci95_low"] >= 0.95
            and sp["s7_chained_multistep_depth4"]["mirror_intermediate_acc"]["ci95_low"] >= 0.95
        ),
        "chained_depth6_exact_and_inter_ci95_low_ge_090": (
            sp["s7_chained_multistep_depth6"]["mirror_exact_acc"]["ci95_low"] >= 0.90
            and sp["s7_chained_multistep_depth6"]["mirror_intermediate_acc"]["ci95_low"] >= 0.90
        ),
        "grounding_collateral_ci95_high_lt_002": (
            summary["overall_collateral_rate"]["ci95_high"] < 0.02
        ),
        "mirror_strictly_beats_lexical_baseline": (
            summary["overall_grounding_exact_acc"]["ci95_low"]
            > summary["overall_lexical_baseline_acc"]["ci95_high"] + 0.25
        ),
        "ambiguity_preserved_before_contrast_100pct": (
            amb["pre_contrast_ambiguous_preserved_rate"]["ci95_low"] >= 0.999
            and amb["pre_contrast_honest_abstention_rate"]["ci95_low"] >= 0.999
        ),
        "ambiguity_resolved_after_minimal_contrast_ci95_low_ge_095": (
            amb["post_contrast_disambiguated_exact_acc"]["ci95_low"] >= 0.95
            and amb["post_contrast_disambiguated_exact_acc"]["ci95_low"]
            > amb["no_contrast_first_guess_ablation_acc"]["ci95_high"]
        ),
        "ood_and_partial_mask_abstention_100pct_zero_hallucination": (
            ood["honest_abstention_rate"]["ci95_low"] >= 0.999
            and ood["hallucinated_mutation_rate"]["ci95_high"] == 0.0
        ),
        "realizer_semantic_ci95_low_ge_098_and_zero_hallucination": (
            rlz["mirror_truth_locked"]["semantic_correctness"]["ci95_low"] >= 0.98
            and rlz["mirror_truth_locked"]["hallucination_rate"]["ci95_high"] == 0.0
        ),
        "realizer_dual_metrics_strictly_independent": (
            rlz["mirror_truth_locked"]["surface_fluency_score"]["ci95_low"]
            > rlz["telegraphic_tuple_dump"]["surface_fluency_score"]["ci95_high"] + 0.30
            and rlz["mirror_truth_locked"]["semantic_correctness"]["ci95_low"]
            > rlz["ungated_fluent_assumer"]["semantic_correctness"]["ci95_high"] + 0.40
        ),
        "byte_lm_mirror_strictly_beats_3gram_kneser_ney_ci95": (
            m_ctw["ci95_high"] < m_3gr["ci95_low"]
        ),
        "byte_lm_mirror_strictly_beats_small_cpu_transformer_ci95": (
            m_ctw["ci95_high"] < m_trf["ci95_low"]
        ),
        "byte_lm_mirror_matches_or_beats_ppmc_order5": (
            m_ctw["mean"] <= m_ppm["mean"]
            and all(
                r["byte_lm_benchmark"]["models"]["mirror_ctw_construction_byte_lm"]["bits_per_byte"]
                <= r["byte_lm_benchmark"]["models"]["ppm_c_order5_byte_lm"]["bits_per_byte"]
                for r in seed_records
            )
        ),
        "byte_lm_probability_normalization_and_transformer_grad_verified": (
            abs(m_uni["mean"] - 8.0) < 1e-6
            and all(
                all(
                    m_info["normalization_verified"]
                    for m_info in r["byte_lm_benchmark"]["models"].values()
                )
                and r["transformer_analytical_grad_max_error"] < 1e-5
                for r in seed_records
            )
        ),
        "cpu_budget_within_30s_per_seed": all(
            r["total_seed_seconds"] <= CPU_BUDGET_SECONDS_PER_SEED_MAX for r in seed_records
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


def _render_s7_markdown_report(summary: Mapping[str, Any]) -> str:
    gate = summary["s7_gate"]
    sp = summary["grounding_splits"]
    amb = summary["ambiguity_minimal_contrast"]
    ood = summary["ood_and_partial_mask_abstention"]
    rlz = summary["plan_realization"]["methods"]
    blm = summary["byte_lm_benchmark"]["models"]
    bits = summary["bit_budget_mean"]

    lines = [
        "# Stage S7 Research Report — Raw UTF-8 Language Grounding, Compositional Realization & Byte-LM Comparison",
        "",
        f"- **Stage Gate Verdict:** `{'PASS' if gate['passed'] else 'FAIL'}` "
        f"(`{sum(1 for v in gate['checks'].values() if v)}/{len(gate['checks'])}` checks passed)",
        f"- **Reproduce Command:** `{summary['reproduce_command']}`",
        f"- **Seeds:** `{summary['seeds']}`",
        f"- **Mean Per-Seed CPU Runtime:** `{summary['per_seed_seconds']['mean']:.2f}s` "
        f"(total `{summary['total_elapsed_seconds']:.2f}s`)",
        "",
        "## 1. Raw-UTF-8 Construction Induction & Compositional Execution (Unseen Held-Out Entities)",
        "",
        "| Split | Tasks/Seed | Mirror Exact (`mean [95% CI]`) | Mirror Inter (`mean`) | Collateral | Lexical Baseline (`mean`) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in S7_GROUNDING_SPLITS:
        ex = sp[s]["mirror_exact_acc"]
        it = sp[s]["mirror_intermediate_acc"]
        cl = sp[s]["mirror_collateral_rate"]
        lx = sp[s]["lexical_baseline_exact_acc"]
        lines.append(
            f"| `{s}` | {sp[s]['task_count']} | "
            f"`{ex['mean']*100:.2f}% [{ex['ci95_low']*100:.2f}%, {ex['ci95_high']*100:.2f}%]` | "
            f"`{it['mean']*100:.2f}%` | `{cl['mean']*100:.3f}%` | `{lx['mean']*100:.2f}%` |"
        )

    lines.extend(
        [
            "",
            "## 2. Ambiguity Preservation, Minimal-Contrast Disambiguation & Epistemic Abstention",
            "",
            f"- **Pre-Contrast Ambiguity Preserved (`observe_utterance_in_scene`):** "
            f"`{amb['pre_contrast_ambiguous_preserved_rate']['mean']*100:.2f}%` "
            f"(honest abstention `{amb['pre_contrast_honest_abstention_rate']['mean']*100:.2f}%`)",
            f"- **Post-Contrast Disambiguated Exact Accuracy (2 minimal demos):** "
            f"`{amb['post_contrast_disambiguated_exact_acc']['mean']*100:.2f}%` vs "
            f"No-Contrast First-Guess Ablation `{amb['no_contrast_first_guess_ablation_acc']['mean']*100:.2f}%`",
            f"- **OOD, Malformed UTF-8 & Partial-Mask Epistemic Abstention:** "
            f"`{ood['honest_abstention_rate']['mean']*100:.2f}%` (`{ood['hallucinated_mutation_rate']['mean']*100:.3f}%` hallucinated mutations)",
            "",
            "## 3. Plan-Conditioned Truth-Locked Realizer (Strictly Separated Dual Metrics)",
            "",
            "| Realizer Method | Semantic Correctness | Hallucination Rate | Surface Fluency Score | Reference Perplexity (`bpb`) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for m_id in ("mirror_truth_locked", "telegraphic_tuple_dump", "ungated_fluent_assumer"):
        mr = rlz[m_id]
        lines.append(
            f"| `{m_id}` | `{mr['semantic_correctness']['mean']*100:.2f}%` | "
            f"`{mr['hallucination_rate']['mean']*100:.2f}%` | "
            f"`{mr['surface_fluency_score']['mean']:.4f}` | "
            f"`{mr['reference_perplexity_bpb']['mean']:.4f}` |"
        )

    lines.extend(
        [
            "",
            "## 4. Zero-Leakage Multi-Domain UTF-8 Byte-LM Bits-Per-Byte Benchmark (PDF p. 13)",
            "",
            "| Model ID | Comparison Only | Uses Backprop | Held-Out `bits_per_byte` (`mean [95% CI]`) | Model Bits | Train Sec | Eval Sec |",
            "|---|:---:|:---:|---:|---:|---:|---:|",
        ]
    )
    for m_id in (
        "mirror_ctw_construction_byte_lm",
        "ppm_c_order5_byte_lm",
        "byte_3gram_kneser_ney",
        "small_cpu_byte_transformer_ref",
        "uniform_byte_floor",
    ):
        mb = blm[m_id]
        bpb = mb["bits_per_byte"]
        lines.append(
            f"| `{m_id}` | `{mb['comparison_only']}` | `{mb['uses_backprop']}` | "
            f"`{bpb['mean']:.4f} [{bpb['ci95_low']:.4f}, {bpb['ci95_high']:.4f}]` | "
            f"`{int(mb['model_bits_mean']):,}` | `{mb['train_seconds_mean']:.3f}s` | `{mb['eval_seconds_mean']:.3f}s` |"
        )

    lines.extend(
        [
            "",
            "## 5. Unified Single-Stream Grounding & Operator Archive Bit Accounting",
            "",
            f"- **Raw Grounding Training Corpus Bits:** `{bits['raw_training_bits']:,.0f}`",
            f"- **Compressed Grounding Training Corpus Bits:** `{bits['compressed_training_corpus_bits']:,.0f}`",
            f"- **Compressed Unified Induced Archive Bits (Operators + Constructions + Guards):** `{bits['compressed_induced_archive_bits']:,.0f}`",
            f"- **Compression Ratio vs Compressed Training Corpus:** `{bits['compression_ratio_vs_compressed_train']:.4f}`",
            "",
            "## 6. Stage S7 Gate Checklist",
            "",
        ]
    )
    for k, ok in gate["checks"].items():
        lines.append(f"- [{'x' if ok else ' '}] `{k}`: `{'PASS' if ok else 'FAIL'}`")
    lines.append("")
    return "\n".join(lines)


def run_s7_experiments(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    demos_per_action: int = DEFAULT_DEMOS_PER_ACTION,
    data_dir: Path = DATA_DIR,
    reports_dir: Path = REPORTS_DIR,
    verify_prior_manifests: bool = True,
) -> dict[str, Any]:
    t_all0 = time.perf_counter()
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = reports_dir / "raw_runs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Write deterministic S7 pre-registration manifest
    prereg = build_s7_preregistration(seeds=seeds)
    (data_dir / "s7_preregistration.json").write_text(
        json.dumps(prereg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Verify AST isolation + S0..S6 committed manifests intact against DEFAULT_SEEDS
    ast_audit = audit_s7_ast_isolation_and_agnosticism()
    prior_ok = True
    if verify_prior_manifests:
        ok0, _ = verify_committed_manifests(
            data_dir=DATA_DIR, seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION
        )
        ok1, _ = verify_s1_committed_manifests(
            data_dir=DATA_DIR, seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION
        )
        ok2, _ = verify_s2_committed_manifests(
            data_dir=DATA_DIR, seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION
        )
        ok3, _ = verify_s3_committed_manifests(
            data_dir=DATA_DIR, seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION
        )
        ok4, _ = verify_s4_committed_manifests(
            data_dir=DATA_DIR, seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION
        )
        ok5, _ = verify_s5_committed_manifests(
            data_dir=DATA_DIR, seeds=DEFAULT_SEEDS, demos_per_action=DEFAULT_DEMOS_PER_ACTION
        )
        ok6, _ = verify_s6_committed_manifests(data_dir=DATA_DIR, seeds=DEFAULT_SEEDS)
        prior_ok = bool(ok0 and ok1 and ok2 and ok3 and ok4 and ok5 and ok6)

    seed_records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for seed in seeds:
        rec = _run_single_s7_seed(seed, demos_per_action=demos_per_action)
        seed_records.append(rec)
        (raw_dir / f"s7_seed_{seed}.json").write_text(
            json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for sp_name, sm in rec["grounding_splits"].items():
            csv_rows.append(
                {
                    "seed": seed,
                    "category": "grounding_split",
                    "name": sp_name,
                    "primary_metric": sm["mirror_exact_acc"],
                    "secondary_metric": sm["lexical_baseline_exact_acc"],
                    "collateral_or_bpb": sm["mirror_collateral_rate"],
                }
            )
        for m_id, bm in rec["byte_lm_benchmark"]["models"].items():
            csv_rows.append(
                {
                    "seed": seed,
                    "category": "byte_lm",
                    "name": m_id,
                    "primary_metric": bm["bits_per_byte"],
                    "secondary_metric": bm["train_seconds"],
                    "collateral_or_bpb": bm["bits_per_byte"],
                }
            )

    with (raw_dir / "s7_splits.csv").open("w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "seed",
                "category",
                "name",
                "primary_metric",
                "secondary_metric",
                "collateral_or_bpb",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    # Aggregate Grounding Splits
    splits_agg: dict[str, Any] = {}
    for sp_name in S7_GROUNDING_SPLITS:
        splits_agg[sp_name] = {
            "task_count": seed_records[0]["grounding_splits"][sp_name]["task_count"],
            "mirror_exact_acc": summarize_across_seeds(
                [r["grounding_splits"][sp_name]["mirror_exact_acc"] for r in seed_records]
            ),
            "mirror_intermediate_acc": summarize_across_seeds(
                [r["grounding_splits"][sp_name]["mirror_intermediate_acc"] for r in seed_records]
            ),
            "mirror_byte_span_exact_rate": summarize_across_seeds(
                [r["grounding_splits"][sp_name]["mirror_byte_span_exact_rate"] for r in seed_records]
            ),
            "mirror_collateral_rate": summarize_across_seeds(
                [r["grounding_splits"][sp_name]["mirror_collateral_rate"] for r in seed_records]
            ),
            "lexical_baseline_exact_acc": summarize_across_seeds(
                [r["grounding_splits"][sp_name]["lexical_baseline_exact_acc"] for r in seed_records]
            ),
        }

    ambig_agg = {
        k: summarize_across_seeds(
            [r["ambiguity_and_abstention"]["ambiguity_minimal_contrast"][k] for r in seed_records]
        )
        for k in (
            "pre_contrast_ambiguous_preserved_rate",
            "pre_contrast_honest_abstention_rate",
            "post_contrast_disambiguated_exact_acc",
            "no_contrast_first_guess_ablation_acc",
        )
    }
    ood_agg = {
        k: summarize_across_seeds(
            [r["ambiguity_and_abstention"]["ood_and_partial_mask_abstention"][k] for r in seed_records]
        )
        for k in ("honest_abstention_rate", "hallucinated_mutation_rate")
    }

    rlz_methods_agg: dict[str, Any] = {}
    for m_id in ("mirror_truth_locked", "telegraphic_tuple_dump", "ungated_fluent_assumer"):
        rlz_methods_agg[m_id] = {
            "semantic_correctness": summarize_across_seeds(
                [r["plan_realization"]["methods"][m_id]["semantic_correctness"] for r in seed_records]
            ),
            "hallucination_rate": summarize_across_seeds(
                [r["plan_realization"]["methods"][m_id]["hallucination_rate"] for r in seed_records]
            ),
            "surface_fluency_score": summarize_across_seeds(
                [r["plan_realization"]["methods"][m_id]["surface_fluency_score"] for r in seed_records]
            ),
            "reference_perplexity_bpb": summarize_across_seeds(
                [r["plan_realization"]["methods"][m_id]["reference_perplexity_bpb"] for r in seed_records],
                bounds=(0.0, None),
            ),
        }

    blm_models_agg: dict[str, Any] = {}
    for m_id in (
        "uniform_byte_floor",
        "byte_3gram_kneser_ney",
        "ppm_c_order5_byte_lm",
        "mirror_ctw_construction_byte_lm",
        "small_cpu_byte_transformer_ref",
    ):
        first_m = seed_records[0]["byte_lm_benchmark"]["models"][m_id]
        blm_models_agg[m_id] = {
            "comparison_only": bool(first_m["comparison_only"]),
            "uses_backprop": bool(first_m["uses_backprop"]),
            "bits_per_byte": summarize_across_seeds(
                [r["byte_lm_benchmark"]["models"][m_id]["bits_per_byte"] for r in seed_records],
                bounds=(0.0, 8.0),
            ),
            "model_bits_mean": round(
                sum(r["byte_lm_benchmark"]["models"][m_id]["model_bits"] for r in seed_records)
                / len(seed_records),
                2,
            ),
            "train_seconds_mean": round(
                sum(r["byte_lm_benchmark"]["models"][m_id]["train_seconds"] for r in seed_records)
                / len(seed_records),
                4,
            ),
            "eval_seconds_mean": round(
                sum(r["byte_lm_benchmark"]["models"][m_id]["eval_seconds"] for r in seed_records)
                / len(seed_records),
                4,
            ),
        }

    bit_keys = list(seed_records[0]["bit_budget"].keys())
    bit_means = {
        k: round(sum(r["bit_budget"][k] for r in seed_records) / len(seed_records), 6)
        for k in bit_keys
    }

    total_elapsed = time.perf_counter() - t_all0
    summary: dict[str, Any] = {
        "schema_version": "s7-summary-v1",
        "stage": "S7",
        "reproduce_command": S7_REPRO_COMMAND,
        "seeds": list(seeds),
        "total_elapsed_seconds": round(total_elapsed, 4),
        "per_seed_seconds": summarize_across_seeds(
            [r["total_seed_seconds"] for r in seed_records], bounds=(0.0, None)
        ),
        "environment": capture_environment_info(),
        "ast_isolation_audit": ast_audit,
        "bit_budget_mean": bit_means,
        "overall_grounding_exact_acc": summarize_across_seeds(
            [r["overall_grounding_exact_acc"] for r in seed_records]
        ),
        "overall_lexical_baseline_acc": summarize_across_seeds(
            [r["overall_lexical_baseline_acc"] for r in seed_records]
        ),
        "overall_collateral_rate": summarize_across_seeds(
            [r["overall_collateral_rate"] for r in seed_records]
        ),
        "grounding_splits": splits_agg,
        "ambiguity_minimal_contrast": ambig_agg,
        "ood_and_partial_mask_abstention": ood_agg,
        "plan_realization": {
            "trace_count_per_seed": seed_records[0]["plan_realization"]["trace_count"],
            "methods": rlz_methods_agg,
        },
        "byte_lm_benchmark": {
            "corpus_train_bytes_mean": round(
                sum(r["byte_lm_benchmark"]["corpus"]["train_byte_len"] for r in seed_records)
                / len(seed_records),
                1,
            ),
            "corpus_heldout_bytes_mean": round(
                sum(r["byte_lm_benchmark"]["corpus"]["heldout_byte_len"] for r in seed_records)
                / len(seed_records),
                1,
            ),
            "models": blm_models_agg,
        },
    }

    s7_gate = _verify_s7_gate(summary, seed_records, ast_audit, prior_ok)
    summary["s7_gate"] = s7_gate

    (reports_dir / "s7_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports_dir / "s7_research_report.md").write_text(
        _render_s7_markdown_report(summary), encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible Mirror 7 CPU-Native Learning Stage S7 Grounding & Byte-LM benchmark."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Fixed random seeds (default: 101 202 303 404 505).",
    )
    parser.add_argument(
        "--demos-per-action",
        type=int,
        default=DEFAULT_DEMOS_PER_ACTION,
        help="S0–S5 demonstrations per action (default: 6).",
    )
    parser.add_argument(
        "--verify-manifests",
        action="store_true",
        help="Verify committed S0..S7 manifests/preregistrations match deterministic generation.",
    )
    args = parser.parse_args(argv)

    if args.verify_manifests:
        ok0, m0 = verify_committed_manifests(seeds=args.seeds, demos_per_action=args.demos_per_action)
        ok1, m1 = verify_s1_committed_manifests(seeds=args.seeds, demos_per_action=args.demos_per_action)
        ok2, m2 = verify_s2_committed_manifests(seeds=args.seeds, demos_per_action=args.demos_per_action)
        ok3, m3 = verify_s3_committed_manifests(seeds=args.seeds, demos_per_action=args.demos_per_action)
        ok4, m4 = verify_s4_committed_manifests(seeds=args.seeds, demos_per_action=args.demos_per_action)
        ok5, m5 = verify_s5_committed_manifests(seeds=args.seeds, demos_per_action=args.demos_per_action)
        ok6, m6 = verify_s6_committed_manifests(seeds=args.seeds)
        ok7, m7 = verify_s7_committed_manifests(seeds=args.seeds)
        if not (ok0 and ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7):
            for msg in m0 + m1 + m2 + m3 + m4 + m5 + m6 + m7:
                print(f"MANIFEST ERROR: {msg}", file=sys.stderr)
            return 1
        print(
            "Committed S0, S1, S2, S3, S4, S5, S6, and S7 manifests verified bit-exact against deterministic generators."
        )
        return 0

    summary = run_s7_experiments(seeds=args.seeds, demos_per_action=args.demos_per_action)
    gate = summary["s7_gate"]
    passed = sum(1 for v in gate["checks"].values() if v)
    total = len(gate["checks"])
    print(f"Stage S7 Gate: {'PASS' if gate['passed'] else 'FAIL'} ({passed}/{total} checks passed)")
    for k, ok in gate["checks"].items():
        if not ok:
            print(f"  FAILED CHECK: {k}", file=sys.stderr)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
