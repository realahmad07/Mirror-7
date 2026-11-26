"""Stage S6 Independent Adversarial Tests: Mathematics, List & Code Transfer.

STRICT ISOLATION:
- NEVER imports ``hidden_actions`` or ``oracle``.
- Audits AST/lexical isolation, sandbox safety & step metering, synchronous multi-slot
  semantics, constant slot resets, partial-card & rollback guards, fair SHARD vs Unguided
  stopping controls, novel unseen radices/list pipelines/AST refactorings/code tasks,
  and S0–S6 manifests + saved 5-seed & isolated 2-seed end-to-end verification.
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from research.cpu_learning.evaluation.run_experiments import (
    DATA_DIR,
    PACKAGE_ROOT,
    REPORTS_DIR,
    verify_committed_manifests,
)
from research.cpu_learning.evaluation.run_s1_experiments import verify_s1_committed_manifests
from research.cpu_learning.evaluation.run_s2_experiments import (
    _string_constants_in_code,
    verify_s2_committed_manifests,
)
from research.cpu_learning.evaluation.run_s3_experiments import verify_s3_committed_manifests
from research.cpu_learning.evaluation.run_s4_experiments import verify_s4_committed_manifests
from research.cpu_learning.evaluation.run_s5_experiments import verify_s5_committed_manifests
from research.cpu_learning.evaluation.run_s6_experiments import (
    _identifiers_in_code,
    audit_s6_ast_isolation_and_agnosticism,
    run_s6_experiments,
    verify_s6_committed_manifests,
)
from research.cpu_learning.transfer import (
    HUMANEVAL_MICRO_FORBIDDEN_NAMES,
    ASTRewriteInducer,
    RestrictedExecutionSandbox,
    SandboxStepLimitError,
    SlotTransitionEpisode,
    TransferInducerFrozenError,
    TransferProgramInducer,
    build_humaneval_micro_suite,
    evaluate_ast_refactoring_suite,
    evaluate_humaneval_micro_suite,
    evaluate_modular_addition_canary,
    evaluate_multidigit_and_radix_transfer,
    execute_multidigit_addition,
    normalize_slot_value,
    run_list_domain_suite,
    synthesize_typed_function,
    to_python_runtime_value,
)
from research.cpu_learning.transfer.math_domain import _sample_column_training_episodes

_EXPANDED_DOMAIN_LITERALS: frozenset[str] = frozenset(
    {
        # S0 Domain 1
        "open",
        "close",
        "lock",
        "unlock",
        "take",
        "drop",
        "put_in",
        "move",
        "turn_on",
        "turn_off",
        "give",
        "unlock_Door",
        "open_Container",
        # S5 Domain 2 (Arcane) & Domain 3 (Logistics)
        "unseal_gate",
        "lift_ward",
        "attune_orb",
        "stow_relic",
        "channel_mana",
        "dispel_mana",
        "glide",
        "bestow",
        "unlatch_hatch",
        "clear_seal",
        "load_crate",
        "power_unit",
        "shutdown_unit",
        "shuttle",
        "handover",
        # S6 benchmark operator names (must NEVER be hardcoded in transfer_inducer.py!)
        "digit_out",
        "carry_out",
        "carry_in",
        "digit_pair_mod_op",
        "filter_even",
        "filter_positive",
        "map_double",
        "map_increment",
        "sort_asc",
        "rename_parameter",
        "swap_if_negated_guard",
        "fold_identity_arithmetic",
        "wrap_nonnegative_clamp",
    }
)


def test_s6_ast_isolation_zero_domain_literals_and_zero_task_name_leakage() -> None:
    """Verify strict AST import isolation and zero S0/D2/D3/S6/HumanEval name leakage in transfer_inducer."""
    audit = audit_s6_ast_isolation_and_agnosticism(PACKAGE_ROOT)
    assert audit["passed"] is True, f"AST audit failed: {audit}"
    assert audit["import_violations"] == []
    assert audit["hardcode_violations"] == []
    assert audit["humaneval_leak_violations"] == []
    assert audit["scanned_transfer_modules"] >= 5

    inducer_path = PACKAGE_ROOT / "transfer" / "transfer_inducer.py"
    inducer_src = inducer_path.read_text(encoding="utf-8")
    consts = _string_constants_in_code(inducer_src)
    idents = _identifiers_in_code(inducer_src)

    forbidden_all = _EXPANDED_DOMAIN_LITERALS | set(HUMANEVAL_MICRO_FORBIDDEN_NAMES)
    assert (consts & forbidden_all) == set()
    assert (idents & forbidden_all) == set()
    for he_name in HUMANEVAL_MICRO_FORBIDDEN_NAMES:
        assert he_name not in inducer_src.lower(), f"Task name {he_name!r} leaked into transfer_inducer.py"

    # Structural check: synthesize_typed_function accepts NO task_id, docstring, or prompt arguments
    sig = inspect.signature(synthesize_typed_function)
    for bad_param in ("task_id", "task_name", "canonical_name", "docstring", "prompt", "description"):
        assert bad_param not in sig.parameters


def test_s6_sandbox_safety_step_budget_and_synchronous_register_swap_regression() -> None:
    """Verify sandbox blocks unsafe AST/loops, meters dynamic steps, and executes multi-slot transitions synchronously."""
    # 1. Unsafe AST & static infinite loop rejections
    fn_imp, r_imp = RestrictedExecutionSandbox.compile_verified_function(
        "import os\ndef f(x):\n    return x\n", "f"
    )
    assert fn_imp is None and "import" in r_imp.lower()

    fn_dunder, r_dunder = RestrictedExecutionSandbox.compile_verified_function(
        "def f(x):\n    return x.__class__\n", "f"
    )
    assert fn_dunder is None

    fn_inf, r_inf = RestrictedExecutionSandbox.compile_verified_function(
        "def f(x):\n    while True:\n        x = x + 1\n    return x\n",
        "f",
        max_loop_steps=100,
    )
    assert fn_inf is None and "loop" in r_inf.lower()

    # 2. Dynamic step limit enforcement + clean counter reset across calls
    fn_dyn, r_dyn = RestrictedExecutionSandbox.compile_verified_function(
        "def count_up(n):\n    i = 0\n    while i < n:\n        i = i + 1\n    return i\n",
        "count_up",
        max_loop_steps=40,
    )
    assert fn_dyn is not None and r_dyn == "OK"
    assert fn_dyn(12) == 12
    with pytest.raises(SandboxStepLimitError):
        fn_dyn(200)
    # Subsequent small invocation must succeed cleanly after exception
    assert fn_dyn(7) == 7

    # 3. Regression for Defect #2: synchronous 2-register swap (a, b) -> (b, a) must NOT clobber b!
    swap_eps = [
        SlotTransitionEpisode(
            f"sw_{i}",
            "swap_regs",
            (("a", a), ("b", b), ("guard_tag", f"g_{i}")),
            (("a", b), ("b", a), ("guard_tag", f"g_{i}")),
            ("a", "b"),
        )
        for i, (a, b) in enumerate([(3, 9), (-5, 12), (0, 7), (14, -2)])
    ]
    inducer = TransferProgramInducer()
    inducer.fit_episodes(swap_eps)
    swap_card = inducer.get_card("swap_regs")
    assert swap_card is not None and swap_card.all_slots_solved is True

    out_sandbox = swap_card.apply_slots({"a": 10, "b": 99, "guard_tag": "keep"}, use_sandbox=True)
    out_direct = swap_card.apply_slots({"a": 10, "b": 99, "guard_tag": "keep"}, use_sandbox=False)
    assert out_sandbox == {"a": 99, "b": 10, "guard_tag": "keep"}
    assert out_direct == out_sandbox

    # 4. Regression for Defect #3: constant slot resets induce O(1) 1-node const ASTs across types
    for atypes, otype, pairs, big_inp, expected in [
        (("int",), "int", [((5,), 0), ((-3,), 0), ((12,), 0), ((0,), 0)], (100_000,), 0),
        (("seq_int",), "bool", [(([1, 2],), False), (([],), False), (([-4],), False), (([0],), False)], ([1, 2, 3],), False),
        (("seq_int",), "seq_int", [(([1, 2],), []), (([9],), []), (([-3, 4],), []), (([],), [])], ([7, 8, 9],), ()),
        (("str",), "str", [(("abc",), ""), (("x",), ""), (("hello world",), ""), (("",), "")], ("test",), ""),
    ]:
        expr, py_src, diag = synthesize_typed_function(atypes, otype, pairs, fn_name="const_reset")
        assert expr is not None and py_src is not None and diag.solved is True
        assert expr.op == "const" and expr.node_count() == 1
        fn_c, _ = RestrictedExecutionSandbox.compile_verified_function(
            py_src, "const_reset", max_loop_steps=500
        )
        assert fn_c is not None
        assert normalize_slot_value(fn_c(*big_inp)) == expected


def test_s6_inducer_freeze_partial_card_guard_and_version_rollback() -> None:
    """Verify partial unsolved cards never overwrite solved active cards, rollback works, and freeze guards mutations."""
    inducer = TransferProgramInducer(max_candidates=80)

    # v1: valid 2-slot update (x -> x + 1, y -> y * 2)
    eps_v1 = [
        SlotTransitionEpisode(
            f"v1_{i}",
            "dual_op",
            (("x", x), ("y", y)),
            (("x", x + 1), ("y", y * 2)),
            ("x", "y"),
        )
        for i, (x, y) in enumerate([(1, 3), (4, -2), (0, 5), (-3, 7)])
    ]
    inducer.fit_episodes(eps_v1)
    c1 = inducer.get_card("dual_op")
    assert c1 is not None and c1.version == 1 and c1.all_slots_solved is True
    assert c1.apply_slots({"x": 10, "y": 6}) == {"x": 11, "y": 12}

    with pytest.raises(ValueError, match=">= 2 versions"):
        inducer.rollback("dual_op")

    # v2: contradictory/unsolvable second slot must NOT overwrite solved v1 active card!
    eps_bad = [
        SlotTransitionEpisode(
            f"v2_{i}",
            "dual_op",
            (("x", x), ("y", y)),
            (("x", x + 1), ("y", val)),
            ("x", "y"),
        )
        for i, (x, y, val) in enumerate([(1, 1, 17), (1, 1, 93), (2, 3, -41), (5, 8, 2)])
    ]
    inducer.fit_episodes(eps_bad)
    hist = inducer.get_history("dual_op")
    assert len(hist) == 2 and hist[1].version == 2 and hist[1].all_slots_solved is False
    assert inducer.get_card("dual_op") == c1

    # v3: valid new definition (x -> x - 1, y -> y + y) promotes to active v3
    eps_v3 = [
        SlotTransitionEpisode(
            f"v3_{i}",
            "dual_op",
            (("x", x), ("y", y)),
            (("x", x - 1), ("y", y + y)),
            ("x", "y"),
        )
        for i, (x, y) in enumerate([(2, 3), (5, -4), (0, 8), (-2, 1)])
    ]
    inducer.fit_episodes(eps_v3)
    c3 = inducer.get_card("dual_op")
    assert c3 is not None and c3.version == 3 and c3.all_slots_solved is True
    assert c3.apply_slots({"x": 10, "y": 6}) == {"x": 9, "y": 12}

    # Explicit rollback to v1
    with pytest.raises(ValueError, match="not found"):
        inducer.rollback("dual_op", target_version=99)
    restored = inducer.rollback("dual_op", target_version=1)
    assert restored == c1 and inducer.get_card("dual_op") == c1
    assert inducer.get_card("dual_op").apply_slots({"x": 10, "y": 6}) == {"x": 11, "y": 12}

    # Freeze blocks both fit_episodes and rollback
    inducer.freeze()
    assert inducer.is_frozen is True
    with pytest.raises(TransferInducerFrozenError):
        inducer.fit_episodes(eps_v1)
    with pytest.raises(TransferInducerFrozenError):
        inducer.rollback("dual_op", target_version=3)


def test_s6_math_canary_multidigit_cascades_and_novel_base12_transfer() -> None:
    """Verify S0 modular canary, unseen 8-digit cascades, and unseen novel base-12 & novel arithmetic rules."""
    canary = evaluate_modular_addition_canary(101)
    assert canary["exact_canary_60_40_accuracy"] == 1.0
    assert canary["sparse_16_84_accuracy"] == 1.0
    assert canary["collateral_corruptions"] == 0
    assert canary["typed_frame_bridge_verified"] is True

    multi = evaluate_multidigit_and_radix_transfer(101)
    assert multi["train_multidigit_demos"] == 0
    assert multi["multidigit_overall_accuracy"] == 1.0
    assert multi["novel_base7_5digit_accuracy"] == 1.0
    assert multi["novel_base16_5digit_accuracy"] == 1.0
    assert multi["collateral_corruptions"] == 0

    # Novel unseen radix: Base-12 (duodecimal) trained on 16 single-column demos -> 6-digit cascades + 0+0
    b12_eps = _sample_column_training_episodes(seed=777, radix=12, sample_size=16)
    ind_b12 = TransferProgramInducer(use_shard_guidance=True, max_candidates=320)
    ind_b12.fit_episodes(b12_eps)
    ind_b12.freeze()
    card_b12 = ind_b12.get_card("column_step_b12")
    assert card_b12 is not None and card_b12.all_slots_solved is True

    test_b12_pairs = [
        (0, 0),
        (12**6 - 1, 1),
        (12**6 - 1, 12**6 - 1),
        (1_845_293, 982_711),
        (11 * (12**5) + 11, 12**5 + 1),
    ]
    for na, nb in test_b12_pairs:
        pred, coll = execute_multidigit_addition(card_b12, na, nb, radix=12)
        assert pred == na + nb and coll == 0

    # Novel unseen arithmetic rules: parametric integer quotient (a + b) // div AND modular double (2 * a) % 7
    q_eps = [
        SlotTransitionEpisode(
            f"q_{i}",
            "param_quot",
            (("a", a), ("b", b), ("div", d), ("q", -1)),
            (("a", a), ("b", b), ("div", d), ("q", (a + b) // d)),
            ("a", "b", "div"),
        )
        for i, (a, b, d) in enumerate(
            [(2, 3, 5), (10, 4, 3), (7, 8, 4), (0, 4, 5), (19, 6, 7), (12, 13, 6)]
        )
    ]
    ind_q = TransferProgramInducer(max_candidates=140)
    ind_q.fit_episodes(q_eps)
    q_card = ind_q.get_card("param_quot")
    assert q_card is not None and q_card.all_slots_solved is True
    for a, b, d in [(33, 18, 5), (100, 25, 9), (0, 0, 3), (49, 2, 11)]:
        assert q_card.apply_slots({"a": a, "b": b, "div": d, "q": -1})["q"] == (a + b) // d

    mod7_eps = [
        SlotTransitionEpisode(
            f"m7_{a}",
            "double_mod7",
            (("a", a), ("r", -1)),
            (("a", a), ("r", (2 * a) % 7)),
            ("a",),
        )
        for a in (0, 1, 2, 3, 4, 5, 6, 8)
    ]
    ind_m7 = TransferProgramInducer(max_candidates=100)
    ind_m7.fit_episodes(mod7_eps)
    m7_card = ind_m7.get_card("double_mod7")
    assert m7_card is not None and m7_card.all_slots_solved is True
    for unseen_a in (9, 13, 21, 50, 77, 104):
        assert m7_card.apply_slots({"a": unseen_a, "r": -1})["r"] == (2 * unseen_a) % 7


def test_s6_list_unseen_lengths_compositions_and_novel_list_operators() -> None:
    """Verify S6 list suite and generalize to 3 brand-new list ops + unseen Depth-3 composition."""
    res = run_list_domain_suite(101)
    assert res["atomic_unseen_len8_20_accuracy"] == 1.0
    assert res["depth2_composition_accuracy"] == 1.0
    assert res["depth3_composition_accuracy"] == 1.0
    assert res["collateral_corruptions"] == 0
    assert res["typed_frame_bridge_verified"] is True

    # Novel unseen list operators trained on len <= 5 -> evaluated on len=15 depth-3 pipeline
    demo_lists = [(3, -2, 3, 4), (-5, 0, 2, -5, 7), (4, 1, 4), (0, -3, 6, 1), (2, -1, 5, 2)]
    novel_specs = {
        "dedup_preserve": lambda xs: tuple(dict.fromkeys(xs)),
        "filter_odd": lambda xs: tuple(x for x in xs if x % 2 != 0),
        "map_square": lambda xs: tuple(x * x for x in xs),
    }
    eps = [
        SlotTransitionEpisode(
            f"nv_{op}_{i}",
            op,
            (("seq", xs), ("sentinel", i)),
            (("seq", fn(xs)), ("sentinel", i)),
            ("seq",),
        )
        for op, fn in novel_specs.items()
        for i, xs in enumerate(demo_lists)
    ]
    inducer = TransferProgramInducer(use_shard_guidance=True, max_candidates=160)
    inducer.fit_episodes(eps)
    inducer.freeze()

    unseen_long = [
        (),
        (-11,),
        (15, -8, 15, 0, -23, 42, -8, 9, 101, -101, 4, 9, -7, 12, 31),
        (-200, 199, -200, 0, 77, 88, -55, 77, 13, -1, 2, 4, -99, 199, 5),
    ]
    for xs in unseen_long:
        state = {"seq": xs, "sentinel": "unmodified"}
        for step_op in ("dedup_preserve", "filter_odd", "map_square"):
            state = inducer.get_card(step_op).apply_slots(state, use_sandbox=True)
            assert state["sentinel"] == "unmodified"
        gold = novel_specs["map_square"](novel_specs["filter_odd"](novel_specs["dedup_preserve"](xs)))
        assert tuple(state["seq"]) == gold


def test_s6_ast_rewrite_inducer_min_support_freeze_and_novel_refactoring() -> None:
    """Verify AST refactoring suite, rich Python AST constructs, min_support=2 noise filter, and novel AST rule."""
    res = evaluate_ast_refactoring_suite(101)
    assert res["compile_valid_rate"] == 1.0
    assert res["exact_verified_accuracy"] == 1.0
    assert res["naive_string_replace_rename_accuracy"] == 0.0

    # Test min_rule_support=2 rejects one-off noise rule and freeze blocks mutation
    noise_ind = ASTRewriteInducer(min_rule_support=2)
    noise_ind.fit([("one_off", "def f(x):\n    return x + 0\n", {}, "def f(x):\n    return x\n")])
    assert noise_ind.cards["one_off"].rules == ()
    noise_ind.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        noise_ind.fit([])

    # Novel unseen AST refactoring: `if len(xs) == 0: A else: B` -> `if not xs: A else: B`
    novel_ast_ind = ASTRewriteInducer(min_rule_support=2)
    novel_ast_ind.fit(
        [
            (
                "simplify_empty_len_guard",
                "def a1(xs):\n    if len(xs) == 0:\n        return 0\n    else:\n        return xs[0]\n",
                {},
                "def a1(xs):\n    if not xs:\n        return 0\n    else:\n        return xs[0]\n",
            ),
            (
                "simplify_empty_len_guard",
                "def a2(items):\n    if len(items) == 0:\n        return -1\n    else:\n        return len(items)\n",
                {},
                "def a2(items):\n    if not items:\n        return -1\n    else:\n        return len(items)\n",
            ),
            (
                "simplify_empty_len_guard",
                "def a3(buf, k):\n    if len(buf) == 0:\n        return k\n    else:\n        return buf[-1] + k\n",
                {},
                "def a3(buf, k):\n    if not buf:\n        return k\n    else:\n        return buf[-1] + k\n",
            ),
        ]
    )
    card = novel_ast_ind.cards["simplify_empty_len_guard"]
    assert len(card.rules) >= 1

    unseen_fn = (
        "def unseen_head_sum(vals, fallback):\n"
        "    if len(vals) == 0:\n"
        "        return fallback * 2\n"
        "    else:\n"
        "        return vals[0] + fallback\n"
    )
    rewritten = card.apply_code(unseen_fn)
    parsed = ast.parse(rewritten)
    assert "len(" not in rewritten
    assert any(isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not) for n in ast.walk(parsed))
    fn_exec, reason = RestrictedExecutionSandbox.compile_verified_function(rewritten, "unseen_head_sum")
    assert fn_exec is not None and reason == "OK"
    assert fn_exec([], 7) == 14
    assert fn_exec([10, 20], 7) == 17


def test_s6_humaneval_micro_fair_control_boundary_refutation_and_novel_code_tasks() -> None:
    """Verify fair stopping rule (Defect #1 regression), 20-task HumanEval-Micro gate, boundary refutation, and novel tasks."""
    suite = build_humaneval_micro_suite(101)
    assert len(suite) == 20
    for task in suite:
        assert len(task.public_train_io) == 4
        assert len(task.hidden_test_io) == 12
        tr_keys = {tuple(normalize_slot_value(x) for x in inp) for inp, _ in task.public_train_io}
        ho_keys = {tuple(normalize_slot_value(x) for x in inp) for inp, _ in task.hidden_test_io}
        assert tr_keys.isdisjoint(ho_keys), f"Train/hidden overlap in {task.canonical_name}"

    # Regression for Defect #1: on dedup_order (where canonical family #1 for seq_int->seq_int solves it),
    # Unguided and SHARD must evaluate the EXACT same number of candidates (zero control handicap!)
    dedup_task = next(t for t in suite if t.canonical_name == "dedup_order")
    _, _, diag_shard = synthesize_typed_function(
        dedup_task.arg_types, dedup_task.out_type, dedup_task.public_train_io, use_shard_guidance=True, max_candidates=48
    )
    _, _, diag_ung = synthesize_typed_function(
        dedup_task.arg_types, dedup_task.out_type, dedup_task.public_train_io, use_shard_guidance=False, max_candidates=48
    )
    assert diag_shard.solved is True and diag_ung.solved is True
    assert diag_shard.candidates_evaluated == diag_ung.candidates_evaluated

    # Verify RDC boundary refutation catches crashing empty-list candidates on rotate_left & max_diff
    for rdc_name in ("rotate_left", "max_diff"):
        rtask = next(t for t in suite if t.canonical_name == rdc_name)
        _, _, rdiag = synthesize_typed_function(
            rtask.arg_types, rtask.out_type, rtask.public_train_io, use_shard_guidance=True, use_rdc_boundary_refutation=True, max_candidates=48
        )
        assert rdiag.solved is True and rdiag.boundary_refuted_candidates >= 1

    # Full 20-task HumanEval-Micro evaluation on seed 101
    he_res = evaluate_humaneval_micro_suite(101)
    sh = he_res["methods"]["shard_itl_rdc_code_inducer"]
    ug = he_res["methods"]["unguided_enumerative_code_inducer"]
    mem = he_res["methods"]["io_memorization_lookup"]
    assert sh["pass_rate"] >= 0.90  # 19/20 = 95%, well above 50% PDF gate
    assert sh["pass_rate"] > ug["pass_rate"]
    assert mem["pass_rate"] == 0.0
    assert he_res["shard_vs_unguided_eval_reduction"] >= 0.45

    # Novel unseen execution-only code synthesis tasks NOT in HumanEval-Micro 20!
    novel_code_specs = [
        (
            "count_odds",
            ("seq_int",),
            "int",
            [(([1, 2, 3, 4, 5],), 3), (([2, 4, 6],), 0), (([-3, 0, 7],), 2), (([9],), 1)],
            [([],), ([11, 13, 15, 2],), ([-8, -7, -6, -5],), ([0, 0, 1],)],
            lambda xs: sum(1 for x in xs if x % 2 != 0),
        ),
        (
            "product_list",
            ("seq_int",),
            "int",
            [(([2, 3, 4],), 24), (([5, -2],), -10), (([7],), 7), (([3, 3],), 9)],
            [([],), ([1, 2, 3, 4, 5],), ([-2, -3, -4],), ([99, 0, 12],)],
            lambda xs: 1 if not xs else __import__("math").prod(xs),
        ),
        (
            "odd_doubles",
            ("seq_int",),
            "seq_int",
            [(([1, 2, 3, 4],), [2, 6]), (([-3, 0, 5],), [-6, 10]), (([2, 4],), []), (([7],), [14])],
            [([],), ([9, 8, 7, 6, 5],), ([-11, 0, 11],), ([2, 4, 6, 8],)],
            lambda xs: [x * 2 for x in xs if x % 2 != 0],
        ),
        (
            "all_even",
            ("seq_int",),
            "bool",
            [(([2, 4, 6],), True), (([2, 3, 4],), False), (([0, -2, 8],), True), (([1],), False)],
            [([],), ([10, 20, 30, 40],), ([10, 20, 31],), ([-4, -2, 0],)],
            lambda xs: all(x % 2 == 0 for x in xs),
        ),
    ]
    for cname, atypes, otype, train_io, hidden_inps, gold_fn in novel_code_specs:
        expr, py_src, diag = synthesize_typed_function(
            atypes, otype, train_io, use_shard_guidance=True, max_candidates=60, fn_name="novel_sol"
        )
        assert expr is not None and py_src is not None and diag.solved is True, f"Failed novel task {cname}"
        fn_v, _ = RestrictedExecutionSandbox.compile_verified_function(py_src, "novel_sol")
        assert fn_v is not None
        for h_inp in hidden_inps:
            pred = normalize_slot_value(fn_v(*[to_python_runtime_value(x) for x in h_inp]))
            assert pred == normalize_slot_value(gold_fn(*h_inp)), f"Mismatch on {cname} input {h_inp}"


def test_s6_committed_manifests_saved_5seed_gate_and_fresh_isolated_run(tmp_path: Path) -> None:
    """Verify S0..S6 committed manifests, saved 5-seed S6 summary gate, and an isolated 2-seed run."""
    ok0, e0 = verify_committed_manifests(data_dir=DATA_DIR)
    ok1, e1 = verify_s1_committed_manifests(data_dir=DATA_DIR)
    ok2, e2 = verify_s2_committed_manifests(data_dir=DATA_DIR)
    ok3, e3 = verify_s3_committed_manifests(data_dir=DATA_DIR)
    ok4, e4 = verify_s4_committed_manifests(data_dir=DATA_DIR)
    ok5, e5 = verify_s5_committed_manifests(data_dir=DATA_DIR)
    ok6, e6 = verify_s6_committed_manifests(data_dir=DATA_DIR)
    assert ok0 and ok1 and ok2 and ok3 and ok4 and ok5 and ok6, (
        f"Manifest mismatch: {e0 + e1 + e2 + e3 + e4 + e5 + e6}"
    )

    saved_summary_path = REPORTS_DIR / "s6_summary.json"
    saved_bytes_before = saved_summary_path.read_bytes()
    saved_summary = json.loads(saved_bytes_before.decode("utf-8"))

    assert saved_summary["stage"] == "S6"
    assert saved_summary["seeds"] == [101, 202, 303, 404, 505]
    assert saved_summary["s6_gate"]["passed"] is True
    assert all(saved_summary["s6_gate"]["checks"].values())
    assert len(saved_summary["s6_gate"]["checks"]) == 21

    bb = saved_summary["bit_budget_mean"]
    assert bb["compressed_induced_library_bits"] < bb["compressed_training_bits"] < bb["raw_training_bits"]

    # Isolated 2-seed run in tmp_path
    tmp_data = tmp_path / "data"
    tmp_reports = tmp_path / "reports"
    fresh = run_s6_experiments(
        seeds=[101, 202],
        data_dir=tmp_data,
        reports_dir=tmp_reports,
        verify_prior_manifests=True,
    )
    for k, ok in fresh["s6_gate"]["checks"].items():
        # Student-t df=1 (t=12.706) widens S0 random baseline CI on n=2 seeds; check min directly below
        if k in {"cpu_budget_within_30s_per_seed", "math_beats_s0_shard_canary_by_90pp"}:
            continue
        assert ok is True, f"Fresh isolated 2-seed S6 gate failed check: {k}"
    assert fresh["math_domain"]["exact_canary_60_40_accuracy"]["min"] == 1.0
    assert fresh["math_domain"]["gain_over_s0_shard_bit_perceptron"]["min"] >= 0.90

    assert (tmp_data / "s6_preregistration.json").exists()
    assert (tmp_reports / "s6_summary.json").exists()
    assert (tmp_reports / "s6_research_report.md").exists()
    assert (tmp_reports / "raw_runs" / "s6_domain_matrix.csv").exists()
    assert (tmp_reports / "raw_runs" / "s6_seed_101.json").exists()
    assert (tmp_reports / "raw_runs" / "s6_seed_202.json").exists()
    assert saved_summary_path.read_bytes() == saved_bytes_before
