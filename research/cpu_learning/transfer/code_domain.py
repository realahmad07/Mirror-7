"""Stage S6 Transfer Domain 3: Python AST Refactoring Induction & 20-Task HumanEval-Micro.

STRICT ISOLATION & EXECUTION-ONLY EVALUATION (PDF pp. 12, 13, 17):
- **Part A — Python AST Refactoring Inducer:** Induces generalizable structural AST pattern-rewrite
  cards (``ASTRewriteCard``) strictly by anti-unifying minimal AST subtree diffs across 4–6
  ``(before_code, params) -> after_code`` training episodes (ZERO action-name dispatch). Verified
  on unseen Python functions via ``ZeroTrustSandboxJudge.is_ast_safe``, ``compile()``, and hidden
  sandbox execution equivalence tests (compared against a naive regex/string-replace baseline).
- **Part B — 20-Task Execution-Only HumanEval-Micro Suite (PDF p. 13 Gate >= 50%):**
  Passes ONLY ``(arg_types, out_type, 4 small I/O pairs)`` to ``synthesize_typed_function``
  (ZERO task IDs, ZERO function names, ZERO docstrings — lexical cheating is structurally
  impossible). Evaluates synthesized standalone Python functions inside ``RestrictedExecutionSandbox``
  on ``12`` unseen edge/random/stress test cases per task, comparing:
  1. ``shard_itl_rdc_code_inducer`` (SHARD 10,240-bit behavioral HV ranking + ITL + RDC MDL)
  2. ``unguided_enumerative_code_inducer`` (identical grammar & budget without SHARD ranking)
  3. ``io_memorization_lookup`` (exact training-pair lookup table)
  alongside Mirror's documented ``0/20`` external MBPP baseline reference.
"""
from __future__ import annotations

import ast
import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from mirror_crystal.grammar_and_sandbox import ZeroTrustSandboxJudge
from research.cpu_learning.simulator.state_graph import (
    canonical_json_bytes,
    compressed_bit_length,
)
from research.cpu_learning.transfer.transfer_inducer import (
    RestrictedExecutionSandbox,
    SlotTransitionEpisode,
    normalize_slot_value,
    synthesize_typed_function,
    to_python_runtime_value,
)

CODE_DOMAIN_SCHEMA_VERSION: str = "s6-code-domain-v1"
HUMANEVAL_MICRO_VERSION: str = "s6-humaneval-micro-20-v1"
HUMANEVAL_CANDIDATE_BUDGET: int = 48
PRIOR_EXTERNAL_MBPP_BASELINE: dict[str, Any] = {
    "benchmark": "MBPP_20_external_prior",
    "solved": 0,
    "total": 20,
    "pass_rate": 0.0,
}


# ===========================================================================
# PART A: Structural AST Tree-Diff Anti-Unification Refactoring Inducer
# ===========================================================================


def _ast_to_sexpr(node: Any) -> Any:
    """Convert a Python ``ast.AST`` subtree into a canonical hashable S-expression."""
    if isinstance(node, ast.AST):
        # Ignore source line/column metadata and Load/Store context noise
        if isinstance(node, (ast.Load, ast.Store, ast.Del)):
            return ("CTX", type(node).__name__)
        fields: list[tuple[str, Any]] = []
        for fname, fval in ast.iter_fields(node):
            if fname == "ctx":
                continue
            fields.append((fname, _ast_to_sexpr(fval)))
        return ("AST", type(node).__name__, tuple(fields))
    if isinstance(node, list):
        return ("LIST", tuple(_ast_to_sexpr(x) for x in node))
    return ("LIT", type(node).__name__, node)


def _sexpr_to_ast(sexpr: Any) -> Any:
    """Reconstruct a valid ``ast.AST`` subtree from a canonical S-expression."""
    tag = sexpr[0]
    if tag == "LIT":
        return sexpr[2]
    if tag == "LIST":
        return [_sexpr_to_ast(x) for x in sexpr[1]]
    if tag == "CTX":
        cls = getattr(ast, sexpr[1])
        return cls()
    if tag == "AST":
        cls_name = sexpr[1]
        cls = getattr(ast, cls_name)
        kwargs = {fname: _sexpr_to_ast(fval) for fname, fval in sexpr[2]}
        if cls_name == "Name" and "ctx" not in kwargs:
            kwargs["ctx"] = ast.Load()
        node = cls(**kwargs)
        return node
    raise ValueError(f"Invalid AST S-expression tag: {tag!r}")


def _fix_store_contexts(tree: ast.AST) -> ast.AST:
    """Ensure assignment targets carry ``ast.Store()`` context before ``compile()``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    t.ctx = ast.Store()
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            node.target.ctx = ast.Store()
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            node.target.ctx = ast.Store()
    ast.fix_missing_locations(tree)
    return tree


def _contains_sexpr_subtree(haystack: Any, needle: Any) -> bool:
    if haystack == needle:
        return True
    if isinstance(haystack, tuple):
        return any(_contains_sexpr_subtree(item, needle) for item in haystack)
    return False


def _collect_minimal_sexpr_diffs(lhs: Any, rhs: Any) -> list[tuple[Any, Any]]:
    """Isolate minimal changed AST S-expression subtree pairs ``(lhs_sub, rhs_sub)``."""
    if lhs == rhs:
        return []
    if (
        isinstance(lhs, tuple)
        and isinstance(rhs, tuple)
        and lhs[0] == "AST"
        and rhs[0] == "AST"
        and lhs[1] == rhs[1]
    ):
        l_fields = dict(lhs[2])
        r_fields = dict(rhs[2])
        if set(l_fields.keys()) == set(r_fields.keys()):
            # 1. Direct scalar field change (e.g., Name.id or arg.arg) -> keep this AST node!
            has_scalar_diff = any(
                l_fields[k] != r_fields[k]
                and isinstance(l_fields[k], tuple)
                and l_fields[k][0] == "LIT"
                for k in l_fields
            )
            changed_keys = [k for k in l_fields if l_fields[k] != r_fields[k]]
            if not has_scalar_diff and changed_keys:
                # 2. Check cross-field subtree movement (e.g., If.body <-> If.orelse)
                cross_field = any(
                    k1 != k2 and _contains_sexpr_subtree(r_fields[k2], l_fields[k1])
                    for k1 in changed_keys
                    for k2 in changed_keys
                )
                # 3. Check pure wrapper embedding (where drilling down would turn lhs into a bare $HOLE)
                pure_wrapper = any(
                    l_fields[k] != r_fields[k] and _contains_sexpr_subtree(r_fields[k], l_fields[k])
                    for k in changed_keys
                )
                if not cross_field and not pure_wrapper:
                    out_d: list[tuple[Any, Any]] = []
                    for k in changed_keys:
                        out_d.extend(_collect_minimal_sexpr_diffs(l_fields[k], r_fields[k]))
                    return out_d
    if (
        isinstance(lhs, tuple)
        and isinstance(rhs, tuple)
        and lhs[0] == "LIST"
        and rhs[0] == "LIST"
        and len(lhs[1]) == len(rhs[1])
    ):
        out: list[tuple[Any, Any]] = []
        for a, b in zip(lhs[1], rhs[1]):
            out.extend(_collect_minimal_sexpr_diffs(a, b))
        return out
    return [(lhs, rhs)]


def _lift_sexpr_pair(
    lhs: Any,
    rhs: Any,
    params: Mapping[str, str],
) -> tuple[Any, Any]:
    """Anti-unify a single ``(lhs, rhs)`` AST diff pair into a reusable pattern->replacement rule.

    1. Replaces any scalar string matching ``params[pname]`` with ``("$PARAM", pname)``.
    2. Replaces maximal common AST/LIST subtrees shared between ``lhs`` and ``rhs`` with
       positional meta-variables ``("$HOLE", idx)``.
    """
    val_to_param = {str(v): str(k) for k, v in params.items()}

    def _sub_params(sexpr: Any) -> Any:
        if isinstance(sexpr, tuple) and sexpr:
            if len(sexpr) == 3 and sexpr[0] == "LIT" and isinstance(sexpr[2], str) and sexpr[2] in val_to_param:
                return ("$PARAM", val_to_param[sexpr[2]])
            return tuple(_sub_params(x) for x in sexpr)
        return sexpr

    lhs_p = _sub_params(lhs)
    rhs_p = _sub_params(rhs)

    # Collect non-trivial subtrees in rhs_p so shared subtrees with lhs_p become $HOLE_i
    rhs_subtrees: set[Any] = set()

    def _gather(s: Any) -> None:
        if isinstance(s, tuple) and s:
            if isinstance(s[0], str) and s[0] in {"AST", "LIST"}:
                rhs_subtrees.add(s)
            for item in s:
                if isinstance(item, tuple):
                    _gather(item)

    _gather(rhs_p)
    # Exclude the root itself so the rewrite isn't trivial $HOLE_0 -> $HOLE_0
    rhs_subtrees.discard(lhs_p)

    hole_map: dict[Any, str] = {}

    def _abstract_lhs(s: Any) -> Any:
        if isinstance(s, tuple) and s:
            if s[0] in {"AST", "LIST"} and s in rhs_subtrees:
                h_name = hole_map.get(s)
                if h_name is None:
                    h_name = f"$H{len(hole_map)}"
                    hole_map[s] = h_name
                return ("$HOLE", h_name)
            if s[0] == "AST":
                return ("AST", s[1], tuple(( fn, _abstract_lhs(fv)) for fn, fv in s[2]))
            if s[0] == "LIST":
                return ("LIST", tuple(_abstract_lhs(x) for x in s[1]))
        return s

    lhs_pat = _abstract_lhs(lhs_p)

    def _abstract_rhs(s: Any) -> Any:
        if s in hole_map:
            return ("$HOLE", hole_map[s])
        if isinstance(s, tuple) and s:
            if s[0] == "AST":
                return ("AST", s[1], tuple((fn, _abstract_rhs(fv)) for fn, fv in s[2]))
            if s[0] == "LIST":
                return ("LIST", tuple(_abstract_rhs(x) for x in s[1]))
        return s

    rhs_pat = _abstract_rhs(rhs_p)
    return (lhs_pat, rhs_pat)


def _match_pattern(
    pat: Any,
    target: Any,
    params: Mapping[str, str],
    bindings: dict[str, Any],
) -> bool:
    if isinstance(pat, tuple) and pat:
        if pat[0] == "$HOLE":
            h = pat[1]
            if h in bindings:
                return bindings[h] == target
            bindings[h] = target
            return True
        if pat[0] == "$PARAM":
            expected_str = params.get(pat[1])
            return (
                isinstance(target, tuple)
                and target[0] == "LIT"
                and target[2] == expected_str
            )
        if not isinstance(target, tuple) or len(pat) != len(target) or pat[0] != target[0]:
            return False
        if pat[0] == "AST":
            if pat[1] != target[1] or len(pat[2]) != len(target[2]):
                return False
            for (f1, v1), (f2, v2) in zip(pat[2], target[2]):
                if f1 != f2 or not _match_pattern(v1, v2, params, bindings):
                    return False
            return True
        if pat[0] == "LIST":
            if len(pat[1]) != len(target[1]):
                return False
            return all(_match_pattern(a, b, params, bindings) for a, b in zip(pat[1], target[1]))
        return pat == target
    return pat == target


def _instantiate_pattern(
    pat: Any,
    params: Mapping[str, str],
    bindings: Mapping[str, Any],
) -> Any:
    if isinstance(pat, tuple) and pat:
        if pat[0] == "$HOLE":
            return bindings[pat[1]]
        if pat[0] == "$PARAM":
            val = params[pat[1]]
            return ("LIT", "str", val)
        if pat[0] == "AST":
            return (
                "AST",
                pat[1],
                tuple((fn, _instantiate_pattern(fv, params, bindings)) for fn, fv in pat[2]),
            )
        if pat[0] == "LIST":
            return ("LIST", tuple(_instantiate_pattern(x, params, bindings) for x in pat[1]))
    return pat


def _is_wrapping_rule(lhs_pat: Any, rhs_pat: Any) -> bool:
    """Return True if ``rhs_pat`` embeds the entire ``lhs_pat`` structure inside a larger wrapper
    (single-pass rewrite, e.g. wrapping Return) vs algebraic reduction/swap (bottom-up fixpoint)."""
    def _depth(s: Any) -> int:
        if not isinstance(s, tuple) or not s:
            return 0
        if s[0] == "AST":
            return 1 + max((_depth(fv) for _fn, fv in s[2]), default=0)
        if s[0] == "LIST":
            return max((_depth(x) for x in s[1]), default=0)
        return 0

    return _depth(rhs_pat) > _depth(lhs_pat)


@dataclass(frozen=True)
class ASTRewriteCard:
    """Induced structural Python AST pattern-rewrite operator card."""

    card_id: str
    action_id: str
    rules: tuple[tuple[Any, Any], ...]
    single_pass: bool
    support_episodes: int

    def compressed_bits(self) -> int:
        return compressed_bit_length(canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "action_id": self.action_id,
            "rule_count": len(self.rules),
            "single_pass": self.single_pass,
            "support_episodes": self.support_episodes,
            "rules_repr": [repr(r) for r in self.rules],
        }

    def apply_code(self, source_code: str, params: Mapping[str, str] | None = None) -> str:
        p_map = dict(params or {})
        tree = ast.parse(source_code)
        sexpr = _ast_to_sexpr(tree)

        def _rewrite_once(node_s: Any) -> tuple[Any, bool]:
            changed = False
            # Transform children bottom-up first
            if isinstance(node_s, tuple) and node_s:
                if node_s[0] == "AST":
                    new_fields = []
                    for fn, fv in node_s[2]:
                        nfv, ch = _rewrite_once(fv)
                        new_fields.append((fn, nfv))
                        changed = changed or ch
                    node_s = ("AST", node_s[1], tuple(new_fields))
                elif node_s[0] == "LIST":
                    new_items = []
                    for x in node_s[1]:
                        nx, ch = _rewrite_once(x)
                        new_items.append(nx)
                        changed = changed or ch
                    node_s = ("LIST", tuple(new_items))

            for lhs_pat, rhs_pat in self.rules:
                bindings: dict[str, Any] = {}
                if _match_pattern(lhs_pat, node_s, p_map, bindings):
                    replaced = _instantiate_pattern(rhs_pat, p_map, bindings)
                    if replaced != node_s:
                        return (replaced, True)
            return (node_s, changed)

        if self.single_pass:
            sexpr, _ = _rewrite_once(sexpr)
        else:
            for _ in range(16):
                sexpr, ch = _rewrite_once(sexpr)
                if not ch:
                    break

        new_tree = _fix_store_contexts(_sexpr_to_ast(sexpr))
        return ast.unparse(new_tree)


class ASTRewriteInducer:
    """Domain-agnostic before/after Python AST anti-unification inducer (RDC min_support=2)."""

    def __init__(self, *, min_rule_support: int = 2) -> None:
        self.min_rule_support = int(min_rule_support)
        self._cards: dict[str, ASTRewriteCard] = {}
        self._frozen = False

    def freeze(self) -> None:
        self._frozen = True

    @property
    def cards(self) -> dict[str, ASTRewriteCard]:
        return dict(self._cards)

    def fit(
        self,
        episodes: Sequence[tuple[str, str, Mapping[str, str], str]],
    ) -> dict[str, ASTRewriteCard]:
        if self._frozen:
            raise RuntimeError("ASTRewriteInducer is frozen")
        grouped: dict[str, list[tuple[str, Mapping[str, str], str]]] = {}
        for aid, b_code, prms, a_code in episodes:
            grouped.setdefault(aid, []).append((b_code, prms, a_code))

        for aid, ep_list in sorted(grouped.items()):
            rule_counts: dict[tuple[Any, Any], int] = {}
            rule_order: list[tuple[Any, Any]] = []
            for b_code, prms, a_code in ep_list:
                b_s = _ast_to_sexpr(ast.parse(b_code))
                a_s = _ast_to_sexpr(ast.parse(a_code))
                diffs = _collect_minimal_sexpr_diffs(b_s, a_s)
                seen_in_ep: set[tuple[Any, Any]] = set()
                for lhs, rhs in diffs:
                    rule = _lift_sexpr_pair(lhs, rhs, prms)
                    if rule not in seen_in_ep:
                        seen_in_ep.add(rule)
                        if rule not in rule_counts:
                            rule_order.append(rule)
                        rule_counts[rule] = rule_counts.get(rule, 0) + 1

            promoted = [r for r in rule_order if rule_counts[r] >= self.min_rule_support]
            single_pass = any(_is_wrapping_rule(lp, rp) for lp, rp in promoted)
            self._cards[aid] = ASTRewriteCard(
                card_id=f"ast_card_{aid}_v1",
                action_id=aid,
                rules=tuple(promoted),
                single_pass=single_pass,
                support_episodes=len(ep_list),
            )
        return dict(self._cards)


def build_ast_training_episodes() -> list[tuple[str, str, dict[str, str], str]]:
    """Construct deterministic before/after Python AST refactoring training episodes."""
    return [
        # 1. rename_parameter (5 demos)
        ("rename_parameter", "def f1(a, b):\n    return a + b\n", {"old_name": "a", "new_name": "lhs"}, "def f1(lhs, b):\n    return lhs + b\n"),
        ("rename_parameter", "def f2(n):\n    return n * n\n", {"old_name": "n", "new_name": "val"}, "def f2(val):\n    return val * val\n"),
        ("rename_parameter", "def f3(k, m):\n    t = k - m\n    return t + k\n", {"old_name": "k", "new_name": "first"}, "def f3(first, m):\n    t = first - m\n    return t + first\n"),
        ("rename_parameter", "def f4(u, v):\n    return max(u, v)\n", {"old_name": "v", "new_name": "rhs"}, "def f4(u, rhs):\n    return max(u, rhs)\n"),
        ("rename_parameter", "def f5(p):\n    p = p + 1\n    return p\n", {"old_name": "p", "new_name": "cnt"}, "def f5(cnt):\n    cnt = cnt + 1\n    return cnt\n"),
        # 2. swap_if_negated_guard (5 demos)
        ("swap_if_negated_guard", "def g1(x):\n    if not (x > 0):\n        return -1\n    else:\n        return 1\n", {}, "def g1(x):\n    if x > 0:\n        return 1\n    else:\n        return -1\n"),
        ("swap_if_negated_guard", "def g2(a, b):\n    if not (a == b):\n        return a\n    else:\n        return b\n", {}, "def g2(a, b):\n    if a == b:\n        return b\n    else:\n        return a\n"),
        ("swap_if_negated_guard", "def g3(n):\n    if not (n % 2 == 0):\n        return n + 1\n    else:\n        return n\n", {}, "def g3(n):\n    if n % 2 == 0:\n        return n\n    else:\n        return n + 1\n"),
        ("swap_if_negated_guard", "def g4(z):\n    if not (z < 10):\n        return 10\n    else:\n        return z\n", {}, "def g4(z):\n    if z < 10:\n        return z\n    else:\n        return 10\n"),
        ("swap_if_negated_guard", "def g5(u, v):\n    if not (u >= v):\n        return v - u\n    else:\n        return u - v\n", {}, "def g5(u, v):\n    if u >= v:\n        return u - v\n    else:\n        return v - u\n"),
        # 3. fold_identity_arithmetic (6 demos, >=2 support per algebraic neutral pattern)
        ("fold_identity_arithmetic", "def h1(x):\n    return (x + 0) + (0 + x)\n", {}, "def h1(x):\n    return x + x\n"),
        ("fold_identity_arithmetic", "def h2(y):\n    return (y - 0) + (y * 1)\n", {}, "def h2(y):\n    return y + y\n"),
        ("fold_identity_arithmetic", "def h3(a, b):\n    return (1 * a) + (b + 0)\n", {}, "def h3(a, b):\n    return a + b\n"),
        ("fold_identity_arithmetic", "def h4(a, b):\n    return (0 + a) - (b - 0)\n", {}, "def h4(a, b):\n    return a - b\n"),
        ("fold_identity_arithmetic", "def h5(z):\n    return (z * 1) + (1 * z)\n", {}, "def h5(z):\n    return z + z\n"),
        ("fold_identity_arithmetic", "def h6(m):\n    return (m - 0) + (0 + m)\n", {}, "def h6(m):\n    return m + m\n"),
        # 4. wrap_nonnegative_clamp (5 demos)
        ("wrap_nonnegative_clamp", "def c1(x):\n    return x - 5\n", {}, "def c1(x):\n    return max(0, x - 5)\n"),
        ("wrap_nonnegative_clamp", "def c2(a, b):\n    return a - b\n", {}, "def c2(a, b):\n    return max(0, a - b)\n"),
        ("wrap_nonnegative_clamp", "def c3(n):\n    return n * 3 - 10\n", {}, "def c3(n):\n    return max(0, n * 3 - 10)\n"),
        ("wrap_nonnegative_clamp", "def c4(u, v):\n    return u + v\n", {}, "def c4(u, v):\n    return max(0, u + v)\n"),
        ("wrap_nonnegative_clamp", "def c5(k):\n    return -k\n", {}, "def c5(k):\n    return max(0, -k)\n"),
    ]


def evaluate_ast_refactoring_suite(seed: int) -> dict[str, Any]:
    """Train ``ASTRewriteInducer`` on 5–6 small demos/operator and evaluate on unseen functions."""
    train_eps = build_ast_training_episodes()

    inducer = ASTRewriteInducer(min_rule_support=2)
    inducer.fit(train_eps)
    inducer.freeze()
    cards = inducer.cards

    rng = random.Random(seed * 509 + 41)
    test_cases: list[dict[str, Any]] = []

    # Unseen test cases per AST refactoring operator (6 unseen functions/op = 24 unseen AST tasks/seed)
    for idx in range(6):
        c_bias = rng.randint(2, 15)
        # 1. rename_parameter — includes string literal containing old_name to catch naive regex corruption!
        fn_rp = f"unseen_rp_{seed}_{idx}"
        test_cases.append(
            {
                "action_id": "rename_parameter",
                "fn_name": fn_rp,
                "before_code": (
                    f"def {fn_rp}(x, scale):\n"
                    f"    label = 'axis_x'\n"
                    f"    return (x * scale + {c_bias}, label)\n"
                ),
                "params": {"old_name": "x", "new_name": "item_val"},
                "gold_fn": lambda a, b, cb=c_bias: (a * b + cb, "axis_x"),
                "inputs": [(rng.randint(-20, 20), rng.randint(-5, 5)) for _ in range(8)],
                "ast_check": lambda tree: any(
                    isinstance(n, ast.arg) and n.arg == "item_val" for n in ast.walk(tree)
                )
                and not any(isinstance(n, ast.arg) and n.arg == "x" for n in ast.walk(tree)),
            }
        )

        # 2. swap_if_negated_guard
        fn_sw = f"unseen_sw_{seed}_{idx}"
        test_cases.append(
            {
                "action_id": "swap_if_negated_guard",
                "fn_name": fn_sw,
                "before_code": (
                    f"def {fn_sw}(p, q):\n"
                    f"    if not (p * 2 > q + {c_bias}):\n"
                    f"        return q - p\n"
                    f"    else:\n"
                    f"        return p + q + {c_bias}\n"
                ),
                "params": {},
                "gold_fn": lambda p, q, cb=c_bias: (p + q + cb) if (p * 2 > q + cb) else (q - p),
                "inputs": [(rng.randint(-15, 25), rng.randint(-15, 25)) for _ in range(8)],
                "ast_check": lambda tree: not any(
                    isinstance(n, ast.If)
                    and isinstance(n.test, ast.UnaryOp)
                    and isinstance(n.test.op, ast.Not)
                    for n in ast.walk(tree)
                ),
            }
        )

        # 3. fold_identity_arithmetic (deeply nested unseen multi-step arithmetic!)
        fn_fd = f"unseen_fd_{seed}_{idx}"
        test_cases.append(
            {
                "action_id": "fold_identity_arithmetic",
                "fn_name": fn_fd,
                "before_code": (
                    f"def {fn_fd}(a, b, c):\n"
                    f"    t = ((a + 0) * 1 - 0) + (0 + (1 * b))\n"
                    f"    return (t * 1 + (c - 0)) + 0\n"
                ),
                "params": {},
                "gold_fn": lambda a, b, c: a + b + c,
                "inputs": [
                    (rng.randint(-30, 30), rng.randint(-30, 30), rng.randint(-30, 30))
                    for _ in range(8)
                ],
                "ast_check": lambda tree: not any(
                    isinstance(n, ast.Constant) and n.value in (0, 1) for n in ast.walk(tree)
                ),
            }
        )

        # 4. wrap_nonnegative_clamp
        fn_cl = f"unseen_cl_{seed}_{idx}"
        test_cases.append(
            {
                "action_id": "wrap_nonnegative_clamp",
                "fn_name": fn_cl,
                "before_code": (
                    f"def {fn_cl}(a, b):\n"
                    f"    return (a - b) - {c_bias}\n"
                ),
                "params": {},
                "gold_fn": lambda a, b, cb=c_bias: max(0, (a - b) - cb),
                "inputs": [(rng.randint(-20, 30), rng.randint(-20, 30)) for _ in range(8)],
                "ast_check": lambda tree: any(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == "max"
                    for n in ast.walk(tree)
                ),
            }
        )

    per_op_hits: dict[str, list[int]] = {}
    compile_valid = 0
    exec_verified = 0
    naive_replace_hits = 0

    for tc in test_cases:
        aid = tc["action_id"]
        card = cards[aid]
        out_src = card.apply_code(tc["before_code"], tc["params"])
        fn_exec, _reason = RestrictedExecutionSandbox.compile_verified_function(
            out_src, tc["fn_name"]
        )
        ok_compile = fn_exec is not None
        if ok_compile:
            compile_valid += 1
        ok_ast = False
        if ok_compile:
            parsed_out = ast.parse(out_src)
            ok_ast = bool(tc["ast_check"](parsed_out))

        ok_exec = False
        if ok_compile and ok_ast and fn_exec is not None:
            ok_exec = all(
                fn_exec(*inp) == tc["gold_fn"](*inp) for inp in tc["inputs"]
            )
        if ok_exec:
            exec_verified += 1
        per_op_hits.setdefault(aid, []).append(1 if ok_exec else 0)

        # Compare naive string replace on rename_parameter (corrupts string literal 'x'!)
        if aid == "rename_parameter":
            naive_src = tc["before_code"].replace(
                tc["params"]["old_name"], tc["params"]["new_name"]
            )
            n_fn, _ = RestrictedExecutionSandbox.compile_verified_function(
                naive_src, tc["fn_name"]
            )
            if n_fn is not None and all(n_fn(*inp) == tc["gold_fn"](*inp) for inp in tc["inputs"]):
                naive_replace_hits += 1

    total = len(test_cases)
    return {
        "total_unseen_tasks": total,
        "compile_valid_rate": round(compile_valid / float(total), 6),
        "exact_verified_accuracy": round(exec_verified / float(total), 6),
        "per_operator_accuracy": {
            k: round(sum(v) / float(len(v)), 6) for k, v in sorted(per_op_hits.items())
        },
        "naive_string_replace_rename_accuracy": round(naive_replace_hits / 6.0, 6),
        "library_compressed_bits": sum(c.compressed_bits() for c in cards.values()),
        "executable_cards": {k: c.to_dict() for k, c in sorted(cards.items())},
    }


# ===========================================================================
# PART B: Execution-Only 20-Task HumanEval-Micro Suite (PDF p. 13 Gate >= 50%)
# ===========================================================================


@dataclass(frozen=True)
class HumanEvalMicroTask:
    """Strict execution-only synthesis benchmark task (public = types + 4 I/O pairs ONLY)."""

    task_id: str
    canonical_name: str
    arg_types: tuple[str, ...]
    out_type: str
    public_train_io: tuple[tuple[tuple[Any, ...], Any], ...]
    hidden_test_io: tuple[tuple[tuple[Any, ...], Any], ...]


HUMANEVAL_MICRO_FORBIDDEN_NAMES: tuple[str, ...] = (
    "gcd",
    "factorial",
    "is_palindrome",
    "sum_evens",
    "max_diff",
    "count_vowels",
    "dedup_order",
    "dot_product",
    "clamp_list",
    "running_sum",
    "second_largest",
    "rotate_left",
    "all_positive",
    "abs_sum",
    "even_squares",
    "count_greater",
    "reverse_words",
    "is_sorted_asc",
    "alternating_sum",
    "common_elements",
)


def build_humaneval_micro_suite(seed: int) -> list[HumanEvalMicroTask]:
    """Construct the pre-registered 20-task HumanEval-Micro suite with 4 public training
    I/O pairs and 12 strictly disjoint unseen edge/random/stress test cases per task."""
    rng = random.Random(seed * 887 + 131)

    def _mk(
        idx: int,
        cname: str,
        arg_types: tuple[str, ...],
        out_type: str,
        train_inputs: Sequence[tuple[Any, ...]],
        hidden_inputs: Sequence[tuple[Any, ...]],
        gold_fn: Callable[..., Any],
    ) -> HumanEvalMicroTask:
        tr_set = {tuple(normalize_slot_value(x) for x in inp) for inp in train_inputs}
        filtered_hidden: list[tuple[Any, ...]] = []
        for h_inp in hidden_inputs:
            nh = tuple(normalize_slot_value(x) for x in h_inp)
            if nh not in tr_set and nh not in {
                tuple(normalize_slot_value(x) for x in z) for z in filtered_hidden
            }:
                filtered_hidden.append(h_inp)
            if len(filtered_hidden) == 12:
                break
        assert len(train_inputs) == 4, f"{cname} must have 4 public demos"
        assert len(filtered_hidden) == 12, f"{cname} must have 12 disjoint hidden tests (got {len(filtered_hidden)})"

        tr_io = tuple((inp, gold_fn(*inp)) for inp in train_inputs)
        ho_io = tuple((inp, gold_fn(*inp)) for inp in filtered_hidden)
        return HumanEvalMicroTask(
            task_id=f"he_micro_{idx:02d}",
            canonical_name=cname,
            arg_types=arg_types,
            out_type=out_type,
            public_train_io=tr_io,
            hidden_test_io=ho_io,
        )

    tasks: list[HumanEvalMicroTask] = [
        # 1. gcd
        _mk(
            1,
            "gcd",
            ("int", "int"),
            "int",
            [(12, 8), (15, 25), (14, 21), (9, 4)],
            [(48, 18), (7, 13), (100, 75), (81, 27), (17, 17), (56, 98), (120, 45), (35, 64), (270, 192), (1071, 462), (24, 36), (99, 33)]
            + [(rng.randint(20, 200), rng.randint(20, 200)) for _ in range(4)],
            math.gcd,
        ),
        # 2. factorial
        _mk(
            2,
            "factorial",
            ("int",),
            "int",
            [(1,), (3,), (4,), (5,)],
            [(0,), (2,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,)],
            math.factorial,
        ),
        # 3. is_palindrome
        _mk(
            3,
            "is_palindrome",
            ("str",),
            "bool",
            [("level",), ("solar",), ("noon",), ("planet",)],
            [("",), ("a",), ("racecar",), ("radar",), ("python",), ("civic",), ("ab",), ("abba",), ("abcba",), ("abca",), ("kayak",), ("mirror",)],
            lambda s: s == s[::-1],
        ),
        # 4. sum_evens
        _mk(
            4,
            "sum_evens",
            ("seq_int",),
            "int",
            [([1, 2, 4, 5],), ([3, 7],), ([-2, 6, 1],), ([0, 8, -4],)],
            [([],), ([9],), ([10],), ([-8, -6, -3, 4],), ([2, 4, 6, 8, 10, 12],), ([1, 3, 5, 7, 9],)]
            + [([rng.randint(-50, 50) for _ in range(rng.randint(6, 14))],) for _ in range(10)],
            lambda xs: sum(x for x in xs if x % 2 == 0),
        ),
        # 5. max_diff (includes empty/singleton hidden edge cases testing RDC boundary refutation!)
        _mk(
            5,
            "max_diff",
            ("seq_int",),
            "int",
            [([3, 9, 2],), ([-4, 5, 0],), ([10, 10, 4],), ([1, 8, 3, 6],)],
            [([],), ([42],), ([-7, -7, -7],), ([-20, -5],), ([100, -100, 0, 50],), ([0, 1],)]
            + [([rng.randint(-100, 100) for _ in range(rng.randint(6, 15))],) for _ in range(10)],
            lambda xs: 0 if not xs else (max(xs) - min(xs)),
        ),
        # 6. count_vowels (honest underdetermined lexical task — zero vowel constants in learner!)
        _mk(
            6,
            "count_vowels",
            ("str",),
            "int",
            [("apple",), ("sky",), ("boat",), ("tree",)],
            [("",), ("rhythm",), ("education",), ("queue",), ("AEIOU",), ("planet",), ("orbit",), ("crystal",), ("audio",), ("ubuntu",), ("lynx",), ("sequoia",)],
            lambda s: sum(1 for ch in s if ch.lower() in "aeiou"),
        ),
        # 7. dedup_order
        _mk(
            7,
            "dedup_order",
            ("seq_int",),
            "seq_int",
            [([3, 1, 3, 2],), ([5, 5, 5],), ([-1, 0, -1, 4],), ([2, 8, 2, 8, 1],)],
            [([],), ([7],), ([9, 8, 7, 9, 8, 7],), ([-5, -5, 0, 3, 0, -5, 9],)]
            + [([rng.randint(-8, 8) for _ in range(rng.randint(8, 16))],) for _ in range(12)],
            lambda xs: list(dict.fromkeys(xs)),
        ),
        # 8. dot_product
        _mk(
            8,
            "dot_product",
            ("seq_int", "seq_int"),
            "int",
            [([1, 2, 3], [4, 5, 6]), ([2, -1], [3, 7]), ([0, 4], [9, 2]), ([-2, 3, 1], [5, 2, -4])],
            [([], []), ([6], [-3]), ([1, 0, -1, 2], [4, 9, 3, -5]), ([-4, -2, -1], [-3, 5, -2])]
            + [
                (
                    [rng.randint(-15, 15) for _ in range(8)],
                    [rng.randint(-15, 15) for _ in range(8)],
                )
                for _ in range(12)
            ],
            lambda xs, ys: sum(a * b for a, b in zip(xs, ys)),
        ),
        # 9. clamp_list
        _mk(
            9,
            "clamp_list",
            ("seq_int", "int", "int"),
            "seq_int",
            [([-3, 2, 9], 0, 5), ([1, 8, 4], 2, 6), ([-10, -2, 3], -5, 0), ([5, 0, 12], 1, 10)],
            [([], -3, 3), ([100], -10, 10), ([-50, 0, 50], -20, 20), ([4, 4, 4], 5, 9)]
            + [([rng.randint(-40, 40) for _ in range(10)], -12, 18) for _ in range(12)],
            lambda xs, lo, hi: [min(hi, max(lo, x)) for x in xs],
        ),
        # 10. running_sum
        _mk(
            10,
            "running_sum",
            ("seq_int",),
            "seq_int",
            [([1, 2, 3],), ([4, -1, 5],), ([-2, -3, 6],), ([0, 7, -2, 1],)],
            [([],), ([9],), ([-5, 5, -5, 5],), ([10, 20, 30, 40, 50],)]
            + [([rng.randint(-25, 25) for _ in range(rng.randint(7, 14))],) for _ in range(12)],
            lambda xs: [sum(xs[: i + 1]) for i in range(len(xs))],
        ),
        # 11. second_largest
        _mk(
            11,
            "second_largest",
            ("seq_int",),
            "int",
            [([4, 9, 2],), ([10, 10, 7, 3],), ([-5, -1, -9],), ([3, 8, 8, 5, 1],)],
            [([1, 2],), ([99, 100, 100],), ([-100, -50, -50, -200],), ([5, 4, 3, 2, 1, 5],)]
            + [
                (list({rng.randint(-80, 80) for _ in range(8)} | {101, -101}) + [101],)
                for _ in range(12)
            ],
            lambda xs: sorted(set(xs))[-2],
        ),
        # 12. rotate_left (includes empty-list & k > len(xs) hidden stress cases!)
        _mk(
            12,
            "rotate_left",
            ("seq_int", "int"),
            "seq_int",
            [([1, 2, 3, 4], 1), ([5, 6, 7], 2), ([8, 9, 10], 4), ([2, 4, 6, 8], 3)],
            [([], 0), ([], 5), ([42], 7), ([1, 2, 3, 4, 5], 0), ([1, 2, 3, 4, 5], 5), ([1, 2, 3, 4, 5], 13)]
            + [([rng.randint(-30, 30) for _ in range(9)], rng.randint(1, 25)) for _ in range(10)],
            lambda xs, k: [] if not xs else xs[k % len(xs) :] + xs[: k % len(xs)],
        ),
        # 13. all_positive
        _mk(
            13,
            "all_positive",
            ("seq_int",),
            "bool",
            [([2, 5, 1],), ([3, 0, 4],), ([7, -1, 2],), ([9, 4],)],
            [([],), ([1],), ([0],), ([-1],), ([5, 8, 12, 3, 19],), ([5, 8, 0, 3],), ([-3, -2, -1],)]
            + [([rng.randint(1, 40) for _ in range(8)],) for _ in range(4)]
            + [([rng.randint(-10, 40) for _ in range(8)] + [0],) for _ in range(4)],
            lambda xs: all(x > 0 for x in xs),
        ),
        # 14. abs_sum
        _mk(
            14,
            "abs_sum",
            ("seq_int",),
            "int",
            [([-3, 4, -2],), ([5, 0, -1],), ([-7, -2],), ([1, -8, 3],)],
            [([],), ([0],), ([-99],), ([-10, -20, -30, 40],)]
            + [([rng.randint(-60, 60) for _ in range(rng.randint(7, 14))],) for _ in range(12)],
            lambda xs: sum(abs(x) for x in xs),
        ),
        # 15. even_squares
        _mk(
            15,
            "even_squares",
            ("seq_int",),
            "seq_int",
            [([1, 2, 3, 4],), ([-2, 5, 6],), ([3, 7],), ([0, -4, 1],)],
            [([],), ([2],), ([5],), ([-6, -3, 0, 4, 9, 10],)]
            + [([rng.randint(-25, 25) for _ in range(rng.randint(8, 14))],) for _ in range(12)],
            lambda xs: [x * x for x in xs if x % 2 == 0],
        ),
        # 16. count_greater
        _mk(
            16,
            "count_greater",
            ("seq_int", "int"),
            "int",
            [([1, 5, 8, 3], 4), ([2, 2, 9], 2), ([-5, 0, 3], -1), ([4, 1, 2], 10)],
            [([], 0), ([5, 5, 5], 5), ([5, 5, 5], 4), ([-10, -3, -1, -8], -5)]
            + [([rng.randint(-30, 30) for _ in range(12)], rng.randint(-10, 10)) for _ in range(12)],
            lambda xs, th: sum(1 for x in xs if x > th),
        ),
        # 17. reverse_words
        _mk(
            17,
            "reverse_words",
            ("str",),
            "str",
            [("alpha beta gamma",), ("north south",), ("one",), ("red green blue yellow",)],
            [("",), ("solo",), ("first second",), ("a b c d e f",), ("  spaced   tokens  here ",), ("cpu native symbolic learner",), ("deepmind mirror seven",), ("x y z",), ("ten nine eight seven",), ("left right",), ("up down Charm",), ("sun moon stars",)],
            lambda s: " ".join(s.split()[::-1]),
        ),
        # 18. is_sorted_asc
        _mk(
            18,
            "is_sorted_asc",
            ("seq_int",),
            "bool",
            [([1, 3, 5],), ([2, 2, 7],), ([4, 1, 6],), ([-3, 0, 2],)],
            [([],), ([8],), ([1, 2, 3, 4, 5, 6],), ([6, 5, 4],), ([1, 3, 2, 4],), ([-10, -10, -5, 0, 0, 9],)]
            + [([rng.randint(-20, 20) for _ in range(9)],) for _ in range(10)],
            lambda xs: list(xs) == sorted(xs),
        ),
        # 19. alternating_sum
        _mk(
            19,
            "alternating_sum",
            ("seq_int",),
            "int",
            [([5, 2, 4],), ([10, 3],), ([1, 2, 3, 4],), ([-2, 5, -1],)],
            [([],), ([7],), ([-8],), ([10, 10, 10, 10],), ([6, -2, 3, -4, 5],)]
            + [([rng.randint(-30, 30) for _ in range(rng.randint(6, 13))],) for _ in range(12)],
            lambda xs: sum(x if i % 2 == 0 else -x for i, x in enumerate(xs)),
        ),
        # 20. common_elements
        _mk(
            20,
            "common_elements",
            ("seq_int", "seq_int"),
            "seq_int",
            [
                ([1, 2, 3, 2], [2, 4, 1]),
                ([5, 6], [7, 8]),
                ([-1, 0, 3, -1], [3, -1, 9]),
                ([4, 2, 4, 8], [8, 4]),
            ],
            [([], [1, 2]), ([1, 2], []), ([3, 3, 3], [3]), ([-5, 2, -5, 7, 9], [9, -5, 0])]
            + [
                (
                    [rng.randint(-10, 10) for _ in range(10)],
                    [rng.randint(-10, 10) for _ in range(8)],
                )
                for _ in range(12)
            ],
            lambda xs, ys: [x for x in dict.fromkeys(xs) if x in ys],
        ),
    ]
    return tasks


def _verify_program_on_hidden_tests(
    py_source: str | None,
    fn_name: str,
    hidden_test_io: Sequence[tuple[tuple[Any, ...], Any]],
) -> bool:
    if py_source is None:
        return False
    compiled_fn, _reason = RestrictedExecutionSandbox.compile_verified_function(py_source, fn_name)
    if compiled_fn is None:
        return False
    for xs, gold_y in hidden_test_io:
        try:
            pred = normalize_slot_value(
                compiled_fn(*[copy.deepcopy(to_python_runtime_value(x)) for x in xs])
            )
        except Exception:
            return False
        norm_gold = normalize_slot_value(gold_y)
        if pred != norm_gold or type(pred) is not type(norm_gold):
            return False
    return True


def evaluate_humaneval_micro_suite(
    seed: int,
    *,
    candidate_budget: int = HUMANEVAL_CANDIDATE_BUDGET,
) -> dict[str, Any]:
    """Evaluate all 20 HumanEval-Micro tasks head-to-head across SHARD+ITL+RDC, Unguided, and Lookup."""
    suite = build_humaneval_micro_suite(seed)
    methods_out: dict[str, Any] = {}

    for method_name, use_shard in (
        ("shard_itl_rdc_code_inducer", True),
        ("unguided_enumerative_code_inducer", False),
    ):
        solved_count = 0
        total_evals = 0
        total_bits = 0
        boundary_refuted_total = 0
        per_task: dict[str, Any] = {}
        solved_asts: dict[str, Any] = {}

        for task in suite:
            # STRICT PUBLIC INTERFACE: pass ONLY (arg_types, out_type, public_train_io) — NO task name!
            expr, py_src, diag = synthesize_typed_function(
                task.arg_types,
                task.out_type,
                task.public_train_io,
                use_shard_guidance=use_shard,
                use_rdc_boundary_refutation=True,
                max_candidates=candidate_budget,
                fn_name="candidate_solution",
                target_slot_label="return_value",
            )
            passed_hidden = _verify_program_on_hidden_tests(
                py_src, "candidate_solution", task.hidden_test_io
            )
            if passed_hidden and expr is not None:
                solved_count += 1
                total_bits += diag.selected_program_bits
                solved_asts[task.task_id] = expr.to_dict()
            total_evals += diag.candidates_evaluated
            boundary_refuted_total += diag.boundary_refuted_candidates
            per_task[task.canonical_name] = {
                "task_id": task.task_id,
                "passed_hidden_12_of_12": passed_hidden,
                "candidates_evaluated": diag.candidates_evaluated,
                "boundary_refuted_candidates": diag.boundary_refuted_candidates,
                "program_bits": diag.selected_program_bits,
            }

        methods_out[method_name] = {
            "solved_tasks": solved_count,
            "total_tasks": len(suite),
            "pass_rate": round(solved_count / float(len(suite)), 6),
            "mean_candidates_evaluated": round(total_evals / float(len(suite)), 4),
            "total_candidates_evaluated": total_evals,
            "total_boundary_refuted": boundary_refuted_total,
            "total_induced_program_bits": total_bits,
            "solved_executable_asts": solved_asts,
            "per_task": per_task,
        }

    # Baseline 3: Pure I/O Memorization Lookup (returns default on unseen inputs)
    mem_solved = 0
    for task in suite:
        table = {
            tuple(normalize_slot_value(x) for x in xs): normalize_slot_value(y)
            for xs, y in task.public_train_io
        }
        all_ok = True
        for xs, gold_y in task.hidden_test_io:
            key = tuple(normalize_slot_value(x) for x in xs)
            if key not in table or table[key] != normalize_slot_value(gold_y):
                all_ok = False
                break
        if all_ok:
            mem_solved += 1

    methods_out["io_memorization_lookup"] = {
        "solved_tasks": mem_solved,
        "total_tasks": len(suite),
        "pass_rate": round(mem_solved / float(len(suite)), 6),
        "mean_candidates_evaluated": 0.0,
        "total_candidates_evaluated": 0,
    }

    shard_m = methods_out["shard_itl_rdc_code_inducer"]
    ung_m = methods_out["unguided_enumerative_code_inducer"]
    eval_reduction = round(
        1.0 - (shard_m["total_candidates_evaluated"] / float(max(1, ung_m["total_candidates_evaluated"]))),
        6,
    )
    pass_gain_vs_unguided = round(shard_m["pass_rate"] - ung_m["pass_rate"], 6)

    return {
        "schema_version": HUMANEVAL_MICRO_VERSION,
        "candidate_budget": candidate_budget,
        "prior_external_mbpp_reference": dict(PRIOR_EXTERNAL_MBPP_BASELINE),
        "methods": methods_out,
        "shard_vs_unguided_pass_gain": pass_gain_vs_unguided,
        "shard_vs_unguided_eval_reduction": eval_reduction,
    }


def run_code_domain_suite(seed: int) -> dict[str, Any]:
    """Run both Part A (Python AST Refactorings) and Part B (20-Task HumanEval-Micro)."""
    return {
        "schema_version": CODE_DOMAIN_SCHEMA_VERSION,
        "seed": seed,
        "ast_refactoring": evaluate_ast_refactoring_suite(seed),
        "humaneval_micro": evaluate_humaneval_micro_suite(seed),
    }
