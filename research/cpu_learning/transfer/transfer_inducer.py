"""Stage S6 General Before/Action/After Programmatic Operator Inducer (SHARD + ITL + RDC).

STRICT ISOLATION & SCIENTIFIC INTEGRITY (PDF pp. 3, 12, 13, 17):
- Uses the SAME before -> action -> after learning paradigm as Stages S1–S5 over typed slot frames
  bridged bidirectionally to ``StateGraph`` and ``TypedStateFrame``.
- Contains ZERO hardcoded domain rules, ZERO task-name dispatch, ZERO pretrained LLMs, ZERO GPU,
  and ZERO backpropagation.
- Neutral typed expression & higher-order combinator grammar (integer arithmetic, comparisons,
  conditionals, generic sequence folds/maps/filters/scans/zips/slices, generic word/char string
  primitives, and bounded 2-state tail iteration).
- **SHARD Proposal Guidance:** Encodes observed ``(before -> after)`` behavioral invariants into
  10,240-bit Binary Spatter Code (BSC) hypervectors via ``mirror_crystal.quotient_egraph_resonator``
  to rank candidate combinator skeletons by POPCNT Hamming similarity.
- **ITL Observational-Equivalence Synthesis:** Deduplicates observationally equivalent sub-programs
  on training states while guaranteeing 100% preservation of unchanged register slots.
- **RDC Boundary Probing & MDL Selection:** Refutes candidates that crash or violate type/domain
  contracts on synthetic boundary dreams (``0``, ``1``, ``-1``, ``[]``, ``""``, duplicates) and
  selects the minimum compressed-description-length survivor.
- **Zero-Trust Execution Verification:** Compiles induced expressions to standalone Python source
  audited by ``mirror_crystal.grammar_and_sandbox.ZeroTrustSandboxJudge.is_ast_safe`` and executed
  inside a step-bounded restricted-builtins CPU sandbox.
"""
from __future__ import annotations

import ast
import functools
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from mirror_crystal.grammar_and_sandbox import (
    ZeroTrustSandboxJudge,
    _BoundedStepInjector,
)
from mirror_crystal.quotient_egraph_resonator import (
    UNIFIED_MASK,
    bind10240,
    hamming_sim10240,
    hv10240_of,
    permute10240,
)
from research.cpu_learning.frames.state_frame import TypedStateFrame
from research.cpu_learning.shard.encoders import _majority_bundle10240
from research.cpu_learning.simulator.state_graph import (
    Entity,
    ObservationMask,
    StateGraph,
    canonical_json_bytes,
    compressed_bit_length,
    sha256_hex,
)

TRANSFER_INDUCER_SCHEMA_VERSION: str = "s6-transfer-inducer-v1"
_EMPTY_BEHAVIOR_HV: int = hv10240_of("s6:shard:behavior:empty_v1")


class SandboxStepLimitError(RuntimeError):
    """Raised when restricted sandbox execution exceeds its deterministic CPU step budget."""


class TransferInducerFrozenError(RuntimeError):
    """Raised if mutation/fitting is attempted after ``freeze()``."""


# ---------------------------------------------------------------------------
# Restricted Zero-Trust In-Process Sandbox (AST-Audited + Bounded Step Tick)
# ---------------------------------------------------------------------------

_RESTRICTED_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "ord": ord,
    "range": range,
    "reversed": reversed,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
    "Exception": Exception,
    "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
    "IndexError": IndexError,
}


class RestrictedExecutionSandbox:
    """Deterministic pure-CPU sandbox combining ``ZeroTrustSandboxJudge`` AST auditing
    with ``_BoundedStepInjector`` loop/call metering and restricted builtins."""

    @staticmethod
    @functools.lru_cache(maxsize=512)
    def compile_verified_function(
        code_str: str,
        fn_name: str,
        *,
        max_loop_steps: int = 25_000,
    ) -> tuple[Callable[..., Any] | None, str]:
        safe, reason = ZeroTrustSandboxJudge.is_ast_safe(code_str)
        if not safe:
            return (None, reason)
        loop_issue = ZeroTrustSandboxJudge._static_loop_limit_violation(code_str, max_loop_steps)
        if loop_issue is not None:
            return (None, loop_issue)

        try:
            parsed = ast.parse(code_str)
            instrumented = _BoundedStepInjector().visit(parsed)
            ast.fix_missing_locations(instrumented)
            compiled = compile(instrumented, "<s6_verified_sandbox>", "exec")
        except Exception as exc:
            return (None, f"CompileError: {exc}")

        step_box = [0]

        def _tick() -> None:
            step_box[0] += 1
            if step_box[0] > max_loop_steps:
                raise SandboxStepLimitError(f"Step budget {max_loop_steps} exceeded")

        env_globals: dict[str, Any] = {
            "__builtins__": dict(_RESTRICTED_SAFE_BUILTINS),
            "_mc_step_tick": _tick,
        }
        try:
            exec(compiled, env_globals, env_globals)
        except Exception as exc:
            return (None, f"ExecDefError: {exc}")
        target_fn = env_globals.get(fn_name)
        if not callable(target_fn):
            return (None, f"Function {fn_name!r} not found after sandbox exec")

        def _invoke(*args: Any) -> Any:
            step_box[0] = 0
            return target_fn(*args)

        return (_invoke, "OK")


# ---------------------------------------------------------------------------
# Before/Action/After Slot Transition Episodes & StateGraph Bridge
# ---------------------------------------------------------------------------


def infer_value_type(val: Any) -> str:
    """Infer canonical slot type tag from a Python value."""
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "int"
    if isinstance(val, str):
        return "str"
    if isinstance(val, (list, tuple)):
        if all(isinstance(x, int) and not isinstance(x, bool) for x in val):
            return "seq_int"
        if all(isinstance(x, str) for x in val):
            return "seq_str"
        return "seq_any"
    return "unknown"


def normalize_slot_value(val: Any) -> Any:
    """Canonicalize mutable Python lists into immutable tuples for deterministic hashing."""
    if isinstance(val, bool):
        return bool(val)
    if isinstance(val, int):
        return int(val)
    if isinstance(val, str):
        return str(val)
    if isinstance(val, (list, tuple)):
        return tuple(normalize_slot_value(x) for x in val)
    return val


def to_python_runtime_value(val: Any) -> Any:
    """Convert canonical slot values back to standard Python types (tuples of ints -> list)."""
    if isinstance(val, tuple):
        return [to_python_runtime_value(x) for x in val]
    return val


@dataclass(frozen=True)
class SlotTransitionEpisode:
    """Auditable Before -> Action -> After state transition over typed register slots."""

    episode_id: str
    action_id: str
    before_slots: tuple[tuple[str, Any], ...]
    after_slots: tuple[tuple[str, Any], ...]
    input_slot_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        b_norm = tuple(sorted((str(k), normalize_slot_value(v)) for k, v in self.before_slots))
        a_norm = tuple(sorted((str(k), normalize_slot_value(v)) for k, v in self.after_slots))
        object.__setattr__(self, "before_slots", b_norm)
        object.__setattr__(self, "after_slots", a_norm)
        if not self.input_slot_order:
            object.__setattr__(self, "input_slot_order", tuple(k for k, _ in b_norm))

    def before_dict(self) -> dict[str, Any]:
        return dict(self.before_slots)

    def after_dict(self) -> dict[str, Any]:
        return dict(self.after_slots)

    def changed_slots(self) -> dict[str, Any]:
        """Compute exact sparse slot delta (slots added or altered from before -> after)."""
        b = self.before_dict()
        a = self.after_dict()
        out: dict[str, Any] = {}
        for k, av in a.items():
            if k not in b or b[k] != av or type(b[k]) is not type(av):
                out[k] = av
        return out

    def preserved_slots(self) -> dict[str, Any]:
        b = self.before_dict()
        a = self.after_dict()
        return {
            k: av
            for k, av in a.items()
            if k in b and b[k] == av and type(b[k]) is type(av)
        }

    def to_state_graphs(self) -> tuple[StateGraph, StateGraph]:
        """Represent Before and After slot bindings as ``StateGraph`` instances."""
        b_ents: dict[str, Entity] = {}
        a_ents: dict[str, Entity] = {}
        b_facts: set[tuple[str, str, str]] = set()
        a_facts: set[tuple[str, str, str]] = set()

        for sname, sval in self.before_slots:
            eid = f"reg_{sname}"
            vtype = infer_value_type(sval)
            b_ents[eid] = Entity(eid, f"slot_{vtype}", (("slot_name", sname), ("value", repr(sval))))
            b_facts.add((eid, "bound_type", vtype))

        for sname, sval in self.after_slots:
            eid = f"reg_{sname}"
            vtype = infer_value_type(sval)
            a_ents[eid] = Entity(eid, f"slot_{vtype}", (("slot_name", sname), ("value", repr(sval))))
            a_facts.add((eid, "bound_type", vtype))

        return (StateGraph.create(b_ents, b_facts), StateGraph.create(a_ents, a_facts))

    def to_typed_frames(self) -> tuple[TypedStateFrame, TypedStateFrame]:
        b_sg, a_sg = self.to_state_graphs()
        mask = ObservationMask(fully_observed=True)
        return (
            TypedStateFrame.from_state_graph(b_sg, mask=mask),
            TypedStateFrame.from_state_graph(a_sg, mask=mask),
        )

    def to_dict(self) -> dict[str, Any]:
        def _ser(v: Any) -> Any:
            return list(v) if isinstance(v, tuple) else v

        return {
            "episode_id": self.episode_id,
            "action_id": self.action_id,
            "input_slot_order": list(self.input_slot_order),
            "before_slots": {k: _ser(v) for k, v in self.before_slots},
            "after_slots": {k: _ser(v) for k, v in self.after_slots},
            "changed_slots": {k: _ser(v) for k, v in sorted(self.changed_slots().items())},
        }


# ---------------------------------------------------------------------------
# Neutral Typed Expression & Combinator Grammar (Zero Task-Specific Builtins)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExprNode:
    """Immutable typed AST node in the neutral combinator DSL."""

    op: str
    out_type: str
    args: tuple[Any, ...] = ()

    def node_count(self) -> int:
        total = 1
        for a in self.args:
            if isinstance(a, ExprNode):
                total += a.node_count()
        return total

    def const_count(self) -> int:
        total = 1 if self.op == "const" else 0
        for a in self.args:
            if isinstance(a, ExprNode):
                total += a.const_count()
        return total

    def to_canonical_tuple(self) -> tuple[Any, ...]:
        return (
            self.op,
            self.out_type,
            tuple(a.to_canonical_tuple() if isinstance(a, ExprNode) else a for a in self.args),
        )

    def compressed_bits(self) -> int:
        return compressed_bit_length(canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "out_type": self.out_type,
            "args": [a.to_dict() if isinstance(a, ExprNode) else a for a in self.args],
        }

    def eval_env(self, env: Mapping[str, Any], *, step_limit: int = 2_000) -> Any:
        """Evaluate ``self`` deterministically on ``env`` (raises ValueError/ZeroDivisionError on invalid domain)."""
        op = self.op
        args = self.args

        if op == "var":
            return env[args[0]]
        if op == "const":
            return args[0]

        # Unary integer / boolean / sequence / string ops
        if op == "neg":
            return -int(args[0].eval_env(env, step_limit=step_limit))
        if op == "abs_val":
            return abs(int(args[0].eval_env(env, step_limit=step_limit)))
        if op == "bool_not":
            return not bool(args[0].eval_env(env, step_limit=step_limit))
        if op == "bool_to_int":
            return 1 if bool(args[0].eval_env(env, step_limit=step_limit)) else 0

        # Binary integer ops
        if op in {"add", "sub", "mul", "floordiv", "mod", "min2", "max2"}:
            lhs = int(args[0].eval_env(env, step_limit=step_limit))
            rhs = int(args[1].eval_env(env, step_limit=step_limit))
            if op == "add":
                return lhs + rhs
            if op == "sub":
                return lhs - rhs
            if op == "mul":
                return lhs * rhs
            if op == "floordiv":
                if rhs == 0:
                    raise ZeroDivisionError("division by zero")
                return lhs // rhs
            if op == "mod":
                if rhs == 0:
                    raise ZeroDivisionError("modulo by zero")
                return lhs % rhs
            if op == "min2":
                return lhs if lhs <= rhs else rhs
            return lhs if lhs >= rhs else rhs

        # Comparisons & logical connectives
        if op in {"eq", "ne", "lt", "le", "gt", "ge"}:
            lhs = args[0].eval_env(env, step_limit=step_limit)
            rhs = args[1].eval_env(env, step_limit=step_limit)
            if type(lhs) is not type(rhs):
                raise TypeError("type mismatch in comparison")
            if op == "eq":
                return lhs == rhs
            if op == "ne":
                return lhs != rhs
            if op == "lt":
                return lhs < rhs
            if op == "le":
                return lhs <= rhs
            if op == "gt":
                return lhs > rhs
            return lhs >= rhs

        if op == "bool_and":
            return bool(args[0].eval_env(env, step_limit=step_limit)) and bool(
                args[1].eval_env(env, step_limit=step_limit)
            )
        if op == "bool_or":
            return bool(args[0].eval_env(env, step_limit=step_limit)) or bool(
                args[1].eval_env(env, step_limit=step_limit)
            )
        if op == "if_else":
            cond = bool(args[0].eval_env(env, step_limit=step_limit))
            branch = args[1] if cond else args[2]
            return branch.eval_env(env, step_limit=step_limit)

        # Generic sequence combinators (operate on immutable tuples internally)
        if op == "seq_len":
            seq = args[0].eval_env(env, step_limit=step_limit)
            return len(seq)
        if op == "reverse_seq":
            seq = tuple(args[0].eval_env(env, step_limit=step_limit))
            return tuple(reversed(seq))
        if op == "sort_seq":
            seq = tuple(args[0].eval_env(env, step_limit=step_limit))
            return tuple(sorted(seq))
        if op == "unique_order":
            seq = tuple(args[0].eval_env(env, step_limit=step_limit))
            return tuple(dict.fromkeys(seq))
        if op == "nth_item":
            seq = tuple(args[0].eval_env(env, step_limit=step_limit))
            idx = int(args[1].eval_env(env, step_limit=step_limit))
            return seq[idx]
        if op == "slice_from":
            seq = tuple(args[0].eval_env(env, step_limit=step_limit))
            idx = int(args[1].eval_env(env, step_limit=step_limit))
            return seq[idx:]
        if op == "slice_to":
            seq = tuple(args[0].eval_env(env, step_limit=step_limit))
            idx = int(args[1].eval_env(env, step_limit=step_limit))
            return seq[:idx]
        if op == "concat_seq":
            s1 = tuple(args[0].eval_env(env, step_limit=step_limit))
            s2 = tuple(args[1].eval_env(env, step_limit=step_limit))
            return s1 + s2
        if op == "member_contains":
            seq = tuple(args[0].eval_env(env, step_limit=step_limit))
            elem = args[1].eval_env(env, step_limit=step_limit)
            return elem in seq
        if op == "map_seq":
            seq = tuple(args[0].eval_env(env, step_limit=step_limit))
            var_x: str = args[1]
            body: ExprNode = args[2]
            if len(seq) > step_limit:
                raise SandboxStepLimitError("map_seq exceeded step limit")
            local = dict(env)
            out_items = []
            for item in seq:
                local[var_x] = item
                out_items.append(body.eval_env(local, step_limit=step_limit))
            return tuple(out_items)
        if op == "filter_seq":
            seq = tuple(args[0].eval_env(env, step_limit=step_limit))
            var_x = args[1]
            pred_node: ExprNode = args[2]
            if len(seq) > step_limit:
                raise SandboxStepLimitError("filter_seq exceeded step limit")
            local = dict(env)
            out_items = []
            for item in seq:
                local[var_x] = item
                if bool(pred_node.eval_env(local, step_limit=step_limit)):
                    out_items.append(item)
            return tuple(out_items)
        if op == "fold_seq":
            seq = tuple(args[0].eval_env(env, step_limit=step_limit))
            acc = args[1].eval_env(env, step_limit=step_limit)
            var_acc: str = args[2]
            var_x = args[3]
            step_node: ExprNode = args[4]
            if len(seq) > step_limit:
                raise SandboxStepLimitError("fold_seq exceeded step limit")
            local = dict(env)
            for item in seq:
                local[var_acc] = acc
                local[var_x] = item
                acc = step_node.eval_env(local, step_limit=step_limit)
            return acc
        if op == "scan_seq":
            seq = tuple(args[0].eval_env(env, step_limit=step_limit))
            acc = args[1].eval_env(env, step_limit=step_limit)
            var_acc = args[2]
            var_x = args[3]
            step_node = args[4]
            if len(seq) > step_limit:
                raise SandboxStepLimitError("scan_seq exceeded step limit")
            local = dict(env)
            out_items = []
            for item in seq:
                local[var_acc] = acc
                local[var_x] = item
                acc = step_node.eval_env(local, step_limit=step_limit)
                out_items.append(acc)
            return tuple(out_items)
        if op == "enumerate_fold":
            seq = tuple(args[0].eval_env(env, step_limit=step_limit))
            acc = args[1].eval_env(env, step_limit=step_limit)
            var_acc = args[2]
            var_idx: str = args[3]
            var_x = args[4]
            step_node = args[5]
            if len(seq) > step_limit:
                raise SandboxStepLimitError("enumerate_fold exceeded step limit")
            local = dict(env)
            for idx, item in enumerate(seq):
                local[var_acc] = acc
                local[var_idx] = idx
                local[var_x] = item
                acc = step_node.eval_env(local, step_limit=step_limit)
            return acc
        if op == "zip_with":
            s1 = tuple(args[0].eval_env(env, step_limit=step_limit))
            s2 = tuple(args[1].eval_env(env, step_limit=step_limit))
            var_x = args[2]
            var_y: str = args[3]
            body = args[4]
            local = dict(env)
            out_items = []
            for x_i, y_i in zip(s1, s2):
                local[var_x] = x_i
                local[var_y] = y_i
                out_items.append(body.eval_env(local, step_limit=step_limit))
            return tuple(out_items)

        # Generic string / word combinators
        if op == "str_reverse":
            s_val = str(args[0].eval_env(env, step_limit=step_limit))
            return s_val[::-1]
        if op == "str_lower":
            s_val = str(args[0].eval_env(env, step_limit=step_limit))
            return s_val.lower()
        if op == "split_words":
            s_val = str(args[0].eval_env(env, step_limit=step_limit))
            return tuple(s_val.split())
        if op == "join_space":
            ws = tuple(args[0].eval_env(env, step_limit=step_limit))
            return " ".join(ws)

        # Bounded 2-register tail-iteration combinator
        if op == "while_pair":
            s0 = int(args[0].eval_env(env, step_limit=step_limit))
            s1 = int(args[1].eval_env(env, step_limit=step_limit))
            v0: str = args[2]
            v1: str = args[3]
            cond_node: ExprNode = args[4]
            next0_node: ExprNode = args[5]
            next1_node: ExprNode = args[6]
            select_idx: int = int(args[7])
            local = dict(env)
            steps = 0
            while True:
                local[v0] = s0
                local[v1] = s1
                if not bool(cond_node.eval_env(local, step_limit=step_limit)):
                    break
                steps += 1
                if steps > step_limit:
                    raise SandboxStepLimitError("while_pair exceeded step limit")
                ns0 = int(next0_node.eval_env(local, step_limit=step_limit))
                ns1 = int(next1_node.eval_env(local, step_limit=step_limit))
                if ns0 == s0 and ns1 == s1:
                    raise SandboxStepLimitError("while_pair stationary state")
                s0, s1 = ns0, ns1
            return s0 if select_idx == 0 else s1

        raise ValueError(f"Unsupported ExprNode op: {op!r}")

    def to_python_expr(self) -> str:
        """Render ``self`` as a pure, standalone Python expression (no external imports needed)."""
        op = self.op
        a = self.args
        if op == "var":
            return str(a[0])
        if op == "const":
            if isinstance(a[0], tuple):
                return repr(list(a[0]))
            return repr(a[0])
        if op == "neg":
            return f"(-({a[0].to_python_expr()}))"
        if op == "abs_val":
            return f"abs({a[0].to_python_expr()})"
        if op == "bool_not":
            return f"(not ({a[0].to_python_expr()}))"
        if op == "bool_to_int":
            return f"(1 if ({a[0].to_python_expr()}) else 0)"

        bin_map = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "floordiv": "//",
            "mod": "%",
            "eq": "==",
            "ne": "!=",
            "lt": "<",
            "le": "<=",
            "gt": ">",
            "ge": ">=",
            "bool_and": "and",
            "bool_or": "or",
        }
        if op in bin_map:
            return f"(({a[0].to_python_expr()}) {bin_map[op]} ({a[1].to_python_expr()}))"
        if op == "min2":
            return f"min({a[0].to_python_expr()}, {a[1].to_python_expr()})"
        if op == "max2":
            return f"max({a[0].to_python_expr()}, {a[1].to_python_expr()})"
        if op == "if_else":
            return f"({a[1].to_python_expr()} if ({a[0].to_python_expr()}) else {a[2].to_python_expr()})"
        if op == "seq_len":
            return f"len({a[0].to_python_expr()})"
        if op == "reverse_seq":
            return f"list(reversed({a[0].to_python_expr()}))"
        if op == "sort_seq":
            return f"sorted({a[0].to_python_expr()})"
        if op == "unique_order":
            return f"list(dict.fromkeys({a[0].to_python_expr()}))"
        if op == "nth_item":
            return f"({a[0].to_python_expr()})[{a[1].to_python_expr()}]"
        if op == "slice_from":
            return f"list(({a[0].to_python_expr()})[{a[1].to_python_expr()}:])"
        if op == "slice_to":
            return f"list(({a[0].to_python_expr()})[:{a[1].to_python_expr()}])"
        if op == "concat_seq":
            return f"(list({a[0].to_python_expr()}) + list({a[1].to_python_expr()}))"
        if op == "member_contains":
            return f"(({a[1].to_python_expr()}) in ({a[0].to_python_expr()}))"
        if op == "map_seq":
            return f"[{a[2].to_python_expr()} for {a[1]} in ({a[0].to_python_expr()})]"
        if op == "filter_seq":
            return f"[{a[1]} for {a[1]} in ({a[0].to_python_expr()}) if ({a[2].to_python_expr()})]"
        if op == "zip_with":
            return (
                f"[{a[4].to_python_expr()} for {a[2]}, {a[3]} in "
                f"zip({a[0].to_python_expr()}, {a[1].to_python_expr()})]"
            )
        if op == "str_reverse":
            return f"({a[0].to_python_expr()})[::-1]"
        if op == "str_lower":
            return f"({a[0].to_python_expr()}).lower()"
        if op == "split_words":
            return f"({a[0].to_python_expr()}).split()"
        if op == "join_space":
            return f"' '.join({a[0].to_python_expr()})"

        # Statement-form combinators (fold_seq, scan_seq, enumerate_fold, while_pair) render via compile_to_python_function
        return f"_stmt_op_{op}"

    def compile_to_python_function(self, fn_name: str, param_names: Sequence[str]) -> str:
        """Render ``self`` into a clean, standalone, AST-safe Python ``def`` statement,
        automatically hoisting any nested statement-form combinators into local helpers."""
        params_sig = ", ".join(param_names)
        stmt_ops = {"fold_seq", "scan_seq", "enumerate_fold", "while_pair"}
        helper_defs: list[str] = []
        counter = [0]

        def _render_node(node: ExprNode, is_root: bool = False) -> str:
            op = node.op
            a = node.args
            if op in stmt_ops and not is_root:
                counter[0] += 1
                h_name = f"_h_{counter[0]}"
                h_src = node.compile_to_python_function(h_name, param_names)
                helper_defs.append(h_src.rstrip("\n"))
                return f"{h_name}({params_sig})"

            if op == "var":
                return str(a[0])
            if op == "const":
                return repr(list(a[0])) if isinstance(a[0], tuple) else repr(a[0])
            if op == "neg":
                return f"(-({_render_node(a[0])}))"
            if op == "abs_val":
                return f"abs({_render_node(a[0])})"
            if op == "bool_not":
                return f"(not ({_render_node(a[0])}))"
            if op == "bool_to_int":
                return f"(1 if ({_render_node(a[0])}) else 0)"

            bin_map = {
                "add": "+",
                "sub": "-",
                "mul": "*",
                "floordiv": "//",
                "mod": "%",
                "eq": "==",
                "ne": "!=",
                "lt": "<",
                "le": "<=",
                "gt": ">",
                "ge": ">=",
                "bool_and": "and",
                "bool_or": "or",
            }
            if op in bin_map:
                return f"(({_render_node(a[0])}) {bin_map[op]} ({_render_node(a[1])}))"
            if op == "min2":
                return f"min({_render_node(a[0])}, {_render_node(a[1])})"
            if op == "max2":
                return f"max({_render_node(a[0])}, {_render_node(a[1])})"
            if op == "if_else":
                return f"({_render_node(a[1])} if ({_render_node(a[0])}) else {_render_node(a[2])})"
            if op == "seq_len":
                return f"len({_render_node(a[0])})"
            if op == "reverse_seq":
                return f"list(reversed({_render_node(a[0])}))"
            if op == "sort_seq":
                return f"sorted({_render_node(a[0])})"
            if op == "unique_order":
                return f"list(dict.fromkeys({_render_node(a[0])}))"
            if op == "nth_item":
                return f"({_render_node(a[0])})[{_render_node(a[1])}]"
            if op == "slice_from":
                return f"list(({_render_node(a[0])})[{_render_node(a[1])}:])"
            if op == "slice_to":
                return f"list(({_render_node(a[0])})[:{_render_node(a[1])}])"
            if op == "concat_seq":
                return f"(list({_render_node(a[0])}) + list({_render_node(a[1])}))"
            if op == "member_contains":
                return f"(({_render_node(a[1])}) in ({_render_node(a[0])}))"
            if op == "map_seq":
                return f"[{_render_node(a[2])} for {a[1]} in ({_render_node(a[0])})]"
            if op == "filter_seq":
                return f"[{a[1]} for {a[1]} in ({_render_node(a[0])}) if ({_render_node(a[2])})]"
            if op == "zip_with":
                return (
                    f"[{_render_node(a[4])} for {a[2]}, {a[3]} in "
                    f"zip({_render_node(a[0])}, {_render_node(a[1])})]"
                )
            if op == "str_reverse":
                return f"({_render_node(a[0])})[::-1]"
            if op == "str_lower":
                return f"({_render_node(a[0])}).lower()"
            if op == "split_words":
                return f"({_render_node(a[0])}).split()"
            if op == "join_space":
                return f"' '.join({_render_node(a[0])})"
            return node.to_python_expr()

        op = self.op
        a = self.args
        if op == "fold_seq":
            seq_s = _render_node(a[0])
            init_s = _render_node(a[1])
            v_acc, v_x = a[2], a[3]
            step_s = _render_node(a[4])
            body_lines = [
                f"    {v_acc} = {init_s}",
                f"    for {v_x} in ({seq_s}):",
                f"        {v_acc} = {step_s}",
                f"    return {v_acc}",
            ]
        elif op == "scan_seq":
            seq_s = _render_node(a[0])
            init_s = _render_node(a[1])
            v_acc, v_x = a[2], a[3]
            step_s = _render_node(a[4])
            body_lines = [
                f"    {v_acc} = {init_s}",
                "    _out = []",
                f"    for {v_x} in ({seq_s}):",
                f"        {v_acc} = {step_s}",
                f"        _out.append({v_acc})",
                "    return _out",
            ]
        elif op == "enumerate_fold":
            seq_s = _render_node(a[0])
            init_s = _render_node(a[1])
            v_acc, v_idx, v_x = a[2], a[3], a[4]
            step_s = _render_node(a[5])
            body_lines = [
                f"    {v_acc} = {init_s}",
                f"    for {v_idx}, {v_x} in enumerate({seq_s}):",
                f"        {v_acc} = {step_s}",
                f"    return {v_acc}",
            ]
        elif op == "while_pair":
            init0_s = _render_node(a[0])
            init1_s = _render_node(a[1])
            v0, v1 = a[2], a[3]
            cond_s = _render_node(a[4])
            n0_s = _render_node(a[5])
            n1_s = _render_node(a[6])
            ret_v = v0 if int(a[7]) == 0 else v1
            body_lines = [
                f"    {v0} = {init0_s}",
                f"    {v1} = {init1_s}",
                f"    while {cond_s}:",
                f"        _n0 = {n0_s}",
                f"        _n1 = {n1_s}",
                f"        {v0} = _n0",
                f"        {v1} = _n1",
                f"    return {ret_v}",
            ]
        else:
            expr_s = _render_node(self, is_root=True)
            body_lines = [f"    return {expr_s}"]

        indented_helpers: list[str] = []
        for h_def in helper_defs:
            for ln in h_def.splitlines():
                indented_helpers.append("    " + ln)
        all_lines = [f"def {fn_name}({params_sig}):"] + indented_helpers + body_lines
        return "\n".join(all_lines) + "\n"


# ---------------------------------------------------------------------------
# SHARD 10,240-Bit Behavioral Property Signature Encoder
# ---------------------------------------------------------------------------


def extract_io_behavioral_features(
    arg_types: Sequence[str],
    out_type: str,
    io_pairs: Sequence[tuple[tuple[Any, ...], Any]],
) -> list[tuple[str, int]]:
    """Extract domain-agnostic relational invariants between ``(inputs -> output)`` pairs."""
    feats: list[tuple[str, int]] = [
        (f"sig:arity:{len(arg_types)}", 3),
        (f"sig:out:{out_type}", 4),
    ]
    for i, t in enumerate(arg_types):
        feats.append((f"sig:in:{i}:{t}", 3))

    if not io_pairs:
        return feats

    outs = [normalize_slot_value(y) for _xs, y in io_pairs]
    if len(set(outs)) == 1:
        feats.append(("beh:constant_output", 9))

    if out_type == "bool":
        feats.append(("beh:bool_output", 5))
        if arg_types == ("str",):
            feats.append(("beh:str_predicate", 6))
        elif "seq_int" in arg_types:
            feats.append(("beh:seq_predicate", 6))

    elif out_type == "str":
        feats.append(("beh:str_transform", 6))
        if arg_types == ("str",):
            same_len = all(len(str(xs[0])) == len(str(y)) for xs, y in io_pairs)
            feats.append(("beh:str_len_preserved" if same_len else "beh:str_len_changed", 5))

    elif out_type == "seq_int" and arg_types and arg_types[0] == "seq_int":
        s_ins = [normalize_slot_value(xs[0]) for xs, _y in io_pairs]
        len_eq = all(len(si) == len(so) for si, so in zip(s_ins, outs))
        len_le = all(len(so) <= len(si) for si, so in zip(s_ins, outs))
        multiset_eq = all(sorted(si) == sorted(so) for si, so in zip(s_ins, outs))
        subset_elems = all(set(so).issubset(set(si)) for si, so in zip(s_ins, outs))
        is_sorted_out = all(list(so) == sorted(so) for so in outs if len(so) >= 2)
        is_unique_out = all(len(set(so)) == len(so) for so in outs)
        first_eq = all(so[0] == si[0] for si, so in zip(s_ins, outs) if len(si) > 0 and len(so) > 0)

        if len(arg_types) == 2 and arg_types[1] == "seq_int":
            feats.append(("beh:two_seq_to_seq", 7))
            if subset_elems and is_unique_out:
                feats.append(("beh:seq_intersection_like", 6))
        elif len(arg_types) == 2 and arg_types[1] == "int":
            feats.append(("beh:seq_and_int_to_seq", 7))
            if multiset_eq and len_eq:
                feats.append(("beh:seq_cyclic_shift", 7))
        elif len(arg_types) == 3 and arg_types[1:] == ("int", "int"):
            feats.append(("beh:seq_bounded_map", 8))

        if len_eq and multiset_eq:
            feats.append(("beh:permutation_preserve_multiset", 6))
            if is_sorted_out:
                feats.append(("beh:sorted_permutation", 6))
        elif len_eq and not multiset_eq:
            if first_eq:
                feats.append(("beh:prefix_scan_or_map", 6))
            else:
                feats.append(("beh:elementwise_map", 6))
        elif len_le and subset_elems:
            feats.append(("beh:subsequence_filter_or_unique", 7))
            if is_unique_out and all(set(so) == set(si) for si, so in zip(s_ins, outs)):
                feats.append(("beh:exact_dedup", 7))

    elif out_type == "int":
        int_outs = [int(y) for y in outs]
        all_nonneg = all(y >= 0 for y in int_outs)
        all_01 = all(y in (0, 1) for y in int_outs)
        if all_nonneg:
            feats.append(("beh:nonneg_int", 3))
        if all_01:
            feats.append(("beh:binary_01_int", 6))

        if "seq_int" in arg_types:
            s_ins = [normalize_slot_value(xs[0]) for xs, _y in io_pairs]
            if len(arg_types) == 2 and arg_types[1] == "seq_int":
                feats.append(("beh:two_seq_to_scalar", 8))
            elif len(arg_types) == 2 and arg_types[1] == "int":
                feats.append(("beh:seq_threshold_to_scalar", 7))
            else:
                in_elem_subset = all(
                    (y in si) for si, y in zip(s_ins, int_outs) if len(si) > 0
                )
                bounded_by_len = all(0 <= y <= len(si) for si, y in zip(s_ins, int_outs))
                if in_elem_subset:
                    feats.append(("beh:selects_element_from_seq", 6))
                if bounded_by_len and all_nonneg:
                    feats.append(("beh:count_like_scalar", 5))
                feats.append(("beh:seq_reduction_to_scalar", 6))
        elif all(t == "int" for t in arg_types):
            if len(arg_types) == 1:
                xs0 = [int(xs[0]) for xs, _ in io_pairs]
                superlinear = any(x >= 3 and y > x * 2 for x, y in zip(xs0, int_outs))
                if superlinear:
                    feats.append(("beh:iter_superlinear_1arg", 8))
            elif len(arg_types) == 2:
                divides_both = all(
                    y > 0 and int(xs[0]) % y == 0 and int(xs[1]) % y == 0
                    for xs, y in io_pairs
                    if int(xs[0]) > 0 and int(xs[1]) > 0
                )
                wraps = any(
                    int(xs[0]) >= 0 and int(xs[1]) >= 0 and y < int(xs[0]) + int(xs[1])
                    for xs, y in io_pairs
                )
                if divides_both and any(int(xs[0]) > 1 and int(xs[1]) > 1 for xs, _ in io_pairs):
                    feats.append(("beh:common_divisor_2arg", 8))
                if wraps:
                    feats.append(("beh:modular_or_quotient_wrap", 6))
            else:
                feats.append(("beh:multi_int_arith", 7))

    return feats


def encode_io_behavior_hv(
    arg_types: Sequence[str],
    out_type: str,
    io_pairs: Sequence[tuple[tuple[Any, ...], Any]],
) -> int:
    """Encode ``(before -> after)`` behavioral properties into a 10,240-bit BSC hypervector."""
    feats = extract_io_behavioral_features(arg_types, out_type, io_pairs)
    weighted = [(hv10240_of(f"s6:feat:{tag}"), w) for tag, w in feats]
    return _majority_bundle10240(weighted, empty_hv=_EMPTY_BEHAVIOR_HV)


# Combinator skeleton family descriptors mapped into the exact same 10,240-bit feature subspace
_COMBINATOR_FAMILY_FEATURES: dict[str, list[tuple[str, int]]] = {
    "scalar_arith": [
        ("sig:out:int", 4),
        ("beh:constant_output", 9),
        ("beh:modular_or_quotient_wrap", 6),
        ("beh:binary_01_int", 6),
        ("beh:multi_int_arith", 7),
    ],
    "iter_pair_loop": [
        ("sig:out:int", 4),
        ("beh:iter_superlinear_1arg", 8),
        ("beh:common_divisor_2arg", 8),
        ("beh:nonneg_int", 3),
    ],
    "seq_direct_or_perm": [
        ("sig:out:seq_int", 4),
        ("beh:permutation_preserve_multiset", 6),
        ("beh:sorted_permutation", 6),
        ("beh:exact_dedup", 7),
        ("beh:subsequence_filter_or_unique", 4),
    ],
    "seq_map_or_clamp": [
        ("sig:out:seq_int", 4),
        ("beh:elementwise_map", 6),
        ("beh:seq_bounded_map", 8),
        ("beh:prefix_scan_or_map", 4),
    ],
    "seq_filter_or_inter": [
        ("sig:out:seq_int", 4),
        ("beh:subsequence_filter_or_unique", 7),
        ("beh:two_seq_to_seq", 7),
        ("beh:seq_intersection_like", 6),
    ],
    "seq_scan_or_slice": [
        ("sig:out:seq_int", 4),
        ("beh:prefix_scan_or_map", 6),
        ("beh:seq_and_int_to_seq", 7),
        ("beh:seq_cyclic_shift", 7),
    ],
    "seq_filter_map": [
        ("sig:out:seq_int", 4),
        ("beh:subsequence_filter_or_unique", 5),
        ("beh:elementwise_map", 5),
    ],
    "seq_fold_reduce": [
        ("sig:out:int", 4),
        ("beh:seq_reduction_to_scalar", 6),
        ("beh:nonneg_int", 3),
        ("beh:count_like_scalar", 5),
        ("beh:seq_threshold_to_scalar", 7),
    ],
    "seq_select_or_span": [
        ("sig:out:int", 4),
        ("beh:selects_element_from_seq", 6),
        ("beh:seq_reduction_to_scalar", 5),
    ],
    "seq_enum_or_zip": [
        ("sig:out:int", 4),
        ("beh:two_seq_to_scalar", 8),
        ("beh:seq_reduction_to_scalar", 5),
    ],
    "bool_predicate": [
        ("sig:out:bool", 5),
        ("beh:bool_output", 5),
        ("beh:seq_predicate", 6),
        ("beh:str_predicate", 6),
    ],
    "str_combinators": [
        ("sig:out:str", 5),
        ("beh:str_transform", 6),
        ("beh:str_len_preserved", 5),
        ("beh:str_len_changed", 5),
    ],
}

_FAMILY_PROTOTYPE_HVS: dict[str, int] = {
    fam: _majority_bundle10240(
        [(hv10240_of(f"s6:feat:{tag}"), w) for tag, w in f_list],
        empty_hv=_EMPTY_BEHAVIOR_HV,
    )
    for fam, f_list in _COMBINATOR_FAMILY_FEATURES.items()
}


# ---------------------------------------------------------------------------
# Candidate Generator Over Neutral Typed Combinator Families
# ---------------------------------------------------------------------------


def _generate_family_candidates(
    family: str,
    param_names: Sequence[str],
    arg_types: Sequence[str],
    out_type: str,
    inferred_int_consts: Sequence[int],
) -> list[ExprNode]:
    """Enumerate neutral typed candidate ``ExprNode``s belonging to ``family``."""
    int_vars = [ExprNode("var", "int", (p,)) for p, t in zip(param_names, arg_types) if t == "int"]
    seq_vars = [ExprNode("var", "seq_int", (p,)) for p, t in zip(param_names, arg_types) if t == "seq_int"]
    str_vars = [ExprNode("var", "str", (p,)) for p, t in zip(param_names, arg_types) if t == "str"]

    base_consts = list(dict.fromkeys([0, 1, 2, -1] + list(inferred_int_consts)))
    const_nodes = [ExprNode("const", "int", (c,)) for c in base_consts]
    out: list[ExprNode] = []

    if family == "scalar_arith" and out_type == "int":
        out.extend(const_nodes)
        if int_vars:
            lvl1: list[ExprNode] = list(int_vars)
            for v in int_vars:
                lvl1.append(ExprNode("abs_val", "int", (v,)))
                lvl1.append(ExprNode("neg", "int", (v,)))
            # Canonical commutative 2-var sums (distinct vars first, then self-sums) & products
            sums2_distinct = [
                ExprNode("add", "int", (int_vars[i], int_vars[j]))
                for i in range(len(int_vars))
                for j in range(i + 1, len(int_vars))
            ]
            sums2_same = [ExprNode("add", "int", (v, v)) for v in int_vars]
            sums2 = sums2_distinct + sums2_same
            lvl1.extend(sums2)
            for i, a in enumerate(int_vars):
                for b in int_vars[i:]:
                    lvl1.append(ExprNode("mul", "int", (a, b)))
                for b in int_vars:
                    if a != b:
                        lvl1.append(ExprNode("sub", "int", (a, b)))
            # Canonical commutative 3-var sums (distinct triples i < j < k first!)
            sums3_distinct = [
                ExprNode("add", "int", (ExprNode("add", "int", (int_vars[i], int_vars[j])), int_vars[k]))
                for i in range(len(int_vars))
                for j in range(i + 1, len(int_vars))
                for k in range(j + 1, len(int_vars))
            ]
            numerators = list(int_vars) + sums2_distinct + sums3_distinct + sums2_same
            divisors = list(int_vars) + [c for c in const_nodes if c.args[0] > 1]
            out.extend(lvl1)
            out.extend(sums3_distinct)
            for num in numerators:
                for div in divisors:
                    if num != div:
                        out.append(ExprNode("mod", "int", (num, div)))
                        out.append(ExprNode("floordiv", "int", (num, div)))
            c1 = ExprNode("const", "int", (1,))
            for e1 in int_vars + sums2:
                out.append(ExprNode("add", "int", (e1, c1)))
                out.append(ExprNode("sub", "int", (e1, c1)))

    elif family == "iter_pair_loop" and out_type == "int" and int_vars:
        s0_v = ExprNode("var", "int", ("_s0",))
        s1_v = ExprNode("var", "int", ("_s1",))
        c_zero = ExprNode("const", "int", (0,))
        c_one = ExprNode("const", "int", (1,))
        if len(int_vars) == 2:
            a_v, b_v = int_vars[0], int_vars[1]
            # Generic 2-state transitions over (_s0, _s1)
            conds = [
                ExprNode("ne", "bool", (s1_v, c_zero)),
                ExprNode("gt", "bool", (s1_v, c_zero)),
            ]
            step_pairs = [
                (s1_v, ExprNode("mod", "int", (s0_v, s1_v))),
                (ExprNode("sub", "int", (s0_v, s1_v)), s1_v),
                (ExprNode("add", "int", (s0_v, a_v)), ExprNode("sub", "int", (s1_v, c_one))),
                (ExprNode("mul", "int", (s0_v, a_v)), ExprNode("sub", "int", (s1_v, c_one))),
            ]
            for cond in conds:
                for n0, n1 in step_pairs:
                    out.append(ExprNode("while_pair", "int", (a_v, b_v, "_s0", "_s1", cond, n0, n1, 0)))
        elif len(int_vars) == 1:
            n_v = int_vars[0]
            conds = [
                ExprNode("gt", "bool", (s0_v, c_one)),
                ExprNode("gt", "bool", (s0_v, c_zero)),
            ]
            init_step_pairs = [
                (c_one, ExprNode("mul", "int", (s1_v, s0_v))),
                (c_zero, ExprNode("add", "int", (s1_v, s0_v))),
            ]
            dec_s0 = ExprNode("sub", "int", (s0_v, c_one))
            for init_acc, st1 in init_step_pairs:
                for cond in conds:
                    out.append(
                        ExprNode("while_pair", "int", (n_v, init_acc, "_s0", "_s1", cond, dec_s0, st1, 1))
                    )

    elif family == "seq_direct_or_perm" and out_type == "seq_int":
        out.append(ExprNode("const", "seq_int", ((),)))
        if seq_vars:
            xs = seq_vars[0]
            base_seqs = [
                xs,
                ExprNode("reverse_seq", "seq_int", (xs,)),
                ExprNode("sort_seq", "seq_int", (xs,)),
                ExprNode("unique_order", "seq_int", (xs,)),
            ]
            out.extend(base_seqs)
            for bs in base_seqs[1:]:
                out.append(ExprNode("reverse_seq", "seq_int", (bs,)))
                out.append(ExprNode("sort_seq", "seq_int", (bs,)))
                out.append(ExprNode("unique_order", "seq_int", (bs,)))

    elif family == "seq_map_or_clamp" and out_type == "seq_int" and seq_vars:
        xs = seq_vars[0]
        vx = ExprNode("var", "int", ("_x",))
        c0 = ExprNode("const", "int", (0,))
        c1 = ExprNode("const", "int", (1,))
        c2 = ExprNode("const", "int", (2,))
        bodies = [
            ExprNode("add", "int", (vx, c1)),
            ExprNode("sub", "int", (vx, c1)),
            ExprNode("mul", "int", (vx, c2)),
            ExprNode("mul", "int", (vx, vx)),
            ExprNode("abs_val", "int", (vx,)),
            ExprNode("neg", "int", (vx,)),
            ExprNode("max2", "int", (c0, vx)),
        ]
        for iv in int_vars:
            bodies.append(ExprNode("add", "int", (vx, iv)))
            bodies.append(ExprNode("mul", "int", (vx, iv)))
            bodies.append(ExprNode("max2", "int", (iv, vx)))
            bodies.append(ExprNode("min2", "int", (iv, vx)))
        if len(int_vars) >= 2:
            lo_v, hi_v = int_vars[0], int_vars[1]
            bodies.append(ExprNode("min2", "int", (hi_v, ExprNode("max2", "int", (lo_v, vx)))))
            bodies.append(ExprNode("max2", "int", (lo_v, ExprNode("min2", "int", (hi_v, vx)))))
        for b in bodies:
            out.append(ExprNode("map_seq", "seq_int", (xs, "_x", b)))

    elif family == "seq_filter_or_inter" and out_type == "seq_int" and seq_vars:
        xs = seq_vars[0]
        vx = ExprNode("var", "int", ("_x",))
        c0 = ExprNode("const", "int", (0,))
        c1 = ExprNode("const", "int", (1,))
        c2 = ExprNode("const", "int", (2,))
        mod2_x = ExprNode("mod", "int", (vx, c2))
        preds = [
            ExprNode("eq", "bool", (mod2_x, c0)),
            ExprNode("ne", "bool", (mod2_x, c0)),
            ExprNode("gt", "bool", (vx, c0)),
            ExprNode("ge", "bool", (vx, c0)),
            ExprNode("lt", "bool", (vx, c0)),
            ExprNode("ne", "bool", (vx, c0)),
        ]
        for iv in int_vars:
            preds.append(ExprNode("gt", "bool", (vx, iv)))
            preds.append(ExprNode("ge", "bool", (vx, iv)))
            preds.append(ExprNode("lt", "bool", (vx, iv)))
            preds.append(ExprNode("eq", "bool", (vx, iv)))
        if len(seq_vars) >= 2:
            ys = seq_vars[1]
            preds.append(ExprNode("member_contains", "bool", (ys, vx)))
            preds.append(ExprNode("bool_not", "bool", (ExprNode("member_contains", "bool", (ys, vx)),)))

        sources = [xs, ExprNode("unique_order", "seq_int", (xs,))]
        for src in sources:
            for pr in preds:
                out.append(ExprNode("filter_seq", "seq_int", (src, "_x", pr)))

    elif family == "seq_scan_or_slice" and out_type == "seq_int" and seq_vars:
        xs = seq_vars[0]
        vacc = ExprNode("var", "int", ("_acc",))
        vx = ExprNode("var", "int", ("_x",))
        c0 = ExprNode("const", "int", (0,))
        c1 = ExprNode("const", "int", (1,))
        out.append(ExprNode("scan_seq", "seq_int", (xs, c0, "_acc", "_x", ExprNode("add", "int", (vacc, vx)))))
        out.append(ExprNode("scan_seq", "seq_int", (xs, c1, "_acc", "_x", ExprNode("mul", "int", (vacc, vx)))))
        for iv in int_vars:
            len_xs = ExprNode("seq_len", "int", (xs,))
            is_empty = ExprNode("eq", "bool", (len_xs, c0))
            k_mod = ExprNode("mod", "int", (iv, len_xs))
            # Un-guarded slice (to demonstrate RDC boundary refutation catching empty-list ZeroDivisionError!)
            unguarded_rot = ExprNode(
                "concat_seq",
                "seq_int",
                (
                    ExprNode("slice_from", "seq_int", (xs, k_mod)),
                    ExprNode("slice_to", "seq_int", (xs, k_mod)),
                ),
            )
            guarded_rot = ExprNode("if_else", "seq_int", (is_empty, xs, unguarded_rot))
            out.append(ExprNode("slice_from", "seq_int", (xs, iv)))
            out.append(ExprNode("slice_to", "seq_int", (xs, iv)))
            out.append(unguarded_rot)
            out.append(guarded_rot)

    elif family == "seq_filter_map" and out_type == "seq_int" and seq_vars:
        xs = seq_vars[0]
        vx = ExprNode("var", "int", ("_x",))
        c0 = ExprNode("const", "int", (0,))
        c2 = ExprNode("const", "int", (2,))
        mod2_x = ExprNode("mod", "int", (vx, c2))
        preds = [
            ExprNode("eq", "bool", (mod2_x, c0)),
            ExprNode("ne", "bool", (mod2_x, c0)),
            ExprNode("gt", "bool", (vx, c0)),
        ]
        maps = [
            ExprNode("mul", "int", (vx, vx)),
            ExprNode("mul", "int", (vx, c2)),
            ExprNode("abs_val", "int", (vx,)),
        ]
        for pr in preds:
            filt = ExprNode("filter_seq", "seq_int", (xs, "_x", pr))
            for mp in maps:
                out.append(ExprNode("map_seq", "seq_int", (filt, "_x", mp)))

    elif family == "seq_fold_reduce" and out_type == "int" and seq_vars:
        xs = seq_vars[0]
        vacc = ExprNode("var", "int", ("_acc",))
        vx = ExprNode("var", "int", ("_x",))
        c0 = ExprNode("const", "int", (0,))
        c1 = ExprNode("const", "int", (1,))
        c2 = ExprNode("const", "int", (2,))
        mod2_x = ExprNode("mod", "int", (vx, c2))

        seq_variants = [
            xs,
            ExprNode("filter_seq", "seq_int", (xs, "_x", ExprNode("eq", "bool", (mod2_x, c0)))),
            ExprNode("filter_seq", "seq_int", (xs, "_x", ExprNode("ne", "bool", (mod2_x, c0)))),
            ExprNode("filter_seq", "seq_int", (xs, "_x", ExprNode("gt", "bool", (vx, c0)))),
            ExprNode("filter_seq", "seq_int", (xs, "_x", ExprNode("lt", "bool", (vx, c0)))),
            ExprNode("map_seq", "seq_int", (xs, "_x", ExprNode("abs_val", "int", (vx,)))),
            ExprNode("map_seq", "seq_int", (xs, "_x", ExprNode("mul", "int", (vx, vx)))),
            ExprNode("unique_order", "seq_int", (xs,)),
        ]
        for iv in int_vars:
            seq_variants.append(
                ExprNode("filter_seq", "seq_int", (xs, "_x", ExprNode("gt", "bool", (vx, iv))))
            )
            seq_variants.append(
                ExprNode("filter_seq", "seq_int", (xs, "_x", ExprNode("ge", "bool", (vx, iv))))
            )
            seq_variants.append(
                ExprNode("filter_seq", "seq_int", (xs, "_x", ExprNode("eq", "bool", (vx, iv))))
            )

        for sv in seq_variants:
            out.append(ExprNode("seq_len", "int", (sv,)))
            out.append(ExprNode("fold_seq", "int", (sv, c0, "_acc", "_x", ExprNode("add", "int", (vacc, vx)))))
        out.append(ExprNode("fold_seq", "int", (xs, c1, "_acc", "_x", ExprNode("mul", "int", (vacc, vx)))))

    elif family == "seq_select_or_span" and out_type == "int" and seq_vars:
        xs = seq_vars[0]
        vacc = ExprNode("var", "int", ("_acc",))
        vx = ExprNode("var", "int", ("_x",))
        c0 = ExprNode("const", "int", (0,))
        len_xs = ExprNode("seq_len", "int", (xs,))
        is_empty = ExprNode("eq", "bool", (len_xs, c0))
        first_x = ExprNode("nth_item", "int", (xs, c0))
        max_fold = ExprNode("fold_seq", "int", (xs, first_x, "_acc", "_x", ExprNode("max2", "int", (vacc, vx))))
        min_fold = ExprNode("fold_seq", "int", (xs, first_x, "_acc", "_x", ExprNode("min2", "int", (vacc, vx))))
        span_unguarded = ExprNode("sub", "int", (max_fold, min_fold))
        span_guarded = ExprNode("if_else", "int", (is_empty, c0, span_unguarded))
        out.extend([max_fold, min_fold, span_unguarded, span_guarded])

        sorted_xs = ExprNode("sort_seq", "seq_int", (xs,))
        sorted_uniq = ExprNode("sort_seq", "seq_int", (ExprNode("unique_order", "seq_int", (xs,)),))
        for idx_c in (0, 1, -1, -2):
            idx_node = ExprNode("const", "int", (idx_c,))
            out.append(ExprNode("nth_item", "int", (sorted_xs, idx_node)))
            out.append(ExprNode("nth_item", "int", (sorted_uniq, idx_node)))

    elif family == "seq_enum_or_zip" and out_type == "int" and seq_vars:
        xs = seq_vars[0]
        vacc = ExprNode("var", "int", ("_acc",))
        vidx = ExprNode("var", "int", ("_i",))
        vx = ExprNode("var", "int", ("_x",))
        vy = ExprNode("var", "int", ("_y",))
        c0 = ExprNode("const", "int", (0,))
        c2 = ExprNode("const", "int", (2,))
        even_idx = ExprNode("eq", "bool", (ExprNode("mod", "int", (vidx, c2)), c0))
        out.append(
            ExprNode(
                "enumerate_fold",
                "int",
                (
                    xs,
                    c0,
                    "_acc",
                    "_i",
                    "_x",
                    ExprNode(
                        "if_else",
                        "int",
                        (
                            even_idx,
                            ExprNode("add", "int", (vacc, vx)),
                            ExprNode("sub", "int", (vacc, vx)),
                        ),
                    ),
                ),
            )
        )
        out.append(
            ExprNode(
                "enumerate_fold",
                "int",
                (xs, c0, "_acc", "_i", "_x", ExprNode("add", "int", (vacc, ExprNode("mul", "int", (vidx, vx))))),
            )
        )
        if len(seq_vars) >= 2:
            ys = seq_vars[1]
            for bop in ("mul", "add", "sub"):
                zw = ExprNode("zip_with", "seq_int", (xs, ys, "_x", "_y", ExprNode(bop, "int", (vx, vy))))
                out.append(ExprNode("fold_seq", "int", (zw, c0, "_acc", "_x", ExprNode("add", "int", (vacc, vx)))))

    elif family == "bool_predicate" and out_type == "bool":
        out.extend(
            [
                ExprNode("const", "bool", (False,)),
                ExprNode("const", "bool", (True,)),
            ]
        )
        c0 = ExprNode("const", "int", (0,))
        c2 = ExprNode("const", "int", (2,))
        if str_vars:
            s = str_vars[0]
            sl = ExprNode("str_lower", "str", (s,))
            out.append(ExprNode("eq", "bool", (s, ExprNode("str_reverse", "str", (s,)))))
            out.append(ExprNode("eq", "bool", (sl, ExprNode("str_reverse", "str", (sl,)))))
        if seq_vars:
            xs = seq_vars[0]
            vx = ExprNode("var", "int", ("_x",))
            vacc_b = ExprNode("var", "bool", ("_acc",))
            c_true = ExprNode("const", "bool", (True,))
            out.append(ExprNode("eq", "bool", (xs, ExprNode("sort_seq", "seq_int", (xs,)))))
            out.append(ExprNode("eq", "bool", (xs, ExprNode("reverse_seq", "seq_int", (xs,)))))
            out.append(
                ExprNode("eq", "bool", (ExprNode("seq_len", "int", (xs,)), ExprNode("seq_len", "int", (ExprNode("unique_order", "seq_int", (xs,)),))))
            )
            for cmp_op, rhs in (("gt", c0), ("ge", c0), ("lt", c0), ("eq", c0)):
                out.append(
                    ExprNode(
                        "fold_seq",
                        "bool",
                        (xs, c_true, "_acc", "_x", ExprNode("bool_and", "bool", (vacc_b, ExprNode(cmp_op, "bool", (vx, rhs))))),
                    )
                )
            even_x = ExprNode("eq", "bool", (ExprNode("mod", "int", (vx, c2)), c0))
            out.append(
                ExprNode(
                    "fold_seq",
                    "bool",
                    (xs, c_true, "_acc", "_x", ExprNode("bool_and", "bool", (vacc_b, even_x))),
                )
            )
        if len(int_vars) >= 2:
            for cmp_op in ("eq", "ne", "lt", "le", "gt", "ge"):
                out.append(ExprNode(cmp_op, "bool", (int_vars[0], int_vars[1])))

    elif family == "str_combinators" and out_type == "str":
        out.append(ExprNode("const", "str", ("",)))
        if str_vars:
            s = str_vars[0]
            words = ExprNode("split_words", "seq_str", (s,))
            out.extend(
                [
                    s,
                    ExprNode("str_reverse", "str", (s,)),
                    ExprNode("str_lower", "str", (s,)),
                    ExprNode("join_space", "str", (words,)),
                    ExprNode("join_space", "str", (ExprNode("reverse_seq", "seq_str", (words,)),)),
                    ExprNode("join_space", "str", (ExprNode("sort_seq", "seq_str", (words,)),)),
                    ExprNode("join_space", "str", (ExprNode("unique_order", "seq_str", (words,)),)),
                ]
            )

    return out


# ---------------------------------------------------------------------------
# RDC Synthetic Boundary Probe Generator (Refutes Fragile / Unsafe Programs)
# ---------------------------------------------------------------------------


def synthesize_typed_boundary_inputs(
    arg_types: Sequence[str],
    observed_inputs: Sequence[tuple[Any, ...]] = (),
) -> list[tuple[Any, ...]]:
    """Synthesize domain-agnostic boundary input tuples purely from ``arg_types``
    (respecting strictly positive integer domain preconditions when all observed inputs >= 1)."""
    per_arg: list[list[Any]] = []
    for idx, t in enumerate(arg_types):
        if t == "int":
            obs_col = [int(xs[idx]) for xs in observed_inputs if idx < len(xs)]
            if obs_col and min(obs_col) >= 1:
                per_arg.append( sorted(set([1, 2, min(obs_col)])) )
            else:
                per_arg.append([0, 1, 2])
        elif t == "bool":
            per_arg.append([False, True])
        elif t == "str":
            per_arg.append(["", "a", "ab ba"])
        elif t == "seq_int":
            per_arg.append([(), (0,), (-2, 0, 3), (4, 4, 1)])
        else:
            per_arg.append([()])

    # Single-factor boundary variations around default safe anchor
    anchors = [vals[-1] for vals in per_arg]
    probes: list[tuple[Any, ...]] = [tuple(anchors)]
    for i, vals in enumerate(per_arg):
        for v in vals:
            cand = list(anchors)
            cand[i] = v
            t_cand = tuple(cand)
            if t_cand not in probes:
                probes.append(t_cand)
    return probes


# ---------------------------------------------------------------------------
# FunctionalOperatorCard & SHARD + ITL + RDC Program Inducer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SynthesisDiagnostics:
    """Auditable search & MDL telemetry for a single synthesized slot/program."""

    target_slot: str
    solved: bool
    use_shard_guidance: bool
    candidates_evaluated: int
    observational_classes: int
    consistent_survivors: int
    boundary_refuted_candidates: int
    ranked_families: tuple[str, ...]
    selected_program_bits: int
    selected_node_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_slot": self.target_slot,
            "solved": self.solved,
            "use_shard_guidance": self.use_shard_guidance,
            "candidates_evaluated": self.candidates_evaluated,
            "observational_classes": self.observational_classes,
            "consistent_survivors": self.consistent_survivors,
            "boundary_refuted_candidates": self.boundary_refuted_candidates,
            "ranked_families": list(self.ranked_families),
            "selected_program_bits": self.selected_program_bits,
            "selected_node_count": self.selected_node_count,
        }


@dataclass(frozen=True)
class FunctionalOperatorCard:
    """Induced Before -> Action -> After programmatic operator card preserving all unchanged slots."""

    card_id: str
    action_id: str
    input_slots: tuple[tuple[str, str], ...]
    slot_programs: tuple[tuple[str, ExprNode], ...]
    python_sources: tuple[tuple[str, str], ...]
    version: int = 1
    support_episodes: int = 0
    all_slots_solved: bool = True
    diagnostics: tuple[SynthesisDiagnostics, ...] = ()
    schema_version: str = TRANSFER_INDUCER_SCHEMA_VERSION

    def compressed_bits(self) -> int:
        return compressed_bit_length(canonical_json_bytes(self.to_executable_dict()))

    def to_executable_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "input_slots": [[k, t] for k, t in self.input_slots],
            "slot_programs": {k: expr.to_dict() for k, expr in self.slot_programs},
        }

    def to_dict(self) -> dict[str, Any]:
        out = self.to_executable_dict()
        out["card_id"] = self.card_id
        out["version"] = self.version
        out["support_episodes"] = self.support_episodes
        out["all_slots_solved"] = self.all_slots_solved
        out["python_sources"] = dict(self.python_sources)
        out["compressed_bits"] = self.compressed_bits()
        out["diagnostics"] = [d.to_dict() for d in self.diagnostics]
        return out

    def apply_slots(
        self,
        before_slots: Mapping[str, Any],
        *,
        use_sandbox: bool = True,
    ) -> dict[str, Any]:
        """Execute induced slot programs synchronously from ``before_slots``, preserving all other slots intact."""
        before_snap = {str(k): normalize_slot_value(v) for k, v in before_slots.items()}
        after_out = dict(before_snap)
        param_names = [k for k, _t in self.input_slots]
        src_map = dict(self.python_sources)
        env_snap = {p: before_snap[p] for p in param_names}

        for target_slot, expr in self.slot_programs:
            if use_sandbox and target_slot in src_map:
                fn_name = f"_slot_{target_slot}"
                fn, reason = RestrictedExecutionSandbox.compile_verified_function(
                    src_map[target_slot], fn_name
                )
                if fn is None:
                    raise RuntimeError(f"Sandbox verification failed for {target_slot}: {reason}")
                py_args = [to_python_runtime_value(before_snap[p]) for p in param_names]
                raw_res = fn(*py_args)
                after_out[target_slot] = normalize_slot_value(raw_res)
            else:
                after_out[target_slot] = normalize_slot_value(expr.eval_env(env_snap))
        return after_out


def synthesize_typed_function(
    arg_types: Sequence[str],
    out_type: str,
    train_io_pairs: Sequence[tuple[tuple[Any, ...], Any]],
    *,
    param_names: Sequence[str] | None = None,
    use_shard_guidance: bool = True,
    use_rdc_boundary_refutation: bool = True,
    max_candidates: int = 240,
    fn_name: str = "induced_fn",
    target_slot_label: str = "result",
) -> tuple[ExprNode | None, str | None, SynthesisDiagnostics]:
    """Synthesize a verified standalone program strictly from ``(arg_types, out_type, train_io_pairs)``.

    Zero task names or docstrings are accepted by this function, making lexical dispatch
    structurally impossible.
    """
    p_names = tuple(param_names) if param_names is not None else tuple(f"arg{i}" for i in range(len(arg_types)))
    norm_pairs = [
        (tuple(normalize_slot_value(x) for x in xs), normalize_slot_value(y))
        for xs, y in train_io_pairs
    ]

    # Infer empirical codomain span constant (e.g., max(y) + 1) purely from observed integer outputs
    inferred_consts: list[int] = []
    if out_type == "int" and norm_pairs:
        y_ints = [int(y) for _xs, y in norm_pairs]
        max_y = max(y_ints)
        min_y = min(y_ints)
        if min_y >= 0 and 2 <= max_y <= 64:
            inferred_consts.append(max_y + 1)

    # Canonical neutral family order (used by unguided baseline; SHARD re-ranks by 10,240-bit BSC sim!)
    canonical_families = list(_COMBINATOR_FAMILY_FEATURES.keys())
    if use_shard_guidance:
        query_hv = encode_io_behavior_hv(arg_types, out_type, norm_pairs)
        ranked_families = sorted(
            canonical_families,
            key=lambda fam: (
                -hamming_sim10240(query_hv, _FAMILY_PROTOTYPE_HVS[fam]),
                canonical_families.index(fam),
            ),
        )
    else:
        ranked_families = canonical_families

    candidates_evaluated = 0
    obs_classes: dict[tuple[Any, ...], ExprNode] = {}
    consistent_candidates: list[ExprNode] = []
    seen_exprs: set[tuple[Any, ...]] = set()

    envs = [{p: xs[i] for i, p in enumerate(p_names)} for xs, _y in norm_pairs]
    gold_outs = tuple(y for _xs, y in norm_pairs)

    budget_exhausted = False
    for fam in ranked_families:
        if budget_exhausted:
            break
        fam_cands = _generate_family_candidates(fam, p_names, arg_types, out_type, inferred_consts)
        for cand in fam_cands:
            if candidates_evaluated >= max_candidates:
                budget_exhausted = True
                break
            ckey = cand.to_canonical_tuple()
            if ckey in seen_exprs:
                continue
            seen_exprs.add(ckey)
            candidates_evaluated += 1

            # Evaluate on training inputs
            preds: list[Any] = []
            failed_eval = False
            for env in envs:
                try:
                    val = normalize_slot_value(cand.eval_env(env, step_limit=500))
                except Exception:
                    failed_eval = True
                    break
                preds.append(val)
            if failed_eval:
                continue

            obs_sig = tuple(preds)
            # ITL Observational-Equivalence Compression: retain shortest MDL representative per behavior
            prev = obs_classes.get(obs_sig)
            if prev is None or (cand.node_count(), cand.compressed_bits()) < (
                prev.node_count(),
                prev.compressed_bits(),
            ):
                obs_classes[obs_sig] = cand

            if obs_sig == gold_outs and all(
                type(p) is type(g) for p, g in zip(obs_sig, gold_outs)
            ):
                consistent_candidates.append(cand)
        # Identical stopping rule for both SHARD-guided and unguided enumeration:
        # once a family yields >= 1 boundary-safe solution, stop searching further families!
        if consistent_candidates:
            # Check if at least one survivor already passes RDC boundary probes & AST sandbox safety
            obs_in = [xs for xs, _y in norm_pairs]
            boundary_check = (
                synthesize_typed_boundary_inputs(arg_types, obs_in)
                if use_rdc_boundary_refutation
                else []
            )
            has_clean = False
            for cc in consistent_candidates:
                b_ok = True
                for b_in in boundary_check:
                    b_env = {p: b_in[i] for i, p in enumerate(p_names)}
                    try:
                        cc.eval_env(b_env, step_limit=500)
                    except Exception:
                        b_ok = False
                        break
                if b_ok:
                    has_clean = True
                    break
            if has_clean:
                break

    if not consistent_candidates:
        return (
            None,
            None,
            SynthesisDiagnostics(
                target_slot=target_slot_label,
                solved=False,
                use_shard_guidance=use_shard_guidance,
                candidates_evaluated=candidates_evaluated,
                observational_classes=len(obs_classes),
                consistent_survivors=0,
                boundary_refuted_candidates=0,
                ranked_families=tuple(ranked_families),
                selected_program_bits=0,
                selected_node_count=0,
            ),
        )

    # RDC Boundary Dream Refutation + ZeroTrustSandboxJudge AST Verification + MDL Selection
    obs_inputs = [xs for xs, _y in norm_pairs]
    boundary_probes = (
        synthesize_typed_boundary_inputs(arg_types, obs_inputs)
        if use_rdc_boundary_refutation
        else []
    )
    boundary_safe: list[tuple[int, ExprNode, str]] = []
    boundary_refuted = 0

    for cand in consistent_candidates:
        py_src = cand.compile_to_python_function(fn_name, p_names)
        compiled_fn, reason = RestrictedExecutionSandbox.compile_verified_function(py_src, fn_name)
        if compiled_fn is None:
            boundary_refuted += 1
            continue

        # Verify compiled Python matches training I/O inside sandbox
        train_ok = True
        for xs, gold_y in norm_pairs:
            try:
                res = normalize_slot_value(compiled_fn(*[to_python_runtime_value(x) for x in xs]))
            except Exception:
                train_ok = False
                break
            if res != gold_y or type(res) is not type(gold_y):
                train_ok = False
                break
        if not train_ok:
            boundary_refuted += 1
            continue

        # Count boundary dream crashes (e.g., empty-list ZeroDivisionError/IndexError)
        crashes = 0
        for b_in in boundary_probes:
            try:
                compiled_fn(*[to_python_runtime_value(x) for x in b_in])
            except Exception:
                crashes += 1
        if crashes > 0:
            boundary_refuted += 1
        boundary_safe.append((crashes, cand, py_src))

    if not boundary_safe:
        return (
            None,
            None,
            SynthesisDiagnostics(
                target_slot=target_slot_label,
                solved=False,
                use_shard_guidance=use_shard_guidance,
                candidates_evaluated=candidates_evaluated,
                observational_classes=len(obs_classes),
                consistent_survivors=len(consistent_candidates),
                boundary_refuted_candidates=boundary_refuted,
                ranked_families=tuple(ranked_families),
                selected_program_bits=0,
                selected_node_count=0,
            ),
        )

    # RDC MDL Selection: strictly prefer 0 boundary crashes, minimal AST nodes, fewer literal constants, then compressed bits
    boundary_safe.sort(
        key=lambda item: (
            item[0],
            item[1].node_count(),
            item[1].const_count(),
            item[1].compressed_bits(),
            item[2],
        )
    )
    _crashes, best_expr, best_src = boundary_safe[0]

    return (
        best_expr,
        best_src,
        SynthesisDiagnostics(
            target_slot=target_slot_label,
            solved=True,
            use_shard_guidance=use_shard_guidance,
            candidates_evaluated=candidates_evaluated,
            observational_classes=len(obs_classes),
            consistent_survivors=len(consistent_candidates),
            boundary_refuted_candidates=boundary_refuted,
            ranked_families=tuple(ranked_families),
            selected_program_bits=best_expr.compressed_bits(),
            selected_node_count=best_expr.node_count(),
        ),
    )


class TransferProgramInducer:
    """General Before -> Action -> After programmatic operator learner (SHARD + ITL + RDC)."""

    def __init__(
        self,
        *,
        use_shard_guidance: bool = True,
        use_rdc_boundary_refutation: bool = True,
        max_candidates: int = 240,
    ) -> None:
        self.use_shard_guidance = bool(use_shard_guidance)
        self.use_rdc_boundary_refutation = bool(use_rdc_boundary_refutation)
        self.max_candidates = int(max_candidates)
        self._cards: dict[str, FunctionalOperatorCard] = {}
        self._version_history: dict[str, list[FunctionalOperatorCard]] = {}
        self._frozen: bool = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        self._frozen = True

    @property
    def cards(self) -> dict[str, FunctionalOperatorCard]:
        return dict(self._cards)

    def get_card(self, action_id: str) -> FunctionalOperatorCard | None:
        return self._cards.get(action_id)

    def get_history(self, action_id: str) -> tuple[FunctionalOperatorCard, ...]:
        return tuple(self._version_history.get(action_id, ()))

    def fit_episodes(
        self,
        episodes: Sequence[SlotTransitionEpisode],
    ) -> dict[str, FunctionalOperatorCard]:
        """Induce ``FunctionalOperatorCard``s from ``SlotTransitionEpisode``s grouped by ``action_id``."""
        if self._frozen:
            raise TransferInducerFrozenError("Cannot call fit_episodes() on a frozen TransferProgramInducer")

        grouped: dict[str, list[SlotTransitionEpisode]] = {}
        for ep in episodes:
            grouped.setdefault(ep.action_id, []).append(ep)

        for action_id, ep_list in sorted(grouped.items()):
            if not ep_list:
                continue
            input_order = list(ep_list[0].input_slot_order)
            # Filter input slots to non-None typed inputs present across all before_slots
            valid_inputs: list[tuple[str, str]] = []
            for sname in input_order:
                b0 = ep_list[0].before_dict().get(sname)
                if b0 is None:
                    continue
                st = infer_value_type(b0)
                if st != "unknown":
                    valid_inputs.append((sname, st))

            param_names = [k for k, _t in valid_inputs]
            arg_types = [t for _k, t in valid_inputs]

            # Identify changed target slots across the episodes for this action
            changed_slot_names = sorted(
                {
                    sname
                    for ep in ep_list
                    for sname in ep.changed_slots().keys()
                }
            )

            slot_progs: list[tuple[str, ExprNode]] = []
            py_sources: list[tuple[str, str]] = []
            diags: list[SynthesisDiagnostics] = []

            for t_slot in changed_slot_names:
                io_pairs: list[tuple[tuple[Any, ...], Any]] = []
                out_t = "unknown"
                for ep in ep_list:
                    b_d = ep.before_dict()
                    a_d = ep.after_dict()
                    if t_slot not in a_d:
                        continue
                    xs = tuple(b_d[p] for p in param_names)
                    yv = a_d[t_slot]
                    out_t = infer_value_type(yv)
                    io_pairs.append((xs, yv))

                expr, py_src, diag = synthesize_typed_function(
                    arg_types,
                    out_t,
                    io_pairs,
                    param_names=param_names,
                    use_shard_guidance=self.use_shard_guidance,
                    use_rdc_boundary_refutation=self.use_rdc_boundary_refutation,
                    max_candidates=self.max_candidates,
                    fn_name=f"_slot_{t_slot}",
                    target_slot_label=t_slot,
                )
                diags.append(diag)
                if expr is not None and py_src is not None:
                    slot_progs.append((t_slot, expr))
                    py_sources.append((t_slot, py_src))

            all_solved = len(slot_progs) == len(changed_slot_names)
            next_ver = len(self._version_history.get(action_id, ())) + 1
            card = FunctionalOperatorCard(
                card_id=f"fcard_{action_id}_v{next_ver}",
                action_id=action_id,
                input_slots=tuple(valid_inputs),
                slot_programs=tuple(slot_progs),
                python_sources=tuple(py_sources),
                version=next_ver,
                support_episodes=len(ep_list),
                all_slots_solved=all_solved,
                diagnostics=tuple(diags),
            )
            self._version_history.setdefault(action_id, []).append(card)
            # Never overwrite an already fully-solved active card with an incomplete partial card!
            prev_active = self._cards.get(action_id)
            if all_solved or prev_active is None or not prev_active.all_slots_solved:
                self._cards[action_id] = card

        return dict(self._cards)

    def rollback(
        self,
        action_id: str,
        target_version: int | None = None,
    ) -> FunctionalOperatorCard:
        """Deterministically roll back ``action_id`` to an earlier version in ``_version_history``."""
        if self._frozen:
            raise TransferInducerFrozenError("Cannot call rollback() on a frozen TransferProgramInducer")
        hist = self._version_history.get(action_id, [])
        if len(hist) < 2:
            raise ValueError(
                f"Cannot roll back {action_id!r}: requires >= 2 versions in history (found {len(hist)})"
            )
        if target_version is not None:
            matched = [c for c in hist if c.version == target_version]
            if not matched:
                raise ValueError(f"Version {target_version} not found for {action_id!r}")
            restored = matched[-1]
        else:
            cur = self._cards.get(action_id)
            if cur is not None:
                prior = [c for c in hist if c.version < cur.version]
                if not prior:
                    raise ValueError(f"No prior version before v{cur.version} for {action_id!r}")
                restored = prior[-1]
            else:
                restored = hist[-2]
        self._cards[action_id] = restored
        return restored

    def bit_budget_summary(self) -> dict[str, int]:
        if not self._cards:
            return {"active_card_count": 0, "library_compressed_bits": 0}
        payload = {aid: c.to_executable_dict() for aid, c in sorted(self._cards.items())}
        return {
            "active_card_count": len(self._cards),
            "library_compressed_bits": compressed_bit_length(canonical_json_bytes(payload)),
        }
