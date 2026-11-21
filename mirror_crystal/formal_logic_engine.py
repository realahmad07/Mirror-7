"""Native Pure-Python Formal Deductive Logic & Constraint Solver (Zero 3rd-Party Dependencies).

Provides 4 mathematically sound, zero-hallucination engines:
1. Propositional DPLL SAT solver (with unit propagation & assignment verification)
2. Syllogism & Horn-clause forward-chaining prover (with proof trace)
3. Transitive ordering & inequality chain prover (e.g., "A > B, B > C, C = 5. Is A > 5?")
4. Bounded finite-domain integer constraint solver (safe AST evaluation, no eval/exec)
"""
from __future__ import annotations

import ast
from collections import defaultdict, deque
from dataclasses import dataclass
import itertools
import re
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class LogicProofResult:
    matched: bool
    kind: str = ""  # "syllogism", "ordering", "constraint", "sat", "undetermined"
    proven: bool | None = None
    answer: Any = None
    proof_steps: tuple[str, ...] = ()
    explanation: str = ""


# ---------------- 1. Propositional DPLL SAT Solver ----------------

def _verify_sat(clauses: Sequence[Sequence[int]], model: Mapping[int, bool]) -> bool:
    for clause in clauses:
        if not any((lit > 0 and model.get(abs(lit), False)) or (lit < 0 and not model.get(abs(lit), False)) for lit in clause):
            return False
    return True


def solve_sat(clauses: Sequence[Sequence[int]]) -> dict[int, bool] | None:
    """Pure-Python DPLL SAT solver over integer CNF clauses (positive=True, negative=False)."""
    cnf = [tuple(c) for c in clauses]

    def simplify(current_cnf: list[tuple[int, ...]], lit: int) -> list[tuple[int, ...]] | None:
        new_cnf: list[tuple[int, ...]] = []
        for clause in current_cnf:
            if lit in clause:
                continue
            reduced = tuple(x for x in clause if x != -lit)
            if not reduced:
                return None
            new_cnf.append(reduced)
        return new_cnf

    def dpll(current_cnf: list[tuple[int, ...]], assign: dict[int, bool]) -> dict[int, bool] | None:
        # Unit propagation
        while True:
            unit = next((c[0] for c in current_cnf if len(c) == 1), None)
            if unit is None:
                break
            var = abs(unit)
            val = unit > 0
            if var in assign and assign[var] != val:
                return None
            assign[var] = val
            nxt = simplify(current_cnf, unit)
            if nxt is None:
                return None
            current_cnf = nxt

        if not current_cnf:
            return assign

        # Pick first unassigned variable
        var = abs(current_cnf[0][0])
        for val in (True, False):
            lit = var if val else -var
            nxt = simplify(current_cnf, lit)
            if nxt is not None:
                res = dpll(nxt, {**assign, var: val})
                if res is not None:
                    return res
        return None

    model = dpll(cnf, {})
    if model is not None and _verify_sat(cnf, model):
        return model
    return None


# ---------------- 2. Syllogism & Horn-Clause Prover ----------------

def _singularize(term: str) -> str:
    t = re.sub(r"^(?:a|an|the)\s+", "", term.strip().casefold())
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def prove_syllogism_prompt(prompt: str) -> LogicProofResult:
    """Prove transitive syllogisms: 'All A are B. All B are C. X is A. Is X C?'"""
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", prompt.strip()) if s.strip()]
    if len(sentences) < 2:
        return LogicProofResult(False)

    graph: dict[str, list[tuple[str, str]]] = defaultdict(list)  # src -> [(dst, raw_rule)]
    question: tuple[str, str] | None = None

    for sent in sentences:
        # "All X are Y" / "Every X is Y"
        m_all = re.match(r"^(?:all|every)\s+(.+?)\s+(?:are|is)\s+(.+)$", sent, flags=re.IGNORECASE)
        if m_all:
            src, dst = _singularize(m_all.group(1)), _singularize(m_all.group(2))
            graph[src].append((dst, sent))
            continue

        # "If X then Y"
        m_if = re.match(r"^if\s+(.+?),?\s+then\s+(.+)$", sent, flags=re.IGNORECASE)
        if m_if:
            src, dst = _singularize(m_if.group(1)), _singularize(m_if.group(2))
            graph[src].append((dst, sent))
            continue

        # Query: "Is X Y?" / "Are X Y?"
        m_q = re.match(r"^(?:is|are)\s+(.+?)\s+(?:a\s+|an\s+)?(.+)$", sent, flags=re.IGNORECASE)
        if m_q:
            question = (_singularize(m_q.group(1)), _singularize(m_q.group(2)))
            continue

        # Fact: "X is a Y" / "X is Y"
        m_fact = re.match(r"^(.+?)\s+(?:is|are)\s+(?:a\s+|an\s+)?(.+)$", sent, flags=re.IGNORECASE)
        if m_fact:
            src, dst = _singularize(m_fact.group(1)), _singularize(m_fact.group(2))
            graph[src].append((dst, sent))

    if question is None or not graph:
        return LogicProofResult(False)

    start, goal = question
    q: deque[tuple[str, list[str]]] = deque([(start, [])])
    visited = {start}
    while q:
        curr, path = q.popleft()
        if curr == goal and path:
            return LogicProofResult(
                matched=True,
                kind="syllogism",
                proven=True,
                answer=True,
                proof_steps=tuple(path),
                explanation=f"PROVEN True: '{start}' -> '{goal}' via {len(path)} deductive step(s): " + " => ".join(path),
            )
        for nxt, rule_txt in graph.get(curr, ()):
            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, path + [f"{curr} -> {nxt} ({rule_txt})"]))

    return LogicProofResult(
        matched=True,
        kind="syllogism",
        proven=None,
        answer=None,
        explanation=f"UNDETERMINED: no deductive proof chain from '{start}' to '{goal}'.",
    )


# ---------------- 3. Transitive Ordering & Inequality Solver ----------------

def prove_ordering_prompt(prompt: str) -> LogicProofResult:
    """Prove ordering chains like 'A > B, B > C, C = 5. Is A > 5?'"""
    m_query = re.search(
        r"\b(?:is|does|prove)\s+([A-Za-z0-9_.]+)\s*(>=|<=|>|<|==|=)\s*([A-Za-z0-9_.]+)\s*\??$",
        prompt.strip(),
        flags=re.IGNORECASE,
    )
    if not m_query:
        return LogicProofResult(False)

    q_lhs = m_query.group(1).strip().rstrip(".")
    q_op = m_query.group(2).strip()
    q_rhs = m_query.group(3).strip().rstrip(".")
    premise_text = prompt[: m_query.start()]

    # Directed graph: u -> (v, strict: bool, label: str) meaning u >= v (or u > v if strict)
    gt_graph: dict[str, list[tuple[str, bool, str]]] = defaultdict(list)
    eq_pairs: list[tuple[str, str]] = []

    for m in re.finditer(r"([A-Za-z0-9_.]+)\s*(>=|<=|>|<|==|=)\s*([A-Za-z0-9_.]+)", premise_text):
        u = m.group(1).strip().rstrip(".")
        op = m.group(2).strip()
        v = m.group(3).strip().rstrip(".")
        if op == ">":
            gt_graph[u].append((v, True, f"{u} > {v}"))
        elif op == ">=":
            gt_graph[u].append((v, False, f"{u} >= {v}"))
        elif op == "<":
            gt_graph[v].append((u, True, f"{v} > {u}"))
        elif op == "<=":
            gt_graph[v].append((u, False, f"{v} >= {u}"))
        elif op in {"=", "=="}:
            gt_graph[u].append((v, False, f"{u} = {v}"))
            gt_graph[v].append((u, False, f"{v} = {u}"))
            eq_pairs.append((u, v))

    if not gt_graph:
        return LogicProofResult(False)

    # Normalize query to (src, dst, need_strict)
    if q_op == ">":
        src, dst, need_strict = q_lhs, q_rhs, True
    elif q_op == ">=":
        src, dst, need_strict = q_lhs, q_rhs, False
    elif q_op == "<":
        src, dst, need_strict = q_rhs, q_lhs, True
    elif q_op == "<=":
        src, dst, need_strict = q_rhs, q_lhs, False
    else:
        return LogicProofResult(False)

    # BFS tracking (node, has_strict_edge, path)
    queue: deque[tuple[str, bool, list[str]]] = deque([(src, False, [])])
    visited: set[tuple[str, bool]] = {(src, False)}
    while queue:
        curr, saw_strict, path = queue.popleft()
        if curr == dst and path and (not need_strict or saw_strict):
            return LogicProofResult(
                matched=True,
                kind="ordering",
                proven=True,
                answer=True,
                proof_steps=tuple(path),
                explanation=f"PROVEN True: {q_lhs} {q_op} {q_rhs} via chain: " + " , ".join(path),
            )
        for nxt, is_strict, label in gt_graph.get(curr, ()):
            nxt_strict = saw_strict or is_strict
            if (nxt, nxt_strict) not in visited:
                visited.add((nxt, nxt_strict))
                queue.append((nxt, nxt_strict, path + [label]))

    return LogicProofResult(
        matched=True,
        kind="ordering",
        proven=None,
        answer=None,
        explanation=f"UNDETERMINED: cannot deduce {q_lhs} {q_op} {q_rhs} from given inequalities.",
    )


# ---------------- 4. Bounded Finite-Domain Integer Constraint Solver ----------------

def _safe_eval_constraint(node: ast.AST, env: Mapping[str, int]) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval_constraint(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"unknown var {node.id}")
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _safe_eval_constraint(node.operand, env)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp):
        l = _safe_eval_constraint(node.left, env)
        r = _safe_eval_constraint(node.right, env)
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        if isinstance(node.op, ast.Mult):
            return l * r
        if isinstance(node.op, ast.Mod) and r != 0:
            return l % r
        raise ValueError("unsupported binop")
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        l = _safe_eval_constraint(node.left, env)
        r = _safe_eval_constraint(node.comparators[0], env)
        op = node.ops[0]
        if isinstance(op, ast.Eq):
            return l == r
        if isinstance(op, ast.NotEq):
            return l != r
        if isinstance(op, ast.Lt):
            return l < r
        if isinstance(op, ast.LtE):
            return l <= r
        if isinstance(op, ast.Gt):
            return l > r
        if isinstance(op, ast.GtE):
            return l >= r
    raise ValueError("unsupported constraint AST")


def solve_constraint_prompt(prompt: str) -> LogicProofResult:
    """Solve bounded integer puzzles: 'Find x, y in 1..20 where x + y = 15 and x = 2 * y'."""
    m = re.match(
        r"^\s*(?:find|solve)\s+(?:integers?\s+)?([A-Za-z_][A-Za-z0-9_,\s]*?)\s+in\s+(-?\d+)\s*\.\.\s*(-?\d+)\s+(?:where|such\s+that)\s+(.+?)\s*\.?\s*$",
        prompt.strip(),
        flags=re.IGNORECASE,
    )
    if not m:
        return LogicProofResult(False)

    vars_list = [v.strip() for v in m.group(1).split(",") if v.strip()]
    lo, hi = int(m.group(2)), int(m.group(3))
    if not 1 <= len(vars_list) <= 3 or (hi - lo + 1) ** len(vars_list) > 50000:
        return LogicProofResult(False, explanation="constraint domain exceeds safe search bound")

    cond_strs = [c.strip() for c in re.split(r"\s+\band\b\s+|,", m.group(4)) if c.strip()]
    parsed_conds: list[ast.AST] = []
    for cs in cond_strs:
        # Normalize single '=' to '==' and implicit multiplication '2y' -> '2*y'
        norm = re.sub(r"(?<![<>!=])=(?!=)", "==", cs)
        norm = re.sub(r"(\d)\s*([A-Za-z])", r"\1*\2", norm)
        try:
            parsed_conds.append(ast.parse(norm, mode="eval"))
        except SyntaxError:
            return LogicProofResult(False)

    solutions: list[dict[str, int]] = []
    for combo in itertools.product(range(lo, hi + 1), repeat=len(vars_list)):
        env = dict(zip(vars_list, combo))
        try:
            if all(bool(_safe_eval_constraint(cond, env)) for cond in parsed_conds):
                solutions.append(env)
                if len(solutions) >= 5:
                    break
        except ValueError:
            return LogicProofResult(False)

    if not solutions:
        return LogicProofResult(
            matched=True,
            kind="constraint",
            proven=False,
            answer=[],
            explanation=f"UNSATISFIABLE: no integer assignment in {lo}..{hi} satisfies all constraints.",
        )

    first = solutions[0]
    sol_txt = ", ".join(f"{k}={v}" for k, v in first.items())
    return LogicProofResult(
        matched=True,
        kind="constraint",
        proven=True,
        answer=first,
        proof_steps=tuple(cond_strs),
        explanation=f"PROVEN constraint solution: {sol_txt} (verified against {len(cond_strs)} constraint(s)).",
    )


def solve_formal_logic_prompt(prompt: str) -> LogicProofResult:
    """Try finite-domain constraints, then ordering chains, then syllogisms."""
    for fn in (solve_constraint_prompt, prove_ordering_prompt, prove_syllogism_prompt):
        res = fn(prompt)
        if res.matched:
            return res
    return LogicProofResult(False)
