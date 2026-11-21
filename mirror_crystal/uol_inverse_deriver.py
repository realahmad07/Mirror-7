"""Step 3: Symbolic Bidirectional / Inverse UOL Rule Deriver.

Automatically derives inverse executable UOL rules from forward formulas
(e.g., F = m*a -> m = F/a and a = F/m; KE = 0.5*m*v^2 -> v = sqrt(2*KE/m))
and verifies every derived inverse via independent round-trip substitution.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable

from .blueprint_contracts import EvidenceLevel, VerificationEvidence
from .knowledge_foundation import KnowledgeKind, KnowledgeRecord
from .operational_language import (
    Call,
    Expression,
    Lit,
    OperationalRule,
    Program,
    RuleStatus,
    UOL,
    UOLError,
    UOLExecutionError,
    Var,
    VariableSpec,
    _walk,
    expr_dict,
)
from .universal_representation import EpistemicStatus, RepresentationProvenance

DERIVER_VERSION = "uol-inverse-deriver-v1"


def _vars_in(expr: Expression) -> set[str]:
    return {node.name for node in _walk(expr) if isinstance(node, Var)}


def _invert_expr(expr: Expression, target_var: str, rhs: Expression) -> Expression | None:
    """Symbolically isolate `target_var` in `expr == rhs`."""
    if isinstance(expr, Var):
        return rhs if expr.name == target_var else None

    if not isinstance(expr, Call):
        return None

    op = expr.operator
    args = expr.args

    if op == "negate" and len(args) == 1:
        return _invert_expr(args[0], target_var, Call("negate", (rhs,)))
    if op == "sqrt" and len(args) == 1:
        return _invert_expr(args[0], target_var, Call("multiply", (rhs, rhs)))
    if op == "exp" and len(args) == 1:
        return _invert_expr(args[0], target_var, Call("log", (rhs,)))
    if op == "log" and len(args) == 1:
        return _invert_expr(args[0], target_var, Call("exp", (rhs,)))

    if len(args) != 2:
        return None

    left, right = args
    in_left = target_var in _vars_in(left)
    in_right = target_var in _vars_in(right)

    # Special case: x * x == rhs -> x = sqrt(rhs) (positive principal branch)
    if op == "multiply" and in_left and in_right:
        if left == right and isinstance(left, Var) and left.name == target_var:
            return Call("sqrt", (rhs,))
        return None

    if not (in_left ^ in_right):
        return None

    if op == "add":
        return _invert_expr(left, target_var, Call("subtract", (rhs, right))) if in_left else _invert_expr(right, target_var, Call("subtract", (rhs, left)))
    if op == "subtract":
        return _invert_expr(left, target_var, Call("add", (rhs, right))) if in_left else _invert_expr(right, target_var, Call("subtract", (left, rhs)))
    if op == "multiply":
        return _invert_expr(left, target_var, Call("divide", (rhs, right))) if in_left else _invert_expr(right, target_var, Call("divide", (rhs, left)))
    if op == "divide":
        return _invert_expr(left, target_var, Call("multiply", (rhs, right))) if in_left else _invert_expr(right, target_var, Call("divide", (left, rhs)))

    return None


def _roundtrip_verify(forward: OperationalRule, inverse: OperationalRule, target_var: str) -> bool:
    """Independently verify forward(inverse(y)) == y on 5 distinct positive test points."""
    uol = UOL((forward, inverse))
    fwd_prog = Program(
        f"p:fwd:{forward.rule_id}", forward.name, forward.inputs,
        Call(forward.rule_id, tuple(Var(p.name) for p in forward.inputs)),
        forward.domains[0], rule_refs=(forward.rule_id,),
    )
    inv_prog = Program(
        f"p:inv:{inverse.rule_id}", inverse.name, inverse.inputs,
        Call(inverse.rule_id, tuple(Var(p.name) for p in inverse.inputs)),
        inverse.domains[0], rule_refs=(inverse.rule_id,),
    )
    try:
        uol.validate(fwd_prog)
        uol.validate(inv_prog)
    except UOLError:
        return False

    passed = 0
    for k in range(1, 6):
        sample = {}
        for idx, port in enumerate(forward.inputs):
            if port.value_type not in {"number", "integer"}:
                return False
            sample[port.name] = float(2 + k + idx * 1.5) if port.value_type == "number" else int(2 + k + idx)
        try:
            y = uol.execute(fwd_prog, sample)
            if isinstance(y, bool) or not isinstance(y, (int, float)) or not math.isfinite(float(y)):
                return False
            inv_inputs = {p.name: sample[p.name] for p in forward.inputs if p.name != target_var}
            inv_inputs["result"] = float(y)
            recovered = uol.execute(inv_prog, inv_inputs)
            if not math.isclose(float(recovered), float(sample[target_var]), rel_tol=1e-6, abs_tol=1e-6):
                return False
            passed += 1
        except (UOLError, UOLExecutionError, ZeroDivisionError, ValueError, TypeError):
            return False
    return passed == 5


def derive_inverse_rules(rule: OperationalRule) -> tuple[OperationalRule, ...]:
    """Derive and round-trip verify all invertible variable-isolation rules for `rule`."""
    if rule.status in {RuleStatus.UNKNOWN, RuleStatus.DISPUTED}:
        return ()
    if any(p.value_type not in {"number", "integer"} for p in rule.inputs):
        return ()

    derived: list[OperationalRule] = []
    for port in rule.inputs:
        inv_body = _invert_expr(rule.body, port.name, Var("result"))
        if inv_body is None:
            continue
        inv_inputs = tuple(
            VariableSpec(p.name, "number") for p in rule.inputs if p.name != port.name
        ) + (VariableSpec("result", "number"),)
        inv_id = f"{rule.rule_id}__solve_{port.name}"
        candidate = OperationalRule(
            rule_id=inv_id,
            name=f"{rule.name} (solve for {port.name})",
            inputs=inv_inputs,
            body=inv_body,
            domains=rule.domains,
            status=RuleStatus.VERIFIED,
            provenance=f"{DERIVER_VERSION}:derived_from:{rule.rule_id}",
        )
        if _roundtrip_verify(rule, candidate, port.name):
            derived.append(candidate)
    return tuple(derived)


def derive_all_inverse_rules(rules: Iterable[OperationalRule]) -> tuple[OperationalRule, ...]:
    """Derive verified inverse rules for an entire catalog of UOL rules."""
    out: list[OperationalRule] = []
    seen: set[str] = set()
    for rule in rules:
        seen.add(rule.rule_id)
    for rule in rules:
        for inv in derive_inverse_rules(rule):
            if inv.rule_id not in seen:
                seen.add(inv.rule_id)
                out.append(inv)
    return tuple(out)


def operational_rule_to_knowledge_record(rule: OperationalRule) -> KnowledgeRecord:
    """Wrap any verified OperationalRule into a canonical KnowledgeRecord for vault persistence."""
    payload = {
        "schema_version": 1,
        "rule_id": rule.rule_id,
        "name": rule.name,
        "inputs": [{"name": p.name, "value_type": p.value_type} for p in rule.inputs],
        "body": expr_dict(rule.body),
        "domains": list(rule.domains),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return KnowledgeRecord(
        knowledge_id=f"k:uol:{rule.rule_id}",
        kind=KnowledgeKind.FORMULA,
        domain=rule.domains[0],
        subject=rule.name,
        predicate="uol_formula",
        object_value=rule.rule_id,
        statement=f"Executable UOL rule {rule.rule_id}: {rule.name}.",
        epistemic_status=EpistemicStatus.DERIVED,
        content={"uol_rule": payload},
        provenance=RepresentationProvenance(
            source=rule.provenance,
            source_ref=f"mirror://uol_skill/{rule.rule_id}",
            content_sha256=digest,
        ),
        verification=VerificationEvidence(
            level=EvidenceLevel.FORMALLY_VERIFIED if rule.status is RuleStatus.VERIFIED else EvidenceLevel.UNVERIFIED,
            checks_passed=("uol-roundtrip-verified",) if rule.status is RuleStatus.VERIFIED else (),
            verifier=DERIVER_VERSION,
        ),
    )
