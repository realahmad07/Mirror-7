"""1-Shot Live Formula Teaching: Safe Math Expression -> Verified UOL Rule + Auto-Inverses.

Allows users to teach Mirror new executable formulas in plain text, e.g.:
    "Learn formula: torque = force * lever_arm"
    "Teach rule: escape_velocity = sqrt(2 * g * r)"

Every formula is parsed via a strict safe AST whitelist (never eval/exec),
compiled into a typed UOL OperationalRule, automatically inverted via
derive_inverse_rules (5-point round-trip verified), and optionally persisted
into a .dcrystal vault and wired live into UOLGraphPlanner.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .disk_vault import DiskCrystalVault
from .operational_language import (
    Call,
    Expression,
    Lit,
    OperationalRule,
    RuleStatus,
    UOL,
    UOLError,
    Var,
    VariableSpec,
)
from .uol_inverse_deriver import derive_inverse_rules
from .uol_skill_store import persist_uol_rules_to_vault

_BIN_OPS: dict[type, str] = {
    ast.Add: "add",
    ast.Sub: "subtract",
    ast.Mult: "multiply",
    ast.Div: "divide",
    ast.Mod: "modulo",
    ast.Pow: "power",
}

_SAFE_FUNCS: dict[str, tuple[str, int]] = {
    "sqrt": ("sqrt", 1),
    "abs": ("absolute", 1),
    "log": ("log", 1),
    "exp": ("exp", 1),
    "sin": ("sin", 1),
    "cos": ("cos", 1),
    "floor": ("floor", 1),
    "ceil": ("ceil", 1),
    "round": ("round", 1),
    "min": ("minimum", 2),
    "max": ("maximum", 2),
}


@dataclass(frozen=True)
class TaughtFormulaResult:
    ok: bool
    target_quantity: str = ""
    forward_rule: OperationalRule | None = None
    inverse_rules: tuple[OperationalRule, ...] = ()
    persisted: bool = False
    message: str = ""

    @property
    def all_rules(self) -> tuple[OperationalRule, ...]:
        if self.forward_rule is None:
            return ()
        return (self.forward_rule,) + self.inverse_rules


def _ast_to_uol(node: ast.AST, seen_vars: list[str]) -> Expression:
    if isinstance(node, ast.Expression):
        return _ast_to_uol(node.body, seen_vars)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UOLError("formula constants must be numbers")
        return Lit(node.value)
    if isinstance(node, ast.Name):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", node.id):
            raise UOLError(f"invalid variable name: {node.id}")
        if node.id not in seen_vars:
            seen_vars.append(node.id)
        return Var(node.id)
    if isinstance(node, ast.UnaryOp):
        operand = _ast_to_uol(node.operand, seen_vars)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return Call("negate", (operand,))
        raise UOLError(f"unsupported unary op: {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        op_name = _BIN_OPS.get(type(node.op))
        if op_name is None:
            raise UOLError(f"unsupported binary op: {type(node.op).__name__}")
        left = _ast_to_uol(node.left, seen_vars)
        right = _ast_to_uol(node.right, seen_vars)
        return Call(op_name, (left, right))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise UOLError("only simple whitelisted math calls are allowed")
        spec = _SAFE_FUNCS.get(node.func.id.casefold())
        if spec is None:
            raise UOLError(f"disallowed function call: {node.func.id}")
        op_name, arity = spec
        if len(node.args) != arity:
            raise UOLError(f"{node.func.id} expects {arity} argument(s)")
        return Call(op_name, tuple(_ast_to_uol(a, seen_vars) for a in node.args))
    raise UOLError(f"unsupported formula syntax: {type(node).__name__}")


def compile_formula_to_uol(
    target_name: str,
    expr_str: str,
    *,
    domain: str = "taught",
    vault_path: str | Path | None = None,
) -> TaughtFormulaResult:
    """Safely compile `target_name = expr_str` into a verified UOL rule + inverse rules."""
    clean_name = re.sub(r"[^A-Za-z0-9_]+", "_", target_name.strip().casefold()).strip("_")
    if not clean_name or not re.fullmatch(r"[a-z_][a-z0-9_]{0,63}", clean_name):
        return TaughtFormulaResult(False, message=f"invalid formula name: {target_name!r}")

    # Allow caret `^` as power operator in user-friendly math notation
    normalized_expr = expr_str.strip().replace("^", "**")
    if not normalized_expr:
        return TaughtFormulaResult(False, message="empty formula expression")

    try:
        tree = ast.parse(normalized_expr, mode="eval")
        seen_vars: list[str] = []
        body = _ast_to_uol(tree, seen_vars)
    except (SyntaxError, UOLError) as exc:
        return TaughtFormulaResult(False, message=f"formula parse error: {exc}")

    if not seen_vars:
        return TaughtFormulaResult(False, message="formula must reference at least one input variable")

    rule_id = f"{domain.strip().casefold()}.{clean_name}"
    inputs = tuple(VariableSpec(v, "number") for v in seen_vars)
    fwd = OperationalRule(
        rule_id=rule_id,
        name=clean_name.replace("_", " ").title(),
        inputs=inputs,
        body=body,
        domains=(domain.strip().casefold(), "general"),
        status=RuleStatus.VERIFIED,
        provenance=f"uol-formula-teacher:{clean_name}={expr_str.strip()}",
    )

    # Validate the rule inside a fresh UOL instance
    try:
        UOL((fwd,))
    except UOLError as exc:
        return TaughtFormulaResult(False, message=f"UOL validation failed: {exc}")

    inverses = derive_inverse_rules(fwd)

    persisted = False
    if vault_path is not None:
        vault = DiskCrystalVault(vault_path)
        try:
            persist_uol_rules_to_vault(vault, (fwd,) + inverses)
            persisted = True
        finally:
            vault.close()

    inv_names = ", ".join(r.rule_id for r in inverses) if inverses else "none"
    return TaughtFormulaResult(
        ok=True,
        target_quantity=clean_name,
        forward_rule=fwd,
        inverse_rules=inverses,
        persisted=persisted,
        message=(
            f"Learned verified UOL formula '{rule_id}' ({', '.join(seen_vars)} -> {clean_name}) "
            f"+ {len(inverses)} round-trip verified inverse rule(s): [{inv_names}]."
        ),
    )


def parse_teach_formula_prompt(prompt: str) -> tuple[str, str] | None:
    """Detect prompts like 'Learn formula: torque = force * lever_arm'."""
    m = re.match(
        r"^\s*(?:learn|teach|define|add)\s+(?:formula|rule|equation)\s*[:\-]?\s*([A-Za-z_][A-Za-z0-9_\s]*?)\s*=\s*(.+?)\s*$",
        prompt.strip(),
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()
