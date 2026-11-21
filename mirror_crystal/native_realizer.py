"""100% Native, Zero-Third-Party Hybrid Realization Layer for Mirror 7.

Renders Mirror's structured verified outputs (multi-step UOL graph derivations,
single UOL executions, TDD-verified code, verified facts, clarifications, and
honest abstentions) into clear, human-readable natural language without any
external or third-party LLM (no Qwen, no Llama, no remote APIs).

Invariant: Every verified value is truth-locked; realization never fabricates
facts or upgrades unverified candidates into verified truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .uol_graph_planner import GraphPlanResult, PlannedStep


def _fmt_num(val: float | int) -> str:
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, int):
        return str(val)
    if abs(val - round(val)) < 1e-9 and abs(val) < 1e12:
        return str(int(round(val)))
    return f"{val:.6g}"


@dataclass(frozen=True)
class RealizationOutput:
    text: str
    style: str
    verified: bool
    truth_locked: bool
    realizer_engine: str = "mirror-native-hybrid-realizer-v1"
    uses_third_party_model: bool = False
    step_count: int = 0


class NativeHybridRealizer:
    """Zero-dependency native realization layer with strict truth-lock checks."""

    ENGINE_ID = "mirror-native-hybrid-realizer-v1"
    USES_THIRD_PARTY_MODEL = False

    def __init__(self, *, default_style: str = "step_by_step") -> None:
        if default_style not in {"concise", "step_by_step", "technical"}:
            raise ValueError("style must be 'concise', 'step_by_step', or 'technical'")
        self.default_style = default_style

    def realize_graph_plan(
        self,
        plan: GraphPlanResult,
        *,
        style: str | None = None,
    ) -> RealizationOutput:
        mode = style or self.default_style
        if not plan.solved or plan.output is None:
            text = (
                f"I could not derive '{plan.target_quantity}' from the given inputs "
                f"({', '.join(f'{k}={_fmt_num(v)}' for k, v in sorted(plan.known_inputs.items())) or 'none'}). "
                f"Reason: {plan.reason}."
            )
            return RealizationOutput(
                text=text,
                style=mode,
                verified=False,
                truth_locked=True,
                step_count=0,
            )

        val_str = _fmt_num(plan.output)
        unit_suffix = f" {plan.unit}" if plan.unit else ""
        label = plan.target_quantity.replace("_", " ")

        if mode == "concise":
            text = f"{label.capitalize()} = {val_str}{unit_suffix} (verified via {len(plan.steps)}-step UOL plan)."
        elif mode == "technical":
            chain = " -> ".join(plan.rule_ids) if plan.rule_ids else "direct"
            text = (
                f"[VERIFIED UOL] target={plan.target_quantity} value={val_str}{unit_suffix} "
                f"hops={len(plan.steps)} chain=[{chain}]"
            )
        else:
            lines = [
                f"{label.capitalize()} = {val_str}{unit_suffix} "
                f"(verified via {len(plan.steps)}-step UOL derivation)"
            ]
            if plan.steps:
                lines.append("")
                lines.append("Steps:")
                for step in plan.steps:
                    in_str = ", ".join(f"{k}={_fmt_num(v)}" for k, v in step.inputs.items())
                    step_unit = f" {step.output_unit}" if step.output_unit else ""
                    lines.append(
                        f"{step.step_index}. {step.rule_name} ({step.rule_id}): "
                        f"{in_str} -> {step.output_quantity} = {_fmt_num(step.output_value)}{step_unit}"
                    )
            text = "\n".join(lines)

        # Truth-lock check: the exact formatted verified value MUST appear in the realized text
        truth_locked = val_str in text
        return RealizationOutput(
            text=text,
            style=mode,
            verified=True,
            truth_locked=truth_locked,
            step_count=len(plan.steps),
        )

    def realize_orchestrator_response(
        self,
        payload: Mapping[str, Any],
        *,
        style: str | None = None,
    ) -> RealizationOutput:
        """Render any UnifiedCognitiveOrchestrator response payload natively."""
        mode = style or self.default_style
        qualification = str(payload.get("answer_qualification", "no_candidate"))
        raw_answer = str(payload.get("answer", "")).strip()
        uol_route = payload.get("uol_route")

        # 1. Multi-step or single UOL route
        if isinstance(uol_route, Mapping) and uol_route.get("matched"):
            steps_data = uol_route.get("steps") or ()
            out_val = uol_route.get("output")
            unit = str(uol_route.get("unit", ""))
            rule_ids = tuple(str(r) for r in uol_route.get("rule_ids", ()))
            val_str = _fmt_num(out_val) if isinstance(out_val, (int, float)) else str(out_val)
            unit_suffix = f" {unit}" if unit else ""

            if mode == "concise" or not steps_data:
                text = f"Verified result: {val_str}{unit_suffix} (via {' -> '.join(rule_ids)})."
            else:
                lines = [f"Verified result: {val_str}{unit_suffix}", "", "Derivation:"]
                for s in steps_data:
                    in_str = ", ".join(f"{k}={_fmt_num(v)}" for k, v in s.get("inputs", {}).items())
                    su = f" {s['output_unit']}" if s.get("output_unit") else ""
                    lines.append(
                        f"{s['step_index']}. {s['rule_name']} ({s['rule_id']}): "
                        f"{in_str} -> {s['output_quantity']} = {_fmt_num(s['output_value'])}{su}"
                    )
                text = "\n".join(lines)
            return RealizationOutput(
                text=text,
                style=mode,
                verified=True,
                truth_locked=val_str in text,
                step_count=len(steps_data) or len(rule_ids),
            )

        # 2. Verified task code / execution / factual synthesis
        if qualification in {"verified", "verified_task_result"}:
            return RealizationOutput(
                text=raw_answer,
                style=mode,
                verified=True,
                truth_locked=True,
            )

        # 3. Clarification required
        if qualification == "clarification_required":
            return RealizationOutput(
                text=raw_answer or "More information is required before Mirror can proceed.",
                style=mode,
                verified=False,
                truth_locked=True,
            )

        # 4. Unverified candidate — never upgrade to verified!
        if qualification == "unverified_candidate":
            cand = str(payload.get("candidate_answer") or raw_answer)
            text = f"Unverified candidate (not independently proven):\n{cand}"
            return RealizationOutput(
                text=text,
                style=mode,
                verified=False,
                truth_locked=True,
            )

        # 5. Honest abstention
        return RealizationOutput(
            text=raw_answer or "Mirror abstains: insufficient independently verified evidence.",
            style=mode,
            verified=False,
            truth_locked=True,
        )
