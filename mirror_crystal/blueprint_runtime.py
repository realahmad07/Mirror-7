"""Dependency-ordered blueprint composition with explicit condition verification."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Iterable, Mapping

from .blueprint_contracts import Blueprint, BlueprintDependencyGraph, BlueprintValidationError


class BlueprintCompositionError(ValueError):
    """Raised when a composition prerequisite or verification gate is not satisfied."""


@dataclass(frozen=True)
class BlueprintCompositionResult:
    answer: str
    blueprint_ids: tuple[str, ...]
    verified_postconditions: tuple[str, ...]


class BlueprintCompositionEngine:
    """Compose only dependency-complete blueprints with verified pre/postconditions."""

    @staticmethod
    def _render(template: str, values: Mapping[str, Any]) -> str:
        names = re.findall(r"\{([A-Za-z][A-Za-z0-9_]*)\}", template)
        missing = sorted({name for name in names if name not in values})
        if missing:
            raise BlueprintCompositionError("missing template values: " + ", ".join(missing))
        return re.sub(
            r"\{([A-Za-z][A-Za-z0-9_]*)\}",
            lambda match: str(values[match.group(1)]),
            template,
        )

    def compose(
        self,
        blueprint_id: str,
        blueprints: Iterable[Blueprint],
        *,
        values: Mapping[str, Any] | None = None,
        verified_conditions: Iterable[str] = (),
        postcondition_verifier: Callable[[Blueprint, str, str, Mapping[str, Any]], bool] | None = None,
        executor: Callable[[Blueprint, Mapping[str, Any]], str] | None = None,
    ) -> BlueprintCompositionResult:
        graph = BlueprintDependencyGraph(blueprints)
        try:
            ordered = graph.resolve(blueprint_id)
        except BlueprintValidationError as exc:
            raise BlueprintCompositionError(str(exc)) from exc

        state: dict[str, Any] = dict(values or {})
        outputs: dict[str, str] = {}
        condition_ledger = set(str(x) for x in verified_conditions)
        answer_parts: list[str] = []
        verified_posts: list[str] = []

        for blueprint in ordered:
            missing = sorted(set(blueprint.preconditions) - condition_ledger)
            if missing:
                raise BlueprintCompositionError(
                    f"{blueprint.blueprint_id} unmet preconditions: " + ", ".join(missing)
                )
            state["blueprint_outputs"] = dict(outputs)
            if executor is not None:
                rendered = str(executor(blueprint, state))
            else:
                if blueprint.derived_expr:
                    raise BlueprintCompositionError(
                        f"{blueprint.blueprint_id} has a derived expression and requires an explicit safe executor"
                    )
                rendered = self._render(blueprint.response_template, state)
            if not rendered.strip():
                raise BlueprintCompositionError(f"{blueprint.blueprint_id} produced an empty result")

            for condition in blueprint.postconditions:
                if postcondition_verifier is None or not postcondition_verifier(
                    blueprint, condition, rendered, state
                ):
                    raise BlueprintCompositionError(
                        f"{blueprint.blueprint_id} postcondition not independently satisfied: {condition}"
                    )
                condition_ledger.add(condition)
                verified_posts.append(condition)

            outputs[blueprint.blueprint_id] = rendered
            state[f"output:{blueprint.blueprint_id}"] = rendered
            answer_parts.append(rendered)

        return BlueprintCompositionResult(
            answer="\n\n".join(answer_parts),
            blueprint_ids=tuple(item.blueprint_id for item in ordered),
            verified_postconditions=tuple(verified_posts),
        )