"""Runtime bridge for Mirror's universal task and UOL construction layers.

The bridge enriches every backend Q&A request with a typed task analysis and a
bounded capability-composition proposal. Explicit UOL synthesis is available
through a separate API and requires typed inputs plus disjoint train/held-out
examples. Candidates remain UNVERIFIED and are never executed or promoted here.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from mirror_crystal.capability_composition import CapabilityCompositionPlanner
from mirror_crystal.capability_matching import CapabilityMatcher
from mirror_crystal.capability_synthesizer import (
    CapabilitySynthesizer,
    Example,
    SynthesisGoal,
)
from mirror_crystal.knowledge_uol import KnowledgeRuleImporter
from mirror_crystal.operational_language import UOL, VariableSpec, program_dict
from mirror_crystal.task_understanding import TaskUnderstandingEngine


class UniversalIntelligencePipeline:
    """Connect task parsing, knowledge-to-UOL import, capability matching and planning."""

    def __init__(
        self,
        *,
        vault_provider: Callable[[], Any | None],
        max_knowledge_records: int = 500,
        max_capabilities: int = 500,
    ) -> None:
        if not callable(vault_provider):
            raise TypeError("vault_provider must be callable")
        for name, value in (
            ("max_knowledge_records", max_knowledge_records),
            ("max_capabilities", max_capabilities),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5000:
                raise ValueError(f"{name} must be an integer between 1 and 5000")
        self._vault_provider = vault_provider
        self.max_knowledge_records = max_knowledge_records
        self.max_capabilities = max_capabilities
        self.task_engine = TaskUnderstandingEngine()
        self.matcher = CapabilityMatcher()
        self.composer = CapabilityCompositionPlanner()

    def _knowledge(self, vault: Any, domain: str):
        listing = getattr(vault, "list_knowledge_records", None)
        if not callable(listing):
            return KnowledgeRuleImporter().import_records(())
        records = listing(
            domain=domain if domain != "general" else None,
            limit=self.max_knowledge_records,
        )
        return KnowledgeRuleImporter().import_records(records)

    def analyze(
        self,
        prompt: str,
        *,
        domain_hint: str | None = None,
        known_inputs: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
        explicit_constraints=(),
        acceptance_criteria=(),
    ) -> dict[str, Any]:
        """Return a task contract, imported-rule summary, matches and a non-executed plan."""
        task = self.task_engine.interpret(
            prompt,
            domain_hint=domain_hint,
            known_inputs=known_inputs,
            context=context,
            explicit_constraints=explicit_constraints,
            acceptance_criteria=acceptance_criteria,
        )
        vault = self._vault_provider()
        if vault is None:
            return {
                "status": "unavailable",
                "task": task.to_dict(),
                "knowledge_to_uol": {"rules_imported": 0, "rules_skipped": 0, "rules_rejected": 0},
                "capability_matching": {"matches": [], "unmatched_subtasks": []},
                "composition_plan": None,
                "execution": {
                    "attempted": False,
                    "reason": "no persistent capability vault is configured",
                },
            }

        imported = self._knowledge(vault, task.domain)
        language = UOL(imported.rules)
        listing = getattr(vault, "list_capabilities", None)
        capabilities = tuple(
            listing(limit=self.max_capabilities) if callable(listing) else ()
        )
        matches = self.matcher.match(task, capabilities)
        plan = self.composer.compose(task, capabilities, match_report=matches)
        if task.status.value == "needs_clarification":
            status = "needs_clarification"
        else:
            status = "plan_ready" if plan.ready else "blocked"

        return {
            "status": status,
            "task": task.to_dict(),
            "knowledge_to_uol": {
                "rules_imported": imported.imported_count,
                "rules_skipped": imported.skipped_count,
                "rules_rejected": imported.rejected_count,
                "uol_catalog_size": len(language.rules),
            },
            "capability_matching": {
                "match_count": len(matches.matches),
                "eligible_match_count": sum(1 for item in matches.matches if item.eligible),
                "unmatched_subtasks": list(matches.unmatched_subtasks),
                "matches": [
                    {
                        "subtask_id": item.subtask_id,
                        "capability_id": item.capability.capability_id,
                        "name": item.capability.name,
                        "score": item.score,
                        "eligible": item.eligible,
                        "matched_inputs": list(item.matched_inputs),
                        "missing_inputs": list(item.missing_inputs),
                        "blockers": list(item.blockers),
                    }
                    for item in matches.matches[:100]
                ],
            },
            "composition_plan": plan.to_dict(),
            "execution": {
                "attempted": False,
                "reason": (
                    "planning is advisory; only an explicitly registered native handler "
                    "and independent verifier may authorize execution and promotion"
                ),
            },
        }

    @staticmethod
    def _examples(value: Any, field_name: str) -> tuple[Example, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{field_name} must be an array")
        examples = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise ValueError(f"{field_name}[{index}] must be an object")
            inputs = item.get("inputs")
            if not isinstance(inputs, Mapping):
                raise ValueError(f"{field_name}[{index}].inputs must be an object")
            if "expected" not in item:
                raise ValueError(f"{field_name}[{index}] must include expected")
            examples.append(Example(dict(inputs), item["expected"]))
        return tuple(examples)

    @classmethod
    def _goal(cls, payload: Mapping[str, Any]) -> tuple[SynthesisGoal, str, str]:
        required = {"name", "domain", "inputs", "output_type", "training", "validation"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError("missing synthesis goal fields: " + ", ".join(missing))
        if not isinstance(payload["inputs"], (list, tuple)):
            raise ValueError("inputs must be an array")
        input_specs = []
        for index, item in enumerate(payload["inputs"]):
            if not isinstance(item, Mapping):
                raise ValueError(f"inputs[{index}] must be an object")
            name, value_type = item.get("name"), item.get("value_type")
            if not isinstance(name, str) or not isinstance(value_type, str):
                raise ValueError(f"inputs[{index}] requires string name and value_type")
            input_specs.append(VariableSpec(name, value_type))
        name, domain, output_type = payload["name"], payload["domain"], payload["output_type"]
        if not all(isinstance(item, str) and item.strip() for item in (name, domain, output_type)):
            raise ValueError("name, domain and output_type must be non-empty strings")
        goal = SynthesisGoal(
            name=name,
            domain=domain,
            inputs=tuple(input_specs),
            output_type=output_type,
            training=cls._examples(payload["training"], "training"),
            validation=cls._examples(payload["validation"], "validation"),
            constants=tuple(payload.get("constants", (0, 1, 2, -1))),
            max_depth=payload.get("max_depth", 2),
            candidate_budget=payload.get("candidate_budget", 20000),
            include_assumed_rules=payload.get("include_assumed_rules", True),
            enable_conditionals=payload.get("enable_conditionals", True),
            enable_map=payload.get("enable_map", True),
            enable_fold=payload.get("enable_fold", True),
        )
        intent = payload.get("intent", "any")
        output_name = payload.get("output_name", "result")
        if not isinstance(intent, str) or not intent.strip():
            raise ValueError("intent must be a non-empty string")
        if not isinstance(output_name, str) or not output_name.strip():
            raise ValueError("output_name must be a non-empty string")
        return goal, intent, output_name

    def synthesize(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Synthesize a bounded UOL candidate; never execute, verify or promote it."""
        if not isinstance(payload, Mapping):
            raise ValueError("goal must be a JSON object")
        goal, intent, output_name = self._goal(payload)
        vault = self._vault_provider()
        imported = self._knowledge(vault, goal.domain) if vault is not None else (
            KnowledgeRuleImporter().import_records(())
        )
        language = UOL(imported.rules)
        synthesizer = CapabilitySynthesizer(language)
        report = synthesizer.synthesize(goal)
        summary = {
            "status": report.status.value,
            "tested_candidates": report.tested,
            "training_passed": report.train_passed,
            "validation_passed": report.validation_passed,
            "rules_imported": imported.imported_count,
            "rounds": report.rounds,
            "blockers": list(report.blockers),
            "counterexamples": [
                {
                    "inputs": dict(item.inputs),
                    "expected": item.expected,
                    "actual": item.actual,
                    "round_found": item.round_found,
                    "reason": item.reason,
                }
                for item in report.counterexamples
            ],
            "probes_passed": report.probes_passed,
        }
        if not report.succeeded:
            return summary

        candidate = synthesizer.to_candidate_capability(
            report, intent=intent, output_name=output_name
        )
        summary.update({
            "status": "candidate_unverified",
            "program": program_dict(report.program),
            "candidate_capability": candidate.to_dict(),
            "verification": "UNVERIFIED",
            "execution_allowed": False,
            "promotion_allowed": False,
            "next_gate": (
                "Run the candidate with an approved host handler, collect execution-bound "
                "independent domain evidence, then satisfy the repeated-observation learning gate."
            ),
        })
        return summary
