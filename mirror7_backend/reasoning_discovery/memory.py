"""
Strategy Memory Module.
Stores successful strategies, their provenance, domain applicability, and known limitations.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from .schema import ReasoningStrategy, ExecutionResult


class StrategyMemory:
    """Persistent storage for discovered strategies and their operational profiles."""

    def __init__(self, storage_path: str | Path | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self._strategies: dict[str, ReasoningStrategy] = {}
        self._domain_index: dict[str, list[str]] = {}
        self._execution_records: dict[str, list[dict[str, Any]]] = {}
        self._known_limitations: dict[str, list[str]] = {}

        if self.storage_path and self.storage_path.exists():
            self.load()

    def store_strategy(
        self,
        strategy: ReasoningStrategy,
        domain: str,
        execution_result: ExecutionResult | None = None,
        limitations: list[str] | None = None,
    ) -> None:
        self._strategies[strategy.strategy_id] = strategy
        self._domain_index.setdefault(domain, [])
        if strategy.strategy_id not in self._domain_index[domain]:
            self._domain_index[domain].append(strategy.strategy_id)

        if execution_result:
            rec = {
                "passed": execution_result.passed_tests,
                "total": execution_result.total_tests,
                "time_ms": execution_result.execution_time_ms,
                "is_correct": execution_result.is_correct,
            }
            self._execution_records.setdefault(strategy.strategy_id, []).append(rec)

        if limitations:
            self._known_limitations.setdefault(strategy.strategy_id, []).extend(limitations)

        if self.storage_path:
            self.save()

    def get_strategy(self, strategy_id: str) -> ReasoningStrategy | None:
        return self._strategies.get(strategy_id)

    def find_by_domain(self, domain: str) -> list[ReasoningStrategy]:
        ids = self._domain_index.get(domain, [])
        return [self._strategies[i] for i in ids if i in self._strategies]

    def list_all(self) -> list[ReasoningStrategy]:
        return list(self._strategies.values())

    def get_limitations(self, strategy_id: str) -> list[str]:
        return self._known_limitations.get(strategy_id, [])

    def save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "strategies": {
                sid: {
                    "strategy_id": s.strategy_id,
                    "name": s.name,
                    "procedure_code": s.procedure_code,
                    "entry_point": s.entry_point,
                    "rationale": s.rationale,
                    "version": s.version,
                    "parent_strategy_id": s.parent_strategy_id,
                    "complexity_class": s.complexity_class,
                    "tags": list(s.tags),
                }
                for sid, s in self._strategies.items()
            },
            "domain_index": self._domain_index,
            "execution_records": self._execution_records,
            "limitations": self._known_limitations,
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        with open(self.storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sid, s_dict in data.get("strategies", {}).items():
            strat = ReasoningStrategy(
                strategy_id=s_dict["strategy_id"],
                name=s_dict["name"],
                procedure_code=s_dict["procedure_code"],
                entry_point=s_dict["entry_point"],
                rationale=s_dict["rationale"],
                version=s_dict.get("version", 1),
                parent_strategy_id=s_dict.get("parent_strategy_id"),
                complexity_class=s_dict.get("complexity_class", "unknown"),
                tags=tuple(s_dict.get("tags", ())),
            )
            self._strategies[sid] = strat
        self._domain_index = data.get("domain_index", {})
        self._execution_records = data.get("execution_records", {})
        self._known_limitations = data.get("limitations", {})
