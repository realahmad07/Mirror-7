"""
Mirror 7 — Phase 35: Action Model

Explicit, inspectable representation of:
  state → action → predicted consequence

Provides:
  ActionSpec      — immutable description of an action (preconditions, effects, reversible, risk)
  ActionOutcome   — record of one execution (predicted vs actual effect)
  ActionMemory    — experiential store: (state, action) → observed outcomes
  ActionLearner   — online learner: updates predictions from observed discrepancies
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


class ActionStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class EffectMatch(Enum):
    EXACT = "exact"
    PARTIAL = "partial"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


class ReversibilityClass(Enum):
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"
    CONDITIONAL = "conditional"


def _dict_to_fset(d: Dict[str, Any]) -> FrozenSet[Tuple[str, Any]]:
    return frozenset((k, v) for k, v in sorted(d.items()))


def _compute_action_hash(name, preconditions, expected_effects, reversibility):
    blob = json.dumps({
        "name": name,
        "preconditions": sorted(str(x) for x in preconditions),
        "expected_effects": sorted(str(x) for x in expected_effects),
        "reversibility": reversibility,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _compute_outcome_hash(action_name, state_key, predicted_effects, actual_effects, status):
    blob = json.dumps({
        "action": action_name,
        "state": state_key,
        "predicted": sorted(str(x) for x in predicted_effects),
        "actual": sorted(str(x) for x in actual_effects),
        "status": status,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True)
class ActionSpec:
    name: str
    preconditions: FrozenSet[Tuple[str, Any]]
    expected_effects: FrozenSet[Tuple[str, Any]]
    reversibility: ReversibilityClass
    risk_level: float
    description: str
    action_hash: str

    def __post_init__(self):
        if not self.name:
            raise ValueError("ActionSpec name cannot be empty")
        if not 0.0 <= self.risk_level <= 1.0:
            raise ValueError(f"risk_level must be in [0,1], got {self.risk_level}")

    def check_preconditions(self, state):
        return all(state.get(key) == value for key, value in self.preconditions)

    def predict_effects(self, state):
        predicted = dict(state)
        for key, value in self.expected_effects:
            predicted[key] = value
        return predicted

    def to_dict(self):
        return {
            "name": self.name,
            "preconditions": sorted(list(self.preconditions), key=str),
            "expected_effects": sorted(list(self.expected_effects), key=str),
            "reversibility": self.reversibility.value,
            "risk_level": round(self.risk_level, 6),
            "description": self.description,
            "action_hash": self.action_hash,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            preconditions=frozenset(tuple(x) for x in data["preconditions"]),
            expected_effects=frozenset(tuple(x) for x in data["expected_effects"]),
            reversibility=ReversibilityClass(data["reversibility"]),
            risk_level=float(data["risk_level"]),
            description=data["description"],
            action_hash=data["action_hash"],
        )

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def create(cls, name, preconditions, expected_effects,
               reversibility=ReversibilityClass.REVERSIBLE,
               risk_level=0.0, description=""):
        prec_fset = _dict_to_fset(preconditions)
        eff_fset = _dict_to_fset(expected_effects)
        ahash = _compute_action_hash(name, prec_fset, eff_fset, reversibility.value)
        return cls(name, prec_fset, eff_fset, reversibility, risk_level, description, ahash)


@dataclass(frozen=True)
class ActionOutcome:
    action_name: str
    state_before: FrozenSet[Tuple[str, Any]]
    predicted_effects: FrozenSet[Tuple[str, Any]]
    actual_effects: FrozenSet[Tuple[str, Any]]
    status: ActionStatus
    effect_match: EffectMatch
    discrepancy: FrozenSet[Tuple[str, Any]]
    timestamp: float
    outcome_hash: str

    def to_dict(self):
        return {
            "action_name": self.action_name,
            "state_before": sorted(list(self.state_before), key=str),
            "predicted_effects": sorted(list(self.predicted_effects), key=str),
            "actual_effects": sorted(list(self.actual_effects), key=str),
            "status": self.status.value,
            "effect_match": self.effect_match.value,
            "discrepancy": sorted(list(self.discrepancy), key=str),
            "timestamp": self.timestamp,
            "outcome_hash": self.outcome_hash,
        }

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True)


def compute_outcome(action, state_before, state_after, status=ActionStatus.SUCCESS):
    state_fset = _dict_to_fset(state_before)
    pred_eff = action.expected_effects
    actual_changes = {}
    for key in set(state_before) | set(state_after):
        before_val = state_before.get(key)
        after_val = state_after.get(key)
        if before_val != after_val:
            actual_changes[key] = after_val
    actual_eff = _dict_to_fset(actual_changes)

    pred_dict = dict(pred_eff)
    actual_dict = dict(actual_eff)
    disc_items = []
    all_effect_keys = set(pred_dict) | set(actual_dict)
    for key in sorted(all_effect_keys):
        pv = pred_dict.get(key)
        av = actual_dict.get(key)
        if pv != av:
            disc_items.append((key, (pv, av)))
    disc_fset = frozenset(disc_items)

    if status == ActionStatus.INVALID:
        match_class = EffectMatch.UNKNOWN
    elif not pred_eff and not actual_eff:
        match_class = EffectMatch.EXACT
    elif pred_eff == actual_eff:
        match_class = EffectMatch.EXACT
    elif len(disc_items) < len(all_effect_keys):
        match_class = EffectMatch.PARTIAL
    else:
        match_class = EffectMatch.MISMATCH

    ohash = _compute_outcome_hash(
        action.name, str(sorted(state_fset)), pred_eff, actual_eff, status.value
    )
    return ActionOutcome(
        action.name, state_fset, pred_eff, actual_eff, status, match_class,
        disc_fset, time.time(), ohash
    )


class ActionMemory:
    def __init__(self):
        self._records = {}
        self._seen_hashes = set()

    def record(self, outcome):
        if outcome.outcome_hash in self._seen_hashes:
            return False
        self._seen_hashes.add(outcome.outcome_hash)
        key = (outcome.action_name, str(sorted(outcome.state_before)))
        self._records.setdefault(key, []).append(outcome)
        return True

    def outcomes_for(self, action_name, state):
        key = (action_name, str(sorted(_dict_to_fset(state))))
        return list(self._records.get(key, []))

    def all_actions(self):
        return sorted(set(a for a, _ in self._records))

    def predict_effects(self, action_name, state, spec=None):
        observed = self.outcomes_for(action_name, state)
        if not observed:
            if spec is not None:
                return spec.expected_effects, 0.5
            return None, 0.0
        votes = {}
        for outcome in observed:
            votes[outcome.actual_effects] = votes.get(outcome.actual_effects, 0) + 1
        best_pattern = max(votes, key=lambda key: votes[key])
        return best_pattern, votes[best_pattern] / len(observed)

    def success_rate(self, action_name, state):
        observed = self.outcomes_for(action_name, state)
        if not observed:
            return 0.5
        return sum(o.status == ActionStatus.SUCCESS for o in observed) / len(observed)

    def total_observations(self):
        return sum(len(v) for v in self._records.values())

    def to_dict(self):
        return {
            "version": "1.0",
            "records": {
                f"{key[0]}|{key[1]}": [o.to_dict() for o in value]
                for key, value in sorted(self._records.items())
            },
        }

    def to_json(self, indent=None):
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)


class ActionLearner:
    def __init__(self):
        self.memory = ActionMemory()
        self._specs = {}
        self._update_count = 0

    def register_action(self, spec):
        self._specs[spec.name] = spec

    def get_spec(self, name):
        return self._specs.get(name)

    def execute_and_learn(self, action_name, state_before, state_after):
        spec = self._specs.get(action_name)
        if spec is not None and not spec.check_preconditions(state_before):
            status = ActionStatus.INVALID
        elif state_before == state_after:
            status = ActionStatus.FAILURE
        else:
            status = ActionStatus.SUCCESS

        outcome = compute_outcome(
            spec if spec is not None else ActionSpec.create(action_name, {}, {}),
            state_before, state_after, status
        )
        self.memory.record(outcome)
        self._update_count += 1
        return outcome

    def predict(self, action_name, state):
        spec = self._specs.get(action_name)
        predicted_eff, confidence = self.memory.predict_effects(action_name, state, spec)
        if predicted_eff is None:
            return None, 0.0
        predicted_state = dict(state)
        for key, value in predicted_eff:
            predicted_state[key] = value
        return predicted_state, confidence

    def discrepancy_report(self, outcome):
        spec = self._specs.get(outcome.action_name)
        return {
            "action": outcome.action_name,
            "status": outcome.status.value,
            "effect_match": outcome.effect_match.value,
            "predicted_effects": dict(spec.expected_effects) if spec else {},
            "actual_effects": dict(outcome.actual_effects),
            "discrepancy": dict(outcome.discrepancy),
            "update_count": self._update_count,
        }

    def explain_action(self, action_name):
        spec = self._specs.get(action_name)
        observed = self.memory.all_actions()
        return {
            "action": action_name,
            "registered_spec": spec.to_dict() if spec else None,
            "total_observations": self.memory.total_observations(),
            "all_observed_actions": observed,
            "in_memory": action_name in observed,
        }
