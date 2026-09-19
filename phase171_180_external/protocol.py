"""Transport-neutral external task/evaluation boundary for Mirror 7."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib, json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

@dataclass(frozen=True)
class ExternalEpisode:
    episode_id: str
    observations: Tuple[str, ...]
    actions: Tuple[str, ...]
    outcomes: Tuple[str, ...]
    goal: Optional[str]
    legal_actions: Tuple[str, ...]
    metadata: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "observations": list(self.observations),
            "actions": list(self.actions),
            "outcomes": list(self.outcomes),
            "goal": self.goal,
            "legal_actions": list(self.legal_actions),
            "metadata": dict(self.metadata or {}),
        }

@dataclass(frozen=True)
class ExternalTaskPack:
    pack_id: str
    version: str
    episodes: Tuple[ExternalEpisode, ...]
    public_metadata: Mapping[str, Any]
    evaluator_id: str

    def public_view(self) -> Dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "version": self.version,
            "episodes": [e.to_dict() for e in self.episodes],
            "public_metadata": dict(self.public_metadata),
        }

    def secret_view(self) -> Dict[str, Any]:
        return {
            "evaluator_id": self.evaluator_id,
            "task_pack_fingerprint": task_pack_fingerprint(self),
            "episode_ids": [e.episode_id for e in self.episodes],
        }

@dataclass(frozen=True)
class SealedTaskView:
    episode_id: str
    observation_history: Tuple[str, ...]
    action_history: Tuple[str, ...]
    outcome_history: Tuple[str, ...]
    goal: Optional[str]
    legal_actions: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class EvaluatorResult:
    passed: bool
    completed_episodes: int
    total_episodes: int
    success_rate: float
    contamination: bool
    pack_fingerprint: str
    reason: str

_FORBIDDEN_AGENT_KEYS = frozenset({
    "expected", "answer", "answers", "hidden", "secret",
    "solution", "solutions", "evaluator_id", "task_pack_fingerprint",
})

def build_pack(pack_id: str, version: str, episodes: Iterable[ExternalEpisode],
               *, public_metadata: Optional[Mapping[str, Any]] = None,
               evaluator_id: str = "external-evaluator") -> ExternalTaskPack:
    episode_tuple = tuple(episodes)
    if not episode_tuple:
        raise ValueError("task pack must contain at least one episode")
    ids = [e.episode_id for e in episode_tuple]
    if any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("episode ids must be non-empty and unique")
    return ExternalTaskPack(
        pack_id, version, episode_tuple, dict(public_metadata or {}), evaluator_id
    )

def sanitize_for_agent(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively reject evaluator-only fields instead of silently stripping them."""
    def walk(value: Any, path: str) -> Any:
        if isinstance(value, Mapping):
            output = {}
            for key, child in value.items():
                key_s = str(key).lower()
                if key_s in _FORBIDDEN_AGENT_KEYS:
                    raise ValueError(f"forbidden evaluator field at {path}/{key}")
                output[str(key)] = walk(child, f"{path}/{key}")
            return output
        if isinstance(value, (list, tuple)):
            return [walk(v, f"{path}[]") for v in value]
        return value
    return walk(dict(payload), "")

def task_pack_fingerprint(pack: ExternalTaskPack) -> str:
    canonical = json.dumps(
        pack.public_view(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()

def make_sealed_view(episode: ExternalEpisode) -> SealedTaskView:
    # Human-readable target labels remain evaluator-side. The sealed agent view
    # receives deterministic opaque action handles instead.
    opaque = tuple(f"action_{i:03d}" for i in range(len(episode.legal_actions)))
    return SealedTaskView(
        episode.episode_id, episode.observations, episode.actions,
        episode.outcomes, episode.goal, opaque
    )

def save_pack(pack: ExternalTaskPack, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pack_id": pack.pack_id,
        "version": pack.version,
        "episodes": [e.to_dict() for e in pack.episodes],
        "public_metadata": dict(pack.public_metadata),
        "evaluator_id": pack.evaluator_id,
    }
    target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

def load_pack(path: str | Path) -> ExternalTaskPack:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("task pack root must be an object")
    episodes = []
    for raw in payload.get("episodes", []):
        if not isinstance(raw, Mapping):
            raise ValueError("episode entry must be an object")
        episodes.append(ExternalEpisode(
            str(raw["episode_id"]),
            tuple(map(str, raw.get("observations", []))),
            tuple(map(str, raw.get("actions", []))),
            tuple(map(str, raw.get("outcomes", []))),
            None if raw.get("goal") is None else str(raw["goal"]),
            tuple(map(str, raw.get("legal_actions", []))),
            dict(raw.get("metadata", {})),
        ))
    return build_pack(
        str(payload["pack_id"]), str(payload["version"]), episodes,
        public_metadata=dict(payload.get("public_metadata", {})),
        evaluator_id=str(payload.get("evaluator_id", "external-evaluator")),
    )

def evaluate_episode_outcomes(expected_success: Sequence[bool],
                              completed_episode_ids: Sequence[str],
                              total_episodes: int, *, contamination: bool,
                              pack_fingerprint: str) -> EvaluatorResult:
    completed = len(set(completed_episode_ids))
    successes = sum(bool(x) for x in expected_success)
    success_rate = successes / max(1, total_episodes)
    passed = (
        total_episodes > 0 and completed == total_episodes
        and successes == total_episodes and not contamination
    )
    return EvaluatorResult(
        passed, completed, total_episodes, success_rate, bool(contamination),
        pack_fingerprint, "pass" if passed else "external_gate_failed"
    )
