import hashlib
from typing import Any, Dict, List, Optional, Tuple

class SafetyViolationError(Exception):
    pass

class SafetyValidator:
    def __init__(self, allowed_actions: set):
        self.allowed_actions = allowed_actions

    def sanitize(self, action: str, args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        if action not in self.allowed_actions:
            raise SafetyViolationError(f"Action '{action}' is strictly prohibited.")
        # Ensure args only contain simple types
        for k, v in args.items():
            if not isinstance(v, (int, float, str, bool)):
                raise SafetyViolationError(f"Argument '{k}' contains forbidden type {type(v)}.")
        return action, args

class TrajectoryReproducer:
    def __init__(self, seed: int):
        self.seed = seed
        self.history = []
        self._hash = hashlib.sha256(str(seed).encode())
        
    def record_transition(self, state: Dict[str, Any], action: str, observation: Any):
        record = f"{sorted(state.items())}|{action}|{observation}"
        self.history.append(record)
        self._hash.update(record.encode())
        
    def get_fingerprint(self) -> str:
        return self._hash.hexdigest()

class ValidatedAgent:
    def __init__(self, allowed_actions: List[str], seed: int):
        self.validator = SafetyValidator(set(allowed_actions))
        self.reproducer = TrajectoryReproducer(seed)
        self.state = {"step": 0}

    def act(self, action: str, args: Dict[str, Any], env_obs: Any):
        try:
            safe_action, safe_args = self.validator.sanitize(action, args)
            self.state["step"] += 1
            self.reproducer.record_transition(self.state, safe_action, env_obs)
            return True
        except SafetyViolationError:
            # Fail closed
            return False
