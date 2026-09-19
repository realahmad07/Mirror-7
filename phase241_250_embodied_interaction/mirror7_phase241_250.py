from typing import Any, Dict, List, Optional, Tuple

class InvalidActionError(Exception):
    pass

class SafetyBoundary:
    def __init__(self, allowed_actions: List[str]):
        self.allowed_actions = set(allowed_actions)
        
    def validate(self, action: str):
        if action not in self.allowed_actions:
            raise InvalidActionError(f"Action '{action}' is not permitted.")

class ExternalEnvironmentAdapter:
    def __init__(self):
        self.state = {"position": (0, 0), "objects": ["apple", "box"]}
        self.history = []
        
    def execute(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        self.history.append((action, args))
        if action == "move":
            dx, dy = args.get("dx", 0), args.get("dy", 0)
            x, y = self.state["position"]
            self.state["position"] = (x + dx, y + dy)
            return {"status": "success", "observation": f"moved to {self.state['position']}"}
        elif action == "look":
            return {"status": "success", "observation": self.state["objects"]}
        else:
            return {"status": "error", "observation": "unknown effect"}

class EmbodiedAgent:
    def __init__(self, env: ExternalEnvironmentAdapter, allowed_actions: List[str]):
        self.env = env
        self.safety = SafetyBoundary(allowed_actions)
        self.internal_state = {"known_position": None, "seen_objects": []}
        
    def act(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        # Fail closed on invalid actions
        self.safety.validate(action)
        
        result = self.env.execute(action, args)
        if result["status"] == "success":
            # Update internal state based on partial observation
            if action == "move":
                pass # Wait for explicit observation or assume success
            elif action == "look":
                self.internal_state["seen_objects"] = result["observation"]
        return result
