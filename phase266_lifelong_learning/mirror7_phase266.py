from typing import Dict, List, Optional, Tuple

class LifelongMemory:
    """Consolidates and protects valid rules from interference across environments."""
    
    def __init__(self, capacity: int = 100):
        if not isinstance(capacity,int) or isinstance(capacity,bool) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        # rules: (state, action) -> (next_state, confidence, last_used, env_id)
        self.rules = {}
        self.time = 0
        
    def observe(self, state: tuple, action: str, next_state: tuple, env_id: str):
        self.time += 1
        key = (state, action)
        
        if key in self.rules:
            old_next, conf, _, old_env = self.rules[key]
            if old_next == next_state:
                self.rules[key] = (next_state, min(1.0, conf + 0.1), self.time, env_id)
            else:
                context_key = (state, action, env_id)
                self.rules[context_key] = (next_state, 0.5, self.time, env_id)
        else:
            self.rules[key] = (next_state, 0.5, self.time, env_id)
            
        self._evict()
        
    def predict(self, state: tuple, action: str, current_env: Optional[str] = None) -> Optional[tuple]:
        if current_env:
            context_key = (state, action, current_env)
            if context_key in self.rules:
                return self.rules[context_key][0]
        key = (state, action)
        if key in self.rules:
            return self.rules[key][0]
        return None

    def _evict(self):
        if len(self.rules) > self.capacity:
            sorted_rules = sorted(self.rules.items(), key=lambda x: (x[1][1], x[1][2]))
            del self.rules[sorted_rules[0][0]]
