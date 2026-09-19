from typing import Dict, List, Any, Tuple

class TransitionRule:
    def __init__(self, conditions: Dict[str, Any], action: str, effects: Dict[str, Any]):
        self.conditions = conditions
        self.action = action
        self.effects = effects
        self.support = 0
        self.violations = 0
        self.creation_step = 0
        
    @property
    def confidence(self) -> float:
        return (self.support + 1) / (self.support + self.violations + 2)

class AdaptiveWorldModelBuilder:
    def __init__(self, max_rules: int = 20):
        self.rules: List[TransitionRule] = []
        self.max_rules = max_rules
        self.step = 0
        
    def predict(self, state: Dict[str, Any], action: str) -> Tuple[Dict[str, Any], float]:
        matching = []
        for r in self.rules:
            if (r.action == '*' or r.action == action) and all(state.get(k) == v for k, v in r.conditions.items()):
                matching.append(r)
                
        if not matching:
            return state.copy(), 0.1
            
        best_rule = max(matching, key=lambda r: r.confidence)
        pred_state = state.copy()
        pred_state.update(best_rule.effects)
        return pred_state, best_rule.confidence
        
    def observe_transition(self, state: Dict[str, Any], action: str, next_state: Dict[str, Any]) -> int:
        self.step += 1
        used_rules = [r for r in self.rules if (r.action == '*' or r.action == action) and all(state.get(k) == v for k, v in r.conditions.items())]
        effects = {k: v for k, v in next_state.items() if state.get(k) != v}
        surprise = len(effects)
        for r in used_rules:
            if all(next_state.get(k) == v for k, v in r.effects.items()):
                r.support += 1
                surprise = max(0, surprise - len(r.effects))
            else:
                r.violations += 1
        if surprise > 0:
            if len(self.rules) >= self.max_rules:
                self.rules.sort(key=lambda x: x.confidence)
                self.rules.pop(0)
            new_rule = TransitionRule(state.copy(), action, effects)
            new_rule.creation_step = self.step
            self.rules.append(new_rule)
        return surprise
        
    def get_rules(self) -> List[TransitionRule]:
        return self.rules
        
    def explain_prediction(self, state: Dict[str, Any], action: str) -> List[TransitionRule]:
        return [r for r in self.rules if (r.action == '*' or r.action == action) and all(state.get(k) == v for k, v in r.conditions.items())]
