import math
from typing import Dict, List, Set, Any, Tuple

class CausalHypothesis:
    def __init__(self, name: str, variables: Set[str], edges: List[Tuple[str, str]], parameters: Dict[Tuple[str, str], float], prior: float):
        self.name = name
        self.variables = variables
        self.edges = edges
        self.parameters = parameters
        self.prior = prior
        self.posterior = prior
        self.predictions: Dict[str, Dict[str, float]] = {}

    def to_dict(self):
        return {
            "name": self.name,
            "variables": list(self.variables),
            "edges": self.edges,
            "parameters": {f"{k[0]}->{k[1]}": v for k, v in self.parameters.items()},
            "prior": self.prior,
            "posterior": self.posterior
        }

class CausalExperimentDesigner:
    def __init__(self, max_hypotheses: int = 20):
        self.hypotheses: List[CausalHypothesis] = []
        self.max_hypotheses = max_hypotheses

    def add_hypothesis(self, hypothesis: CausalHypothesis) -> bool:
        if len(self.hypotheses) >= self.max_hypotheses:
            return False
        self.hypotheses.append(hypothesis)
        self._normalize_posteriors()
        return True

    def observe(self, data: Dict[str, Any]) -> None:
        for h in self.hypotheses:
            likelihood = self.likelihood(h, data)
            h.posterior *= likelihood
        self._normalize_posteriors()

    def likelihood(self, h: CausalHypothesis, data: Dict[str, Any]) -> float:
        if not data:
            return 1.0
        likelihood = 1.0
        for var, value in data.items():
            if var not in h.variables:
                continue
            incoming = [e for e in h.edges if e[1] == var]
            if not incoming:
                continue
            for parent, child in incoming:
                p = h.parameters.get((parent, child), 0.5)
                if value:
                    likelihood *= p
                else:
                    likelihood *= (1 - p)
        return max(likelihood, 1e-12)

    def _normalize_posteriors(self):
        total = sum(h.posterior for h in self.hypotheses)
        if total > 0:
            for h in self.hypotheses:
                h.posterior /= total

    def suggest_experiment(self, possible_interventions: List[str]) -> str:
        if not self.hypotheses or not possible_interventions:
            return "none"
        best_intervention = possible_interventions[0]
        best_eig = -1.0
        current_entropy = self._entropy([h.posterior for h in self.hypotheses])
        for intervention in possible_interventions:
            eig = 0.0
            for outcome in [0, 1]:
                outcome_prob = 0.5
                posteriors = []
                for h in self.hypotheses:
                    mock_data = {intervention: outcome}
                    lik = self.likelihood(h, mock_data)
                    posteriors.append(h.posterior * lik)
                total = sum(posteriors)
                if total > 0:
                    posteriors = [p / total for p in posteriors]
                new_entropy = self._entropy(posteriors)
                eig += outcome_prob * (current_entropy - new_entropy)
            if eig > best_eig:
                best_eig = eig
                best_intervention = intervention
        if best_eig < 1e-5 and len(possible_interventions) > 0:
            return "indistinguishable"
        return best_intervention

    def get_posteriors(self) -> Dict[str, float]:
        return {h.name: h.posterior for h in self.hypotheses}

    def _entropy(self, probs: List[float]) -> float:
        return -sum(p * math.log2(p) if p > 0 else 0 for p in probs)
