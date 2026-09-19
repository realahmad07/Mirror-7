import math
import random
from typing import Dict, List, Set, Optional, Any

class KnownFact:
    def __init__(self, topic: str, content: Any, confidence: float, last_updated: int, source: str):
        self.topic = topic
        self.content = content
        self.confidence = confidence
        self.last_updated = last_updated
        self.source = source
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "content": self.content,
            "confidence": self.confidence,
            "last_updated": self.last_updated,
            "source": self.source
        }

class Question:
    def __init__(self, topic: str, importance: float, difficulty: float, dependencies: List[str] = None):
        self.topic = topic
        self.importance = importance
        self.difficulty = difficulty
        self.dependencies = dependencies or []
        self.attempts = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "importance": self.importance,
            "difficulty": self.difficulty,
            "dependencies": self.dependencies,
            "attempts": self.attempts
        }

class ResearchAction:
    def __init__(self, action_type: str, target_topic: str, expected_info_gain: float, estimated_cost: float, rationale: str):
        self.action_type = action_type
        self.target_topic = target_topic
        self.expected_info_gain = expected_info_gain
        self.estimated_cost = estimated_cost
        self.rationale = rationale

class AutonomousResearch:
    def __init__(self, max_questions: int = 100, max_history: int = 500):
        self.max_questions = max_questions
        self.max_history = max_history
        
        self.known_facts: Dict[str, KnownFact] = {}
        self.open_questions: List[Question] = []
        self.uncertainty_map: Dict[str, float] = {}
        self.exploration_history: List[ResearchAction] = []
        
        self.time_step = 0
        self.epsilon_0 = 0.2
        self.decay_rate = 100

    def add_question(self, question: Question):
        if len(self.open_questions) >= self.max_questions:
            # Drop lowest importance
            self.open_questions.sort(key=lambda q: q.importance, reverse=True)
            self.open_questions.pop()
        self.open_questions.append(question)
        if question.topic not in self.uncertainty_map:
            self.uncertainty_map[question.topic] = 1.0

    def add_fact(self, fact: KnownFact):
        self.known_facts[fact.topic] = fact
        self.uncertainty_map[fact.topic] = 1.0 - fact.confidence
        # Remove resolved questions
        if fact.confidence > 0.8:
            self.open_questions = [q for q in self.open_questions if q.topic != fact.topic]

    def _get_epsilon(self) -> float:
        return self.epsilon_0 / (1 + self.time_step / self.decay_rate)

    def suggest_action(self) -> Optional[ResearchAction]:
        self.time_step += 1
        if not self.open_questions and not self.uncertainty_map:
            return None

        # Filter out questions with unmet dependencies
        known_topics = set(self.known_facts.keys())
        valid_questions = []
        for q in self.open_questions:
            can_pursue = True
            for dep in q.dependencies:
                if dep not in known_topics or self.known_facts[dep].confidence < 0.5:
                    can_pursue = False
                    break
            if can_pursue:
                valid_questions.append(q)

        epsilon = self._get_epsilon()
        if valid_questions and random.random() < epsilon:
            # Exploration
            q = random.choice(valid_questions)
            action = ResearchAction("explore", q.topic, self.uncertainty_map.get(q.topic, 1.0), q.difficulty, "exploration")
            return self._record_action(action)

        best_action = None
        best_score = -1.0

        for q in valid_questions:
            uncertainty = self.uncertainty_map.get(q.topic, 1.0)
            expected_gain = uncertainty * 0.5
            cost = max(q.difficulty, 0.1)
            score = (expected_gain * q.importance) / (cost + 1)
            
            # Penalize repeated attempts
            score /= (q.attempts + 1)

            if score > best_score:
                best_score = score
                best_action = ResearchAction("query", q.topic, expected_gain, cost, "exploitation")

        # Fallback to high uncertainty known topics to verify
        if not best_action and self.uncertainty_map:
            items = sorted(self.uncertainty_map.items(), key=lambda x: x[1], reverse=True)
            top_topic = items[0][0]
            if items[0][1] > 0.5:
                best_action = ResearchAction("verify", top_topic, items[0][1]*0.5, 1.0, "verify uncertainty")

        if best_action:
            for q in self.open_questions:
                if q.topic == best_action.target_topic:
                    q.attempts += 1
            return self._record_action(best_action)

        return None

    def _record_action(self, action: ResearchAction) -> ResearchAction:
        self.exploration_history.append(action)
        if len(self.exploration_history) > self.max_history:
            self.exploration_history.pop(0)
        return action

    def report_result(self, action: ResearchAction, fact: Optional[KnownFact]):
        if fact:
            self.add_fact(fact)
        else:
            # Failed action, increase uncertainty or lower importance
            topic = action.target_topic
            if topic in self.uncertainty_map:
                self.uncertainty_map[topic] = min(1.0, self.uncertainty_map[topic] + 0.1)

    def get_uncertainty_map(self) -> Dict[str, float]:
        return self.uncertainty_map.copy()
