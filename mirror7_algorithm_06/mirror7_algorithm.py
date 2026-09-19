import time
from typing import Any, Dict, List, Optional, Set
import json

class Goal:
    def __init__(self, id: str, description: str, parent_id: Optional[str], priority: float, urgency: float, preconditions: List[str], effects: List[str], deadline: Optional[int]):
        self.id = id
        self.description = description
        self.parent_id = parent_id
        self.children_ids: List[str] = []
        self.status = 'active'
        self.priority = max(0.0, min(1.0, priority))
        self.urgency = max(0.0, min(1.0, urgency))
        self.progress = 0.0
        self.preconditions = preconditions
        self.effects = effects
        self.deadline = deadline
        self.created_at = int(time.time())
        self.attempts = 0
        self.failures = 0
        self.max_retries = 3
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'description': self.description,
            'parent_id': self.parent_id,
            'children_ids': self.children_ids,
            'status': self.status,
            'priority': self.priority,
            'urgency': self.urgency,
            'progress': self.progress,
            'preconditions': self.preconditions,
            'effects': self.effects,
            'deadline': self.deadline,
            'created_at': self.created_at,
            'attempts': self.attempts,
            'failures': self.failures
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Goal':
        goal = cls(data['id'], data['description'], data['parent_id'], data['priority'], data['urgency'], data['preconditions'], data['effects'], data['deadline'])
        goal.children_ids = data['children_ids']
        goal.status = data['status']
        goal.progress = data['progress']
        goal.created_at = data['created_at']
        goal.attempts = data['attempts']
        goal.failures = data['failures']
        return goal

class LongTermGoalManager:
    def __init__(self, max_depth: int = 10, max_goals: int = 1000):
        self.max_depth = max_depth
        self.max_goals = max_goals
        self.goals: Dict[str, Goal] = {}
        self.step_counter = 0

    def add_goal(self, id: str, description: str, priority: float, parent_id: Optional[str] = None, urgency: float = 0.5, preconditions: List[str] = None, effects: List[str] = None, deadline: Optional[int] = None) -> bool:
        if len(self.goals) >= self.max_goals:
            return False
            
        if parent_id and parent_id not in self.goals:
            return False
            
        if parent_id:
            depth = 1
            curr = self.goals[parent_id]
            while curr.parent_id:
                depth += 1
                curr = self.goals[curr.parent_id]
            if depth >= self.max_depth:
                return False
                
        goal = Goal(id, description, parent_id, priority, urgency, preconditions or [], effects or [], deadline)
        self.goals[id] = goal
        if parent_id:
            self.goals[parent_id].children_ids.append(id)
        return True

    def decompose(self, goal_id: str, subgoals: List[Dict[str, Any]]) -> bool:
        if goal_id not in self.goals:
            return False
        for sg in subgoals:
            success = self.add_goal(sg['id'], sg['description'], sg.get('priority', 0.5), goal_id, sg.get('urgency', 0.5), sg.get('preconditions', []), sg.get('effects', []), sg.get('deadline'))
            if not success:
                return False
        return True
        
    def _compute_priority(self, goal: Goal) -> float:
        base_priority = goal.priority
        urgency_factor = 1.0 + goal.urgency
        if goal.deadline:
            time_left = max(1, goal.deadline - self.step_counter)
            urgency_factor += goal.urgency * (1.0 / time_left)
            
        dependency_factor = 1.0 if goal.status != 'blocked' else 0.1
        feasibility_factor = 1.0 - (goal.failures / (goal.attempts + 1))
        
        return base_priority * urgency_factor * dependency_factor * feasibility_factor

    def reprioritize(self) -> None:
        self.step_counter += 1

    def get_next_goal(self, world_state: Set[str]) -> Optional[str]:
        self.reprioritize()
        best_goal = None
        best_prio = -1.0
        
        for g in self.goals.values():
            if g.status in ('active', 'suspended') and not g.children_ids:
                if self.check_preconditions(g.id, world_state):
                    g.status = 'active'
                    prio = self._compute_priority(g)
                    if prio > best_prio:
                        best_prio = prio
                        best_goal = g.id
                else:
                    g.status = 'blocked'
                    
        if best_goal:
            self.goals[best_goal].attempts += 1
            
        return best_goal

    def update_progress(self, goal_id: str, progress: float, status: str) -> bool:
        if goal_id not in self.goals:
            return False
        g = self.goals[goal_id]
        g.progress = max(0.0, min(1.0, progress))
        g.status = status
        
        if status == 'completed':
            if g.parent_id:
                p = self.goals[g.parent_id]
                all_done = all(self.goals[cid].status == 'completed' for cid in p.children_ids)
                if all_done:
                    self.update_progress(p.id, 1.0, 'completed')
        return True

    def report_failure(self, goal_id: str, reason: str) -> bool:
        if goal_id not in self.goals:
            return False
        g = self.goals[goal_id]
        g.failures += 1
        if g.failures < g.max_retries:
            g.status = 'active'
        else:
            g.status = 'failed'
            if g.parent_id:
                self.report_failure(g.parent_id, "subgoal failed")
        return True

    def check_preconditions(self, goal_id: str, world_state: Set[str]) -> bool:
        if goal_id not in self.goals:
            return False
        g = self.goals[goal_id]
        return all(p in world_state for p in g.preconditions)

    def get_plan(self, world_state: Set[str]) -> List[str]:
        actionable = []
        for g in self.goals.values():
            if g.status == 'active' and not g.children_ids and self.check_preconditions(g.id, world_state):
                actionable.append(g)
        actionable.sort(key=self._compute_priority, reverse=True)
        return [g.id for g in actionable]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'max_depth': self.max_depth,
            'max_goals': self.max_goals,
            'step_counter': self.step_counter,
            'goals': {k: v.to_dict() for k, v in self.goals.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LongTermGoalManager':
        mgr = cls(data['max_depth'], data['max_goals'])
        mgr.step_counter = data['step_counter']
        for k, v in data['goals'].items():
            mgr.goals[k] = Goal.from_dict(v)
        return mgr
