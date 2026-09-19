"""
Open-World Concept Discovery
Increments a hierarchical tree of concepts based on categorical utility.
"""

import time
import math
from typing import Dict, Any, List, Optional, Tuple, Set

Observation = Dict[str, Any]

class ConceptNode:
    def __init__(self, node_id: str, parent: Optional['ConceptNode'] = None):
        self.node_id = node_id
        self.parent = parent
        self.children: List['ConceptNode'] = []
        self.instance_count: int = 0
        self.feature_counts: Dict[str, Dict[Any, int]] = {}
        self.confidence_score: float = 0.0
        self.creation_timestamp: float = time.time()
        
    def add_instance(self, observation: Observation) -> None:
        self.instance_count += 1
        for feature, value in observation.items():
            if feature not in self.feature_counts:
                self.feature_counts[feature] = {}
            if value not in self.feature_counts[feature]:
                self.feature_counts[feature][value] = 0
            self.feature_counts[feature][value] += 1
            
    def remove_instance(self, observation: Observation) -> None:
        if self.instance_count > 0:
            self.instance_count -= 1
        for feature, value in observation.items():
            if feature in self.feature_counts and value in self.feature_counts[feature]:
                self.feature_counts[feature][value] -= 1
                if self.feature_counts[feature][value] == 0:
                    del self.feature_counts[feature][value]
                if not self.feature_counts[feature]:
                    del self.feature_counts[feature]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "instance_count": self.instance_count,
            "feature_counts": self.feature_counts,
            "children": [child.to_dict() for child in self.children]
        }

class ConceptTree:
    def __init__(self, max_depth: int = 5, max_nodes: int = 100):
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.root = ConceptNode("root")
        self.node_counter = 0
        self.total_nodes = 1
        self.all_nodes: Dict[str, ConceptNode] = {"root": self.root}
        
    def _generate_id(self) -> str:
        self.node_counter += 1
        return f"node_{self.node_counter}"
        
    def _get_probability(self, node: ConceptNode, feature: str, value: Any) -> float:
        if node.instance_count == 0:
            return 0.0
        counts = node.feature_counts.get(feature, {})
        return counts.get(value, 0) / node.instance_count
        
    def _compute_category_utility(self, parent: ConceptNode, children: List[ConceptNode]) -> float:
        if not children:
            return 0.0
        
        k = len(children)
        cu = 0.0
        
        all_features = set()
        for child in children:
            all_features.update(child.feature_counts.keys())
            
        for child in children:
            prob_child = child.instance_count / max(1, parent.instance_count)
            term = 0.0
            for feature in all_features:
                child_values = child.feature_counts.get(feature, {}).keys()
                parent_values = parent.feature_counts.get(feature, {}).keys()
                all_values = set(child_values).union(parent_values)
                
                for val in all_values:
                    p_f_given_c = self._get_probability(child, feature, val)
                    p_f = self._get_probability(parent, feature, val)
                    term += (p_f_given_c ** 2) - (p_f ** 2)
            cu += prob_child * term
            
        return cu / k

    def observe(self, observation: Observation) -> None:
        self.root.add_instance(observation)
        self._incorporate(self.root, observation, 1)
        
    def _incorporate(self, node: ConceptNode, observation: Observation, depth: int) -> None:
        if not node.children:
            if self.total_nodes < self.max_nodes and depth < self.max_depth:
                new_id = self._generate_id()
                new_child = ConceptNode(new_id, node)
                new_child.add_instance(observation)
                node.children.append(new_child)
                self.all_nodes[new_id] = new_child
                self.total_nodes += 1
            return

        best_cu = -1.0
        best_action = "none"
        best_child = None
        
        for child in node.children:
            child.add_instance(observation)
            cu = self._compute_category_utility(node, node.children)
            if cu > best_cu:
                best_cu = cu
                best_action = "incorporate"
                best_child = child
            child.remove_instance(observation)
            
        if self.total_nodes < self.max_nodes and depth < self.max_depth:
            new_id = self._generate_id()
            new_child = ConceptNode(new_id, node)
            new_child.add_instance(observation)
            node.children.append(new_child)
            cu = self._compute_category_utility(node, node.children)
            if cu > best_cu:
                best_cu = cu
                best_action = "create"
            node.children.pop()
            
        if best_action == "incorporate" and best_child is not None:
            best_child.add_instance(observation)
            self._incorporate(best_child, observation, depth + 1)
        elif best_action == "create":
            new_id = self._generate_id()
            new_child = ConceptNode(new_id, node)
            new_child.add_instance(observation)
            node.children.append(new_child)
            self.all_nodes[new_id] = new_child
            self.total_nodes += 1
            
    def classify(self, observation: Observation) -> str:
        return self._classify_recursive(self.root, observation)
        
    def _classify_recursive(self, node: ConceptNode, observation: Observation) -> str:
        if not node.children:
            return node.node_id
            
        best_cu = -1.0
        best_child = node.children[0]
        
        for child in node.children:
            child.add_instance(observation)
            node.add_instance(observation)
            cu = self._compute_category_utility(node, node.children)
            if cu > best_cu:
                best_cu = cu
                best_child = child
            child.remove_instance(observation)
            node.remove_instance(observation)
            
        return self._classify_recursive(best_child, observation)
        
    def get_concepts(self) -> List[str]:
        return list(self.all_nodes.keys())
