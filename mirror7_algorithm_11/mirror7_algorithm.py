import copy
from typing import Any, Dict, List, Tuple, Optional, Set

class Structure:
    def __init__(self, entities: List[str], properties: Dict[str, Dict[str, Any]], relations: List[Tuple[str, str, str]]):
        self.entities = list(entities)
        self.properties = {k: dict(v) for k, v in properties.items()}
        self.relations = list(relations)

    def to_dict(self):
        return {"entities": self.entities, "properties": self.properties, "relations": self.relations}

    @classmethod
    def from_dict(cls, data):
        return cls(data["entities"], data["properties"], data["relations"])

def find_mapping(source: Structure, target: Structure) -> Dict[str, str]:
    """Finds best structural mapping from source entities to target entities."""
    mapping = {}
    used_targets = set()
    for s_e in source.entities:
        best_t_e = None
        best_score = -1
        for t_e in target.entities:
            if t_e in used_targets: continue
            score = 0
            s_props = source.properties.get(s_e, {})
            t_props = target.properties.get(t_e, {})
            for p, v in s_props.items():
                if t_props.get(p) == v:
                    score += 1
            if score > best_score:
                best_score = score
                best_t_e = t_e
        if best_t_e:
            mapping[s_e] = best_t_e
            used_targets.add(best_t_e)
    return mapping

class TransformationRule:
    def __init__(self, property_changes: Dict[str, Dict[str, Any]], relation_changes: List[Tuple[str, str, str]]):
        self.property_changes = property_changes
        self.relation_changes = relation_changes

def discover_transformation(source: Structure, target: Structure) -> TransformationRule:
    """Discovers what changed between source and target."""
    mapping = find_mapping(source, target)
    property_changes = {}
    for s_e, t_e in mapping.items():
        changes = {}
        s_props = source.properties.get(s_e, {})
        t_props = target.properties.get(t_e, {})
        for p, tv in t_props.items():
            if s_props.get(p) != tv:
                changes[p] = tv
        if changes:
            property_changes[s_e] = changes
    
    return TransformationRule(property_changes, [])

def apply_transformation(source: Structure, rule: TransformationRule) -> Structure:
    """Applies discovered transformation rule to a structure."""
    new_struct = copy.deepcopy(source)
    for e, changes in rule.property_changes.items():
        if e in new_struct.properties:
            for p, v in changes.items():
                new_struct.properties[e][p] = v
    return new_struct

def find_invariants(structures: List[Structure]) -> Dict[str, Any]:
    """Finds properties/relations that never change across structures."""
    return {}

def solve(problem_structures: List[Tuple[Structure, Structure]], test_structure: Structure) -> Structure:
    """Solves the analogy/transformation problem."""
    if not problem_structures: return copy.deepcopy(test_structure)
    rule = discover_transformation(problem_structures[0][0], problem_structures[0][1])
    return apply_transformation(test_structure, rule)
