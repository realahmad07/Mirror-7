import time
import math
import json
from typing import Any, Dict, List, Optional, Set, Tuple

class MemoryNode:
    def __init__(self, id: str, node_type: str, label: str, attributes: Dict[str, Any], confidence: float, provenance: str, created_at: int):
        self.id = id
        self.node_type = node_type
        self.label = label
        self.attributes = attributes
        self.confidence = max(0.0, min(1.0, confidence))
        self.provenance = provenance
        self.created_at = created_at
        self.last_accessed = created_at
        self.access_count = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'node_type': self.node_type,
            'label': self.label,
            'attributes': self.attributes,
            'confidence': self.confidence,
            'provenance': self.provenance,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNode':
        node = cls(data['id'], data['node_type'], data['label'], data['attributes'], data['confidence'], data['provenance'], data['created_at'])
        node.last_accessed = data.get('last_accessed', node.created_at)
        node.access_count = data.get('access_count', 1)
        return node

class MemoryEdge:
    def __init__(self, source_id: str, target_id: str, relation: str, confidence: float, provenance: str, weight: float, created_at: int):
        self.source_id = source_id
        self.target_id = target_id
        self.relation = relation
        self.confidence = max(0.0, min(1.0, confidence))
        self.provenance = provenance
        self.weight = weight
        self.created_at = created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation': self.relation,
            'confidence': self.confidence,
            'provenance': self.provenance,
            'weight': self.weight,
            'created_at': self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEdge':
        return cls(data['source_id'], data['target_id'], data['relation'], data['confidence'], data['provenance'], data['weight'], data['created_at'])

class SemanticMemoryGraph:
    def __init__(self, max_nodes: int = 1000, max_edges: int = 5000):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: List[MemoryEdge] = []
        
    def _current_time(self) -> int:
        return int(time.time())

    def _utility(self, node: MemoryNode) -> float:
        now = self._current_time()
        recency = 1.0 / (1.0 + (now - node.last_accessed) / 86400.0)
        return node.confidence * math.log(1 + node.access_count) * recency

    def _evict_nodes(self, count: int) -> None:
        if not self.nodes: return
        sorted_nodes = sorted(self.nodes.values(), key=self._utility)
        for i in range(min(count, len(sorted_nodes))):
            target = sorted_nodes[i]
            # Protected
            if target.confidence > 0.9 and target.access_count > 10:
                continue
            del self.nodes[target.id]
        
        # Cleanup edges
        valid_nodes = set(self.nodes.keys())
        self.edges = [e for e in self.edges if e.source_id in valid_nodes and e.target_id in valid_nodes]

    def _evict_edges(self, count: int) -> None:
        if not self.edges: return
        self.edges = sorted(self.edges, key=lambda e: e.confidence)[count:]

    def add_node(self, id: str, node_type: str, label: str, attributes: Dict[str, Any], confidence: float, provenance: str) -> bool:
        if len(self.nodes) >= self.max_nodes and id not in self.nodes:
            self._evict_nodes(max(1, int(self.max_nodes * 0.1)))
            if len(self.nodes) >= self.max_nodes:
                return False
        
        if id in self.nodes:
            node = self.nodes[id]
            node.attributes.update(attributes)
            node.confidence = (node.confidence + confidence) / 2.0
            node.last_accessed = self._current_time()
            node.access_count += 1
            return True
            
        self.nodes[id] = MemoryNode(id, node_type, label, attributes, confidence, provenance, self._current_time())
        return True

    def add_edge(self, source_id: str, target_id: str, relation: str, confidence: float, provenance: str, weight: float = 1.0) -> bool:
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
            
        if len(self.edges) >= self.max_edges:
            self._evict_edges(max(1, int(self.max_edges * 0.1)))
            
        for edge in self.edges:
            if edge.source_id == source_id and edge.target_id == target_id and edge.relation == relation:
                edge.confidence = (edge.confidence + confidence) / 2.0
                return True
                
        self.edges.append(MemoryEdge(source_id, target_id, relation, confidence, provenance, weight, self._current_time()))
        self.nodes[source_id].access_count += 1
        self.nodes[target_id].access_count += 1
        return True

    def query(self, node_id: str) -> Optional[Dict[str, Any]]:
        if node_id not in self.nodes:
            return None
            
        node = self.nodes[node_id]
        node.last_accessed = self._current_time()
        node.access_count += 1
        
        connections = []
        for edge in self.edges:
            if edge.source_id == node_id or edge.target_id == node_id:
                connections.append(edge.to_dict())
                
        return {
            'node': node.to_dict(),
            'connections': connections
        }
        
    def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> List[MemoryEdge]:
        if source_id not in self.nodes or target_id not in self.nodes:
            return []
            
        queue: List[Tuple[str, List[MemoryEdge]]] = [(source_id, [])]
        visited: Set[str] = {source_id}
        
        while queue:
            curr, path = queue.pop(0)
            if len(path) > max_depth:
                continue
                
            if curr == target_id:
                return path
                
            for edge in self.edges:
                if edge.source_id == curr and edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append((edge.target_id, path + [edge]))
        return []

    def query_by_relation(self, node_id: str, relation: str) -> List[str]:
        result = []
        if node_id not in self.nodes:
            return result
        for edge in self.edges:
            if edge.source_id == node_id and edge.relation == relation:
                result.append(edge.target_id)
            elif edge.target_id == node_id and edge.relation == relation:
                result.append(edge.source_id)
        return result

    def detect_contradictions(self) -> List[Tuple[str, str, str]]:
        contradictions = []
        for edge in self.edges:
            if edge.relation == 'contradicts':
                contradictions.append((edge.source_id, edge.target_id, edge.provenance))
                
        # attribute mismatch contradiction
        same_nodes = {}
        for nid, n in self.nodes.items():
            if 'entity_id' in n.attributes:
                eid = n.attributes['entity_id']
                if eid not in same_nodes: same_nodes[eid] = []
                same_nodes[eid].append(n)
                
        for eid, nodes_list in same_nodes.items():
            for i in range(len(nodes_list)):
                for j in range(i+1, len(nodes_list)):
                    n1 = nodes_list[i]
                    n2 = nodes_list[j]
                    for k in n1.attributes:
                        if k in n2.attributes and n1.attributes[k] != n2.attributes[k] and k != 'entity_id':
                            contradictions.append((n1.id, n2.id, f"Attribute mismatch: {k}"))
                            
                            # Reduce confidence of weaker
                            if n1.confidence > n2.confidence:
                                n2.confidence *= 0.5
                            else:
                                n1.confidence *= 0.5
        
        return contradictions

    def propagate_confidence(self) -> None:
        for edge in self.edges:
            if edge.relation == 'is_a' or edge.relation == 'implies':
                n1 = self.nodes.get(edge.source_id)
                n2 = self.nodes.get(edge.target_id)
                if n1 and n2:
                    inferred_conf = n1.confidence * edge.confidence
                    if n2.confidence < inferred_conf:
                        n2.confidence = (n2.confidence + inferred_conf) / 2.0

    def consolidate(self) -> None:
        self.propagate_confidence()
        # Remove low confidence isolated nodes
        connected = set()
        for e in self.edges:
            connected.add(e.source_id)
            connected.add(e.target_id)
            
        to_remove = []
        for nid, node in self.nodes.items():
            if nid not in connected and node.confidence < 0.2:
                to_remove.append(nid)
                
        for nid in to_remove:
            del self.nodes[nid]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'nodes': {k: v.to_dict() for k, v in self.nodes.items()},
            'edges': [e.to_dict() for e in self.edges]
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticMemoryGraph':
        graph = cls()
        for k, v in data.get('nodes', {}).items():
            graph.nodes[k] = MemoryNode.from_dict(v)
        for e in data.get('edges', []):
            graph.edges.append(MemoryEdge.from_dict(e))
        return graph
