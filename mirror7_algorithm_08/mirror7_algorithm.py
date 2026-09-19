import uuid
import time
import math
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter

class Experience:
    def __init__(self, content: Dict[str, Any], source: str, repetitions: int = 1, confidence: float = 0.5):
        self.content = content
        self.source = source
        self.timestamp = int(time.time())
        self.repetitions = repetitions
        self.confidence = confidence

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        exp = cls(data['content'], data['source'], data.get('repetitions', 1), data.get('confidence', 0.5))
        exp.timestamp = data['timestamp']
        return exp

class Knowledge:
    def __init__(self, content: Dict[str, Any], schema: Dict[str, Any], sources: List[str], 
                 support_count: int, confidence: float):
        self.id = str(uuid.uuid4())
        self.content = content
        self.schema = schema
        self.confidence = confidence
        self.sources = sources
        self.support_count = support_count
        self.contradiction_count = 0
        self.created_at = int(time.time())
        self.last_reinforced = self.created_at
        self.usage_count = 0

    @property
    def utility(self) -> float:
        # utility = confidence * log(1 + support_count) * recency_factor * usage_count
        # Simple recency_factor: inverse of age in seconds (plus 1 to avoid /0)
        age = max(1, int(time.time()) - self.last_reinforced)
        recency = 1.0 / math.log(age + 2)
        return self.confidence * math.log(1 + self.support_count) * recency * max(1, self.usage_count)

    def to_dict(self):
        return {
            'id': self.id,
            'content': self.content,
            'schema': self.schema,
            'confidence': self.confidence,
            'sources': self.sources,
            'support_count': self.support_count,
            'contradiction_count': self.contradiction_count,
            'created_at': self.created_at,
            'last_reinforced': self.last_reinforced,
            'usage_count': self.usage_count
        }

    @classmethod
    def from_dict(cls, data):
        k = cls(data['content'], data['schema'], data['sources'], data['support_count'], data['confidence'])
        k.id = data['id']
        k.contradiction_count = data.get('contradiction_count', 0)
        k.created_at = data['created_at']
        k.last_reinforced = data['last_reinforced']
        k.usage_count = data.get('usage_count', 0)
        return k

class KnowledgeConsolidationEngine:
    def __init__(self, max_buffer: int = 100, max_knowledge: int = 50, 
                 promotion_threshold: int = 3, min_confidence: float = 0.2):
        self.max_buffer = max_buffer
        self.max_knowledge = max_knowledge
        self.promotion_threshold = promotion_threshold
        self.min_confidence = min_confidence
        
        self.stb: List[Experience] = []  # Short-Term Buffer
        self.lts: List[Knowledge] = []   # Long-Term Store

    def add_experience(self, exp: Experience) -> None:
        # Check if matches LTS for immediate reinforcement
        reinforced = False
        for k in self.lts:
            if self._matches_content(k.content, exp.content):
                k.support_count += exp.repetitions
                k.last_reinforced = int(time.time())
                k.sources.append(exp.source)
                k.confidence = k.support_count / (k.support_count + k.contradiction_count + 1)
                reinforced = True
                break
            elif self._is_contradictory(k.content, exp.content):
                k.contradiction_count += exp.repetitions
                k.confidence = k.support_count / (k.support_count + k.contradiction_count + 1)
                
        if not reinforced:
            self.stb.append(exp)
            if len(self.stb) > self.max_buffer:
                self.stb.pop(0)

    def consolidate(self) -> None:
        clusters = self._cluster_stb()
        
        for rep_count, exps in clusters:
            if rep_count >= self.promotion_threshold:
                # Extract common content and schema
                common_content = self._extract_common_content(exps)
                schema = self._extract_schema(exps)
                sources = list(set([e.source for e in exps]))
                
                # Check contradictions with LTS
                contradicted_k = None
                for k in self.lts:
                    if self._is_contradictory(k.content, common_content):
                        contradicted_k = k
                        break
                        
                if contradicted_k:
                    if rep_count > contradicted_k.support_count:
                        # Override
                        self.lts.remove(contradicted_k)
                        self._promote(common_content, schema, sources, rep_count)
                    else:
                        # Existing knowledge wins
                        contradicted_k.contradiction_count += rep_count
                        contradicted_k.confidence = contradicted_k.support_count / (contradicted_k.support_count + contradicted_k.contradiction_count + 1)
                else:
                    self._promote(common_content, schema, sources, rep_count)
                    
        # Demote low confidence
        self.lts = [k for k in self.lts if k.confidence >= self.min_confidence]
        
        # Enforce LTS size
        if len(self.lts) > self.max_knowledge:
            self.lts.sort(key=lambda x: x.utility, reverse=True)
            self.lts = self.lts[:self.max_knowledge]
            
        # Clear promoted from STB (simplified: clear all processed)
        self.stb = []

    def _promote(self, content: Dict[str, Any], schema: Dict[str, Any], sources: List[str], support_count: int) -> None:
        confidence = support_count / (support_count + 1)  # 0 contradictions initially
        k = Knowledge(content, schema, sources, support_count, confidence)
        self.lts.append(k)

    def query_knowledge(self, key_values: Dict[str, Any]) -> List[Knowledge]:
        results = []
        for k in self.lts:
            match = True
            for key, val in key_values.items():
                if k.content.get(key) != val:
                    match = False
                    break
            if match:
                k.usage_count += 1
                results.append(k)
        return results

    def get_stats(self) -> Dict[str, int]:
        return {
            'stb_size': len(self.stb),
            'lts_size': len(self.lts)
        }

    def _matches_content(self, c1: Dict[str, Any], c2: Dict[str, Any]) -> bool:
        return c1 == c2

    def _is_contradictory(self, c1: Dict[str, Any], c2: Dict[str, Any]) -> bool:
        # A simple contradiction: same keys but different values
        common_keys = set(c1.keys()).intersection(set(c2.keys()))
        if not common_keys:
            return False
            
        for key in common_keys:
            if c1[key] != c2[key]:
                return True
        return False

    def _cluster_stb(self) -> List[tuple[int, List[Experience]]]:
        # Simple exact match clustering for content
        clusters = []
        used = set()
        
        for i, e1 in enumerate(self.stb):
            if i in used:
                continue
            
            cluster = [e1]
            used.add(i)
            total_reps = e1.repetitions
            
            for j, e2 in enumerate(self.stb):
                if j in used:
                    continue
                    
                if self._matches_content(e1.content, e2.content):
                    cluster.append(e2)
                    total_reps += e2.repetitions
                    used.add(j)
                    
            clusters.append((total_reps, cluster))
            
        return clusters

    def _extract_common_content(self, exps: List[Experience]) -> Dict[str, Any]:
        if not exps:
            return {}
        # Simple intersection
        first = exps[0].content
        common = {}
        for k, v in first.items():
            if all(e.content.get(k) == v for e in exps):
                common[k] = v
        return common

    def _extract_schema(self, exps: List[Experience]) -> Dict[str, Any]:
        if not exps:
            return {}
        first = exps[0].content
        schema = {}
        for k in first.keys():
            vals = set(str(e.content.get(k)) for e in exps)
            if len(vals) == 1:
                schema[k] = list(vals)[0]
            else:
                schema[k] = '?' # Wildcard
        return schema
