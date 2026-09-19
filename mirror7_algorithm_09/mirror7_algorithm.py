import math
import uuid
from typing import Dict, List, Set, Optional, Tuple, Any

class ModalityToken:
    def __init__(self, modality: str, token_id: str, features: Dict[str, float], timestamp: int):
        self.modality = modality
        self.token_id = token_id
        self.features = features.copy()
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "token_id": self.token_id,
            "features": self.features,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModalityToken':
        return cls(data["modality"], data["token_id"], data["features"], data["timestamp"])

class GroundedConcept:
    def __init__(self, concept_id: str):
        self.concept_id = concept_id
        self.bindings: Dict[str, Set[str]] = {}
        self.co_occurrence_count: int = 0
        self.confidence: float = 0.0
        self.feature_centroid: Dict[str, float] = {}
        self.last_updated: int = 0

    def add_binding(self, modality: str, token_id: str):
        if modality not in self.bindings:
            self.bindings[modality] = set()
        self.bindings[modality].add(token_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "bindings": {m: list(t) for m, t in self.bindings.items()},
            "co_occurrence_count": self.co_occurrence_count,
            "confidence": self.confidence,
            "feature_centroid": self.feature_centroid,
            "last_updated": self.last_updated
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GroundedConcept':
        gc = cls(data["concept_id"])
        gc.bindings = {m: set(t) for m, t in data["bindings"].items()}
        gc.co_occurrence_count = data["co_occurrence_count"]
        gc.confidence = data["confidence"]
        gc.feature_centroid = data["feature_centroid"]
        gc.last_updated = data["last_updated"]
        return gc

class MultimodalGrounding:
    def __init__(self, max_concepts: int = 1000, max_pairs: int = 5000, alignment_threshold: float = 0.5):
        self.max_concepts = max_concepts
        self.max_pairs = max_pairs
        self.alignment_threshold = alignment_threshold
        
        self.concepts: Dict[str, GroundedConcept] = {}
        self.token_to_concept: Dict[Tuple[str, str], str] = {} # (modality, token_id) -> concept_id
        
        self.co_occurrences: Dict[Tuple[Tuple[str, str], Tuple[str, str]], int] = {}
        self.token_counts: Dict[Tuple[str, str], int] = {}
        
        self.time_step: int = 0
        
    def _evict_pairs_if_needed(self):
        if len(self.co_occurrences) > self.max_pairs:
            # Remove pairs with lowest counts
            sorted_pairs = sorted(self.co_occurrences.items(), key=lambda x: x[1])
            to_remove = len(self.co_occurrences) - self.max_pairs
            for k, _ in sorted_pairs[:to_remove]:
                del self.co_occurrences[k]

    def _evict_concepts_if_needed(self):
        if len(self.concepts) > self.max_concepts:
            # Remove concepts with lowest confidence
            sorted_concepts = sorted(self.concepts.items(), key=lambda x: x[1].confidence)
            to_remove = len(self.concepts) - self.max_concepts
            for k, concept in sorted_concepts[:to_remove]:
                # Clean up token_to_concept mappings
                for m, tokens in concept.bindings.items():
                    for t in tokens:
                        if (m, t) in self.token_to_concept and self.token_to_concept[(m, t)] == k:
                            del self.token_to_concept[(m, t)]
                del self.concepts[k]

    def observe(self, tokens: List[ModalityToken]):
        self.time_step += 1
        
        # Update individual counts
        for token in tokens:
            t_key = (token.modality, token.token_id)
            self.token_counts[t_key] = self.token_counts.get(t_key, 0) + 1
            
        # Update co-occurrences
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                tk1 = (tokens[i].modality, tokens[i].token_id)
                tk2 = (tokens[j].modality, tokens[j].token_id)
                
                # Order to ensure consistency
                pair = tuple(sorted([tk1, tk2]))
                self.co_occurrences[pair] = self.co_occurrences.get(pair, 0) + 1
                
                self._process_pair(tokens[i], tokens[j], pair)
                
        self._evict_pairs_if_needed()
        self._evict_concepts_if_needed()

    def _process_pair(self, t1: ModalityToken, t2: ModalityToken, pair: Tuple[Tuple[str, str], Tuple[str, str]]):
        key1 = (t1.modality, t1.token_id)
        key2 = (t2.modality, t2.token_id)
        
        c1 = self.token_to_concept.get(key1)
        c2 = self.token_to_concept.get(key2)
        
        if c1 and c2 and c1 == c2:
            # Already bound to the same concept
            concept = self.concepts[c1]
            concept.co_occurrence_count += 1
            concept.confidence = min(1.0, concept.confidence + 0.1)
            concept.last_updated = self.time_step
            self._update_centroid(concept, [t1, t2])
        elif c1 and not c2:
            self._evaluate_alignment(t2, t1, self.concepts[c1], pair)
        elif c2 and not c1:
            self._evaluate_alignment(t1, t2, self.concepts[c2], pair)
        elif c1 and c2 and c1 != c2:
            # Possible merge scenario, keeping it simple by evaluating one way
            align = self.alignment(key1, key2)
            if align > self.alignment_threshold:
                self._merge_concepts(c1, c2)
        else:
            # Neither has a concept
            align = self.alignment(key1, key2)
            if align > self.alignment_threshold:
                # Create new concept
                concept_id = str(uuid.uuid4())
                concept = GroundedConcept(concept_id)
                concept.add_binding(t1.modality, t1.token_id)
                concept.add_binding(t2.modality, t2.token_id)
                concept.co_occurrence_count = self.co_occurrences[pair]
                concept.confidence = 0.5
                concept.last_updated = self.time_step
                self.concepts[concept_id] = concept
                self.token_to_concept[key1] = concept_id
                self.token_to_concept[key2] = concept_id
                self._update_centroid(concept, [t1, t2])

    def _evaluate_alignment(self, unbound_token: ModalityToken, bound_token: ModalityToken, concept: GroundedConcept, pair: Tuple[Tuple[str, str], Tuple[str, str]]):
        k1 = (unbound_token.modality, unbound_token.token_id)
        k2 = (bound_token.modality, bound_token.token_id)
        align = self.alignment(k1, k2)
        
        if align > self.alignment_threshold:
            concept.add_binding(unbound_token.modality, unbound_token.token_id)
            self.token_to_concept[k1] = concept.concept_id
            concept.co_occurrence_count += 1
            concept.confidence = min(1.0, concept.confidence + 0.1)
            concept.last_updated = self.time_step
            self._update_centroid(concept, [unbound_token])

    def _merge_concepts(self, c1_id: str, c2_id: str):
        c1 = self.concepts.get(c1_id)
        c2 = self.concepts.get(c2_id)
        if not c1 or not c2: return
        
        for m, tokens in c2.bindings.items():
            for t in tokens:
                c1.add_binding(m, t)
                self.token_to_concept[(m, t)] = c1_id
                
        c1.co_occurrence_count += c2.co_occurrence_count
        c1.confidence = min(1.0, (c1.confidence + c2.confidence) / 1.5)
        c1.last_updated = self.time_step
        
        # Merge centroids (simplified unweighted)
        for f, v in c2.feature_centroid.items():
            if f in c1.feature_centroid:
                c1.feature_centroid[f] = (c1.feature_centroid[f] + v) / 2
            else:
                c1.feature_centroid[f] = v
                
        del self.concepts[c2_id]

    def alignment(self, tk1: Tuple[str, str], tk2: Tuple[str, str]) -> float:
        pair = tuple(sorted([tk1, tk2]))
        co_occ = self.co_occurrences.get(pair, 0)
        if co_occ == 0:
            return 0.0
            
        count1 = self.token_counts.get(tk1, 1)
        count2 = self.token_counts.get(tk2, 1)
        
        return co_occ / math.sqrt(count1 * count2)

    def _update_centroid(self, concept: GroundedConcept, tokens: List[ModalityToken]):
        for token in tokens:
            for f, v in token.features.items():
                if f in concept.feature_centroid:
                    concept.feature_centroid[f] = (concept.feature_centroid[f] * 0.9) + (v * 0.1)
                else:
                    concept.feature_centroid[f] = v

    def ground(self, token_modality: str, token_id: str) -> Optional[GroundedConcept]:
        concept_id = self.token_to_concept.get((token_modality, token_id))
        if concept_id:
            return self.concepts.get(concept_id)
        return None

    def retrieve(self, token_modality: str, token_id: str, target_modality: str) -> List[str]:
        concept = self.ground(token_modality, token_id)
        if not concept:
            return []
        return list(concept.bindings.get(target_modality, set()))
