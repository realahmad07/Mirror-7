from typing import Any, Dict, List, Set, Tuple

class StructureExtractor:
    """Discovers internal representations from raw byte/text streams."""
    
    def __init__(self, min_support: int = 2):
        self.min_support = min_support
        self.motifs = {}
        self.history = []

    def observe(self, raw_data: bytes):
        """Processes raw bytes and extracts repeating structures (n-grams/motifs)."""
        self.history.append(raw_data)
        
        # Simple n-gram discovery for bytes
        for n in range(1, min(len(raw_data), 10) + 1):
            for i in range(len(raw_data) - n + 1):
                motif = raw_data[i:i+n]
                if motif in self.motifs:
                    self.motifs[motif] += 1
                else:
                    self.motifs[motif] = 1

    def get_concepts(self) -> List[bytes]:
        """Returns discovered structural motifs that pass the support threshold."""
        return [k for k, v in self.motifs.items() if v >= self.min_support]

    def parse(self, raw_data: bytes) -> List[bytes]:
        """Parses a raw observation into a sequence of known concepts."""
        concepts = self.get_concepts()
        concepts.sort(key=len, reverse=True) # Greedily match longest first
        
        parsed = []
        i = 0
        while i < len(raw_data):
            matched = False
            for c in concepts:
                if raw_data[i:i+len(c)] == c:
                    parsed.append(c)
                    i += len(c)
                    matched = True
                    break
            if not matched:
                # Unrecognized raw byte
                parsed.append(bytes([raw_data[i]]))
                i += 1
        return parsed

class RawGroundedAgent:
    """Agent that uses discovered representations for prediction."""
    
    def __init__(self):
        self.extractor = StructureExtractor()
        self.raw_transitions = []

    def observe_transition(self, before_raw: bytes, action: str, after_raw: bytes):
        self.extractor.observe(before_raw)
        self.extractor.observe(after_raw)
        self.raw_transitions.append((before_raw, action, after_raw))

    def predict(self, state_raw: bytes, action: str) -> bytes:
        parsed_s = tuple(self.extractor.parse(state_raw))
        
        matches=[]
        for b_raw, act, a_raw in self.raw_transitions:
            if act != action: continue
            if tuple(self.extractor.parse(b_raw)) != parsed_s: continue
            matches.append(b"".join(self.extractor.parse(a_raw)))
        unique=[]
        for value in matches:
            if value not in unique: unique.append(value)
        if len(unique)==1: return unique[0]
        return b"" # abstain on unknown or conflicting evidence
