import json
import math
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter

class FailureRecord:
    def __init__(self, component: str, error_type: str, context: Dict[str, Any], 
                 input_snapshot: Any, expected_output: Optional[Any], actual_output: Optional[Any], 
                 severity: str = 'error'):
        self.id = str(uuid.uuid4())
        self.timestamp = int(time.time())
        self.component = component
        self.error_type = error_type
        self.context = context
        self.input_snapshot = input_snapshot
        self.expected_output = expected_output
        self.actual_output = actual_output
        self.severity = severity

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        record = cls(data['component'], data['error_type'], data['context'], 
                     data['input_snapshot'], data.get('expected_output'), data.get('actual_output'), 
                     data.get('severity', 'error'))
        record.id = data['id']
        record.timestamp = data['timestamp']
        return record

class CandidatePatch:
    def __init__(self, target_pattern: str, description: str, patch_type: str, parameters: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.target_pattern = target_pattern
        self.description = description
        self.patch_type = patch_type
        self.parameters = parameters
        self.status = 'proposed'

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        patch = cls(data['target_pattern'], data['description'], data['patch_type'], data['parameters'])
        patch.id = data['id']
        patch.status = data.get('status', 'proposed')
        return patch

class ValidationResult:
    def __init__(self, patch_id: str, success: bool, fixes_failing: bool, breaks_passing: bool, cost: float):
        self.patch_id = patch_id
        self.success = success
        self.fixes_failing = fixes_failing
        self.breaks_passing = breaks_passing
        self.cost = cost

class SelfDebuggingAlgorithm:
    def __init__(self, max_records: int = 50, pattern_threshold: int = 3):
        self.max_records = max_records
        self.pattern_threshold = pattern_threshold
        self.failure_log: List[FailureRecord] = []
        self.passing_context_log: List[Dict[str, Any]] = [] # to calculate information gain

    def log_failure(self, record: FailureRecord) -> None:
        self.failure_log.append(record)
        if len(self.failure_log) > self.max_records:
            self.failure_log.pop(0)

    def log_success(self, context: Dict[str, Any]) -> None:
        self.passing_context_log.append(context)
        if len(self.passing_context_log) > self.max_records:
            self.passing_context_log.pop(0)

    def detect_patterns(self) -> List[Dict[str, Any]]:
        clusters = defaultdict(list)
        for record in self.failure_log:
            key = f"{record.component}::{record.error_type}"
            clusters[key].append(record)
        
        detected_patterns = []
        for key, records in clusters.items():
            if len(records) >= self.pattern_threshold:
                pattern = self._analyze_cluster(key, records)
                if pattern:
                    detected_patterns.append(pattern)
        
        return detected_patterns

    def _entropy(self, labels: List[int]) -> float:
        if not labels:
            return 0.0
        counts = Counter(labels)
        total = len(labels)
        ent = 0.0
        for count in counts.values():
            p = count / total
            ent -= p * math.log2(p)
        return ent

    def _analyze_cluster(self, key: str, records: List[FailureRecord]) -> Optional[Dict[str, Any]]:
        # Root Cause Localization using Information Gain
        # We need all contexts: failures (label 1) and successes (label 0)
        # If we have no successes logged, we just use common context
        
        all_samples = []
        labels = []
        
        for r in records:
            all_samples.append(r.context)
            labels.append(1)
            
        for c in self.passing_context_log:
            all_samples.append(c)
            labels.append(0)
            
        base_entropy = self._entropy(labels)
        
        if not self.passing_context_log:
            # Fallback to pure correlation if no passing logs
            common_context = {}
            if not records:
                return None
            for k in records[0].context.keys():
                vals = set(str(r.context.get(k)) for r in records)
                if len(vals) == 1:
                    common_context[k] = records[0].context[k]
            
            return {
                'pattern_id': key,
                'count': len(records),
                'root_cause_features': common_context,
                'records': [r.id for r in records]
            }

        # Calculate Info Gain for each feature
        features = set()
        for s in all_samples:
            features.update(s.keys())
            
        info_gain = {}
        for feature in features:
            values = set(str(s.get(feature)) for s in all_samples)
            weighted_ent = 0.0
            
            for v in values:
                subset_labels = [labels[i] for i, s in enumerate(all_samples) if str(s.get(feature)) == v]
                if subset_labels:
                    weighted_ent += (len(subset_labels) / len(labels)) * self._entropy(subset_labels)
                    
            ig = base_entropy - weighted_ent
            info_gain[feature] = ig
            
        # Rank features
        ranked_features = sorted(info_gain.items(), key=lambda x: x[1], reverse=True)
        top_features = {}
        
        if ranked_features and ranked_features[0][1] > 0.1:
            top_feature = ranked_features[0][0]
            # Find the value of this feature most associated with failures
            failing_values = [str(r.context.get(top_feature)) for r in records]
            if failing_values:
                most_common = Counter(failing_values).most_common(1)[0][0]
                top_features[top_feature] = most_common

        return {
            'pattern_id': key,
            'count': len(records),
            'root_cause_features': top_features,
            'records': [r.id for r in records]
        }

    def suggest_fix(self, pattern: Dict[str, Any]) -> Optional[CandidatePatch]:
        root_causes = pattern.get('root_cause_features', {})
        if not root_causes:
            return None
            
        key = pattern['pattern_id']
        desc = f"Filter inputs matching context: {root_causes}"
        patch = CandidatePatch(
            target_pattern=key,
            description=desc,
            patch_type='input_filter',
            parameters=root_causes
        )
        return patch

    def validate_fix(self, patch: CandidatePatch, test_cases: List[Dict[str, Any]]) -> ValidationResult:
        fixes_failing = True
        breaks_passing = False
        
        # Test cases format: {'context': {...}, 'is_failing_case': bool}
        for case in test_cases:
            is_failing_case = case.get('is_failing_case', False)
            context = case.get('context', {})
            
            # The filter matches if the context has the specified values
            match_filter = all(str(context.get(k)) == v for k, v in patch.parameters.items())
            
            if is_failing_case:
                if not match_filter:
                    fixes_failing = False
            else:
                if match_filter:
                    breaks_passing = True
                    
        success = fixes_failing and not breaks_passing
        patch.status = 'validated' if success else 'rejected'
        
        return ValidationResult(patch.id, success, fixes_failing, breaks_passing, cost=1.0)
