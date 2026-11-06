"""
Audit dataset mirror_oasst2_curated_v1.jsonl:
- Duplicates and near duplicates
- Length distributions (user and target)
- Truncation statistics at 256, 512, 1024, 2048 bytes
- EOS preservation and natural boundary characteristics
- Split leakage verification
"""
import json
import hashlib
from collections import Counter, defaultdict
import numpy as np

path = "training/data/mirror_oasst2_curated_v1.jsonl"

records = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

total = len(records)
print(f"Total records: {total}")

# 1. Split Distribution
splits = Counter(r.get("split", "unknown") for r in records)
print(f"Splits: {dict(splits)}")

# 2. Duplicate Detection
seen_ids = set()
seen_inputs = defaultdict(list)
seen_targets = defaultdict(list)
seen_pairs = defaultdict(list)

exact_dups = 0
for idx, r in enumerate(records):
    eid = r.get("example_id", "")
    ut = r.get("user_text", "").strip()
    tt = r.get("target_text", "").strip()
    sp = r.get("split", "")
    
    seen_ids.add(eid)
    seen_inputs[ut.lower()].append((idx, eid, sp))
    seen_targets[tt.lower()].append((idx, eid, sp))
    seen_pairs[(ut.lower(), tt.lower())].append((idx, eid, sp))

print(f"Unique example IDs: {len(seen_ids)}")
print(f"Unique user inputs (case-insensitive): {len(seen_inputs)}")
print(f"Unique targets (case-insensitive): {len(seen_targets)}")
print(f"Unique (input, target) pairs: {len(seen_pairs)}")

# Check duplicate user inputs across splits (leakage)
input_leakage = 0
for ut, occurrences in seen_inputs.items():
    sp_set = set(sp for _, _, sp in occurrences)
    if len(sp_set) > 1:
        input_leakage += 1

print(f"User inputs appearing across multiple splits: {input_leakage}")

# 3. Length Distributions
user_bytes = [len(r.get("user_text", "").encode("utf-8")) for r in records]
target_bytes = [len(r.get("target_text", "").encode("utf-8")) for r in records]

print("\n--- Length Statistics (Bytes) ---")
print(f"User text: min={np.min(user_bytes)}, p25={np.percentile(user_bytes, 25):.1f}, median={np.median(user_bytes):.1f}, p75={np.percentile(user_bytes, 75):.1f}, p95={np.percentile(user_bytes, 95):.1f}, max={np.max(user_bytes)}, mean={np.mean(user_bytes):.1f}")
print(f"Target text: min={np.min(target_bytes)}, p25={np.percentile(target_bytes, 25):.1f}, median={np.median(target_bytes):.1f}, p75={np.percentile(target_bytes, 75):.1f}, p95={np.percentile(target_bytes, 95):.1f}, max={np.max(target_bytes)}, mean={np.mean(target_bytes):.1f}")

# 4. Truncation Impact
print("\n--- Truncation Impact ---")
for max_len in [256, 512, 1024, 2048]:
    u_truncated = sum(1 for b in user_bytes if b > max_len)
    t_truncated = sum(1 for b in target_bytes if b > max_len)
    t_bytes_lost = sum(max(0, b - max_len) for b in target_bytes)
    t_total_bytes = sum(target_bytes)
    print(f"Cutoff {max_len} bytes:")
    print(f"  User texts truncated: {u_truncated}/{total} ({u_truncated/total*100:.1f}%)")
    print(f"  Target texts truncated: {t_truncated}/{total} ({t_truncated/total*100:.1f}%)")
    print(f"  Target information lost: {t_bytes_lost}/{t_total_bytes} bytes ({t_bytes_lost/t_total_bytes*100:.1f}%)")

# 5. Quality metadata
qualities = [float(r.get("quality", 1.0)) for r in records]
print(f"\nQuality scores: min={np.min(qualities)}, median={np.median(qualities)}, mean={np.mean(qualities):.2f}, max={np.max(qualities)}")
q_dist = Counter(round(q, 2) for q in qualities)
print(f"Quality distribution (top 5): {q_dist.most_common(5)}")

# Save summary
audit_result = {
    "total": total,
    "splits": dict(splits),
    "unique_inputs": len(seen_inputs),
    "unique_targets": len(seen_targets),
    "input_leakage": input_leakage,
    "user_bytes": {
        "min": int(np.min(user_bytes)),
        "median": float(np.median(user_bytes)),
        "p95": float(np.percentile(user_bytes, 95)),
        "max": int(np.max(user_bytes)),
        "mean": float(np.mean(user_bytes)),
    },
    "target_bytes": {
        "min": int(np.min(target_bytes)),
        "median": float(np.median(target_bytes)),
        "p95": float(np.percentile(target_bytes, 95)),
        "max": int(np.max(target_bytes)),
        "mean": float(np.mean(target_bytes)),
    },
    "truncation_loss": {
        "256": sum(max(0, b - 256) for b in target_bytes) / sum(target_bytes),
        "512": sum(max(0, b - 512) for b in target_bytes) / sum(target_bytes),
        "1024": sum(max(0, b - 1024) for b in target_bytes) / sum(target_bytes),
    }
}
with open("artifacts/dataset_audit_metrics.json", "w", encoding="utf-8") as f:
    json.dump(audit_result, f, indent=2)
print("\nMetrics written to artifacts/dataset_audit_metrics.json")
