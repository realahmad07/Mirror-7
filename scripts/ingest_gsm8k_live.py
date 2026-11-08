"""
Downloads and ingests all 7,473 verified GSM8K math reasoning problems
into Mirror 7's production DiskCrystalVault.
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mirror_crystal.disk_vault import DiskCrystalVault

VAULT_PATH = Path("artifacts/mirror_crystal_all_datasets.dcrystal")
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"


def format_gsm8k_answer(ans_raw: str) -> str:
    clean = re.sub(r"<<[^>]+>>", "", ans_raw)
    if "####" in clean:
        steps, final_ans = clean.rsplit("####", 1)
        return f"{steps.strip()}\n\nTherefore, the answer is: {final_ans.strip()}."
    return clean.strip()


def main():
    print("=== LIVE MIRROR 7 UPGRADE: GSM8K REASONING DATASET ===")
    if not VAULT_PATH.exists():
        print(f"Error: Vault not found at {VAULT_PATH}")
        return

    vault = DiskCrystalVault(VAULT_PATH)
    initial_count = vault.rule_count()
    print(f"Initial rules in vault: {initial_count:,}")

    print(f"Connecting to {GSM8K_URL}...")
    req = urllib.request.Request(GSM8K_URL, headers={"User-Agent": "Mirror7-Ingester/1.0"})
    
    items = []
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        for line in resp:
            raw_str = line.decode("utf-8").strip()
            if not raw_str:
                continue
            row = json.loads(raw_str)
            q = row.get("question", "").strip()
            a = format_gsm8k_answer(row.get("answer", ""))
            if q and a:
                items.append({
                    "prompt": q,
                    "response": a,
                    "domain": "mathematics",
                    "trust_tier": 1,
                })

    print(f"Loaded {len(items):,} GSM8K reasoning problems. Ingesting into DiskCrystalVault...")
    t0 = time.perf_counter()
    stats = vault.ingest_batch(items, batch_size=500, verify_math=False)
    dur = time.perf_counter() - t0

    final_count = vault.rule_count()
    print(f"Ingestion completed in {dur:.2f}s ({len(items)/max(0.01, dur):.1f} items/sec)!")
    print(f"Rules: {initial_count:,} -> {final_count:,} (+{final_count - initial_count:,} new rules)")
    print(f"Stats: {stats}")

    vault.close()
    print("=== UPGRADE COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
