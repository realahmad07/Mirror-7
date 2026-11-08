"""
Fetches verified competition math problems from EleutherAI/hendrycks_math
and OpenAI GSM8K test set, saves to 01_advanced_mathematics.jsonl,
and ingests into Mirror 7's production vault.
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

OUT_FILE = Path(r"c:\Users\Ali Computers\AppData\Local\Temp\01_advanced_mathematics.jsonl")
VAULT_PATH = Path("artifacts/mirror_crystal_all_datasets.dcrystal")

CONFIGS = [
    "algebra",
    "counting_and_probability",
    "number_theory",
    "prealgebra",
    "intermediate_algebra",
    "geometry",
    "precalculus",
]


def clean_math(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<<[^>]+>>", "", str(text))
    if "####" in text:
        steps, ans = text.rsplit("####", 1)
        text = f"{steps.strip()}\n\nTherefore, the answer is: {ans.strip()}."
    return text.strip()


def main():
    print("=== LIVE RECOVERY: FETCHING ADVANCED MATHEMATICS ===")
    collected_items = []
    seen = set()

    # 1. Fetch OpenAI GSM8K test set (1,319 problems)
    print("Fetching OpenAI GSM8K test set...")
    url_gsm_test = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    try:
        req = urllib.request.Request(url_gsm_test, headers={"User-Agent": "Mirror7/1.0"})
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            for line in resp:
                raw_str = line.decode("utf-8").strip()
                if not raw_str:
                    continue
                row = json.loads(raw_str)
                q = clean_math(row.get("question", ""))
                a = clean_math(row.get("answer", ""))
                if q and a and q.lower() not in seen:
                    seen.add(q.lower())
                    collected_items.append({
                        "prompt": q,
                        "response": a,
                        "domain": "mathematics",
                        "tool": "",
                        "test_code": "",
                        "trust_tier": 1,
                    })
        print(f"  -> Added {len(collected_items)} GSM8K problems.")
    except Exception as e:
        print(f"  -> Error fetching GSM8K test: {e}")

    # 2. Fetch Hendrycks MATH across 7 configs (up to 300 items per config)
    print("\nFetching EleutherAI/hendrycks_math across 7 disciplines...")
    for cfg in CONFIGS:
        cfg_count = 0
        for offset in (0, 100, 200):
            url = f"https://datasets-server.huggingface.co/rows?dataset=EleutherAI%2Fhendrycks_math&config={cfg}&split=train&offset={offset}&limit=100"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mirror7/1.0"})
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    rows = data.get("rows", [])
                    if not rows:
                        break
                    for r in rows:
                        row = r.get("row", {})
                        prob = clean_math(row.get("problem", ""))
                        sol = clean_math(row.get("solution", ""))
                        if prob and sol and prob.lower() not in seen:
                            seen.add(prob.lower())
                            collected_items.append({
                                "prompt": prob,
                                "response": sol,
                                "domain": "mathematics",
                                "tool": "",
                                "test_code": "",
                                "trust_tier": 1,
                            })
                            cfg_count += 1
            except Exception as e:
                print(f"  [{cfg} offset={offset}] Error: {e}")
                break
        print(f"  -> {cfg:<25}: {cfg_count:>4} problems")

    print(f"\nTotal Advanced Mathematics Collected: {len(collected_items):,}")

    # Write to 01_advanced_mathematics.jsonl
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for it in collected_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Saved to {OUT_FILE} ({OUT_FILE.stat().st_size / (1024*1024):.2f} MB)")

    # Ingest into production vault
    print("\nIngesting into Mirror 7's production vault...")
    vault = DiskCrystalVault(VAULT_PATH)
    start_rules = vault.rule_count()
    print(f"Initial Vault Rules: {start_rules:,}")

    t0 = time.perf_counter()
    stats = vault.ingest_batch(collected_items, batch_size=1000, verify_math=False)
    dur = time.perf_counter() - t0

    final_rules = vault.rule_count()
    vault.close()

    print("\n=== INGESTION SUMMARY ===")
    print(f"Source File:         {OUT_FILE.name}")
    print(f"Total Processed:     {len(collected_items):,}")
    print(f"New Rules Added:     +{stats.get('new_rules', 0):,}")
    print(f"Duplicates Mapped:   {stats.get('duplicates_skipped', 0):,}")
    print(f"Initial Vault Rules: {start_rules:,}")
    print(f"Final Vault Rules:   {final_rules:,}")
    print(f"Net Rule Increase:   +{final_rules - start_rules:,}")
    print(f"Elapsed Time:        {dur:.2f} seconds ({len(collected_items) / max(0.01, dur):.1f} items/sec)")
    print("=========================")


if __name__ == "__main__":
    main()
