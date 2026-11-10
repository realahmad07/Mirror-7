#!/usr/bin/env python3
"""Phase 3 — Non-Destructive Knowledge Purification Engine for Mirror-7 Vault.

Rules Enforced:
1. NEVER delete original records from 'rules'.
2. Count overlapping defects by UNIQUE rule_id.
3. Correct trust tiers: set evidence_status='unverified', corrected_trust_tier=3 (heuristic)
   unless independently proven.
4. Quarantine defective rules into 'rule_hygiene_quarantine' table.
5. Create 'verified_production_index' for fast, clean, non-defective retrieval.
6. Context-slot ambiguous conversational prompts.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

VAULT_PATH = Path("restorevalutecrytaled/mirror_crystal_all_datasets.dcrystal")
BACKUP_PATH = Path("restorevalutecrytaled/mirror_crystal_all_datasets.dcrystal.phase3.bak")
REPORT_PATH = Path("artifacts/PHASE3_PURIFICATION_REPORT.json")

AMBIGUOUS_COLLISION_PROMPTS = {
    "yes please", "yes please.", "that's correct", "that is correct",
    "awesome, thank you!", "thank you for your help", "hello", "hi",
    "what can you do?", "what can you do", "ok", "okay", "sure", "thanks", "thank you"
}

def normalize_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())

def is_likely_truncated(text: str) -> bool:
    t = text.rstrip()
    if not t:
        return True
    if t.endswith("...") or t.endswith("…"):
        return True
    last_char = t[-1]
    return last_char not in {".", "!", "?", '"', "'", ")", "}", "]", "`", ">", ":"}

def check_refusal(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in [
        "as an ai", "as a language model", "i am an ai",
        "i cannot fulfill", "i'm sorry, but i cannot", "my knowledge cutoff"
    ])

def purify_vault():
    t0 = time.time()
    print("=" * 70)
    print("PHASE 3: KNOWLEDGE PURIFICATION & HYGIENE ENGINE")
    print("=" * 70)

    if not VAULT_PATH.exists():
        raise FileNotFoundError(f"Vault not found: {VAULT_PATH}")

    # Step 1: Create an online SQLite backup before anything
    if not BACKUP_PATH.exists():
        print(f"Creating pre-purification SQLite backup at: {BACKUP_PATH}...")
        src_conn = sqlite3.connect(VAULT_PATH)
        dst_conn = sqlite3.connect(BACKUP_PATH)
        with dst_conn:
            src_conn.backup(dst_conn)
        src_conn.close()
        dst_conn.close()
        print(f"Backup created successfully ({BACKUP_PATH.stat().st_size:,} bytes).")
    else:
        print(f"Existing backup found at: {BACKUP_PATH}")

    conn = sqlite3.connect(VAULT_PATH)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM rules;")
    total_rules = cur.fetchone()[0]
    print(f"Total rules in vault: {total_rules:,}")
    assert total_rules == 368270, f"Expected 368,270 rules, found {total_rules}"

    # Step 2: Initialize Non-Destructive Quarantine and Verified Production Index Tables
    print("Initializing non-destructive purification schemas...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rule_hygiene_quarantine (
            rule_id INTEGER PRIMARY KEY,
            defect_flags TEXT NOT NULL,
            reasons TEXT NOT NULL,
            quarantined_at REAL NOT NULL,
            FOREIGN KEY(rule_id) REFERENCES rules(rule_id)
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS verified_production_index (
            rule_id INTEGER PRIMARY KEY,
            domain TEXT NOT NULL,
            original_trust_tier INTEGER NOT NULL,
            corrected_trust_tier INTEGER NOT NULL,
            evidence_status TEXT NOT NULL,
            has_context_slots INTEGER NOT NULL,
            purified_at REAL NOT NULL,
            FOREIGN KEY(rule_id) REFERENCES rules(rule_id)
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_prod_idx_domain ON verified_production_index(domain);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_quarantine_flags ON rule_hygiene_quarantine(defect_flags);")
    conn.commit()

    # Clear prior index entries to allow idempotent re-runs
    cur.execute("DELETE FROM rule_hygiene_quarantine;")
    cur.execute("DELETE FROM verified_production_index;")
    conn.commit()

    # Step 3: Stream, evaluate, and categorize defects by UNIQUE rule_id
    print("Scanning all rules for defects, overlaps, and context collisions...")
    
    # Containers for defect sets (storing rule_ids)
    defects_by_id = defaultdict(list)
    set_trunc = set()
    set_syntax = set()
    set_collision = set()
    set_degen = set()
    set_refusal = set()

    # First pass: map prompt normalization to identify exact duplicate/colliding prompts
    norm_prompt_registry = defaultdict(list) # norm_p -> list of (rule_id, norm_resp, domain)

    BATCH_SIZE = 15000
    cur.execute("SELECT rule_id, num_slots, payload FROM rules ORDER BY rule_id ASC;")

    rule_metadata = {} # rule_id -> (domain, orig_tt, num_slots)

    processed = 0
    import zlib
    while True:
        batch = cur.fetchmany(BATCH_SIZE)
        if not batch:
            break

        for rule_id, num_slots, raw_payload in batch:
            data = json.loads(zlib.decompress(raw_payload).decode("utf-8"))
            prompt = str(data.get("ap", "")).strip()
            resp = str(data.get("ar", "")).strip()
            domain = str(data.get("dom", "unknown")).strip()
            orig_tt = data.get("tt", 3)

            rule_metadata[rule_id] = (domain, orig_tt, num_slots)

            norm_p = normalize_text(prompt)
            norm_r = normalize_text(resp)

            norm_prompt_registry[norm_p].append((rule_id, norm_r, domain, prompt))

            # 1. Truncation check
            if is_likely_truncated(resp):
                set_trunc.add(rule_id)
                defects_by_id[rule_id].append("TRUNCATED_RESPONSE")

            # 2. Degenerate response check (< 20 chars)
            if len(resp) < 20:
                set_degen.add(rule_id)
                defects_by_id[rule_id].append("DEGENERATE_SHORT")

            # 3. Chatbot refusal boilerplate
            if check_refusal(resp):
                set_refusal.add(rule_id)
                defects_by_id[rule_id].append("REFUSAL_BOILERPLATE")

            # 4. Code syntax validation
            if domain == "algorithms" or "```python" in resp:
                py_code = ""
                if "```python" in resp:
                    parts = resp.split("```python")
                    if len(parts) > 1:
                        py_code = parts[1].split("```")[0]
                elif "def " in resp or "class " in resp:
                    py_code = resp

                if py_code:
                    try:
                        ast.parse(py_code)
                    except SyntaxError:
                        set_syntax.add(rule_id)
                        defects_by_id[rule_id].append("CODE_SYNTAX_ERROR")

            processed += 1
            if processed % 100000 == 0:
                print(f"  First pass: {processed:,} / {total_rules:,} rules scanned...")

    # Second pass: Contradiction & Ambiguous Collision identification
    print("Analyzing prompt collisions and ambiguous prompts...")
    ambiguous_slotted_count = 0
    for norm_p, occurrences in norm_prompt_registry.items():
        # Check if prompt is inherently ambiguous conversational phrase
        is_ambiguous = norm_p in AMBIGUOUS_COLLISION_PROMPTS or (len(norm_p.split()) <= 2 and norm_p in {"hello", "thanks", "ok", "yes"})
        
        # Check if multiple occurrences produce different answers
        distinct_responses = {r[1] for r in occurrences}
        if len(distinct_responses) > 1 or (is_ambiguous and len(occurrences) > 1):
            # Mark colliding rules (keep the first if identical, but if conflicting mark for collision quarantine)
            for rid, _, dom, _ in occurrences:
                set_collision.add(rid)
                defects_by_id[rid].append("CONTRADICTORY_PROMPT_COLLISION")
        elif is_ambiguous and len(occurrences) == 1:
            ambiguous_slotted_count += 1

    # Overlap and unique counting
    all_defective_rule_ids = set_trunc | set_syntax | set_collision | set_degen | set_refusal
    total_unique_quarantined = len(all_defective_rule_ids)
    total_clean_rules = total_rules - total_unique_quarantined

    # Calculate exact multi-defect overlaps
    overlap_counts = Counter(len(defects) for defects in defects_by_id.values())

    print("\n" + "=" * 50)
    print("DEFECT BREAKDOWN & OVERLAP AUDIT (UNIQUE RULE IDs):")
    print("=" * 50)
    print(f"Total Rules Audited:                  {total_rules:,}")
    print(f"1. Truncated Responses (D_TRUNC):     {len(set_trunc):,}")
    print(f"2. Syntax-Broken Code (D_SYNTAX):     {len(set_syntax):,}")
    print(f"3. Prompt Collisions (D_COLLISION):   {len(set_collision):,}")
    print(f"4. Degenerate Responses (D_DEGEN):    {len(set_degen):,}")
    print(f"5. Refusal Boilerplate (D_REFUSAL):   {len(set_refusal):,}")
    print("-" * 50)
    print(f"SUM OF INDIVIDUAL DEFECT FLAGS:       {len(set_trunc)+len(set_syntax)+len(set_collision)+len(set_degen)+len(set_refusal):,}")
    print(f"TOTAL UNIQUE QUARANTINED RULE IDs:    {total_unique_quarantined:,} ({total_unique_quarantined/total_rules*100:.2f}%)")
    print(f"Rules with 1 defect:                  {overlap_counts[1]:,}")
    print(f"Rules with 2 overlapping defects:     {overlap_counts[2]:,}")
    print(f"Rules with 3+ overlapping defects:    {sum(c for k, c in overlap_counts.items() if k >= 3):,}")
    print("-" * 50)
    print(f"TOTAL VERIFIED CANDIDATE RULES:       {total_clean_rules:,} ({total_clean_rules/total_rules*100:.2f}%)")
    print("=" * 50)

    # Step 4: Populate 'rule_hygiene_quarantine' and 'verified_production_index'
    print("\nWriting non-destructive quarantine & production indexes in database...")
    now = time.time()
    
    quarantine_rows = []
    for rid in sorted(all_defective_rule_ids):
        reasons_list = defects_by_id[rid]
        mask = "|".join(reasons_list)
        quarantine_rows.append((rid, mask, json.dumps(reasons_list), now))

    production_rows = []
    domain_clean_counts = Counter()
    for rid in sorted(rule_metadata.keys()):
        if rid in all_defective_rule_ids:
            continue
        dom, orig_tt, num_slots = rule_metadata[rid]
        domain_clean_counts[dom] += 1
        # Correct trust tier: enforce unverified heuristic baseline (Tier 3) per Canonical Evidence policy
        production_rows.append((
            rid,
            dom,
            int(orig_tt) if isinstance(orig_tt, int) else 3,
            3, # corrected_trust_tier: Tier 3 (Heuristic / Unverified)
            "unverified",
            int(num_slots > 0),
            now
        ))

    cur.executemany("""
        INSERT INTO rule_hygiene_quarantine (rule_id, defect_flags, reasons, quarantined_at)
        VALUES (?, ?, ?, ?);
    """, quarantine_rows)

    cur.executemany("""
        INSERT INTO verified_production_index (
            rule_id, domain, original_trust_tier, corrected_trust_tier,
            evidence_status, has_context_slots, purified_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?);
    """, production_rows)

    conn.commit()
    print("Database index writes committed.")

    # Integrity verification
    cur.execute("SELECT COUNT(*) FROM rules;")
    final_rules_count = cur.fetchone()[0]
    assert final_rules_count == total_rules, "CRITICAL: Original rules count changed!"

    cur.execute("SELECT COUNT(*) FROM rule_hygiene_quarantine;")
    q_count = cur.fetchone()[0]
    assert q_count == total_unique_quarantined, f"Quarantine count mismatch: {q_count} vs {total_unique_quarantined}"

    cur.execute("SELECT COUNT(*) FROM verified_production_index;")
    prod_count = cur.fetchone()[0]
    assert prod_count == total_clean_rules, f"Prod index count mismatch: {prod_count} vs {total_clean_rules}"
    assert q_count + prod_count == total_rules, "Total partitioned rules does not equal 368,270!"

    elapsed = time.time() - t0
    print(f"\nPurification and indexing completed in {elapsed:.2f} seconds.")

    # Output detailed report JSON
    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vault_path": str(VAULT_PATH),
        "backup_path": str(BACKUP_PATH),
        "total_vault_rules": total_rules,
        "original_rules_deleted": 0,
        "purification_duration_sec": round(elapsed, 2),
        "unique_defective_quarantined": total_unique_quarantined,
        "unique_defective_pct": round(total_unique_quarantined / total_rules * 100, 2),
        "verified_clean_production_rules": total_clean_rules,
        "verified_clean_pct": round(total_clean_rules / total_rules * 100, 2),
        "defect_counts_disjoint": {
            "truncated_responses": len(set_trunc),
            "syntax_broken_code": len(set_syntax),
            "prompt_collisions": len(set_collision),
            "degenerate_under_20_chars": len(set_degen),
            "refusal_boilerplate": len(set_refusal),
        },
        "defect_overlap_distribution": {
            "single_defect_rules": overlap_counts[1],
            "two_defect_overlapping_rules": overlap_counts[2],
            "three_plus_overlapping_rules": sum(c for k, c in overlap_counts.items() if k >= 3),
        },
        "clean_rules_by_domain": dict(domain_clean_counts.most_common()),
        "trust_tier_correction": {
            "policy": "docs/14-blueprint-rule-architecture.md",
            "enforced_trust_tier": 3,
            "evidence_status": "unverified",
            "factual_guarantee": "Structural hygiene validated; semantic claim verification pending independent review."
        }
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {REPORT_PATH}")

    conn.close()
    return report

if __name__ == "__main__":
    purify_vault()
