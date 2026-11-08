"""
Fetches verified relational and multi-hop logic reasoning datasets:
- HotpotQA (multi-hop relational reasoning)
- ProofWriter (formal deductive proofs & rules)
- BIG-Bench Hard (formal fallacies, temporal sequences, causal judgement, web of lies, date understanding)
- Abductive NLI (cause-and-effect hypothesis evaluation)

Saves to 04_relational_logic.jsonl and ingests into Mirror 7's production vault.
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

OUT_FILE = Path(r"c:\Users\Ali Computers\AppData\Local\Temp\04_relational_logic.jsonl")
VAULT_PATH = Path("artifacts/mirror_crystal_all_datasets.dcrystal")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\r\n", "\n", str(text))
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def fetch_url_json(url: str, timeout: float = 12.0):
    req = urllib.request.Request(url, headers={"User-Agent": "Mirror7/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    print("=== LIVE RECOVERY: FETCHING RELATIONAL & MULTI-HOP LOGIC ===")
    collected_items = []
    seen = set()

    def add_pair(prompt: str, response: str, domain: str = "relational_logic", trust_tier: int = 1):
        p = clean_text(prompt)
        r = clean_text(response)
        if not p or not r or len(p) < 8 or len(r) < 2:
            return False
        p_key = p.lower().strip()
        if p_key in seen:
            return False
        seen.add(p_key)
        collected_items.append({
            "prompt": p,
            "response": r,
            "domain": domain,
            "tool": "",
            "test_code": "",
            "trust_tier": trust_tier,
        })
        return True

    # 1. HotpotQA Multi-Hop Question Answering (Validation split, multiple batches)
    print("\n1. Fetching HotpotQA (Multi-Hop Relational Reasoning)...")
    hotpot_count = 0
    for offset in range(0, 1500, 100):
        url = f"https://datasets-server.huggingface.co/rows?dataset=hotpotqa%2Fhotpot_qa&config=distractor&split=validation&offset={offset}&limit=100"
        try:
            data = fetch_url_json(url)
            rows = data.get("rows", [])
            if not rows:
                break
            for r in rows:
                row = r.get("row", {})
                q = row.get("question", "")
                a = row.get("answer", "")
                # context paragraphs
                context = row.get("context", {})
                titles = context.get("title", [])
                sentences = context.get("sentences", [])
                ctx_snippets = []
                for t, sents in zip(titles[:3], sentences[:3]):
                    ctx_snippets.append(f"[{t}]: {' '.join(sents[:2])}")
                ctx_str = "\n".join(ctx_snippets)
                full_prompt = f"Context:\n{ctx_str}\n\nQuestion: {q}" if ctx_snippets else q
                full_response = f"Based on the relational facts, the answer is: {a}."
                if add_pair(full_prompt, full_response, domain="relational_logic", trust_tier=1):
                    hotpot_count += 1
        except Exception as e:
            print(f"  [Hotpot offset={offset}] Error: {e}")
            break
    print(f"  -> HotpotQA collected: {hotpot_count:,} items")

    # 2. ProofWriter (Formal Deductive Proofs)
    print("\n2. Fetching ProofWriter (Deductive Logic & Proof Trees)...")
    pw_count = 0
    for offset in range(0, 1000, 100):
        url = f"https://datasets-server.huggingface.co/rows?dataset=tasksource%2Fproofwriter&config=default&split=train&offset={offset}&limit=100"
        try:
            data = fetch_url_json(url)
            rows = data.get("rows", [])
            if not rows:
                break
            for r in rows:
                row = r.get("row", {})
                theory = row.get("theory", "")
                q = row.get("question", "")
                ans = str(row.get("answer", ""))
                proofs = row.get("allProofs", "")
                prompt = f"Given the following formal theory:\n{theory}\n\nQuestion: Is it true that {q}?"
                resp_text = f"Answer: {ans}.\nProof deduction:\n{proofs}" if proofs else f"Answer: {ans}."
                if add_pair(prompt, resp_text, domain="deductive_logic", trust_tier=1):
                    pw_count += 1
        except Exception as e:
            print(f"  [ProofWriter offset={offset}] Error: {e}")
            break
    print(f"  -> ProofWriter collected: {pw_count:,} items")

    # 3. BIG-Bench Hard (BBH) Logic Subsets
    bbh_configs = [
        "formal_fallacies",
        "temporal_sequences",
        "causal_judgement",
        "web_of_lies",
        "date_understanding",
        "navigate",
    ]
    print("\n3. Fetching BIG-Bench Hard Logic Tasks...")
    bbh_count = 0
    for cfg in bbh_configs:
        cfg_items = 0
        for offset in (0, 100):
            url = f"https://datasets-server.huggingface.co/rows?dataset=lukaemon%2Fbbh&config={cfg}&split=test&offset={offset}&limit=100"
            try:
                data = fetch_url_json(url)
                rows = data.get("rows", [])
                if not rows:
                    break
                for r in rows:
                    row = r.get("row", {})
                    inp = row.get("input", "")
                    tgt = row.get("target", "")
                    if add_pair(inp, f"Conclusion: {tgt}", domain="relational_logic", trust_tier=1):
                        cfg_items += 1
                        bbh_count += 1
            except Exception as e:
                print(f"  [BBH {cfg} offset={offset}] Error: {e}")
                break
        print(f"  -> {cfg:<25}: {cfg_items:>4} items")
    print(f"  -> Total BBH collected: {bbh_count:,} items")

    # 4. Abductive Reasoning (allenai/art - ANLI)
    print("\n4. Fetching Abductive NLI (Causal & Hypothesis Inference)...")
    art_count = 0
    for offset in range(0, 800, 100):
        url = f"https://datasets-server.huggingface.co/rows?dataset=allenai%2Fart&config=anli&split=train&offset={offset}&limit=100"
        try:
            data = fetch_url_json(url)
            rows = data.get("rows", [])
            if not rows:
                break
            for r in rows:
                row = r.get("row", {})
                obs1 = row.get("observation_1", "")
                obs2 = row.get("observation_2", "")
                h1 = row.get("hypothesis_1", "")
                h2 = row.get("hypothesis_2", "")
                lbl = row.get("label", 1)
                best_h = h1 if lbl == 1 else h2
                prompt = (
                    f"Observation 1: {obs1}\n"
                    f"Observation 2: {obs2}\n"
                    f"Which hypothesis best explains what happened between observation 1 and 2?\n"
                    f"A: {h1}\n"
                    f"B: {h2}"
                )
                response = f"The most plausible explanation is: {best_h} (Option {'A' if lbl == 1 else 'B'})."
                if add_pair(prompt, response, domain="abductive_logic", trust_tier=1):
                    art_count += 1
        except Exception as e:
            print(f"  [ANLI offset={offset}] Error: {e}")
            break
    print(f"  -> Abductive NLI collected: {art_count:,} items")

    total_collected = len(collected_items)
    print(f"\nTotal Relational Logic Items Collected: {total_collected:,}")

    # Write out to 04_relational_logic.jsonl
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for it in collected_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    size_mb = OUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Saved to {OUT_FILE} ({size_mb:.2f} MB)")

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
    print(f"Total Processed:     {total_collected:,}")
    print(f"New Rules Added:     +{stats.get('new_rules', 0):,}")
    print(f"Duplicates Mapped:   {stats.get('duplicates_skipped', 0):,}")
    print(f"Initial Vault Rules: {start_rules:,}")
    print(f"Final Vault Rules:   {final_rules:,}")
    print(f"Net Rule Increase:   +{final_rules - start_rules:,}")
    print(f"Elapsed Time:        {dur:.2f} seconds ({total_collected / max(0.01, dur):.1f} items/sec)")
    print("=========================")


if __name__ == "__main__":
    main()
