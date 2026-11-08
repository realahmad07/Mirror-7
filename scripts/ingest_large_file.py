"""
Streams and ingests any given .jsonl file into Mirror 7's production vault
with batching and progress tracking.
"""
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mirror_crystal.disk_vault import DiskCrystalVault

VAULT_PATH = Path("artifacts/mirror_crystal_all_datasets.dcrystal")
DEFAULT_FILE = Path(r"c:\Users\Ali Computers\AppData\Local\Temp\03_science_and_physics.jsonl")
BATCH_SIZE = 5000


def ingest_file(input_path: Path):
    print(f"=== INGESTING: {input_path.name} ===")
    if not input_path.exists():
        print(f"Error: {input_path} does not exist.")
        return

    vault = DiskCrystalVault(VAULT_PATH)
    start_rules = vault.rule_count()
    print(f"Initial Vault Rules: {start_rules:,}")

    total_processed = 0
    total_new = 0
    total_dupes = 0
    total_rejected = 0

    batch = []
    t_start = time.perf_counter()

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                row = json.loads(line_str)
            except Exception:
                continue

            q = row.get("prompt") or row.get("question") or ""
            a = row.get("response") or row.get("answer") or ""
            if not q or not a:
                continue

            batch.append({
                "prompt": q.strip(),
                "response": a.strip(),
                "domain": row.get("domain", "physics_science"),
                "tool": row.get("tool", ""),
                "test_code": row.get("test_code", ""),
                "trust_tier": int(row.get("trust_tier", 2)),
            })

            if len(batch) >= BATCH_SIZE:
                t0 = time.perf_counter()
                res = vault.ingest_batch(batch, batch_size=2500, verify_math=False)
                dur_batch = time.perf_counter() - t0

                total_processed += res.get("total_processed", len(batch))
                total_new += res.get("new_rules", 0)
                total_dupes += res.get("duplicates_skipped", 0)
                total_rejected += res.get("sandbox_rejected", 0)

                print(
                    f"Progress: {total_processed:>6,} records | "
                    f"New Rules: +{total_new:>6,} | "
                    f"Duplicates: {total_dupes:>6,} | "
                    f"Batch time: {dur_batch:.2f}s"
                )
                batch.clear()
                time.sleep(0.01)

        # Flush final remaining batch
        if batch:
            t0 = time.perf_counter()
            res = vault.ingest_batch(batch, batch_size=2500, verify_math=False)
            dur_batch = time.perf_counter() - t0
            total_processed += res.get("total_processed", len(batch))
            total_new += res.get("new_rules", 0)
            total_dupes += res.get("duplicates_skipped", 0)
            total_rejected += res.get("sandbox_rejected", 0)
            print(
                f"Progress: {total_processed:>6,} records | "
                f"New Rules: +{total_new:>6,} | "
                f"Duplicates: {total_dupes:>6,} | "
                f"Batch time: {dur_batch:.2f}s"
            )
            batch.clear()

    total_dur = time.perf_counter() - t_start
    final_rules = vault.rule_count()
    vault.close()

    print("\n=== INGESTION SUMMARY ===")
    print(f"Source File:         {input_path.name}")
    print(f"Total Processed:     {total_processed:,}")
    print(f"New Rules Added:     +{total_new:,}")
    print(f"Duplicates Mapped:   {total_dupes:,}")
    print(f"Initial Vault Rules: {start_rules:,}")
    print(f"Final Vault Rules:   {final_rules:,}")
    print(f"Net Rule Increase:   +{final_rules - start_rules:,}")
    print(f"Elapsed Time:        {total_dur:.2f} seconds ({total_processed / max(0.01, total_dur):.1f} items/sec)")
    print("=========================")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    ingest_file(target)
