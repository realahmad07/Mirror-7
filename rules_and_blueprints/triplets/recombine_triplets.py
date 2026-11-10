# Universal Dataset Recombiner for Mirror 7 Crystal Parts
import json
from pathlib import Path

parts = [
    Path("mirror_triplets_part1_of_4.jsonl"),
    Path("mirror_triplets_part2_of_4.jsonl"),
    Path("mirror_triplets_part3_of_4.jsonl"),
    Path("mirror_triplets_part4_of_4.jsonl"),
]

present = [p for p in parts if p.exists()]
print(f"Found {len(present)}/4 triplet part files in current directory.")

out_file = Path("mirror_39428_unique_triplets_recombined.jsonl")
count = 0
with open(out_file, "w", encoding="utf-8") as out_fp:
    for p in present:
        with open(p, "r", encoding="utf-8") as in_fp:
            for line in in_fp:
                if line.strip():
                    out_fp.write(line.strip() + "\n")
                    count += 1

print(f"Successfully recombined {count:,} triplets into {out_file.name} ({out_file.stat().st_size / (1024*1024):.2f} MB)")
