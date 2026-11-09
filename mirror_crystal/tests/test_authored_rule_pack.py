from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "research" / "rule_packs" / "mirror_domain_seed_v1.jsonl"


def test_authored_rule_pack_is_balanced_and_explicitly_unverified():
    rows = [json.loads(line) for line in PACK.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 110
    assert set(Counter(row["domain"] for row in rows).values()) == {10}
    prompts = [re.sub(r"[^\\w]+", " ", row["question"].casefold()).strip() for row in rows]
    assert len(prompts) == len(set(prompts))
    assert all(row["trust_tier"] == 3 for row in rows)
    assert all(row["evidence_status"] == "unverified" for row in rows)
    assert all(row["source"] == "mirror_authored_seed_v1" for row in rows)
    assert all(len(row["answer"]) >= 40 for row in rows)
