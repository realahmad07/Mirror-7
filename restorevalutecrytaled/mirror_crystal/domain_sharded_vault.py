"""DomainShardedVault — Multi-Shard Domain Router for Zero-Lock Scaling.

Distributes large dataset rules across domain-specific shards (e.g. physics, math, code, general)
so that writes to one domain never block reads or writes to other domains.
"""
from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Any, Optional

from .disk_vault import DiskCrystalVault, CrystalResponse, CrystalRule


class DomainShardedVault:
    """
    Sharded Crystal Vault router that partitions rules by domain into distinct .dcrystal files.
    Queries check target domain shard or primary vault.
    Writes with domain route directly to that domain's shard, enabling concurrent multi-domain writes.
    """

    def __init__(
        self,
        primary_vault: DiskCrystalVault,
        shards_dir: Path | str = "artifacts/shards",
        lru_rule_cache_size: int = 4096,
    ):
        self.primary = primary_vault
        self.shards_dir = Path(shards_dir)
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.lru_rule_cache_size = lru_rule_cache_size
        self._shards: dict[str, DiskCrystalVault] = {}
        self._lock = threading.RLock()
        self._load_existing_shards()

    def _load_existing_shards(self) -> None:
        with self._lock:
            for p in self.shards_dir.glob("*.dcrystal"):
                dom = p.stem.replace("vault_", "")
                self._shards[dom] = DiskCrystalVault(p, lru_rule_cache_size=self.lru_rule_cache_size)

    def get_or_create_shard(self, domain: str) -> DiskCrystalVault:
        dom_clean = domain.strip().lower().replace(" ", "_")
        with self._lock:
            if dom_clean in self._shards:
                return self._shards[dom_clean]
            shard_path = self.shards_dir / f"vault_{dom_clean}.dcrystal"
            shard = DiskCrystalVault(shard_path, lru_rule_cache_size=self.lru_rule_cache_size)
            self._shards[dom_clean] = shard
            return shard

    def respond(self, prompt: str, session_id: str = "default", domain: Optional[str] = None) -> CrystalResponse:
        # If domain shard is specified, query it first
        if domain:
            dom_clean = domain.strip().lower().replace(" ", "_")
            if dom_clean in self._shards:
                res = self._shards[dom_clean].respond(prompt, session_id=session_id)
                if res.epistemic_tier != "honest_abstain":
                    return res

        # Primary query
        res_primary = self.primary.respond(prompt, session_id=session_id)
        if res_primary.epistemic_tier != "honest_abstain" or res_primary.confidence >= 0.75:
            return res_primary

        # Fallback query across shards if primary abstains
        with self._lock:
            shards_list = list(self._shards.values())

        best_res = res_primary
        for s in shards_list:
            res_s = s.respond(prompt, session_id=session_id)
            if res_s.epistemic_tier != "honest_abstain" and res_s.confidence > best_res.confidence:
                best_res = res_s
                if best_res.confidence >= 0.90:
                    break
        return best_res

    def ingest_pair(
        self,
        user_text: str,
        target_text: str,
        domain: str = "general",
        **kwargs: Any,
    ) -> bool:
        dom_clean = domain.strip().lower().replace(" ", "_")
        if dom_clean not in ("general", "", "all"):
            shard = self.get_or_create_shard(dom_clean)
            return shard.ingest_pair(user_text, target_text, domain=dom_clean, **kwargs)
        return self.primary.ingest_pair(user_text, target_text, domain=domain, **kwargs)

    def rule_count(self) -> int:
        total = self.primary.rule_count()
        with self._lock:
            for s in self._shards.values():
                total += s.rule_count()
        return total

    def list_shards(self) -> list[dict[str, Any]]:
        with self._lock:
            out = [{
                "shard": "primary",
                "domain": "all",
                "rules": self.primary.rule_count(),
                "path": str(self.primary.db_path),
            }]
            for dom, s in self._shards.items():
                out.append({
                    "shard": f"vault_{dom}",
                    "domain": dom,
                    "rules": s.rule_count(),
                    "path": str(s.db_path),
                })
            return out
