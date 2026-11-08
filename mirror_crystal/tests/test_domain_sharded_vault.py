import tempfile
import shutil
import pytest
from pathlib import Path

from mirror_crystal.disk_vault import DiskCrystalVault
from mirror_crystal.domain_sharded_vault import DomainShardedVault


def test_domain_sharded_vault_routing():
    temp_dir = tempfile.mkdtemp()
    try:
        primary_path = Path(temp_dir) / "primary.dcrystal"
        shards_dir = Path(temp_dir) / "shards"

        p_vault = DiskCrystalVault(primary_path)
        router = DomainShardedVault(p_vault, shards_dir=shards_dir)

        # Ingest into distinct shards
        router.ingest_pair("What is photosynthesis?", "Process of plants converting light to energy.", domain="biology")
        router.ingest_pair("State Newton first law", "An object remains at rest unless acted upon.", domain="physics")
        router.ingest_pair("What is Python GIL?", "Global Interpreter Lock in CPython.", domain="programming")

        shards = router.list_shards()
        shard_names = {s["domain"] for s in shards}
        assert "biology" in shard_names
        assert "physics" in shard_names
        assert "programming" in shard_names

        # Direct domain routing
        res_bio = router.respond("What is photosynthesis?", domain="biology")
        assert "photosynthesis" in res_bio.answer.lower() or "converting light" in res_bio.answer.lower()
        assert res_bio.confidence >= 0.85

        # Fallback multi-shard routing
        res_phys = router.respond("State Newton first law")
        assert "object remains" in res_phys.answer.lower()

        p_vault.close()
        for s in router._shards.values():
            s.close()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
