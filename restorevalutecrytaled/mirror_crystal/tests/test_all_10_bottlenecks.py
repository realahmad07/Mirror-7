import tempfile
import shutil
import pytest
from pathlib import Path

from mirror_crystal.disk_vault import DiskCrystalVault
from mirror7_backend.service import BackendService


def test_bottleneck_fixes():
    # The production vault is optional in source control. Exercise the complete
    # production-vault path when the actual SQLite file is present; never create
    # a substitute empty database just to satisfy this integration fixture.
    production_vault = (
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "mirror_crystal_all_datasets.dcrystal"
    )
    if not production_vault.is_file():
        pytest.skip("Production .dcrystal vault is not present in this checkout")

    temp_dir = tempfile.mkdtemp()
    try:
        vault_path = Path(temp_dir) / "test.dcrystal"
        shutil.copy2(production_vault, vault_path)

        vault = DiskCrystalVault(vault_path, lru_rule_cache_size=4096)

        # 1. Test Rule Cache Size
        assert vault.lru_rule_cache_size == 4096

        # 2. Test Feedback API (upvote / downvote)
        r = vault.get_rule(1)
        if r is not None:
            pc, fc = r.pass_count, r.fail_count
            res_up = vault.record_feedback(1, is_positive=True)
            assert res_up["pass_count"] == pc + 1
            res_down = vault.record_feedback(1, is_positive=False)
            assert res_down["fail_count"] == fc + 1

        # 3. Test Coverage Gaps on abstention
        res_abs = vault.respond("Who won the 2045 lunar marathon championship?")
        assert res_abs.epistemic_tier == "honest_abstain"
        gaps = vault.get_coverage_gaps(limit=5)
        assert len(gaps) > 0
        prompt = vault.generate_gap_fill_prompt()
        assert "Topic:" in prompt

        # 4. Test Paraphrase Deduplication
        r1_added = vault.ingest_pair(
            "Who wrote the novel Pride and Prejudice?",
            "Jane Austen wrote Pride and Prejudice, published in 1813.",
            domain="literature"
        )
        assert r1_added is True
        count1 = vault.rule_count()

        # Same answer, different question:
        r2_added = vault.ingest_pair(
            "Which author wrote the novel Pride and Prejudice?",
            "Jane Austen wrote Pride and Prejudice, published in 1813.",
            domain="literature"
        )
        assert r2_added is False  # Folded as synonym bridge!
        assert vault.rule_count() == count1

        # Query paraphrase returns exact answer
        res_para = vault.respond("Which author wrote the novel Pride and Prejudice?")
        assert "Jane Austen" in res_para.answer

        # 5. Test Compaction
        c_res = vault.compact()
        assert "bytes_reclaimed" in c_res
        assert c_res["mb_after"] <= c_res["mb_before"]

        vault.close()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_service_readiness_and_warmup():
    svc = BackendService(enable_crystal_auto_load=True)
    warm_res = svc.warmup()
    assert warm_res["ready"] is True
    assert svc.is_ready is True
    assert isinstance(svc.warmup_ms, (int, float))
    assert svc.warmup_ms >= 0.0
    assert svc.warmup_ms == warm_res["warmup_ms"]
