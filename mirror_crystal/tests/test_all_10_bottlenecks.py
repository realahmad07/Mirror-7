from pathlib import Path

from mirror_crystal.disk_vault import DiskCrystalVault
from mirror7_backend.service import BackendService


def _seed_test_vault(vault: DiskCrystalVault) -> None:
    # A tiny deterministic fixture replaces the workstation-only production vault.
    assert vault.ingest_pair(
        "What is SQLite write-ahead logging?",
        "SQLite WAL records changes before applying them to the main database file.",
        domain="systems",
    )
    assert vault.ingest_pair(
        "What is a deterministic blueprint rule?",
        "A deterministic blueprint rule maps a defined input condition to a reproducible output.",
        domain="general",
    )


def test_bottleneck_fixes(tmp_path: Path):
    vault_path = tmp_path / "test.dcrystal"
    vault = DiskCrystalVault(vault_path, lru_rule_cache_size=4096)
    try:
        _seed_test_vault(vault)

        # 1. Test Rule Cache Size
        assert vault.lru_rule_cache_size == 4096

        # 2. Test Feedback API (upvote / downvote)
        rule = vault.get_rule(1)
        assert rule is not None
        pass_count, fail_count = rule.pass_count, rule.fail_count
        res_up = vault.record_feedback(1, is_positive=True)
        assert res_up["pass_count"] == pass_count + 1
        res_down = vault.record_feedback(1, is_positive=False)
        assert res_down["fail_count"] == fail_count + 1

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
            domain="literature",
        )
        assert r1_added is True
        count1 = vault.rule_count()

        # Same answer, different question:
        r2_added = vault.ingest_pair(
            "Which author wrote the novel Pride and Prejudice?",
            "Jane Austen wrote Pride and Prejudice, published in 1813.",
            domain="literature",
        )
        assert r2_added is False  # Folded as synonym bridge!
        assert vault.rule_count() == count1

        # Query paraphrase returns exact answer
        res_para = vault.respond("Which author wrote the novel Pride and Prejudice?")
        assert "Jane Austen" in res_para.answer

        # 5. Test Compaction
        c_res = vault.compact()
        assert "bytes_reclaimed" in c_res
        # A tiny test DB can materialize its schema pages during VACUUM, so
        # compaction need not shrink it; verify completion and byte accounting.
        assert c_res["bytes_after"] > 0
        assert c_res["bytes_reclaimed"] == max(0, c_res["bytes_before"] - c_res["bytes_after"])
    finally:
        vault.close()


def test_service_readiness_and_warmup(tmp_path: Path):
    vault = DiskCrystalVault(tmp_path / "warmup.dcrystal")
    try:
        _seed_test_vault(vault)
        svc = BackendService(crystal_vault=vault, enable_crystal_auto_load=False)
        warm_res = svc.warmup()
        assert warm_res["ready"] is True
        assert svc.is_ready is True
        assert svc.warmup_ms > 0
        assert warm_res["rules"] == 2
    finally:
        vault.close()
