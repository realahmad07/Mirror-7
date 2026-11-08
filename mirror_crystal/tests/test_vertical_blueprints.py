"""
Unit tests for Specialized Vertical Intelligence Blueprints:
Systems programming, bioinformatics, and quantitative finance.
"""
from pathlib import Path
import tempfile
import pytest

from mirror_crystal.disk_vault import DiskCrystalVault
from mirror_crystal.vertical_blueprints import (
    VerticalBlueprintCurator,
    build_systems_programming_blueprints,
    build_bioinformatics_blueprints,
    build_quantitative_finance_blueprints,
)
from mirror_crystal.grammar_and_sandbox import ZeroTrustSandboxJudge


def test_vertical_candidate_generation():
    sys_recs = build_systems_programming_blueprints()
    bio_recs = build_bioinformatics_blueprints()
    fin_recs = build_quantitative_finance_blueprints()

    assert len(sys_recs) >= 4
    assert len(bio_recs) >= 3
    assert len(fin_recs) >= 4


def test_vertical_code_sandbox_verification():
    bio_recs = build_bioinformatics_blueprints()
    fin_recs = build_quantitative_finance_blueprints()

    # Needleman-Wunsch verification
    nw_rec = next(r for r in bio_recs if "Needleman-Wunsch" in r.prompt)
    ok_nw, err_nw = ZeroTrustSandboxJudge.verify_code_and_test(nw_rec.response, test_code=nw_rec.test_code)
    assert ok_nw is True, f"Needleman-Wunsch sandbox failed: {err_nw}"

    # BWT verification
    bwt_rec = next(r for r in bio_recs if "Burrows-Wheeler" in r.prompt)
    ok_bwt, err_bwt = ZeroTrustSandboxJudge.verify_code_and_test(bwt_rec.response, test_code=bwt_rec.test_code)
    assert ok_bwt is True, f"BWT sandbox failed: {err_bwt}"

    # Black-Scholes verification
    bs_rec = next(r for r in fin_recs if "Black-Scholes" in r.prompt)
    ok_bs, err_bs = ZeroTrustSandboxJudge.verify_code_and_test(bs_rec.response, test_code=bs_rec.test_code)
    assert ok_bs is True, f"Black-Scholes sandbox failed: {err_bs}"

    # Kelly Criterion verification
    kelly_rec = next(r for r in fin_recs if "Kelly" in r.prompt)
    ok_k, err_k = ZeroTrustSandboxJudge.verify_code_and_test(kelly_rec.response, test_code=kelly_rec.test_code)
    assert ok_k is True, f"Kelly sandbox failed: {err_k}"


def test_vertical_curation_and_ingestion():
    with tempfile.TemporaryDirectory() as td:
        vpath = Path(td) / "test_vertical_vault.dcrystal"
        vault = DiskCrystalVault(vpath)
        curator = VerticalBlueprintCurator()
        report = curator.curate_and_ingest(vault, report_path=Path(td) / "report.json")

        assert report["clean_passed"] == 11
        assert report["vault_stats"]["new_rules"] == 11
        assert vault.rule_count() == 11

        # Query verification
        resp = vault.respond("Write a Python function to compute the Needleman-Wunsch global sequence alignment score and traceback.")
        assert "needleman_wunsch" in resp.answer
        assert resp.mode == "direct_crystal_lock"

        resp_fin = vault.respond("Write a Python function to compute European call and put option prices using the Black-Scholes-Merton formula.")
        assert "black_scholes_call_put" in resp_fin.answer

        vault.close()
