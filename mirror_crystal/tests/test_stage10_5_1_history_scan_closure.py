"""Stage 10.5.1: exhaustive Git-history and dynamic secret-classification tests."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

from mirror_eval.stage10_5_independent_trust_validation import GitHistoricalObjectScanner


def _git_repo(root: Path, filename: str, payload: bytes) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "mirror-test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Mirror Scanner Test"], cwd=root, check=True)
    target = root / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    subprocess.run(["git", "add", filename], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scanner fixture"], cwd=root, check=True)
    return target


def test_scans_large_blobs_and_computes_unrevoked_secret_count(tmp_path: Path) -> None:
    pem = (
        b"-----BEGIN " + b"PRIVATE KEY-----\n"
        + b"A" * 96
        + b"\n-----END " + b"PRIVATE KEY-----\n"
    )
    payload = (b"ordinary-binary-content-" * 26000) + pem
    root = tmp_path / "large-blob-repo"
    _git_repo(root, "fixtures/large-secret.bin", payload)

    result = GitHistoricalObjectScanner.scan_all_historical_objects(root)

    assert result["historical_blobs_scanned"] == 1
    assert result["large_blobs_scanned"] == 1
    assert result["total_blob_bytes_scanned"] >= 500_000
    assert result["unrevoked_secrets_in_history"] == 1
    assert result["all_historical_findings_covered_by_crl"] is False
    assert result["working_tree_clean"] is False
    assert any("raw_pem_private_key" in item["matches"] for item in result["findings"])


def test_known_compromised_seed_is_classified_as_crl_covered(tmp_path: Path) -> None:
    seed_marker = b"mirror7.reviewer." + b"elena-vance.external-ai-audit.seed.2026"
    root = tmp_path / "revoked-seed-repo"
    _git_repo(root, "legacy/reviewer-seed.txt", seed_marker)

    result = GitHistoricalObjectScanner.scan_all_historical_objects(root)

    assert result["historical_blobs_scanned"] == 1
    assert result["unrevoked_secrets_in_history"] == 0
    assert result["all_historical_findings_covered_by_crl"] is True
    assert "historical_reviewer_seed" in result["findings"][0]["matches"]


def test_raw_hex_seed_is_derived_and_checked_against_crl(tmp_path: Path) -> None:
    seed_marker = b"mirror7.reviewer." + b"marcus-chen.metrology-institute.seed.2026"
    seed_hex = hashlib.sha256(seed_marker).hexdigest()
    content = f'private_key_hex = "{seed_hex}"\n'.encode("ascii")
    root = tmp_path / "revoked-hex-seed-repo"
    _git_repo(root, "legacy/key_material.py", content)

    result = GitHistoricalObjectScanner.scan_all_historical_objects(root)

    assert result["unrevoked_secrets_in_history"] == 0
    assert result["all_historical_findings_covered_by_crl"] is True
    assert "historical_reviewer_seed" in result["findings"][0]["matches"]


def test_unregistered_ed25519_seed_is_not_reported_as_zero(tmp_path: Path) -> None:
    seed_hex = bytes(range(32)).hex()
    content = f'private_key_hex = "{seed_hex}"\n'.encode("ascii")
    root = tmp_path / "unregistered-seed-repo"
    _git_repo(root, "leak/key_material.py", content)

    result = GitHistoricalObjectScanner.scan_all_historical_objects(root)

    assert result["unrevoked_secrets_in_history"] == 1
    assert result["all_historical_findings_covered_by_crl"] is False
    assert "hardcoded_ed25519_private_key" in result["findings"][0]["matches"]
