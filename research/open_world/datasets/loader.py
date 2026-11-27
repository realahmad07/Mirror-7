"""Runtime SHA-256 verified downloader and parser for license-approved external benchmarks.

POLICY:
- Downloads pinned upstream files strictly into gitignored ``.external_eval_cache/s11_open_world/``.
- Verifies exact byte length and SHA-256 digest before parsing.
- NEVER redistributes external benchmark rows into the Apache-2.0 repository.
- Pure Python 3.11 stdlib only (``csv``, ``hashlib``, ``json``, ``urllib.request``).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

from research.open_world.audit.dataset_license_audit import get_active_dataset_specs

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / ".external_eval_cache" / "s11_open_world"


class DatasetDownloadError(RuntimeError):
    """Raised when a pinned external benchmark cannot be fetched or fails SHA-256 verification."""


def _active_spec_map() -> dict[str, dict[str, Any]]:
    return {spec["dataset_id"]: spec for spec in get_active_dataset_specs()}


def _relative_cache_label(dest: Path) -> str:
    try:
        return dest.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return dest.name


def download_or_load_verified(
    url: str,
    dest: Path,
    expected_sha256: str,
    expected_size_bytes: int | None = None,
    *,
    allow_download: bool = True,
    timeout: int = 90,
) -> tuple[bytes, dict[str, Any]]:
    """Return verified file bytes from local cache or pinned URL, checking exact SHA-256."""
    expected_sha256_low = expected_sha256.casefold()
    source_used = "cache"

    if dest.is_file():
        data = dest.read_bytes()
        observed_sha = hashlib.sha256(data).hexdigest().casefold()
        if observed_sha == expected_sha256_low and (
            expected_size_bytes is None or len(data) == expected_size_bytes
        ):
            return data, {
                "cache_rel_path": _relative_cache_label(dest),
                "url": url,
                "sha256": observed_sha,
                "size_bytes": len(data),
                "source": source_used,
                "verified": True,
            }

    if not allow_download:
        raise DatasetDownloadError(
            f"Verified cached file missing at {dest} and allow_download=False"
        )

    source_used = "downloaded"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mirror7-S11-OpenWorld-Eval/1.0", "Accept": "*/*"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DatasetDownloadError(f"Failed to fetch pinned dataset {url}: {exc}") from exc

    observed_sha = hashlib.sha256(data).hexdigest().casefold()
    if observed_sha != expected_sha256_low:
        raise DatasetDownloadError(
            f"SHA-256 mismatch for {url}: got {observed_sha}, expected {expected_sha256_low}"
        )
    if expected_size_bytes is not None and len(data) != expected_size_bytes:
        raise DatasetDownloadError(
            f"Byte-length mismatch for {url}: got {len(data)}, expected {expected_size_bytes}"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return data, {
        "cache_rel_path": _relative_cache_label(dest),
        "url": url,
        "sha256": observed_sha,
        "size_bytes": len(data),
        "source": source_used,
        "verified": True,
    }


def load_balanced_copa_raw(
    cache_dir: Path = DEFAULT_CACHE_DIR, *, allow_download: bool = True
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    """Load verified Balanced COPA train.csv and test.csv rows."""
    spec = _active_spec_map()["balanced_copa"]
    file_meta: dict[str, Any] = {}
    parsed: dict[str, list[dict[str, str]]] = {}

    for fspec in spec["files"]:
        fname = fspec["name"]
        raw_bytes, meta = download_or_load_verified(
            url=fspec["url"],
            dest=cache_dir / "balanced_copa" / fname,
            expected_sha256=fspec["sha256"],
            expected_size_bytes=fspec["size_bytes"],
            allow_download=allow_download,
        )
        file_meta[fname] = meta
        reader = csv.DictReader(io.StringIO(raw_bytes.decode("utf-8")))
        parsed[fname] = [dict(row) for row in reader]

    return parsed["train.csv"], parsed["test.csv"], {
        "dataset_id": "balanced_copa",
        "license": spec["verified_license"],
        "pinned_revision": spec["pinned_revision"],
        "files": file_meta,
    }


def load_squad_v2_raw(
    cache_dir: Path = DEFAULT_CACHE_DIR, *, allow_download: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load verified SQuAD 2.0 dev-v2.0.json articles."""
    spec = _active_spec_map()["squad_v2"]
    fspec = spec["files"][0]
    raw_bytes, meta = download_or_load_verified(
        url=fspec["url"],
        dest=cache_dir / "squad_v2" / fspec["name"],
        expected_sha256=fspec["sha256"],
        expected_size_bytes=fspec["size_bytes"],
        allow_download=allow_download,
    )
    payload = json.loads(raw_bytes.decode("utf-8"))
    articles = list(payload.get("data") or [])
    return articles, {
        "dataset_id": "squad_v2",
        "license": spec["verified_license"],
        "pinned_revision": spec["pinned_revision"],
        "files": {fspec["name"]: meta},
    }


def load_longmemeval_raw(
    cache_dir: Path = DEFAULT_CACHE_DIR, *, allow_download: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load verified LongMemEval oracle JSON episodes."""
    spec = _active_spec_map()["longmemeval"]
    fspec = spec["files"][0]
    raw_bytes, meta = download_or_load_verified(
        url=fspec["url"],
        dest=cache_dir / "longmemeval" / fspec["name"],
        expected_sha256=fspec["sha256"],
        expected_size_bytes=fspec["size_bytes"],
        allow_download=allow_download,
    )
    episodes = json.loads(raw_bytes.decode("utf-8"))
    return list(episodes), {
        "dataset_id": "longmemeval",
        "license": spec["verified_license"],
        "pinned_revision": spec["pinned_revision"],
        "files": {fspec["name"]: meta},
    }
