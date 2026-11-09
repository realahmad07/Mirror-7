#!/usr/bin/env python3
"""Publish the owner-provided Crystal ZIP bundle into a local Mirror-7 Git checkout.

Example:
  python scripts/publish_crystal_bundle.py --bundle /path/to/Mirror_Crystal_Dataset_Integration_Bundle_v1.zip --replace-legacy-mcrystal --commit --push

The script verifies every bundle SHA-256 entry and confirms that crystal_4.zip
contains the exact same MCRYSTAL bytes as artifacts/mirror_crystal_v1.mcrystal
before deleting that redundant standalone asset. Each ZIP is below GitHub's
100 MB per-file limit, so Git LFS is not required.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(command: list[str], cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def find_bundle_root(extracted: Path) -> Path:
    roots = [p for p in extracted.iterdir() if p.is_dir() and (p / "raw_archives").is_dir()]
    if len(roots) == 1:
        return roots[0]
    if (extracted / "raw_archives").is_dir():
        return extracted
    raise ValueError("Bundle must contain one root folder with raw_archives/.")


def verify_checksums(root: Path) -> None:
    sums = root / "SHA256SUMS.txt"
    if not sums.is_file():
        raise ValueError("Bundle is missing SHA256SUMS.txt.")
    for line_no, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        rel = PurePosixPath(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unsafe path in checksum manifest: {relative}")
        path = root.joinpath(*rel.parts)
        if not path.is_file():
            raise FileNotFoundError(f"Checksum target missing: {relative}")
        actual = sha256(path.read_bytes())
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {relative}: expected {expected}, got {actual}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--branch", default="datasets/crystal-archive-integration-2026-11-10")
    parser.add_argument("--replace-legacy-mcrystal", action="store_true")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    if not (repo / ".git").exists():
        parser.error(f"Not a Git checkout: {repo}")
    current_branch = run(["git", "branch", "--show-current"], repo)
    if current_branch != args.branch:
        parser.error(f"Checkout branch {current_branch!r}; expected {args.branch!r}.")
    if args.push and not args.commit:
        parser.error("--push requires --commit.")

    with tempfile.TemporaryDirectory(prefix="mirror-crystal-publish-") as temp:
        extracted = Path(temp)
        with zipfile.ZipFile(args.bundle, "r") as archive:
            bad = archive.testzip()
            if bad:
                parser.error(f"Corrupt bundle entry: {bad}")
            for name in archive.namelist():
                rel = PurePosixPath(name)
                if rel.is_absolute() or ".." in rel.parts:
                    parser.error(f"Unsafe ZIP path: {name}")
            archive.extractall(extracted)

        root = find_bundle_root(extracted)
        verify_checksums(root)
        raw = root / "raw_archives"
        names = [
            "crystal_1.zip", "crystal_2.zip", "crystal_3.zip", "crystal_4.zip",
            "Mirror_General_Reasoning_10K_Candidate_Pack_v1.zip",
        ]
        for name in names:
            source = raw / name
            if not source.is_file():
                parser.error(f"Required archive missing from bundle: {name}")
            with zipfile.ZipFile(source, "r") as archive:
                bad = archive.testzip()
                if bad:
                    parser.error(f"{name} is corrupt at {bad}")

        with zipfile.ZipFile(raw / "crystal_4.zip", "r") as archive:
            members = [n for n in archive.namelist() if Path(n).name == "mirror_crystal_v1.mcrystal"]
            if len(members) != 1:
                parser.error("crystal_4.zip must contain exactly one MCRYSTAL payload.")
            embedded = archive.read(members[0])

        legacy = repo / "artifacts" / "mirror_crystal_v1.mcrystal"
        if args.replace_legacy_mcrystal:
            if not legacy.is_file() or sha256(legacy.read_bytes()) != sha256(embedded):
                parser.error("Old standalone MCRYSTAL is missing or differs from the embedded payload; refusing deletion.")
            source = (repo / "mirror_crystal" / "engine.py").read_text(encoding="utf-8")
            if "zipfile.is_zipfile(source_path)" not in source:
                parser.error("This checkout does not support ZIP brain loading; refusing to delete the standalone brain.")

        dest = repo / "datasets" / "crystal_archive" / "raw"
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(raw / name, dest / name)

        manifest = {
            "manifest_version": 1,
            "published_at": datetime.now().astimezone().isoformat(),
            "archives": [
                {"file": name, "bytes": (dest / name).stat().st_size, "sha256": sha256((dest / name).read_bytes())}
                for name in names
            ],
            "embedded_mcrystal_sha256": sha256(embedded),
            "note": "Mixed-source collection; see LICENSES.md."
        }
        (repo / "datasets" / "crystal_archive" / "raw_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        readme = repo / "datasets" / "crystal_archive" / "README.md"
        old = readme.read_text(encoding="utf-8") if readme.exists() else "# Crystal archive\n"
        section = """
## Public raw archives

- crystal_1.zip — triplets 1–9,857 and blueprint catalogue.
- crystal_2.zip — triplets 9,858–19,714, taxonomy and rule sample.
- crystal_3.zip — triplets 19,715–29,571 and evaluation report.
- crystal_4.zip — triplets 29,572–39,428 and MCRYSTAL brain.
- General Reasoning 10K candidate pack.

Load the MCRYSTAL brain directly from the archive:
python -m mirror_crystal query --brain datasets/crystal_archive/raw/crystal_4.zip --prompt "What is protocol code 1000?"

See raw_manifest.json for checksums and LICENSES.md for source terms.
"""
        if "## Public raw archives" not in old:
            readme.write_text(old.rstrip() + "\n\n" + section, encoding="utf-8")

        # Exercise Mirror's real runtime against the published ZIP before removing
        # the redundant standalone .mcrystal file.
        run(
            [
                sys.executable, "-m", "pytest", "-q",
                "mirror_crystal/tests/test_mirror_crystal.py",
                "-k", "mcrystal_archive_zip_loads_and_answers_stored_rule",
            ],
            repo,
        )
        run(
            [
                sys.executable, "scripts/test_crystal_archive_responses.py",
                "--brain", str(dest / "crystal_4.zip"),
            ],
            repo,
        )

        if args.replace_legacy_mcrystal:
            legacy.unlink()

        run(["git", "add", "datasets/crystal_archive"], repo)
        if args.replace_legacy_mcrystal:
            run(["git", "add", "-u", "artifacts/mirror_crystal_v1.mcrystal"], repo)
        if args.commit:
            run(["git", "commit", "-m", "Publish Crystal archives and replace duplicate MCRYSTAL file"], repo)
            if args.push:
                run(["git", "push", "origin", args.branch], repo)

    print(json.dumps({
        "published_archive_files": names,
        "raw_directory": str(repo / "datasets" / "crystal_archive" / "raw"),
        "replaced_legacy_mcrystal": bool(args.replace_legacy_mcrystal),
        "committed": bool(args.commit),
        "pushed": bool(args.push),
        "branch": args.branch
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
