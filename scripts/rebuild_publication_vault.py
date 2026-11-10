#!/usr/bin/env python3
"""Rebuild Mirror's production DiskCrystalVault from the published Crystal bundle.

The original source files remain immutable. Exact prompt collisions with conflicting
answers are quarantined; all imported data stays unverified/trust tier 3 unless
independent, record-level evidence exists. Canonical blueprints preserve source
evidence, family/context IDs, and test status in an additive sidecar table.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "rules_and_blueprints"
OUTPUT = ROOT / "artifacts" / "mirror_crystal_all_datasets.dcrystal"
REPORT = ROOT / "artifacts" / "mirror_crystal_rebuild_report.json"
MCRYSTAL = ROOT / "artifacts" / "mirror_crystal_v1.mcrystal"
TRIPLET_FILES = [DATA / "triplets" / f"mirror_triplets_part{i}_of_4.jsonl" for i in range(1, 5)]
ROADMAP = DATA / "blueprints" / "mirror_all_1014_blueprints.jsonl"
SAMPLE_RULES = DATA / "verified_sample_rules" / "sample_verified_rules_pillars.jsonl"
GR_DIR = DATA / "general_reasoning"
GR_CANDIDATES = GR_DIR / "rules" / "general_reasoning_candidates_10240.jsonl"
GR_SOURCE = GR_DIR / "rules" / "general_reasoning_source_supported_26.jsonl"
GR_VALIDATOR = GR_DIR / "tools" / "validate_pack.py"
GR_CONVERTER = GR_DIR / "tools" / "convert_to_canonical_blueprints.py"
MAX_GIT_FILE_BYTES = 90 * 1024 * 1024
HOLDOUT_SPLITS = {"test", "eval", "eval_holdout", "validation", "val", "dev"}

sys.path.insert(0, str(ROOT))

from mirror_crystal import DiskCrystalVault, MirrorCrystalEngine  # noqa: E402
from mirror_crystal.blueprint_contracts import (  # noqa: E402
    Blueprint, BlueprintKind, BlueprintProvenance, EvidenceLevel, VerificationEvidence,
)
from mirror_crystal.blueprint_prompt import normalize_blueprint_text  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256_manifest(base: Path, manifest: Path) -> dict[str, str]:
    verified: dict[str, str] = {}
    for line_no, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"{manifest}:{line_no}: malformed SHA256 entry")
        expected, relative = parts[0], parts[1].strip().lstrip("*")
        target = base / relative
        if not target.is_file():
            raise FileNotFoundError(f"{manifest}:{line_no}: checksum target missing: {target}")
        actual = sha256_file(target)
        if actual.lower() != expected.lower():
            raise ValueError(f"{manifest}:{line_no}: SHA-256 mismatch for {relative}")
        verified[relative] = actual
    return verified


def domain_name(raw: Any) -> str:
    value = str(raw or "general").strip()
    if not value:
        return "general"
    if normalize_blueprint_text(value) == "provide a helpful accurate response":
        return "general_chat"
    return value


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        for key in ("text", "content", "value", "message"):
            if key in value:
                return as_text(value[key])
        return ""
    if isinstance(value, list):
        return "\n".join(part for part in (as_text(item) for item in value) if part)
    return ""


def jsonl_read(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: JSONL record must be an object")
            yield line_no, obj


def create_stage(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE pairs (
            prompt_key TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            response_key TEXT NOT NULL,
            domain TEXT NOT NULL,
            tool TEXT NOT NULL DEFAULT '',
            sources_json TEXT NOT NULL,
            domains_json TEXT NOT NULL,
            conflict INTEGER NOT NULL DEFAULT 0,
            first_source TEXT NOT NULL,
            first_line INTEGER NOT NULL
        );
        CREATE TABLE pair_conflicts (
            prompt_key TEXT NOT NULL,
            response_key TEXT NOT NULL,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            source TEXT NOT NULL,
            source_line INTEGER NOT NULL,
            domain TEXT NOT NULL,
            PRIMARY KEY(prompt_key, response_key)
        );
        CREATE TABLE blueprints (
            prompt_key TEXT PRIMARY KEY,
            blueprint_id TEXT NOT NULL,
            canonical_json TEXT NOT NULL,
            sidecar_json TEXT NOT NULL,
            response_key TEXT NOT NULL,
            source_pack TEXT NOT NULL,
            conflict INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE blueprint_conflicts (
            prompt_key TEXT NOT NULL,
            blueprint_id TEXT NOT NULL,
            canonical_json TEXT NOT NULL,
            sidecar_json TEXT NOT NULL,
            source_pack TEXT NOT NULL,
            response_key TEXT NOT NULL,
            PRIMARY KEY(prompt_key, blueprint_id)
        );
        CREATE TABLE quarantine (
            source TEXT NOT NULL,
            source_id TEXT NOT NULL,
            reason TEXT NOT NULL,
            record_json TEXT NOT NULL
        );
    """)
    return conn


def add_pair(conn: sqlite3.Connection, prompt: str, response: str, domain: str,
             tool: str, source: str, line_no: int, stats: dict[str, int]) -> None:
    prompt = str(prompt or "").strip()
    response = str(response or "").strip()
    key = normalize_blueprint_text(prompt)
    response_key = normalize_blueprint_text(response)
    if not key or not response_key:
        stats["empty_or_incomplete_pairs"] = stats.get("empty_or_incomplete_pairs", 0) + 1
        return
    domain = domain_name(domain)
    existing = conn.execute(
        "SELECT prompt, response, response_key, domain, tool, sources_json, domains_json, conflict, first_source, first_line "
        "FROM pairs WHERE prompt_key=?", (key,)
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO pairs(prompt_key,prompt,response,response_key,domain,tool,sources_json,domains_json,conflict,first_source,first_line) "
            "VALUES(?,?,?,?,?,?,?,?,0,?,?)",
            (key, prompt, response, response_key, domain, str(tool or ""),
             json.dumps([source]), json.dumps([domain]), source, int(line_no))
        )
        stats["staged_unique_prompt_pairs"] = stats.get("staged_unique_prompt_pairs", 0) + 1
        return

    if existing[2] == response_key:
        sources = set(json.loads(existing[5]))
        domains = set(json.loads(existing[6]))
        sources.add(source)
        domains.add(domain)
        conn.execute(
            "UPDATE pairs SET sources_json=?,domains_json=? WHERE prompt_key=?",
            (json.dumps(sorted(sources)), json.dumps(sorted(domains)), key)
        )
        stats["exact_duplicates_collapsed"] = stats.get("exact_duplicates_collapsed", 0) + 1
        return

    conn.execute("UPDATE pairs SET conflict=1 WHERE prompt_key=?", (key,))
    conn.execute(
        "INSERT OR IGNORE INTO pair_conflicts VALUES(?,?,?,?,?,?,?)",
        (key, existing[2], existing[0], existing[1], existing[8], int(existing[9]), existing[3])
    )
    conn.execute(
        "INSERT OR IGNORE INTO pair_conflicts VALUES(?,?,?,?,?,?,?)",
        (key, response_key, prompt, response, source, int(line_no), domain)
    )
    stats["conflicting_prompt_keys"] = stats.get("conflicting_prompt_keys", 0) + 1


def record_quarantine(conn: sqlite3.Connection, source: str, source_id: str,
                      reason: str, record: Any) -> None:
    conn.execute(
        "INSERT INTO quarantine(source,source_id,reason,record_json) VALUES(?,?,?,?)",
        (source, str(source_id), reason,
         json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    )


def process_triplet_file(conn: sqlite3.Connection, path: Path,
                         all_stats: dict[str, Any]) -> None:
    source = path.relative_to(ROOT).as_posix()
    stats = all_stats["sources"].setdefault(source, {
        "records_read": 0, "pairs_seen": 0, "holdout_records_skipped": 0,
        "empty_or_incomplete_pairs": 0, "staged_unique_prompt_pairs": 0,
        "exact_duplicates_collapsed": 0, "conflicting_prompt_keys": 0,
    })
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_no, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                raise ValueError(f"{source}:{line_no}: invalid UTF-8 JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{source}:{line_no}: record must be an object")
            stats["records_read"] += 1
            split = str(obj.get("split") or "").strip().lower()
            if split in HOLDOUT_SPLITS:
                stats["holdout_records_skipped"] += 1
                continue

            signature = ",".join(sorted(str(key) for key in obj.keys()))
            signatures = stats.setdefault("record_field_signatures", {})
            signatures[signature] = int(signatures.get(signature, 0)) + 1

            state = obj.get("state") if isinstance(obj.get("state"), dict) else {}
            outer_domain = (
                state.get("domain") or obj.get("domain") or obj.get("category")
                or obj.get("goal") or "general"
            )

            turns = obj.get("turns")
            if isinstance(turns, list):
                for turn_idx, turn in enumerate(turns):
                    if not isinstance(turn, dict):
                        continue
                    if str(turn.get("split") or "").strip().lower() in HOLDOUT_SPLITS:
                        continue
                    prompt = next((as_text(turn.get(key)) for key in
                                   ("user_text", "user", "prompt", "instruction", "question", "query", "anchor", "input")
                                   if as_text(turn.get(key))), "")
                    response = next((as_text(turn.get(key)) for key in
                                     ("target_text", "response", "completion", "output", "answer",
                                      "positive_response", "positive", "chosen", "preferred", "target")
                                     if as_text(turn.get(key))), "")
                    turn_state = turn.get("state") if isinstance(turn.get("state"), dict) else {}
                    domain = turn_state.get("domain") or turn.get("domain") or outer_domain
                    stats["pairs_seen"] += 1
                    if not prompt or not response:
                        stats["unmapped_records"] = stats.get("unmapped_records", 0) + 1
                    add_pair(conn, prompt, response, domain, turn.get("tool", ""),
                             source, line_no * 1000 + turn_idx, stats)
            else:
                prompt = next((as_text(obj.get(key)) for key in
                               ("user_text", "user", "prompt", "instruction", "question", "query", "anchor", "input_text", "input")
                               if as_text(obj.get(key))), "")
                response = next((as_text(obj.get(key)) for key in
                                 ("target_text", "response", "completion", "output", "answer",
                                  "positive_response", "positive", "chosen", "preferred", "target", "assistant")
                                 if as_text(obj.get(key))), "")
                # Some source triplets use positional [anchor, positive, negative] records.
                if (not prompt or not response):
                    positional = obj.get("triplet") or obj.get("texts") or obj.get("examples")
                    if isinstance(positional, list) and len(positional) >= 2:
                        prompt = prompt or as_text(positional[0])
                        response = response or as_text(positional[1])
                # A structured subject-predicate-object record is rendered as a factual Q/A pair.
                if (not prompt or not response) and all(k in obj for k in ("subject", "predicate", "object")):
                    subject = as_text(obj.get("subject"))
                    predicate = as_text(obj.get("predicate"))
                    object_value = as_text(obj.get("object"))
                    prompt = prompt or f"What is the {predicate} of {subject}?"
                    response = response or f"{subject} {predicate} {object_value}."
                stats["pairs_seen"] += 1
                if not prompt or not response:
                    stats["unmapped_records"] = stats.get("unmapped_records", 0) + 1
                row_source_dataset = as_text(obj.get("source_dataset"))
                row_source = f"{source}#source_dataset={row_source_dataset}" if row_source_dataset else source
                add_pair(conn, prompt, response, outer_domain, as_text(obj.get("tool")),
                         row_source, line_no, stats)
    stats["sha256"] = digest.hexdigest()
    stats["bytes"] = path.stat().st_size


def add_blueprint(conn: sqlite3.Connection, blueprint: Blueprint, sidecar: dict[str, Any],
                  source_pack: str, stats: dict[str, int]) -> None:
    key = normalize_blueprint_text(blueprint.prompt_pattern)
    response_key = normalize_blueprint_text(blueprint.response_template)
    if not key or not response_key:
        stats["invalid_blueprints"] = stats.get("invalid_blueprints", 0) + 1
        return
    canonical = blueprint.canonical_json()
    sidecar_json = json.dumps(sidecar, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    current = conn.execute(
        "SELECT blueprint_id,canonical_json,sidecar_json,response_key,source_pack,conflict "
        "FROM blueprints WHERE prompt_key=?", (key,)
    ).fetchone()
    if current is None:
        conn.execute(
            "INSERT INTO blueprints VALUES(?,?,?,?,?,?,0)",
            (key, blueprint.blueprint_id, canonical, sidecar_json, response_key, source_pack)
        )
        stats["blueprints_staged"] = stats.get("blueprints_staged", 0) + 1
    elif current[3] == response_key:
        stats["exact_blueprint_duplicates_collapsed"] = stats.get("exact_blueprint_duplicates_collapsed", 0) + 1
    else:
        conn.execute("UPDATE blueprints SET conflict=1 WHERE prompt_key=?", (key,))
        conn.execute(
            "INSERT OR IGNORE INTO blueprint_conflicts VALUES(?,?,?,?,?,?)",
            (key, current[0], current[1], current[2], current[4], current[3])
        )
        conn.execute(
            "INSERT OR IGNORE INTO blueprint_conflicts VALUES(?,?,?,?,?,?)",
            (key, blueprint.blueprint_id, canonical, sidecar_json, source_pack, response_key)
        )
        stats["blueprint_conflict_keys"] = stats.get("blueprint_conflict_keys", 0) + 1


def roadmap_blueprint(row: dict[str, Any]) -> tuple[Blueprint, dict[str, Any]]:
    prompt = str(row.get("prompt") or "").strip()
    response = str(row.get("response") or "").strip()
    stage = str(row.get("stage") or "legacy_roadmap")
    domain = str(row.get("domain") or "general")
    tool = str(row.get("tool") or "")
    digest = Blueprint._digest(prompt, response)
    kind = BlueprintKind.PROCEDURE if tool or str(row.get("test_code") or "").strip() else BlueprintKind.FACT
    slots = tuple(sorted(set(re.findall(r"\{(v\d+)\}", prompt)),
                         key=lambda item: int(item[1:])))
    bp = Blueprint(
        blueprint_id=f"roadmap:{row.get('blueprint_id')}",
        kind=kind,
        domain=domain,
        # Stage is a curriculum bucket, not a single semantic intent. Give each
        # distinct prompt its own intent key so unrelated lessons in the same
        # stage are not reported as conflicting answers.
        intent=f"roadmap_{re.sub(r'[^a-z0-9]+', '_', stage.casefold()).strip('_')}_{digest[:16]}",
        prompt_pattern=prompt,
        response_template=response,
        slots=slots,
        provenance=BlueprintProvenance(
            source="mirror_crystal_publication_bundle_v2",
            source_ref=f"roadmap:{row.get('blueprint_id')}:{stage}",
            license_spdx="NOASSERTION",
            content_sha256=digest,
        ),
        verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
        legacy_trust_tier=3,
    )
    sidecar = {
        "source_pack": "roadmap_blueprints_1014",
        "original_trust_tier": row.get("trust_tier"),
        "original_type": row.get("type"),
        "stage": stage,
        "domain": domain,
        "tool": tool,
        "test_code": row.get("test_code", ""),
        "test_status": "not_run",
        "license_spdx": "NOASSERTION",
        "verification_status": "unverified",
    }
    return bp, sidecar


def stage_general_reasoning(conn: sqlite3.Connection, all_stats: dict[str, Any],
                            work_dir: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(GR_VALIDATOR)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"General Reasoning validation failed:\n{result.stdout}\n{result.stderr}")
    all_stats["general_reasoning_validator"] = result.stdout.strip()
    checksum_manifest = GR_DIR / "SHA256SUMS.txt"
    verified_hashes = verify_sha256_manifest(GR_DIR, checksum_manifest)
    all_stats["general_reasoning_checksums_verified"] = len(verified_hashes)

    for source_path, label in (
        (GR_CANDIDATES, "general_reasoning_candidates_10240"),
        (GR_SOURCE, "general_reasoning_source_supported_26"),
    ):
        canonical_path = work_dir / f"{label}.canonical.jsonl"
        sidecar_path = work_dir / f"{label}.sidecar.jsonl"
        converted = subprocess.run(
            [sys.executable, str(GR_CONVERTER), "--input", str(source_path),
             "--output", str(canonical_path), "--sidecar", str(sidecar_path)],
            cwd=ROOT, text=True, capture_output=True,
        )
        if converted.returncode != 0:
            raise RuntimeError(f"Canonical conversion failed for {source_path}:\n{converted.stdout}\n{converted.stderr}")
        input_count = 0
        with canonical_path.open("r", encoding="utf-8") as canon, sidecar_path.open("r", encoding="utf-8") as side:
            for line_no, (canonical_line, side_line) in enumerate(zip(canon, side), 1):
                if not canonical_line.strip():
                    continue
                canonical = json.loads(canonical_line)
                sidecar = json.loads(side_line)
                bp = Blueprint.from_dict(canonical)
                original_status = str(sidecar.get("original_evidence_status") or "unverified")
                sidecar["source_pack"] = label
                sidecar["canonical_verification_status"] = bp.verification.level.value
                sidecar["original_evidence_status"] = original_status
                add_blueprint(conn, bp, sidecar, label, all_stats["blueprints"])
                input_count += 1
        all_stats["sources"][source_path.relative_to(ROOT).as_posix()] = {
            "records_converted": input_count,
            "sha256": sha256_file(source_path),
            "bytes": source_path.stat().st_size,
        }
        if label == "general_reasoning_candidates_10240" and input_count != 10240:
            raise RuntimeError(f"Expected 10,240 candidates, converted {input_count}")
        if label == "general_reasoning_source_supported_26" and input_count != 26:
            raise RuntimeError(f"Expected 26 source-indexed records, converted {input_count}")


def import_rule_sources(stage: sqlite3.Connection, all_stats: dict[str, Any]) -> None:
    # Rehydrate the 6,798-rule in-memory Crystal artifact as low-trust candidates.
    if not MCRYSTAL.is_file():
        raise FileNotFoundError(f"Tracked source Crystal artifact is missing: {MCRYSTAL}")
    engine = MirrorCrystalEngine.load(MCRYSTAL)
    brain_stats = dict(engine.stats)
    brain_rules = engine.rule_index.rules
    bundle_manifest_path = DATA / "bundle_metadata" / "manifest.json"
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    embedded_info = next(
        (item.get("embedded_brain") for item in bundle_manifest.get("archives", [])
         if item.get("embedded_brain")),
        None,
    )
    if not embedded_info:
        raise RuntimeError("Bundle manifest does not identify the embedded MCRYSTAL brain")
    actual_brain_sha = sha256_file(MCRYSTAL)
    if actual_brain_sha != embedded_info.get("sha256"):
        raise RuntimeError("Embedded MCRYSTAL checksum does not match the publication manifest")
    if len(brain_rules) != int(embedded_info.get("rule_count", -1)):
        raise RuntimeError("Embedded MCRYSTAL rule count does not match the publication manifest")
    mstats = {
        "source_rule_records": len(brain_rules),
        "active_rules_staged": 0,
        "inactive_rules_held": 0,
        "empty_or_incomplete_pairs": 0,
        "staged_unique_prompt_pairs": 0,
        "exact_duplicates_collapsed": 0,
        "conflicting_prompt_keys": 0,
    }
    source = MCRYSTAL.relative_to(ROOT).as_posix()
    for rule in brain_rules:
        if not rule.active:
            mstats["inactive_rules_held"] += 1
            continue
        mstats["active_rules_staged"] += 1
        add_pair(stage, rule.abstract_prompt, rule.abstract_response, rule.domain,
                 rule.abstract_tool, source, int(rule.rule_id), mstats)
    mstats["sha256"] = sha256_file(MCRYSTAL)
    mstats["bytes"] = MCRYSTAL.stat().st_size
    mstats["embedded_engine_stats"] = brain_stats
    all_stats["sources"][source] = mstats

    # Stream all four triplet parts. Test/evaluation/validation partitions are not activated.
    for path in TRIPLET_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"Missing triplet part: {path}")
        process_triplet_file(stage, path, all_stats)

    # The 500-row sample is historical reference data. Canary fixtures are kept out of active knowledge.
    sample_source = SAMPLE_RULES.relative_to(ROOT).as_posix()
    sample_stats = {
        "records_read": 0, "pairs_seen": 0, "canary_fixtures_held": 0,
        "empty_or_incomplete_pairs": 0, "staged_unique_prompt_pairs": 0,
        "exact_duplicates_collapsed": 0, "conflicting_prompt_keys": 0,
        "license_status": "NOASSERTION", "evidence_status": "unverified",
    }
    for line_no, row in jsonl_read(SAMPLE_RULES):
        sample_stats["records_read"] += 1
        if str(row.get("domain", "")).startswith("canary_"):
            sample_stats["canary_fixtures_held"] += 1
            record_quarantine(stage, sample_source, row.get("rule_id", line_no),
                              "canary fixture; not domain knowledge", row)
            continue
        sample_stats["pairs_seen"] += 1
        add_pair(stage, row.get("abstract_prompt", ""), row.get("abstract_response", ""),
                 row.get("domain", "general"), "", sample_source, line_no, sample_stats)
    sample_stats["sha256"] = sha256_file(SAMPLE_RULES)
    sample_stats["bytes"] = SAMPLE_RULES.stat().st_size
    if sample_stats["records_read"] != 500:
        raise RuntimeError(f"Expected 500 sample rules, found {sample_stats['records_read']}")
    all_stats["sources"][sample_source] = sample_stats


def stage_roadmap(conn: sqlite3.Connection, all_stats: dict[str, Any]) -> None:
    rows = [row for _, row in jsonl_read(ROADMAP)]
    if len(rows) != 1014:
        raise RuntimeError(f"Expected 1,014 roadmap records, found {len(rows)}")
    counts = Counter(normalize_blueprint_text(str(row.get("prompt") or "")) for row in rows)
    collision_keys = {key for key, count in counts.items() if count > 1}
    collisions = [row for row in rows if normalize_blueprint_text(str(row.get("prompt") or "")) in collision_keys]
    candidates = [row for row in rows if normalize_blueprint_text(str(row.get("prompt") or "")) not in collision_keys]
    if len(candidates) != 997 or len(collisions) != 17:
        raise RuntimeError(f"Roadmap collision audit changed: candidates={len(candidates)}, collisions={len(collisions)}")
    for row in collisions:
        record_quarantine(conn, ROADMAP.relative_to(ROOT).as_posix(),
                          row.get("blueprint_id", "unknown"),
                          "normalized prompt collision; manual review required", row)
    stats = all_stats["blueprints"]
    stats["roadmap_input"] = len(rows)
    stats["roadmap_unique_prompt_candidates"] = len(candidates)
    stats["roadmap_collision_records_held"] = len(collisions)
    stats["roadmap_collision_groups"] = len(collision_keys)
    for row in candidates:
        bp, sidecar = roadmap_blueprint(row)
        add_blueprint(conn, bp, sidecar, "roadmap_blueprints_1014", stats)
    all_stats["sources"][ROADMAP.relative_to(ROOT).as_posix()] = {
        "records_read": len(rows),
        "records_staged": len(candidates),
        "collision_records_held": len(collisions),
        "sha256": sha256_file(ROADMAP),
        "bytes": ROADMAP.stat().st_size,
        "license_status": "NOASSERTION",
    }


def prepare_rule_jsonl(stage: sqlite3.Connection, output: Path) -> tuple[int, int, int]:
    count = 0
    conflicts = int(stage.execute("SELECT COUNT(DISTINCT prompt_key) FROM pairs WHERE conflict=1").fetchone()[0])
    with output.open("w", encoding="utf-8") as handle:
        for row in stage.execute(
            "SELECT prompt,response,domain,tool FROM pairs WHERE conflict=0 ORDER BY prompt_key"
        ):
            prompt, response, domain, tool = row
            handle.write(json.dumps({
                "prompt": prompt, "response": response, "category": domain,
                "tool": tool, "trust_tier": 3, "split": "train",
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    variants = int(stage.execute("SELECT COUNT(*) FROM pair_conflicts").fetchone()[0])
    return count, conflicts, variants


def add_metadata_tables(vault: DiskCrystalVault) -> None:
    vault.conn.executescript("""
        CREATE TABLE IF NOT EXISTS vault_import_manifest (
            manifest_id INTEGER PRIMARY KEY CHECK(manifest_id=1),
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS vault_import_sources (
            source_path TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            source_license_status TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS vault_import_rule_provenance (
            prompt_key TEXT PRIMARY KEY,
            prompt_sha256 TEXT NOT NULL,
            source_files_json TEXT NOT NULL,
            source_domains_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS vault_import_conflicts (
            prompt_key TEXT NOT NULL,
            response_key TEXT NOT NULL,
            source TEXT NOT NULL,
            canonical_json TEXT NOT NULL,
            PRIMARY KEY(prompt_key,response_key,source)
        );
        CREATE TABLE IF NOT EXISTS vault_import_quarantine (
            quarantine_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            source_id TEXT NOT NULL,
            reason TEXT NOT NULL,
            record_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS vault_blueprint_source_metadata (
            blueprint_id TEXT PRIMARY KEY,
            source_pack TEXT NOT NULL,
            original_evidence_status TEXT NOT NULL,
            family_id TEXT NOT NULL,
            context_id TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_vault_blueprint_source_pack
            ON vault_blueprint_source_metadata(source_pack);
    """)


def main() -> int:
    started = time.time()
    for path in [MCRYSTAL, ROADMAP, SAMPLE_RULES, GR_CANDIDATES, GR_SOURCE, GR_VALIDATOR, GR_CONVERTER, *TRIPLET_FILES]:
        if not path.is_file():
            raise FileNotFoundError(f"Required publication input missing: {path}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, Any] = {
        "sources": {},
        "rules": {},
        "blueprints": {},
        "runtime_smoke": {},
        "tests": {},
        "evidence_policy": {
            "all imported legacy/source records are trust_tier 3 or EvidenceLevel.UNVERIFIED",
            "source_supported candidates remain canonical unverified; source status retained in sidecar",
            "test_code in roadmap records is preserved but not executed/promoted",
            "test/eval/validation triplet splits are excluded from live knowledge",
            "canary fixtures and conflicting prompt/answer variants are quarantined",
        },
    }

    with tempfile.TemporaryDirectory(prefix="mirror-vault-rebuild-") as td:
        work_dir = Path(td)
        stage_path = work_dir / "staging.sqlite"
        stage = create_stage(stage_path)
        try:
            import_rule_sources(stage, all_stats)
            stage_roadmap(stage, all_stats)
            stage_general_reasoning(stage, all_stats, work_dir)
            stage.commit()

            bundle_manifest_path = DATA / "bundle_metadata" / "manifest.json"
            bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
            expected_triplets = int(bundle_manifest.get("inventory", {}).get("triplets", -1))
            triplet_sources = [
                (source, meta) for source, meta in all_stats["sources"].items()
                if "/triplets/" in source
            ]
            actual_triplets = sum(int(meta.get("records_read", 0)) for _, meta in triplet_sources)
            mapped_triplet_pairs = sum(int(meta.get("pairs_seen", 0)) for _, meta in triplet_sources)
            unmapped_triplet_records = sum(int(meta.get("unmapped_records", 0)) for _, meta in triplet_sources)
            if actual_triplets != expected_triplets:
                raise RuntimeError(f"Triplet record count mismatch: expected {expected_triplets}, found {actual_triplets}")
            if mapped_triplet_pairs == 0 or unmapped_triplet_records > max(10, expected_triplets // 10):
                signatures = {
                    source: meta.get("record_field_signatures", {})
                    for source, meta in triplet_sources
                }
                raise RuntimeError(
                    f"Triplet schema adapter could not map enough records: pairs_seen={mapped_triplet_pairs}, "
                    f"unmapped={unmapped_triplet_records}, fields={signatures}"
                )
            all_stats["bundle_manifest_inventory"] = bundle_manifest.get("inventory", {})

            prepared_rules = work_dir / "deduplicated_rule_pairs.jsonl"
            pair_count, conflict_keys, conflict_variants = prepare_rule_jsonl(stage, prepared_rules)
            all_stats["rules"].update({
                "source_prompt_keys_before_conflict_filter": int(stage.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]),
                "exact_duplicate_records_collapsed": sum(
                    int(x.get("exact_duplicates_collapsed", 0)) for x in all_stats["sources"].values()
                ),
                "conflicting_prompt_keys_quarantined": conflict_keys,
                "conflicting_response_variants_quarantined": conflict_variants,
                "unique_pairs_submitted_to_engine": pair_count,
            })

            building = OUTPUT.with_name(OUTPUT.name + ".building")
            for sidecar in (Path(str(building) + "-wal"), Path(str(building) + "-shm")):
                sidecar.unlink(missing_ok=True)
            building.unlink(missing_ok=True)
            vault = DiskCrystalVault(building)
            vault.stats.setdefault("synonym_bridges_folded", 0)
            try:
                ingest_result = vault.ingest_jsonl(
                    prepared_rules, batch_flush_rows=2000, fold_synonyms=True
                )
                all_stats["rules"]["engine_ingestion"] = ingest_result
                all_stats["rules"]["active_rule_count"] = vault.rule_count()

                add_metadata_tables(vault)
                with vault.conn:
                    # Retain normalized source provenance for every unique staged pair.
                    for row in stage.execute(
                        "SELECT prompt_key,prompt,sources_json,domains_json FROM pairs WHERE conflict=0"
                    ):
                        key, prompt, sources_json, domains_json = row
                        vault.conn.execute(
                            "INSERT OR REPLACE INTO vault_import_rule_provenance VALUES(?,?,?,?)",
                            (key, hashlib.sha256(key.encode("utf-8")).hexdigest(), sources_json, domains_json)
                        )

                    # Keep all source/rights distinctions beside the production data.
                    for source_path, metadata in all_stats["sources"].items():
                        license_status = str(metadata.get("license_status") or "mixed/see bundle license register")
                        vault.conn.execute(
                            "INSERT OR REPLACE INTO vault_import_sources VALUES(?,?,?,?,?)",
                            (source_path, str(metadata.get("sha256") or ""), int(metadata.get("bytes") or 0),
                             license_status, json.dumps(metadata, ensure_ascii=False, sort_keys=True))
                        )

                    # Exact prompt conflicts are retained for audit but excluded from active retrieval.
                    for row in stage.execute("SELECT prompt_key,response_key,source,prompt,response,domain,source_line FROM pair_conflicts"):
                        key, response_key, source, prompt, response, domain, line_no = row
                        vault.conn.execute(
                            "INSERT OR IGNORE INTO vault_import_conflicts VALUES(?,?,?,?)",
                            (key, response_key, source,
                             json.dumps({"prompt": prompt, "response": response, "domain": domain, "source_line": line_no},
                                        ensure_ascii=False, separators=(",", ":")))
                        )

                    for source, source_id, reason, record_json in stage.execute(
                        "SELECT source,source_id,reason,record_json FROM quarantine"
                    ):
                        vault.conn.execute(
                            "INSERT INTO vault_import_quarantine(source,source_id,reason,record_json) VALUES(?,?,?,?)",
                            (source, source_id, reason, record_json)
                        )

                # Import canonical blueprints only when their prompt is unique across the whole active import.
                bp_counts = Counter()
                imported_blueprint_ids: list[str] = []
                pair_lookup = {
                    row[0]: row[1] for row in stage.execute(
                        "SELECT prompt_key,response_key FROM pairs WHERE conflict=0"
                    )
                }
                blocked_pair_keys = {
                    row[0] for row in stage.execute("SELECT prompt_key FROM pairs WHERE conflict=1")
                }
                blueprint_smoke_id = None
                for row in stage.execute(
                    "SELECT prompt_key,blueprint_id,canonical_json,sidecar_json,response_key,source_pack,conflict "
                    "FROM blueprints ORDER BY prompt_key"
                ):
                    key, blueprint_id, canonical_json, sidecar_json, response_key, source_pack, conflicted = row
                    if conflicted:
                        bp_counts["blueprint_prompt_conflicts_quarantined"] += 1
                        continue
                    if key in blocked_pair_keys:
                        bp_counts["blueprint_vs_conflicting_rule_prompt_quarantined"] += 1
                        continue
                    rule_response_key = pair_lookup.get(key)
                    if rule_response_key is not None:
                        if rule_response_key == response_key:
                            bp_counts["duplicate_blueprints_already_in_rule_store"] += 1
                        else:
                            bp_counts["blueprint_vs_rule_prompt_conflicts_quarantined"] += 1
                        continue
                    blueprint = Blueprint.from_dict(json.loads(canonical_json))
                    if vault.get_blueprint(blueprint.blueprint_id) is not None:
                        raise RuntimeError(f"Duplicate canonical blueprint ID: {blueprint.blueprint_id}")
                    vault.store_blueprint(blueprint)
                    sidecar = json.loads(sidecar_json)
                    vault.conn.execute(
                        "INSERT INTO vault_blueprint_source_metadata VALUES(?,?,?,?,?,?)",
                        (
                            blueprint.blueprint_id,
                            source_pack,
                            str(sidecar.get("original_evidence_status") or "unverified"),
                            str(sidecar.get("family_id") or ""),
                            str(sidecar.get("context_id") or ""),
                            sidecar_json,
                        )
                    )
                    imported_blueprint_ids.append(blueprint.blueprint_id)
                    if (
                        blueprint_smoke_id is None
                        and source_pack == "roadmap_blueprints_1014"
                        and not re.search(r"\{[A-Za-z][A-Za-z0-9_]*\}", blueprint.response_template)
                    ):
                        blueprint_smoke_id = blueprint.blueprint_id
                    bp_counts["canonical_blueprints_imported"] += 1
                    bp_counts[f"imported_from:{source_pack}"] += 1

                # Preserve prompt-collision clusters from the structured blueprint packs.
                for row in stage.execute(
                    "SELECT prompt_key,blueprint_id,canonical_json,sidecar_json,source_pack,response_key "
                    "FROM blueprint_conflicts"
                ):
                    key, blueprint_id, canonical_json, sidecar_json, source_pack, response_key = row
                    vault.conn.execute(
                        "INSERT OR IGNORE INTO vault_import_conflicts VALUES(?,?,?,?)",
                        (key, response_key, source_pack,
                         json.dumps({"blueprint_id": blueprint_id, "canonical_json": canonical_json,
                                     "sidecar": json.loads(sidecar_json)}, ensure_ascii=False))
                    )

                all_stats["blueprints"].update(bp_counts)
                all_stats["blueprints"]["source_prompt_collision_records_held"] = 17
                all_stats["blueprints"]["source_prompt_collision_groups_held"] = 7
                all_stats["blueprints"]["canonical_blueprint_count"] = int(
                    vault.conn.execute("SELECT COUNT(*) FROM blueprints").fetchone()[0]
                )
                all_stats["rules"]["active_rule_count"] = vault.rule_count()
                all_stats["rules"]["domain_counts"] = vault.get_domain_counts()
                all_stats["rules"]["trust_tier_counts"] = vault.get_trust_tier_counts()
                all_stats["rules"]["canonical_blueprints_with_provenance"] = int(
                    vault.conn.execute("SELECT COUNT(*) FROM vault_blueprint_source_metadata").fetchone()[0]
                )

                # Runtime smoke: exact retrieval from imported data and composition of an unverified blueprint.
                first_pair = stage.execute(
                    "SELECT prompt,response FROM pairs WHERE conflict=0 ORDER BY prompt_key LIMIT 1"
                ).fetchone()
                if first_pair is None:
                    raise RuntimeError("No usable rule pairs were available for runtime testing")
                rule_response = vault.respond(first_pair[0])
                if not str(getattr(rule_response, "answer", "")).strip():
                    raise RuntimeError("Imported rule retrieval returned an empty answer")
                all_stats["runtime_smoke"]["rule_query_mode"] = str(getattr(rule_response, "mode", "unknown"))
                all_stats["runtime_smoke"]["rule_query_answer_nonempty"] = True

                if not imported_blueprint_ids:
                    raise RuntimeError("No canonical blueprints were imported")
                imported_id = imported_blueprint_ids[0]
                imported_blueprint = vault.get_blueprint(imported_id)
                if imported_blueprint is None:
                    raise RuntimeError("Imported canonical blueprint could not be read back")
                if imported_blueprint.verification.level is not EvidenceLevel.UNVERIFIED:
                    raise RuntimeError(
                        "Imported source blueprint trust boundary changed: "
                        + imported_blueprint.verification.level.value
                    )
                all_stats["runtime_smoke"]["imported_blueprint_id"] = imported_id
                all_stats["runtime_smoke"]["imported_blueprint_evidence"] = imported_blueprint.verification.level.value

                # Test the persisted composition path with an isolated, placeholder-free
                # fixture. Source records can correctly abstain on collisions or unresolved
                # preconditions; that behavior is audited separately and is never promotion.
                smoke_prompt = "Mirror vault canonical composition smoke check"
                smoke_response = "Mirror stored and composed this isolated rebuild test blueprint."
                smoke_id = "rebuild:canonical-composition-smoke"
                smoke_blueprint = Blueprint(
                    blueprint_id=smoke_id,
                    kind=BlueprintKind.PROCEDURE,
                    domain="rebuild_runtime_smoke",
                    intent="canonical_composition_round_trip",
                    prompt_pattern=smoke_prompt,
                    response_template=smoke_response,
                    provenance=BlueprintProvenance(
                        source="mirror_vault_rebuild_test",
                        source_ref="test://canonical-blueprint-composition",
                        license_spdx="NOASSERTION",
                        content_sha256=Blueprint._digest(smoke_prompt, smoke_response),
                        captured_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    ),
                    verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
                    legacy_trust_tier=3,
                )
                vault.store_blueprint(smoke_blueprint)
                bp_response = vault.respond_blueprint(smoke_id)
                if not str(getattr(bp_response, "answer", "")).strip():
                    raise RuntimeError("Canonical blueprint composition returned an empty answer")
                if str(getattr(bp_response, "epistemic_tier", "")) != "unverified":
                    raise RuntimeError(
                        "Isolated unverified blueprint composition changed its evidence tier: "
                        + str(getattr(bp_response, "epistemic_tier", "missing"))
                    )
                if str(getattr(bp_response, "mode", "")) != "blueprint_composition_unverified":
                    raise RuntimeError(
                        "Isolated blueprint composition did not use the expected runtime path: "
                        + str(getattr(bp_response, "mode", "missing"))
                    )
                with vault.conn:
                    vault.conn.execute("DELETE FROM blueprints WHERE blueprint_id=?", (smoke_id,))
                all_stats["runtime_smoke"]["blueprint_smoke_id"] = smoke_id
                all_stats["runtime_smoke"]["blueprint_query_mode"] = str(getattr(bp_response, "mode", "unknown"))
                all_stats["runtime_smoke"]["blueprint_epistemic_tier"] = str(bp_response.epistemic_tier)
                all_stats["runtime_smoke"]["blueprint_answer_nonempty"] = True
                all_stats["runtime_smoke"]["temporary_smoke_blueprint_removed"] = True

                duplicate_blueprints = int(vault.conn.execute(
                    "SELECT COUNT(*) FROM (SELECT prompt_key FROM blueprints GROUP BY prompt_key HAVING COUNT(*) > 1)"
                ).fetchone()[0])
                if duplicate_blueprints:
                    raise RuntimeError(f"Canonical blueprint prompt duplicates remain: {duplicate_blueprints}")
                all_stats["tests"]["canonical_blueprint_prompt_uniqueness"] = "PASS"

                manifest_payload = {
                    "status": "built_and_smoke_tested",
                    "rule_count": vault.rule_count(),
                    "blueprint_count": int(vault.conn.execute("SELECT COUNT(*) FROM blueprints").fetchone()[0]),
                    "source_files": sorted(all_stats["sources"]),
                    "runtime_smoke": all_stats["runtime_smoke"],
                    "tests": all_stats["tests"],
                    "evidence_policy": all_stats["evidence_policy"],
                }
                vault.conn.execute(
                    "INSERT OR REPLACE INTO vault_import_manifest(manifest_id,payload_json,created_at) VALUES(1,?,?)",
                    (json.dumps(manifest_payload, ensure_ascii=False, sort_keys=True), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                )
                integrity = vault.conn.execute("PRAGMA integrity_check").fetchone()
                if not integrity or integrity[0] != "ok":
                    raise RuntimeError(f"SQLite integrity check failed before close: {integrity}")
                all_stats["tests"]["sqlite_integrity_before_close"] = "PASS"
                all_stats["tests"]["rule_retrieval_smoke"] = "PASS"
                all_stats["tests"]["blueprint_composition_smoke"] = "PASS"
                all_stats["tests"]["blueprint_evidence_boundary"] = "PASS"
                vault.optimize_disk()
            finally:
                vault.close()

            # Re-open the built file to test persistence, then checkpoint and validate standalone SQLite bytes.
            reopened = DiskCrystalVault(building)
            try:
                if reopened.rule_count() != all_stats["rules"]["active_rule_count"]:
                    raise RuntimeError("Rule count changed after reopening the rebuilt vault")
                bp = reopened.get_blueprint(all_stats["runtime_smoke"]["blueprint_id"])
                if bp is None or bp.verification.level is not EvidenceLevel.UNVERIFIED:
                    raise RuntimeError("Reopened canonical blueprint lost its unverified evidence state")
                response_after_reload = reopened.respond(first_pair[0])
                if not str(getattr(response_after_reload, "answer", "")).strip():
                    raise RuntimeError("Reopened vault failed its rule retrieval smoke test")
                integrity = reopened.conn.execute("PRAGMA integrity_check").fetchone()
                if not integrity or integrity[0] != "ok":
                    raise RuntimeError(f"SQLite integrity check failed after reopen: {integrity}")
                all_stats["tests"]["reopen_persistence"] = "PASS"
                all_stats["tests"]["sqlite_integrity_after_reopen"] = "PASS"
                reopened.optimize_disk()
            finally:
                reopened.close()

            # SQLite must be self-contained; no WAL sidecar may be needed at runtime.
            with sqlite3.connect(building) as check_db:
                check = check_db.execute("PRAGMA integrity_check").fetchone()
                if not check or check[0] != "ok":
                    raise RuntimeError(f"Final standalone SQLite integrity check failed: {check}")
                all_stats["rules"]["final_rule_count"] = int(check_db.execute("SELECT COUNT(*) FROM rules").fetchone()[0])
                all_stats["blueprints"]["final_blueprint_count"] = int(check_db.execute("SELECT COUNT(*) FROM blueprints").fetchone()[0])
            os.replace(building, OUTPUT)

        finally:
            stage.close()

    all_stats["vault"] = {
        "path": OUTPUT.relative_to(ROOT).as_posix(),
        "bytes": OUTPUT.stat().st_size,
        "sha256": sha256_file(OUTPUT),
        "size_mb": round(OUTPUT.stat().st_size / (1024 * 1024), 3),
    }
    if OUTPUT.stat().st_size > MAX_GIT_FILE_BYTES:
        raise RuntimeError(
            f"Rebuilt .dcrystal is {OUTPUT.stat().st_size:,} bytes; exceeds conservative GitHub limit "
            f"{MAX_GIT_FILE_BYTES:,}. The build is valid locally, but this workflow will not attempt a failing Git push."
        )

    all_stats["elapsed_seconds"] = round(time.time() - started, 2)
    all_stats["status"] = "PASS"
    REPORT.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": all_stats["status"],
        "vault": all_stats["vault"],
        "rules": all_stats["rules"],
        "blueprints": all_stats["blueprints"],
        "runtime_smoke": all_stats["runtime_smoke"],
        "tests": all_stats["tests"],
        "elapsed_seconds": all_stats["elapsed_seconds"],
        "report": REPORT.relative_to(ROOT).as_posix(),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
