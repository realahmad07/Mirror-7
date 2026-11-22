"""Unified Phase-3 Multi-Shard Admission, Integrity Audit & Report Generator for Mirror 7.

Zero neural parameters, zero GPU, pure Python stdlib.
Admits verified Phase-3 staged JSONL corpora into compact, GitHub-safe (< 35 MB each)
Disk-Backed SQLite FTS5 + Zlib shards under `artifacts/concept_shards/` using
`OpenEndedConceptLatticeEngine.admit_to_shard(...)`, while keeping the primary 63,510-concept
vault (`artifacts/mirror_uol_skills.dcrystal`) 100% read-only and bit-for-bit untouched.

Usage:
    python scripts/harvest_phase3_shards.py \
        --browser-jsonl .../browser_phase3.jsonl \
        --github-jsonl .../github_repos.jsonl \
        --datasets-jsonl .../public_datasets.jsonl \
        --rebuild-shards
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import tracemalloc
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirror_crystal.open_ended_concept_lattice import (
    ConceptNode,
    OpenEndedConceptLatticeEngine,
    _VAULT_SNAPSHOT_CACHE,
    canonical_concept_key,
    content_fingerprint,
    provenance_tier_rank,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_staged_jsonl(jsonl_path: Path | None) -> list[ConceptNode]:
    """Load validated 5-facet ConceptNodes from a Phase-3 staged JSONL file."""
    if jsonl_path is None or not jsonl_path.exists():
        return []
    nodes: list[ConceptNode] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            cid = str(rec.get("concept_id", "")).strip().casefold()
            title = str(rec.get("title", "")).strip()
            cat = str(rec.get("category", "general_knowledge")).strip() or "general_knowledge"
            raw_aliases = rec.get("aliases") or [cid]
            if isinstance(raw_aliases, str):
                raw_aliases = [a.strip() for a in raw_aliases.split("|") if a.strip()]
            aliases = tuple(
                dict.fromkeys([cid, *(str(a).strip().casefold() for a in raw_aliases if str(a).strip())])
            )
            defn = str(rec.get("definition", "")).strip()
            mech = str(rec.get("mechanism", "")).strip()
            apps = str(rec.get("applications", "")).strip()
            trd = str(rec.get("tradeoffs", "")).strip()
            prov = str(rec.get("provenance", "mirror:phase3_staged")).strip()
            qscore = float(rec.get("quality_score", 0.85) or 0.85)
            if not cid or not title or not defn:
                continue
            nodes.append(
                ConceptNode(
                    concept_id=cid,
                    title=title,
                    category=cat,
                    aliases=aliases,
                    definition=defn,
                    mechanism=mech,
                    applications=apps,
                    tradeoffs=trd,
                    provenance=prov,
                    quality_score=round(max(0.0, min(1.0, qscore)), 4),
                )
            )
    return nodes


def _classify_phase3_source(provenance: str) -> str:
    p = (provenance or "").strip().casefold()
    if p.startswith("mirror:concept_lattice:"):
        return "seed_foundational"
    if p.startswith("mirror:domain_agent_harvester"):
        return "curated_20_domain_catalog"
    if p.startswith("browser:"):
        return "browser_verified_web_portals"
    if p.startswith("mitre:attack:"):
        return "github_mitre_attack_stix"
    if p.startswith("mitre:capec:"):
        return "github_mitre_capec_stix"
    if p.startswith("cncf:k8s_glossary:"):
        return "github_kubernetes_glossary"
    if p.startswith("cncf:"):
        return "github_cncf_glossary"
    if p.startswith("owasp:"):
        return "github_owasp_cheatsheets"
    if p.startswith("mdn:"):
        return "github_mdn_web_glossary"
    if p.startswith("nist:"):
        return "standards_nist_csrc_glossary"
    if p.startswith("ietf:"):
        return "standards_ietf_rfc4949"
    if p.startswith("docs:postgresql_glossary"):
        return "github_postgresql_glossary"
    if p.startswith("docs:python_glossary"):
        return "github_python_glossary"
    if p.startswith("docs:rust_glossary"):
        return "github_rust_glossary"
    if p.startswith("docs:python_stdlib"):
        return "python_stdlib_docs_introspector"
    if p.startswith("openalex:"):
        return "dataset_openalex_topics"
    if p.startswith("schema.org:"):
        return "dataset_schema_org_ontology"
    if p.startswith("worldbank:"):
        return "dataset_worldbank_indicators"
    if p.startswith("mesh:"):
        return "dataset_nih_mesh"
    if p.startswith("hf:"):
        return "dataset_huggingface_verified"
    if p.startswith(("ols:", "chembl:", "pubchem:", "uniprot:", "reactome:", "interpro:", "openfda:", "clinicaltrials:", "clinvar:", "plugin:")):
        return "science_plugins_verified"
    if p.startswith("wikipedia:"):
        return "live_wikipedia_api"
    if p.startswith("arxiv:"):
        return "live_arxiv_api"
    return "other_verified"


def _audit_all_databases_readonly(
    primary_vault: Path,
    shard_paths: list[Path],
) -> dict[str, Any]:
    """Independently verify SQLite integrity and global cross-shard zero-duplication in read-only mode."""
    seen_cids: set[str] = set()
    seen_ckeys: set[str] = set()
    seen_fps: set[str] = set()

    dup_cids = 0
    dup_ckeys = 0
    dup_fps = 0

    integrity_results: dict[str, str] = {}
    source_counts: Counter[str] = Counter()
    shard_source_counts: dict[str, dict[str, int]] = {}
    total_distilled_bytes = 0
    total_rows = 0

    all_dbs = [("primary", primary_vault)] + [(p.name, p) for p in shard_paths]
    for label, db_path in all_dbs:
        uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=15.0)
        try:
            ic = str(con.execute("PRAGMA integrity_check").fetchone()[0])
            integrity_results[db_path.name] = ic
            cols = {r[1] for r in con.execute("PRAGMA table_info(mirror_concept_lattice)").fetchall()}
            per_db_sources: Counter[str] = Counter()
            rows = con.execute(
                "SELECT concept_id, canonical_key, content_fp, provenance FROM mirror_concept_lattice"
            ).fetchall()
            for cid, ckey, cfp, prov in rows:
                total_rows += 1
                cid_s = str(cid or "").strip().casefold()
                ck_s = str(ckey or "").strip() or canonical_concept_key(cid_s)
                fp_s = str(cfp or "").strip()
                if cid_s in seen_cids:
                    dup_cids += 1
                else:
                    seen_cids.add(cid_s)
                if ck_s and ck_s in seen_ckeys:
                    dup_ckeys += 1
                elif ck_s:
                    seen_ckeys.add(ck_s)
                if fp_s and fp_s in seen_fps:
                    dup_fps += 1
                elif fp_s:
                    seen_fps.add(fp_s)
                bucket = _classify_phase3_source(str(prov or ""))
                source_counts[bucket] += 1
                per_db_sources[bucket] += 1

            shard_source_counts[db_path.name] = dict(per_db_sources.most_common())
            if "facets_zblob" in cols:
                b_row = con.execute(
                    "SELECT COALESCE(SUM(LENGTH(title) + LENGTH(facets_zblob)), 0) FROM mirror_concept_lattice"
                ).fetchone()
                total_distilled_bytes += int(b_row[0] or 0)
        finally:
            con.close()

    return {
        "total_rows_audited": total_rows,
        "integrity_checks": integrity_results,
        "all_integrity_ok": all(v == "ok" for v in integrity_results.values()),
        "global_duplicate_concept_ids": dup_cids,
        "global_duplicate_canonical_keys": dup_ckeys,
        "global_duplicate_content_fps": dup_fps,
        "global_zero_duplicates": (dup_cids == 0 and dup_ckeys == 0 and dup_fps == 0),
        "source_counts": dict(source_counts.most_common()),
        "per_database_source_counts": shard_source_counts,
        "total_distilled_bytes": total_distilled_bytes,
    }


def _pick_shard_sample_queries(shard_path: Path, count: int = 4) -> list[tuple[str, str]]:
    """Select clean non-digit representative titles from a shard to prove uncached cross-shard O(1) retrieval."""
    if not shard_path.exists():
        return []
    uri = f"file:{shard_path.resolve().as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=10.0)
    try:
        rows = con.execute(
            """
            SELECT concept_id, title FROM mirror_concept_lattice
            WHERE LENGTH(title) BETWEEN 14 AND 44
              AND title LIKE '% %'
              AND title NOT LIKE '%(%'
              AND title NOT LIKE '%:%'
              AND title NOT GLOB '*[0-9]*'
            ORDER BY quality_score DESC, concept_id ASC
            LIMIT 600
            """
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return []
    step = max(1, len(rows) // max(count, 1))
    picked = [rows[i] for i in range(0, len(rows), step)][:count]
    return [(str(cid), f"What is {title}?") for cid, title in picked]


@dataclass
class Phase3ShardReport:
    primary_vault_path: str
    primary_node_count: int
    primary_disk_mb: float
    primary_sha256_before: str
    primary_sha256_after: str
    primary_untouched: bool
    shards_count: int
    shards_node_count: int
    shards_disk_mb: float
    total_unique_concepts: int
    total_domains: int
    combinatorial_cross_pairs: int
    discourse_operator_combinations_12x: int
    discourse_operator_combinations_24x: int
    total_disk_mb: float
    all_shards_under_35mb: bool
    cold_load_ms: float
    warm_load_ms: float
    peak_ram_mb: float
    median_lookup_ms: float
    p90_lookup_ms: float
    sla_cold_start_pass: bool
    sla_ram_pass: bool
    sla_lookup_pass: bool
    elapsed_seconds: float
    shard_admissions: dict[str, dict[str, Any]]
    shards_breakdown: list[dict[str, Any]]
    global_integrity_audit: dict[str, Any]
    cross_shard_probe_results: list[dict[str, Any]]
    domain_counts: dict[str, int]
    source_counts: dict[str, int]


def format_phase3_markdown(report: Phase3ShardReport) -> str:
    lines = [
        "# Mirror 7 Phase-3 Multi-Shard Knowledge Lattice Report",
        "",
        "Zero-parameter, zero-GPU, zero-overwrite, GitHub-safe Multi-Shard Disk-Backed SQLite FTS5 + Zlib Knowledge Lattice combining the untouched 63,510-concept primary vault (`artifacts/mirror_uol_skills.dcrystal`) with modular domain shards (`artifacts/concept_shards/*.dcrystal`).",
        "",
        "## Headline Multi-Shard Telemetry",
        "",
        "| Metric | Verified Value | Phase-3 SLA |",
        "|---|---:|:---:|",
        f"| **Total Unique 5-Facet ConceptNodes (Primary + Shards)** | **{report.total_unique_concepts:,}** | `>= 70,000` |",
        f"| **Primary Vault ConceptNodes (100% Untouched)** | **{report.primary_node_count:,}** | `0` overwrites |",
        f"| **New Phase-3 Multi-Shard ConceptNodes** | **{report.shards_node_count:,}** across **{report.shards_count}** shards | Quality `>= 0.80` |",
        f"| **Knowledge Domains Covered** | **{report.total_domains}** | 20+ canonical |",
        f"| **Combinatorial Cross-Concept Pairs (`N*(N-1)/2`)** | **{report.combinatorial_cross_pairs:,}** | `> 2.45 Billion` |",
        f"| **Open-Ended Discourse Combinations (`Pairs x 12`)** | **{report.discourse_operator_combinations_12x:,}** | `> 29 Billion` |",
        f"| **Bilingual Discourse Combinations (`Pairs x 24`)** | **{report.discourse_operator_combinations_24x:,}** | `> 58 Billion` |",
        f"| **Primary Vault Disk Size (`mirror_uol_skills.dcrystal`)** | **{report.primary_disk_mb:.2f} MB** | Untouched (`SHA-256` match) |",
        f"| **Phase-3 Shards Disk Size (`artifacts/concept_shards/`)** | **{report.shards_disk_mb:.2f} MB** | Every shard `< 35 MB` |",
        f"| **Cold Start Latency (Primary + All Shards)** | **{report.cold_load_ms:.2f} ms** | `< 50 ms` ({'PASS' if report.sla_cold_start_pass else 'FAIL'}) |",
        f"| **Warm Init Latency** | **{report.warm_load_ms:.3f} ms** | Sub-millisecond |",
        f"| **Peak Python Heap RAM (`tracemalloc`)** | **{report.peak_ram_mb:.2f} MB** | `< 25 MB` ({'PASS' if report.sla_ram_pass else 'FAIL'}) |",
        f"| **Uncached O(1) Median Lookup Latency (CPU)** | **{report.median_lookup_ms:.3f} ms** (P90: `{report.p90_lookup_ms:.3f} ms`) | `< 2.0 ms` ({'PASS' if report.sla_lookup_pass else 'FAIL'}) |",
        f"| **Unified Phase-3 Admission & Audit Runtime** | **{report.elapsed_seconds:.2f} s** | Pure Python stdlib |",
        "",
        "## Primary Vault Immutability & Global Cross-Shard Zero-Duplication Audit",
        "",
        "| Integrity Gate | Verified Result |",
        "|---|---|",
        f"| Primary Vault SHA-256 Before | `{report.primary_sha256_before}` |",
        f"| Primary Vault SHA-256 After | `{report.primary_sha256_after}` |",
        f"| Primary Vault Byte-Identical (`0` Overwrites) | **{'YES (100% UNTOUCHED)' if report.primary_untouched else 'NO'}** |",
        f"| SQLite `PRAGMA integrity_check` Across All DBs | **{'ALL OK' if report.global_integrity_audit['all_integrity_ok'] else 'FAILED'}** |",
        f"| Global Duplicate `concept_id` Across Primary + All Shards | **{report.global_integrity_audit['global_duplicate_concept_ids']}** |",
        f"| Global Duplicate `canonical_key` Across Primary + All Shards | **{report.global_integrity_audit['global_duplicate_canonical_keys']}** |",
        f"| Global Duplicate 64-Bit `content_fp` Across Primary + All Shards | **{report.global_integrity_audit['global_duplicate_content_fps']}** |",
        f"| All Phase-3 Shards Strictly Under 35 MB (GitHub-Safe) | **{'PASS' if report.all_shards_under_35mb else 'FAIL'}** |",
        "",
        "## Multi-Shard Topology & Admission Breakdown",
        "",
        "| Shard File | Nodes | Disk Size | Domains | Inserted | Quality Rejected | Duplicates Blocked |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| `mirror_uol_skills.dcrystal` (Primary, Read-Only) | {report.primary_node_count:,} | {report.primary_disk_mb:.2f} MB | {report.total_domains} | — | — | — |",
    ]
    for sh in report.shards_breakdown:
        fname = sh["filename"]
        adm = report.shard_admissions.get(fname, {})
        dups = (
            int(adm.get("skipped_existing_primary_or_shard", 0))
            + int(adm.get("skipped_canonical_duplicate", 0))
            + int(adm.get("skipped_alias_duplicate", 0))
            + int(adm.get("skipped_content_duplicate", 0))
        )
        lines.append(
            f"| `concept_shards/{fname}` | {sh['node_count']:,} | {sh['disk_mb']:.2f} MB | {sh['domains_count']} "
            f"| {int(adm.get('inserted_new', 0)):,} | {int(adm.get('rejected_low_quality', 0)):,} | {dups:,} |"
        )

    lines += [
        "",
        "## Cross-Shard Retrieval Verification Probes",
        "",
        "| Probe Query | Matched | Latency (ms) | Concepts Resolved |",
        "|---|:---:|---:|---|",
    ]
    for pr in report.cross_shard_probe_results:
        cstr = ", ".join(f"`{c}`" for c in pr["concepts_used"][:3]) or "—"
        lines.append(f"| {pr['query']} | {'YES' if pr['matched'] else 'NO'} | {pr['latency_ms']:.3f} | {cstr} |")

    lines += [
        "",
        "## Global Provenance & Source Breakdown (Primary + All Shards)",
        "",
        "| Source Bucket | Verified ConceptNodes |",
        "|---|---:|",
    ]
    for src, cnt in sorted(report.source_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| `{src}` | {cnt:,} |")

    lines += [
        "",
        "## Domain Coverage Breakdown (Primary + All Shards)",
        "",
        "| Knowledge Domain | ConceptNodes |",
        "|---|---:|",
    ]
    for dom, cnt in sorted(report.domain_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| `{dom}` | {cnt:,} |")

    lines += [
        "",
        "## Open-Source & Public Dataset Attributions (Phase 3)",
        "- **MITRE ATT&CK STIX 2.0 (Enterprise, ICS, Mobile) & MITRE CAPEC STIX 2.1**: Public GitHub repository `mitre/cti` (Apache 2.0).",
        "- **MITRE CWE Top 25:2025 & Cloud Native Glossary (Tier-3 Browser Verified)**: Host-verified official portals (`cwe.mitre.org`, `glossary.cncf.io`).",
        "- **CNCF Cloud Native Glossary & Kubernetes Official Glossary**: `cncf/glossary`, `kubernetes/website` (CC BY 4.0).",
        "- **OWASP Cheat Sheet Series**: `OWASP/CheatSheetSeries` (CC BY-SA 4.0).",
        "- **MDN Web Docs Glossary**: `mdn/content` (CC BY-SA 2.5).",
        "- **NIST CSRC Official Glossary & IETF RFC 4949**: US Government Public Domain (`csrc.nist.gov`) & IETF Trust (`rfc-editor.org`).",
        "- **PostgreSQL, Python & Rust Official Glossaries**: `postgres/postgres`, `python/cpython`, `rust-lang/reference`.",
        "- **Public Scientific & Ontology Datasets (`shard_public_datasets.dcrystal`)**: OpenAlex Topics (CC0), Schema.org Ontology (Apache 2.0 / CC-BY-SA), World Bank Indicators (CC BY 4.0), NIH/NLM MeSH (US NLM Public Domain), and strictly filtered HuggingFace scientific datasets.",
    ]
    return "\n".join(lines) + "\n"


def run_phase3_unified_shard_admission(
    *,
    vault_path: Path,
    shard_dir: Path,
    browser_jsonl: Path | None,
    github_jsonl: Path | None,
    datasets_jsonl: Path | None,
    artifacts_dir: Path,
    min_quality: float = 0.80,
    rebuild_shards: bool = True,
) -> Phase3ShardReport:
    t0 = time.perf_counter()
    if not vault_path.exists():
        raise FileNotFoundError(f"Primary vault not found: {vault_path}")

    # Snapshot exact SHA-256 of primary vault BEFORE any operation
    primary_sha_before = _sha256_file(vault_path)
    primary_size_before = vault_path.stat().st_size

    shard_dir.mkdir(parents=True, exist_ok=True)
    shard1_path = shard_dir / "shard_github_engineering.dcrystal"
    shard2_path = shard_dir / "shard_public_datasets.dcrystal"

    if rebuild_shards:
        for sp in (shard1_path, shard2_path):
            for suffix in ("", "-wal", "-shm"):
                cand = Path(str(sp) + suffix)
                if cand.exists():
                    cand.unlink()

    engine = OpenEndedConceptLatticeEngine(vault_path=vault_path, shard_dir=shard_dir)
    shard_admissions: dict[str, dict[str, Any]] = {}

    # 1. Admit Shard 1: Tier-3 browser_phase3.jsonl + Tier-2.8/2.0 github_repos.jsonl
    browser_nodes = load_staged_jsonl(browser_jsonl)
    github_nodes = load_staged_jsonl(github_jsonl)
    # Sort strictly by descending provenance tier so Tier 3 always takes precedence over Tier 2.8 / 2.0
    shard1_nodes = sorted(
        [*browser_nodes, *github_nodes],
        key=lambda nd: (-provenance_tier_rank(nd.provenance), -nd.quality_score, nd.concept_id),
    )
    if shard1_nodes:
        adm1 = engine.admit_to_shard(
            shard1_path,
            shard1_nodes,
            min_quality=min_quality,
            vacuum=True,
        )
        adm1["input_browser_nodes"] = len(browser_nodes)
        adm1["input_github_nodes"] = len(github_nodes)
        shard_admissions[shard1_path.name] = adm1

    # 2. Admit Shard 2: Tier-2.5 public_datasets.jsonl (globally deduplicated against primary + Shard 1!)
    dataset_nodes = load_staged_jsonl(datasets_jsonl)
    dataset_nodes.sort(key=lambda nd: (-provenance_tier_rank(nd.provenance), -nd.quality_score, nd.concept_id))
    if dataset_nodes:
        adm2 = engine.admit_to_shard(
            shard2_path,
            dataset_nodes,
            min_quality=min_quality,
            vacuum=True,
        )
        adm2["input_dataset_nodes"] = len(dataset_nodes)
        shard_admissions[shard2_path.name] = adm2

    if hasattr(engine, "close"):
        engine.close()

    # Release staging lists before clean cold-start & RAM benchmark
    del browser_nodes, github_nodes, shard1_nodes, dataset_nodes

    # Verify primary vault is 100% bit-for-bit untouched
    primary_sha_after = _sha256_file(vault_path)
    primary_size_after = vault_path.stat().st_size
    primary_untouched = (primary_sha_before == primary_sha_after) and (primary_size_before == primary_size_after)
    if not primary_untouched:
        raise RuntimeError(
            f"CRITICAL VIOLATION: Primary vault modified! before={primary_sha_before} after={primary_sha_after}"
        )

    # Independent read-only SQLite integrity + cross-shard 0-duplication audit
    active_shards = sorted(p for p in shard_dir.glob("*.dcrystal") if p.is_file())
    audit = _audit_all_databases_readonly(vault_path, active_shards)

    # Build diverse uncached multi-shard probe queries (Primary + Shard 1 + Shard 2 + Cross-Shard)
    static_probes = [
        "What is Raft consensus?",
        "Explain photosynthesis",
        "Compare Docker and Kubernetes",
        "How do Blockchain and Cryptography work together?",
        "Yaar blokchain kaam kese krta h?",
        "What is Cross-Origin Resource Sharing?",
        "Explain Content Security Policy",
        "Explain CRISPR gene editing",
        "What is System Information Discovery?",
        "What is Authentication Abuse?",
        "Explain Bayes theorem",
        "What is zero-knowledge proof?",
    ]
    shard1_probes = [q for _, q in _pick_shard_sample_queries(shard1_path, count=4)]
    shard2_probes = [q for _, q in _pick_shard_sample_queries(shard2_path, count=4)]
    all_probes = static_probes + shard1_probes + shard2_probes

    # Pass 1: True wall-clock Cold Start (<50 ms) & Uncached O(1) Lookup Latency (<2.0 ms) WITHOUT tracemalloc distortion
    _VAULT_SNAPSHOT_CACHE.clear()
    tc0 = time.perf_counter()
    fresh_engine = OpenEndedConceptLatticeEngine(vault_path=vault_path, shard_dir=shard_dir)
    cold_load_ms = (time.perf_counter() - tc0) * 1000.0

    probe_results: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    for pq in all_probes:
        qs = time.perf_counter()
        res = fresh_engine.route_open_ended(pq)
        lat = (time.perf_counter() - qs) * 1000.0
        latencies_ms.append(lat)
        probe_results.append({
            "query": pq,
            "matched": bool(res.matched),
            "concepts_used": list(res.concepts_used),
            "latency_ms": round(lat, 3),
        })

    telem = fresh_engine.shard_telemetry()
    dom_counts = fresh_engine.domain_distribution()
    total_n = fresh_engine.concept_count()

    warm_samples_ms: list[float] = []
    for _ in range(3):
        tw0 = time.perf_counter()
        warm_engine = OpenEndedConceptLatticeEngine(vault_path=vault_path, shard_dir=shard_dir)
        warm_samples_ms.append((time.perf_counter() - tw0) * 1000.0)
        if hasattr(warm_engine, "close"):
            warm_engine.close()
    warm_samples_ms.sort()
    warm_load_ms = warm_samples_ms[len(warm_samples_ms) // 2]
    if hasattr(fresh_engine, "close"):
        fresh_engine.close()

    # Pass 2: True Peak Python Heap RAM (tracemalloc) across a fresh cold start + multi-shard queries
    _VAULT_SNAPSHOT_CACHE.clear()
    tracemalloc.start()
    mem_engine = OpenEndedConceptLatticeEngine(vault_path=vault_path, shard_dir=shard_dir)
    for pq in all_probes:
        mem_engine.route_open_ended(pq)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if hasattr(mem_engine, "close"):
        mem_engine.close()
    peak_ram_mb = round(peak_bytes / (1024 * 1024), 2)

    latencies_ms.sort()
    median_ms = latencies_ms[len(latencies_ms) // 2] if latencies_ms else 0.0
    p90_idx = min(len(latencies_ms) - 1, int(len(latencies_ms) * 0.90)) if latencies_ms else 0
    p90_ms = latencies_ms[p90_idx] if latencies_ms else 0.0

    pairs = (total_n * (total_n - 1)) // 2
    shards_breakdown = telem.get("shards_breakdown") or []
    all_under_35mb = all(float(sh.get("disk_mb", 0.0)) < 35.0 for sh in shards_breakdown)

    report = Phase3ShardReport(
        primary_vault_path=str(vault_path),
        primary_node_count=int(telem["primary_node_count"]),
        primary_disk_mb=float(telem["primary_disk_mb"]),
        primary_sha256_before=primary_sha_before,
        primary_sha256_after=primary_sha_after,
        primary_untouched=primary_untouched,
        shards_count=int(telem["shards_count"]),
        shards_node_count=int(telem["shards_node_count"]),
        shards_disk_mb=float(telem["shards_disk_mb"]),
        total_unique_concepts=total_n,
        total_domains=len(dom_counts),
        combinatorial_cross_pairs=pairs,
        discourse_operator_combinations_12x=pairs * 12,
        discourse_operator_combinations_24x=pairs * 24,
        total_disk_mb=float(telem["total_disk_mb"]),
        all_shards_under_35mb=all_under_35mb,
        cold_load_ms=round(cold_load_ms, 2),
        warm_load_ms=round(warm_load_ms, 3),
        peak_ram_mb=peak_ram_mb,
        median_lookup_ms=round(median_ms, 3),
        p90_lookup_ms=round(p90_ms, 3),
        sla_cold_start_pass=(cold_load_ms < 50.0),
        sla_ram_pass=(peak_ram_mb < 25.0),
        sla_lookup_pass=(median_ms < 2.0),
        elapsed_seconds=round(time.perf_counter() - t0, 2),
        shard_admissions=shard_admissions,
        shards_breakdown=shards_breakdown,
        global_integrity_audit=audit,
        cross_shard_probe_results=probe_results,
        domain_counts=dom_counts,
        source_counts=audit["source_counts"],
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    json_path = artifacts_dir / "mirror_phase3_shard_report.json"
    md_path = artifacts_dir / "MIRROR_PHASE3_SHARD_REPORT.md"
    json_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(format_phase3_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    default_staging = ROOT.parent / "phase3_harvest"
    parser = argparse.ArgumentParser(
        description="Mirror 7 Phase-3 Unified Multi-Shard Admission & Report Generator"
    )
    parser.add_argument(
        "--vault",
        type=Path,
        default=ROOT / "artifacts" / "mirror_uol_skills.dcrystal",
        help="Read-only primary vault path.",
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=ROOT / "artifacts" / "concept_shards",
        help="Output directory for .dcrystal shards.",
    )
    parser.add_argument(
        "--browser-jsonl",
        type=Path,
        default=default_staging / "browser_phase3.jsonl",
        help="Optional Tier-3 browser-verified JSONL staging file.",
    )
    parser.add_argument(
        "--github-jsonl",
        type=Path,
        default=default_staging / "github_repos.jsonl",
        help="Tier-2.8/2.0 public GitHub repository concepts JSONL.",
    )
    parser.add_argument(
        "--datasets-jsonl",
        type=Path,
        default=default_staging / "public_datasets.jsonl",
        help="Tier-2.5 public datasets concepts JSONL.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ROOT / "artifacts",
        help="Directory for saving mirror_phase3_shard_report.json and MIRROR_PHASE3_SHARD_REPORT.md.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.80,
        help="Minimum quality_score required for shard admission (default: 0.80).",
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Append into existing shards instead of rebuilding target shards cleanly.",
    )
    args = parser.parse_args()

    report = run_phase3_unified_shard_admission(
        vault_path=args.vault,
        shard_dir=args.shard_dir,
        browser_jsonl=args.browser_jsonl if args.browser_jsonl and args.browser_jsonl.exists() else None,
        github_jsonl=args.github_jsonl if args.github_jsonl and args.github_jsonl.exists() else None,
        datasets_jsonl=args.datasets_jsonl if args.datasets_jsonl and args.datasets_jsonl.exists() else None,
        artifacts_dir=args.artifacts_dir,
        min_quality=args.min_quality,
        rebuild_shards=not args.no_rebuild,
    )

    print("=" * 76)
    print("MIRROR 7 PHASE-3 UNIFIED MULTI-SHARD ADMISSION & INTEGRITY AUDIT COMPLETE")
    print("=" * 76)
    print(f"Primary Vault Nodes (Untouched)   : {report.primary_node_count:,} ({report.primary_disk_mb:.2f} MB)")
    print(f"Primary SHA-256 Match (0 edits)   : {report.primary_untouched} ({report.primary_sha256_after[:16]}...)")
    print(f"Phase-3 Shards Mounted            : {report.shards_count} ({report.shards_node_count:,} nodes, {report.shards_disk_mb:.2f} MB)")
    print(f"Total Unique 5-Facet ConceptNodes : {report.total_unique_concepts:,} across {report.total_domains} domains")
    print(f"Combinatorial Cross-Concept Pairs : {report.combinatorial_cross_pairs:,}")
    print(f"Open-Ended Discourse (12x / 24x)  : {report.discourse_operator_combinations_12x:,} / {report.discourse_operator_combinations_24x:,}")
    print(f"Global Cross-Shard Duplicates     : CID={report.global_integrity_audit['global_duplicate_concept_ids']} | CK={report.global_integrity_audit['global_duplicate_canonical_keys']} | FP={report.global_integrity_audit['global_duplicate_content_fps']}")
    print(f"SQLite Integrity Check (All DBs)  : {report.global_integrity_audit['all_integrity_ok']}")
    print(f"Cold Start / Warm Init (ms)       : {report.cold_load_ms:.2f} ms / {report.warm_load_ms:.3f} ms (SLA < 50 ms: {report.sla_cold_start_pass})")
    print(f"Peak Python Heap RAM (tracemalloc): {report.peak_ram_mb:.2f} MB (SLA < 25 MB: {report.sla_ram_pass})")
    print(f"Uncached O(1) Lookup (Med / P90)  : {report.median_lookup_ms:.3f} ms / {report.p90_lookup_ms:.3f} ms (SLA < 2.0 ms: {report.sla_lookup_pass})")
    print(f"All Shards < 35 MB (GitHub-Safe)  : {report.all_shards_under_35mb}")
    print(f"Elapsed Runtime                   : {report.elapsed_seconds:.2f} s")
    for sh in report.shards_breakdown:
        print(f"  - {sh['filename']}: {sh['node_count']:,} nodes, {sh['disk_mb']:.2f} MB, {sh['domains_count']} domains")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
