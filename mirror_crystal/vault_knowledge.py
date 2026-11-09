"""Knowledge-foundation persistence, provenance-preserving lookup and conflict links."""
from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Iterable

from .knowledge_foundation import (
    KnowledgeConflict,
    KnowledgeKind,
    KnowledgeRecord,
    KnowledgeRegistry,
    KnowledgeSearchResult,
)
from .universal_representation import (
    EpistemicStatus,
    RepresentationGraph,
    RepresentationRelation,
    RepresentationProvenance,
    stable_representation_id,
)


class VaultKnowledgeRuntimeMixin:
    """Store Phase 1 knowledge as typed Phase 2 graph nodes."""

    @staticmethod
    def _knowledge_record_from_node(node) -> KnowledgeRecord | None:
        if node.content.get("node_role") != "knowledge_record":
            return None
        payload = node.content.get("knowledge_record")
        if not isinstance(payload, Mapping):
            return None
        return KnowledgeRecord.from_dict(payload)

    def get_knowledge_record(self, knowledge_id: str) -> KnowledgeRecord | None:
        node = self.get_representation_node(f"knowledge:{knowledge_id}")
        if node is None:
            return None
        return self._knowledge_record_from_node(node)

    def store_knowledge_record(self, record: KnowledgeRecord) -> str:
        if not isinstance(record, KnowledgeRecord):
            raise TypeError("record must be KnowledgeRecord")
        node = record.to_representation_node()

        def persist(conn):
            existing = conn.execute(
                "SELECT knowledge_id, knowledge_node_id, object_fingerprint "
                "FROM knowledge_claim_index WHERE claim_key = ? "
                "ORDER BY knowledge_id",
                (record.claim_key,),
            ).fetchall()
            node_added = self._store_node_conn(conn, node)
            relations_added = 0
            for previous_id, previous_node_id, previous_fingerprint in existing:
                if previous_node_id == node.node_id:
                    continue
                if str(previous_fingerprint) == record.object_fingerprint:
                    continue
                endpoint = conn.execute(
                    "SELECT 1 FROM representation_nodes WHERE node_id = ?",
                    (str(previous_node_id),),
                ).fetchone()
                if endpoint is None:
                    raise RuntimeError(
                        f"knowledge claim index has a missing representation node: {previous_node_id}"
                    )
                payload = {
                    "source": node.node_id,
                    "predicate": "contradicts",
                    "target": str(previous_node_id),
                }
                relation = RepresentationRelation(
                    relation_id=stable_representation_id("knowledge-conflict", payload),
                    source_id=node.node_id,
                    predicate="contradicts",
                    target_id=str(previous_node_id),
                    domain=record.domain,
                    status=EpistemicStatus.DISPUTED,
                    attributes={
                        "claim_key": record.claim_key,
                        "new_object_fingerprint": record.object_fingerprint,
                        "existing_object_fingerprint": str(previous_fingerprint),
                    },
                    provenance=RepresentationProvenance(
                        source="mirror_knowledge_conflict_detector",
                        source_ref=record.fingerprint,
                        parent_ids=(record.knowledge_id, str(previous_id)),
                    ),
                )
                relations_added += int(self._store_relation_conn(conn, relation))
            conn.execute(
                "INSERT OR IGNORE INTO knowledge_claim_index "
                "(knowledge_node_id, knowledge_id, claim_key, object_fingerprint, domain, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    node.node_id, record.knowledge_id, record.claim_key,
                    record.object_fingerprint, record.domain, time.time(),
                ),
            )
            indexed = conn.execute(
                "SELECT object_fingerprint FROM knowledge_claim_index WHERE knowledge_node_id = ?",
                (node.node_id,),
            ).fetchone()
            if indexed is None or str(indexed[0]) != record.object_fingerprint:
                raise KnowledgeConflictError(
                    f"knowledge index conflicts with existing content: {record.knowledge_id}"
                )
            return {"nodes_added": int(node_added), "conflict_relations_added": relations_added}

        self._representation_write(persist)
        return record.knowledge_id

    def store_knowledge_records(self, records: Iterable[KnowledgeRecord]) -> dict[str, int]:
        items = tuple(records)
        if any(not isinstance(item, KnowledgeRecord) for item in items):
            raise TypeError("records must contain KnowledgeRecord values")
        added = 0
        conflicts_before = len(self.list_representation_relations(predicate="contradicts", limit=20000))
        for item in items:
            before = self.get_representation_node(f"knowledge:{item.knowledge_id}")
            self.store_knowledge_record(item)
            after = self.get_representation_node(f"knowledge:{item.knowledge_id}")
            added += int(before is None and after is not None)
        conflicts_after = len(self.list_representation_relations(predicate="contradicts", limit=20000))
        return {
            "records_added": added,
            "conflict_relations_added": max(0, conflicts_after - conflicts_before),
        }

    def list_knowledge_records(
        self,
        *,
        domain: str | None = None,
        kind: KnowledgeKind | str | None = None,
        limit: int = 1000,
    ) -> tuple[KnowledgeRecord, ...]:
        nodes = self.list_representation_nodes(
            domain=domain, limit=max(0, min(int(limit), 10000))
        )
        selected_kind = KnowledgeKind(kind) if kind is not None else None
        records = []
        for node in nodes:
            record = self._knowledge_record_from_node(node)
            if record is None:
                continue
            if selected_kind is not None and record.kind is not selected_kind:
                continue
            records.append(record)
        return tuple(sorted(records, key=lambda item: item.knowledge_id))

    def knowledge_conflicts(self, *, domain: str | None = None) -> tuple[KnowledgeConflict, ...]:
        clauses = ["claim_key IN (SELECT claim_key FROM knowledge_claim_index"]
        params = []
        if domain is not None:
            clauses[-1] += " WHERE domain = ?"
            params.append(str(domain).strip().casefold())
        clauses[-1] += " GROUP BY claim_key HAVING COUNT(DISTINCT object_fingerprint) > 1)"
        rows = self.conn.execute(
            "SELECT DISTINCT claim_key FROM knowledge_claim_index WHERE " + clauses[0],
            tuple(params),
        ).fetchall()
        conflicts = []
        for (claim_key,) in rows:
            members = self.conn.execute(
                "SELECT knowledge_id, object_fingerprint FROM knowledge_claim_index "
                "WHERE claim_key = ? ORDER BY knowledge_id",
                (str(claim_key),),
            ).fetchall()
            if len({str(item[1]) for item in members}) < 2:
                continue
            first = self.get_knowledge_record(str(members[0][0]))
            if first is None:
                continue
            conflicts.append(KnowledgeConflict(
                claim_key=str(claim_key),
                domain=first.domain,
                subject=first.subject,
                predicate=first.predicate,
                knowledge_ids=tuple(str(item[0]) for item in members),
                object_fingerprints=tuple(sorted({str(item[1]) for item in members})),
            ))
        return tuple(sorted(conflicts, key=lambda item: item.claim_key))

    def search_knowledge(
        self,
        query: str,
        *,
        domain: str | None = None,
        kind: KnowledgeKind | str | None = None,
        limit: int = 20,
        include_assumptions: bool = True,
    ) -> tuple[KnowledgeSearchResult, ...]:
        registry = KnowledgeRegistry(self.list_knowledge_records(domain=domain, limit=10000))
        return registry.query(
            query,
            domain=domain,
            kind=kind,
            limit=limit,
            include_assumptions=include_assumptions,
        )

    def ingest_blueprints_as_knowledge(self, blueprints: Iterable) -> dict[str, int]:
        """Adapt canonical blueprints while preserving their exact evidence and provenance."""
        items = tuple(blueprints)
        records = tuple(KnowledgeRecord.from_blueprint(item) for item in items)
        return self.store_knowledge_records(records)
