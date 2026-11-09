"""Additive SQLite persistence for the universal representation graph."""
from __future__ import annotations

import json
import time
from typing import Any, Iterable

from .blueprint_contracts import Blueprint
from .universal_representation import (
    EpistemicStatus,
    RepresentationConflictError,
    RepresentationGraph,
    RepresentationKind,
    RepresentationNode,
    RepresentationRelation,
    RepresentationValidationError,
    stable_representation_id,
)


class VaultRepresentationRuntimeMixin:
    """Persist domain-neutral representation nodes and typed relations in Crystal."""

    @staticmethod
    def _node_from_row(row: tuple[Any, ...] | None) -> RepresentationNode | None:
        if row is None:
            return None
        return RepresentationNode.from_dict(json.loads(str(row[0])))

    @staticmethod
    def _relation_from_row(row: tuple[Any, ...] | None) -> RepresentationRelation | None:
        if row is None:
            return None
        return RepresentationRelation.from_dict(json.loads(str(row[0])))

    @staticmethod
    def _store_node_conn(conn, node: RepresentationNode) -> bool:
        canonical = node.canonical_json()
        existing = conn.execute(
            "SELECT canonical_json FROM representation_nodes WHERE node_id = ?",
            (node.node_id,),
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != canonical:
                raise RepresentationConflictError(
                    f"representation node identifier already has different content: {node.node_id}"
                )
            return False
        conn.execute(
            "INSERT INTO representation_nodes "
            "(node_id, domain, kind, epistemic_status, canonical_json, fingerprint, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                node.node_id, node.domain, node.kind.value, node.status.value,
                canonical, node.fingerprint(), time.time(),
            ),
        )
        return True

    @staticmethod
    def _store_relation_conn(conn, relation: RepresentationRelation) -> bool:
        for endpoint in (relation.source_id, relation.target_id):
            present = conn.execute(
                "SELECT 1 FROM representation_nodes WHERE node_id = ?",
                (endpoint,),
            ).fetchone()
            if present is None:
                raise RepresentationValidationError(f"unknown relation endpoint: {endpoint}")
        canonical = relation.canonical_json()
        existing = conn.execute(
            "SELECT canonical_json FROM representation_relations WHERE relation_id = ?",
            (relation.relation_id,),
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != canonical:
                raise RepresentationConflictError(
                    "representation relation identifier already has different content: "
                    + relation.relation_id
                )
            return False
        conn.execute(
            "INSERT INTO representation_relations "
            "(relation_id, source_id, predicate, target_id, domain, canonical_json, fingerprint, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                relation.relation_id, relation.source_id, relation.predicate,
                relation.target_id, relation.domain, canonical, relation.fingerprint(), time.time(),
            ),
        )
        return True

    def _representation_write(self, operation):
        """Run a representation mutation in an explicit SQLite transaction."""
        with self._lock:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                result = operation(conn)
                conn.execute("COMMIT")
                return result
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def store_representation_node(self, node: RepresentationNode) -> str:
        if not isinstance(node, RepresentationNode):
            raise TypeError("node must be a RepresentationNode")
        self._representation_write(lambda conn: self._store_node_conn(conn, node))
        return node.node_id

    def store_representation_relation(self, relation: RepresentationRelation) -> str:
        if not isinstance(relation, RepresentationRelation):
            raise TypeError("relation must be a RepresentationRelation")
        self._representation_write(lambda conn: self._store_relation_conn(conn, relation))
        return relation.relation_id

    def store_representation_graph(self, graph: RepresentationGraph) -> dict[str, int]:
        if not isinstance(graph, RepresentationGraph):
            raise TypeError("graph must be a RepresentationGraph")
        graph.validate()

        def persist(conn):
            nodes_added = sum(int(self._store_node_conn(conn, node)) for node in graph.nodes)
            relations_added = sum(
                int(self._store_relation_conn(conn, relation))
                for relation in graph.relations
            )
            return {"nodes_added": nodes_added, "relations_added": relations_added}

        return self._representation_write(persist)

    def get_representation_node(self, node_id: str) -> RepresentationNode | None:
        row = self.conn.execute(
            "SELECT canonical_json FROM representation_nodes WHERE node_id = ?",
            (str(node_id),),
        ).fetchone()
        return self._node_from_row(row)

    def get_representation_relation(self, relation_id: str) -> RepresentationRelation | None:
        row = self.conn.execute(
            "SELECT canonical_json FROM representation_relations WHERE relation_id = ?",
            (str(relation_id),),
        ).fetchone()
        return self._relation_from_row(row)

    def list_representation_nodes(
        self,
        *,
        domain: str | None = None,
        kind: RepresentationKind | str | None = None,
        status: EpistemicStatus | str | None = None,
        limit: int = 500,
    ) -> tuple[RepresentationNode, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if domain is not None:
            clauses.append("domain = ?")
            params.append(str(domain).strip().casefold())
        if kind is not None:
            clauses.append("kind = ?")
            params.append(RepresentationKind(kind).value)
        if status is not None:
            clauses.append("epistemic_status = ?")
            params.append(EpistemicStatus(status).value)
        bounded_limit = max(0, min(int(limit), 10000))
        if bounded_limit == 0:
            return ()
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.conn.execute(
            "SELECT canonical_json FROM representation_nodes" + where
            + " ORDER BY node_id LIMIT ?",
            (*params, bounded_limit),
        ).fetchall()
        return tuple(RepresentationNode.from_dict(json.loads(row[0])) for row in rows)

    def list_representation_relations(
        self,
        *,
        node_id: str | None = None,
        predicate: str | None = None,
        domain: str | None = None,
        limit: int = 1000,
    ) -> tuple[RepresentationRelation, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if node_id is not None:
            clauses.append("(source_id = ? OR target_id = ?)")
            params.extend((str(node_id), str(node_id)))
        if predicate is not None:
            clauses.append("predicate = ?")
            params.append(str(predicate).strip().casefold())
        if domain is not None:
            clauses.append("domain = ?")
            params.append(str(domain).strip().casefold())
        bounded_limit = max(0, min(int(limit), 20000))
        if bounded_limit == 0:
            return ()
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.conn.execute(
            "SELECT canonical_json FROM representation_relations" + where
            + " ORDER BY relation_id LIMIT ?",
            (*params, bounded_limit),
        ).fetchall()
        return tuple(RepresentationRelation.from_dict(json.loads(row[0])) for row in rows)

    def load_representation_graph(
        self,
        *,
        domain: str | None = None,
        node_ids: Iterable[str] | None = None,
        node_limit: int = 400,
        relation_limit: int = 10000,
    ) -> RepresentationGraph:
        """Load a bounded induced subgraph; the whole vault is never loaded by default."""
        maximum = max(0, min(int(node_limit), 400))
        if node_ids is None:
            nodes = self.list_representation_nodes(domain=domain, limit=maximum)
        else:
            selected = tuple(dict.fromkeys(str(value) for value in node_ids))[:maximum]
            nodes = tuple(
                node for node_id in selected
                if (node := self.get_representation_node(node_id)) is not None
                and (domain is None or node.domain == domain.strip().casefold())
            )
        graph = RepresentationGraph(nodes=nodes)
        selected_ids = {node.node_id for node in graph.nodes}
        if not selected_ids:
            return graph
        qmarks = ",".join("?" for _ in selected_ids)
        ordered_ids = sorted(selected_ids)
        rows = self.conn.execute(
            "SELECT canonical_json FROM representation_relations "
            "WHERE source_id IN (" + qmarks + ") AND target_id IN (" + qmarks + ") "
            "ORDER BY relation_id LIMIT ?",
            (*ordered_ids, *ordered_ids, max(0, min(int(relation_limit), 20000))),
        ).fetchall()
        for row in rows:
            graph.add_relation(RepresentationRelation.from_dict(json.loads(row[0])))
        return graph

    def ingest_blueprints_as_representations(
        self,
        blueprints: Iterable[Blueprint],
    ) -> dict[str, int]:
        """Index blueprints without changing their storage or evidence tier."""
        items = tuple(blueprints)
        nodes = {item.blueprint_id: RepresentationNode.from_blueprint(item) for item in items}
        graph = RepresentationGraph(nodes=nodes.values())
        for blueprint in items:
            source = nodes[blueprint.blueprint_id]
            for dependency_id in blueprint.dependencies:
                target_id = f"blueprint:{dependency_id}"
                target = nodes.get(dependency_id)
                if target is None:
                    target = self.get_representation_node(target_id)
                if target is None:
                    continue
                graph.add_node(target)
                payload = {
                    "source": source.node_id,
                    "predicate": "depends_on",
                    "target": target.node_id,
                }
                graph.add_relation(RepresentationRelation(
                    relation_id=stable_representation_id("relation", payload),
                    source_id=source.node_id,
                    predicate="depends_on",
                    target_id=target.node_id,
                    domain=blueprint.domain,
                    status=source.status,
                    provenance=source.provenance,
                    verification=source.verification,
                ))
        return self.store_representation_graph(graph)
