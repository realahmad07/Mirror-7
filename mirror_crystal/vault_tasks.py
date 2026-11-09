"""Persist understood task specifications in the universal representation graph."""
from __future__ import annotations

from .task_understanding import TaskSpec
from .universal_representation import (
    EpistemicStatus,
    RepresentationGraph,
    RepresentationKind,
    RepresentationNode,
    RepresentationProvenance,
    RepresentationRelation,
    stable_representation_id,
)


class VaultTaskRuntimeMixin:
    """Explicit task persistence without auto-recording private user requests."""

    def store_task_spec(self, task: TaskSpec) -> dict[str, int]:
        if not isinstance(task, TaskSpec):
            raise TypeError("task must be a TaskSpec")
        root_id = f"task:{task.task_id}"
        provenance = RepresentationProvenance(
            source="mirror_task_understanding",
            source_ref=task.fingerprint,
        )
        graph_nodes = [RepresentationNode(
            node_id=root_id,
            kind=RepresentationKind.GOAL,
            domain=task.domain,
            label=task.goal[:240],
            content={"node_role": "task", "task_spec": task.to_dict()},
            value_type="record",
            status=EpistemicStatus.ASSERTED,
            attributes={"task_status": task.status.value, "intent": task.intent.value},
            provenance=provenance,
        )]
        relations = []
        subtask_node_ids: dict[str, str] = {}
        for subtask in task.subtasks:
            node_id = f"task-subtask:{task.task_id}:{subtask.subtask_id}"
            subtask_node_ids[subtask.subtask_id] = node_id
            graph_nodes.append(RepresentationNode(
                node_id=node_id,
                kind=RepresentationKind.GOAL,
                domain=task.domain,
                label=subtask.goal[:240],
                content={"node_role": "subtask", "subtask": subtask.to_dict()},
                value_type="record",
                status=EpistemicStatus.ASSERTED,
                provenance=RepresentationProvenance(
                    source="mirror_task_understanding",
                    source_ref=task.fingerprint,
                    parent_ids=(root_id,),
                ),
            ))
            edge_payload = {"source": root_id, "predicate": "decomposes_into", "target": node_id}
            relations.append(RepresentationRelation(
                relation_id=stable_representation_id("task-relation", edge_payload),
                source_id=root_id,
                predicate="decomposes_into",
                target_id=node_id,
                domain=task.domain,
                status=EpistemicStatus.ASSERTED,
                provenance=provenance,
            ))
        for subtask in task.subtasks:
            for dependency_id in subtask.depends_on:
                source_id = subtask_node_ids[subtask.subtask_id]
                target_id = subtask_node_ids[dependency_id]
                edge_payload = {
                    "source": source_id,
                    "predicate": "depends_on",
                    "target": target_id,
                }
                relations.append(RepresentationRelation(
                    relation_id=stable_representation_id("task-relation", edge_payload),
                    source_id=source_id,
                    predicate="depends_on",
                    target_id=target_id,
                    domain=task.domain,
                    status=EpistemicStatus.ASSERTED,
                    provenance=provenance,
                ))
        return self.store_representation_graph(RepresentationGraph(graph_nodes, relations))

    def get_task_spec(self, task_id: str) -> TaskSpec | None:
        node = self.get_representation_node(f"task:{task_id}")
        if node is None:
            return None
        if node.kind is not RepresentationKind.GOAL:
            return None
        if node.content.get("node_role") != "task":
            return None
        payload = node.content.get("task_spec")
        if not isinstance(payload, dict):
            return None
        return TaskSpec.from_dict(payload)
