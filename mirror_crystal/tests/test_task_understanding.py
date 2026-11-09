from __future__ import annotations

import pytest

from mirror_crystal import (
    ConstraintKind,
    DiskCrystalVault,
    TaskConstraint,
    TaskIntent,
    TaskSpec,
    TaskStatus,
    TaskSubtask,
    TaskUnderstandingEngine,
    TaskUnderstandingError,
)


def test_understands_build_request_and_preserves_exact_goal():
    request = "Build a calculator that supports addition and multiplication."
    task = TaskUnderstandingEngine().interpret(request)
    assert task.intent is TaskIntent.BUILD
    assert task.domain == "programming"
    assert task.goal == request
    assert task.request == request
    assert task.status is TaskStatus.READY
    assert task.subtasks == ()


def test_understands_non_programming_domains_with_one_shared_interpreter():
    examples = (
        ("Calculate the eigenvalues of this matrix.", "mathematics", TaskIntent.CALCULATE),
        ("Analyze the quantum wavefunction.", "physics", TaskIntent.ANALYZE),
        ("Explain how DNA replication works.", "biology", TaskIntent.EXPLAIN),
        ("Analyze this molecular reaction.", "chemistry", TaskIntent.ANALYZE),
        ("Design a structural bridge.", "engineering", TaskIntent.DESIGN),
        ("Summarize this portfolio's financial risk.", "finance", TaskIntent.SUMMARIZE),
    )
    engine = TaskUnderstandingEngine()
    for request, domain, intent in examples:
        result = engine.interpret(request)
        assert result.domain == domain
        assert result.intent is intent


def test_explicit_domain_hint_wins_over_lexical_guess():
    task = TaskUnderstandingEngine().interpret(
        "Analyze a Python model of a biological cell.",
        domain_hint="biology",
    )
    assert task.domain == "biology"


def test_unknown_intent_is_not_fabricated_and_requests_clarification():
    task = TaskUnderstandingEngine().interpret("Something about that.")
    assert task.intent is TaskIntent.UNKNOWN
    assert task.status is TaskStatus.NEEDS_CLARIFICATION
    assert any("action verb" in item for item in task.unknowns)


def test_short_unresolved_reference_needs_context():
    task = TaskUnderstandingEngine().interpret("Please fix it.")
    assert task.intent is TaskIntent.DIAGNOSE
    assert task.status is TaskStatus.NEEDS_CLARIFICATION
    assert any("referent" in item for item in task.unknowns)


def test_context_resolves_short_reference_for_understanding():
    task = TaskUnderstandingEngine().interpret(
        "Please fix it.",
        context={"referent": "the provided Python function"},
    )
    assert task.status is TaskStatus.READY


def test_extracts_explicit_constraints_without_inventing_them():
    request = "Build the API. It must use Python and must not store passwords. Keep the current database schema."
    task = TaskUnderstandingEngine().interpret(request)
    assert any(item.kind is ConstraintKind.MUST for item in task.constraints)
    assert any(item.kind is ConstraintKind.MUST_NOT for item in task.constraints)
    assert any(item.kind is ConstraintKind.PRESERVE for item in task.constraints)
    assert all(item.source == "user_request" for item in task.constraints)


def test_explicit_context_and_inputs_are_preserved_as_typed_json():
    task = TaskUnderstandingEngine().interpret(
        "Calculate the total.",
        domain_hint="mathematics",
        known_inputs={"values": [1, 2, 3], "include_tax": False, "rate": 0.05},
        context={"currency": "PKR", "source": "user_supplied"},
    )
    assert task.known_inputs["include_tax"] is False
    assert task.known_inputs["values"] == (1, 2, 3)
    assert task.context["currency"] == "PKR"


def test_requested_output_formats_are_recorded():
    task = TaskUnderstandingEngine().interpret("Create a report and return JSON.")
    assert task.requested_outputs == ("json",)


def test_assumptions_and_acceptance_criteria_are_only_recorded_when_explicit():
    task = TaskUnderstandingEngine().interpret(
        "Build a parser. Assume the input is UTF-8. Done when all fixture cases pass."
    )
    assert task.assumptions == ("Assume the input is UTF-8.",)
    assert task.acceptance_criteria == ("all fixture cases pass",)


def test_supplied_acceptance_criteria_and_constraints_are_included():
    task = TaskUnderstandingEngine().interpret(
        "Design a battery enclosure.",
        explicit_constraints=("Maximum mass is 2 kg.",),
        acceptance_criteria=("Survives vibration test.",),
    )
    assert task.acceptance_criteria == ("Survives vibration test.",)
    assert any(item.source == "caller_context" for item in task.constraints)


def test_sequential_actions_become_dependency_linked_subtasks():
    request = "First analyze the requirements, then design the system, and then implement it."
    task = TaskUnderstandingEngine().interpret(request, context={"referent": "the system"})
    assert len(task.subtasks) == 3
    ordered = task.topological_order()
    assert ordered == tuple(item.subtask_id for item in task.subtasks)
    assert task.subtasks[0].depends_on == ()
    assert task.subtasks[1].depends_on == (task.subtasks[0].subtask_id,)
    assert task.subtasks[2].depends_on == (task.subtasks[1].subtask_id,)
    assert task.subtasks[0].intent is TaskIntent.ANALYZE
    assert task.subtasks[1].intent is TaskIntent.DESIGN
    assert task.subtasks[2].intent is TaskIntent.IMPLEMENT


def test_parallel_bullets_become_independent_subtasks():
    request = """Complete these tasks:
- Analyze the measurements.
- Compare the two models.
- Summarize the results."""
    task = TaskUnderstandingEngine().interpret(request)
    assert len(task.subtasks) == 3
    assert all(not item.depends_on for item in task.subtasks)


def test_single_action_is_not_artificially_decomposed():
    task = TaskUnderstandingEngine().interpret("Explain how photosynthesis works.")
    assert task.subtasks == ()


def test_task_spec_round_trip_and_fingerprint_are_deterministic():
    engine = TaskUnderstandingEngine()
    first = engine.interpret(
        "Analyze the data, then summarize the results.",
        known_inputs={"rows": [{"x": 1}, {"x": 2}]},
    )
    second = TaskSpec.from_dict(first.to_dict())
    assert second == first
    assert second.canonical_json() == first.canonical_json()
    assert second.fingerprint == first.fingerprint


def test_task_spec_rejects_unknown_dependencies_and_cycles():
    with pytest.raises(TaskUnderstandingError, match="unknown dependencies"):
        TaskSpec(
            task_id="missing",
            request="Do work",
            goal="Do work",
            intent=TaskIntent.UNKNOWN,
            subtasks=(TaskSubtask("a", "A", depends_on=("b",)),),
        )
    with pytest.raises(TaskUnderstandingError, match="cycle"):
        TaskSpec(
            task_id="cycle",
            request="Do work",
            goal="Do work",
            intent=TaskIntent.UNKNOWN,
            subtasks=(
                TaskSubtask("a", "A", depends_on=("b",)),
                TaskSubtask("b", "B", depends_on=("a",)),
            ),
        )


def test_subtask_rejects_negative_order_and_self_dependency():
    with pytest.raises(TaskUnderstandingError, match="non-negative"):
        TaskSubtask("a", "A", sequence_index=-1)
    with pytest.raises(TaskUnderstandingError, match="depend on itself"):
        TaskSubtask("a", "A", depends_on=("a",))


def test_task_persistence_reuses_representation_graph_and_survives_reopen(tmp_path):
    task = TaskUnderstandingEngine().interpret(
        "Analyze the request, then design the response.",
        domain_hint="general",
    )
    path = tmp_path / "task-spec.dcrystal"
    vault = DiskCrystalVault(path)
    try:
        assert vault.store_task_spec(task) == {
            "nodes_added": 3,
            "relations_added": 3,
        }
        assert vault.store_task_spec(task) == {
            "nodes_added": 0,
            "relations_added": 0,
        }
        loaded = vault.get_task_spec(task.task_id)
        assert loaded == task
        graph = vault.load_representation_graph(domain="general")
        assert len(graph.nodes) == 3
        assert sorted(edge.predicate for edge in graph.relations) == [
            "decomposes_into", "decomposes_into", "depends_on"
        ]
    finally:
        vault.close()

    reopened = DiskCrystalVault(path)
    try:
        assert reopened.get_task_spec(task.task_id) == task
    finally:
        reopened.close()


def test_task_persistence_does_not_auto_record_requests(tmp_path):
    path = tmp_path / "explicit-only.dcrystal"
    vault = DiskCrystalVault(path)
    try:
        assert vault.list_representation_nodes(domain="general") == ()
    finally:
        vault.close()
