from mirror_crystal.blueprint_contracts import EvidenceLevel, VerificationEvidence
from mirror_crystal.capability_matching import CapabilityMatcher
from mirror_crystal.capability_model import CapabilitySpec, PortContract
from mirror_crystal.task_understanding import TaskIntent, TaskSpec, TaskStatus, TaskSubtask
from mirror_crystal.universal_representation import RepresentationProvenance


def verified():
    return VerificationEvidence(
        level=EvidenceLevel.EXECUTION_TESTED,
        checks_passed=("contract-test",),
        verifier="pytest",
    )


def cap(identifier, *, domain="mathematics", intent="calculate", evidence=None, inputs=()):
    return CapabilitySpec(
        capability_id=identifier,
        name=identifier,
        description="test capability",
        domains=(domain,),
        intents=(intent,),
        inputs=tuple(inputs),
        outputs=(PortContract("result", "number"),),
        provenance=RepresentationProvenance(source="matcher-test"),
        verification=evidence or verified(),
    )


def task(**kwargs):
    values = dict(
        task_id="calc-task",
        request="Calculate a value",
        goal="Calculate a value",
        intent=TaskIntent.CALCULATE,
        domain="mathematics",
        status=TaskStatus.READY,
        known_inputs={"left": 3, "right": 4},
    )
    values.update(kwargs)
    return TaskSpec(**values)


def test_matcher_prefers_exact_domain_and_intent_and_exposes_evidence():
    exact = cap("exact")
    generic = cap("generic", domain="general", intent="any")
    unverified = cap(
        "unverified",
        evidence=VerificationEvidence(level=EvidenceLevel.UNVERIFIED),
    )
    report = CapabilityMatcher().match(task(), (generic, unverified, exact))
    matches = report.for_subtask("task:calc-task")
    assert matches[0].capability.capability_id == "exact"
    assert matches[0].eligible
    assert not next(x for x in matches if x.capability.capability_id == "unverified").eligible


def test_missing_inputs_are_reported_without_guessing_a_binding():
    candidate = cap(
        "two-inputs",
        inputs=(PortContract("left", "number"), PortContract("third", "number")),
    )
    report = CapabilityMatcher().match(task(), (candidate,))
    match = report.for_subtask("task:calc-task")[0]
    assert match.matched_inputs == ("left",)
    assert match.missing_inputs == ("third",)
    assert match.eligible


def test_named_input_with_wrong_type_rejects_candidate():
    candidate = cap("integer-only", inputs=(PortContract("left", "integer"),))
    report = CapabilityMatcher().match(
        task(known_inputs={"left": True}),
        (candidate,),
    )
    assert report.matches == ()
    assert report.unmatched_subtasks == ("task:calc-task",)


def test_domain_and_intent_mismatch_are_not_suggested():
    candidate = cap("wrong", domain="biology", intent="analyze")
    report = CapabilityMatcher().match(task(), (candidate,))
    assert not report.matches
    assert report.unmatched_subtasks == ("task:calc-task",)


def test_matcher_handles_each_subtask_independently():
    first = cap("build", intent="build", domain="programming")
    second = cap("test", intent="test", domain="programming")
    spec = TaskSpec(
        task_id="two-step",
        request="Build it then test it",
        goal="Build and test",
        intent=TaskIntent.BUILD,
        domain="programming",
        subtasks=(
            TaskSubtask("s1", "Build it", TaskIntent.BUILD, 0),
            TaskSubtask("s2", "Test it", TaskIntent.TEST, 1, ("s1",)),
        ),
    )
    report = CapabilityMatcher().match(spec, (first, second))
    assert [x.capability.capability_id for x in report.for_subtask("s1")] == ["build"]
    assert [x.capability.capability_id for x in report.for_subtask("s2")] == ["test"]
