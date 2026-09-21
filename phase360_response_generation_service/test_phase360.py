from types import SimpleNamespace

from phase355_backend_realization_contract import BackendRealizationContract
from phase360_response_generation_service import (
    make_generation_record,
    record_to_mapping,
)


def test_record_captures_generated_response():
    result = SimpleNamespace(
        sequence=3,
        realization_contract=BackendRealizationContract(
            mode="explain", context={}, actions=(), observation="x"
        ),
        response="answer",
    )
    record = make_generation_record(result)
    assert record.sequence == 3
    assert record.generated is True
    assert record.mode == "explain"
    assert record.response == "answer"


def test_record_handles_no_generation():
    result = SimpleNamespace(
        sequence=4, realization_contract=None, response=None
    )
    record = make_generation_record(result)
    assert record.generated is False
    assert record.mode is None


def test_mapping_is_read_only_view_data():
    result = SimpleNamespace(
        sequence=1,
        realization_contract=BackendRealizationContract(
            mode="debug", context={}, actions=(), observation="x"
        ),
        response={"text": "fixed"},
    )
    mapping = record_to_mapping(make_generation_record(result))
    assert mapping["mode"] == "debug"
    assert mapping["response"] == {"text": "fixed"}
