import pytest

from phase355_backend_realization_contract import BackendRealizationContract
from phase357_response_generation_interface import (
    build_response_generation_request,
    generate_response,
)


def make_contract():
    return BackendRealizationContract(
        mode="explain",
        context={"entities": ("Python",), "operations": ("explain",)},
        actions=("inspect",),
        observation="Explain Python",
    )


def test_build_request_preserves_verified_contract():
    contract = make_contract()
    request = build_response_generation_request(contract)
    assert request is not None
    assert request.contract is contract


def test_missing_contract_fails_closed():
    assert build_response_generation_request(None) is None
    assert generate_response(None, lambda request: "ignored") is None


def test_generation_is_delegated_without_invention():
    request = build_response_generation_request(make_contract())
    seen = []

    def generator(received):
        seen.append(received.contract)
        return "generated"

    assert generate_response(request, generator) == "generated"
    assert seen == [request.contract]


def test_non_callable_generator_rejected():
    request = build_response_generation_request(make_contract())
    with pytest.raises(TypeError):
        generate_response(request, None)


def test_generator_receives_no_unverified_actions():
    contract = make_contract()
    request = build_response_generation_request(contract)

    def generator(received):
        assert received.contract.actions == ("inspect",)
        return received.contract.observation

    assert generate_response(request, generator) == "Explain Python"
