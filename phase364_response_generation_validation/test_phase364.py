import pytest

from phase362_response_generation_config import ResponseGenerationConfig
from phase364_response_generation_validation import validate_generated_response


def test_valid_string_passes_unchanged():
    value = "hello"
    assert validate_generated_response(value, mode="explain", config=ResponseGenerationConfig()) == value


def test_mapping_response_passes_unchanged():
    value = {"text": "hello", "score": 1.0}
    assert validate_generated_response(value, mode="explain", config=ResponseGenerationConfig()) == value


def test_none_is_rejected():
    with pytest.raises(ValueError):
        validate_generated_response(None, mode="explain", config=ResponseGenerationConfig())


def test_empty_string_is_rejected_by_default():
    with pytest.raises(ValueError):
        validate_generated_response("", mode="explain", config=ResponseGenerationConfig())


def test_length_limit_is_enforced():
    with pytest.raises(ValueError):
        validate_generated_response(
            "abcdef",
            mode="explain",
            config=ResponseGenerationConfig(max_output_chars=5),
        )


def test_allowed_modes_are_enforced():
    config = ResponseGenerationConfig(allowed_modes=("explain",))
    with pytest.raises(ValueError):
        validate_generated_response("ok", mode="create", config=config)
