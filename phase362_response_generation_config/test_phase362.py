import pytest

from phase362_response_generation_config import ResponseGenerationConfig, make_generation_config


def test_defaults_are_safe_and_immutable():
    config = ResponseGenerationConfig()
    assert config.max_output_chars == 8192
    assert config.reject_empty is True
    assert config.allowed_modes == ()


def test_factory_normalizes_modes():
    config = make_generation_config(allowed_modes=["explain", "create"])
    assert config.allowed_modes == ("explain", "create")


@pytest.mark.parametrize("kwargs", [
    {"max_output_chars": 0},
    {"max_output_chars": -1},
    {"allowed_modes": ["explain", "explain"]},
    {"allowed_modes": [""]},
])
def test_invalid_configuration_fails_closed(kwargs):
    with pytest.raises(ValueError):
        ResponseGenerationConfig(**kwargs)
