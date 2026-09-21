import pytest

from phase363_response_model_loading_boundary import load_response_model


class Model:
    def generate(self, request):
        return "ok"


class Loader:
    def load(self):
        return Model()


def test_loader_returns_valid_model():
    loaded = load_response_model(Loader())
    assert loaded.model.generate(None) == "ok"


def test_invalid_loader_fails_closed():
    with pytest.raises(TypeError):
        load_response_model(object())


def test_invalid_loaded_model_fails_closed():
    class BadLoader:
        def load(self):
            return object()

    with pytest.raises(TypeError):
        load_response_model(BadLoader())
