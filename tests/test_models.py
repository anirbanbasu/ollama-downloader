import pytest
from ollama_downloader import data_models
from pathlib import Path


def test_ollama_library_default():
    ollama_library_model = data_models.OllamaLibrary()


def test_ollama_library_valid_path():
    ollama_library_model = data_models.OllamaLibrary(models_path=Path("."))


def test_ollama_library_invalid_path():
    with pytest.raises(ValueError):
        ollama_library_model = data_models.OllamaLibrary(
            models_path=Path("/unstandard")
        )


def test_ollama_library_assign_valid_path():
    ollama_library_model = data_models.OllamaLibrary()

    ollama_library_model.models_path = Path(".")


def test_ollama_libary_assign_invalid_path():
    ollama_library_model = data_models.OllamaLibrary()

    with pytest.raises(ValueError):
        ollama_library_model.models_path = Path("/unstandard")
