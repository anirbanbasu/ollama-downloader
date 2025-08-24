import pytest
from ollama_downloader import data_models

def test_ollama_server_default():
    server_model = data_models.OllamaServer()

def test_ollama_server_valid_url():
    server_model = data_models.OllamaServer(url="http://192.168.1.1:11434")

def test_ollama_server_invalid_url():
    with pytest.raises(ValueError):
        server_model = data_models.OllamaServer(url="http://:11434")

def test_ollama_server_assign_valid_url():
    server_model = data_models.OllamaServer()

    server_model.url = "http://192.168.1.1:11434"
    server_model.url = "https://mypage.com:11434"

def test_ollama_server_assign_invalid_url():
    server_model = data_models.OllamaServer()

    with pytest.raises(ValueError):
        server_model.url = "http://:11434"
