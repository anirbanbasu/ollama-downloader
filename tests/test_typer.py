from typer.testing import CliRunner
import subprocess
import sys

from ollama_downloader.cli import app
from ollama_downloader.data_models import AppSettings

runner = CliRunner()


def test_show_config():
    result = runner.invoke(app, ["show-config"])
    assert result.exit_code == 0
    settings = AppSettings()
    # Assert that we can read the local settings -- after all, this is what the show-config command does
    assert settings.read_settings() is True
    # This is a bit fragile, as the indentation matching depends on the print_json implementation of Rich, which defaults to 2.
    assert settings.model_dump_json(indent=2) == result.output.strip()


def test_list_models():
    result = runner.invoke(app, ["list-models"])
    assert result.exit_code == 0
    # Expect at least few known models to be listed
    assert "gpt-oss" in result.output.lower()
    assert "llama" in result.output.lower()
    assert "granite" in result.output.lower()
    assert "gemma" in result.output.lower()
    assert "deepseek" in result.output.lower()


def test_list_tags():
    # Pass the --update flag to ensure we have the latest tags in case the cache is stale or non-existent
    result = runner.invoke(app, ["list-tags", "gpt-oss", "--update"])
    assert result.exit_code == 0
    # Expect at least two known tags to be listed for the gpt-oss model
    assert ":latest" in result.output.lower()
    assert ":20b" in result.output.lower()


def test_model_download():
    # Let's try downloading the smallest possible model to stop the test from taking too long
    model_tag = "all-minilm:22m"
    # Typer's CliRunner is unable to handle to cleanup of temp directories properly.
    # Hence, we will invoke the CLI via subprocess instead.
    result = subprocess.run(
        [sys.executable, "-m", "ollama_downloader.cli", "model-download", model_tag],
        capture_output=True,
        text=True,
        env={"LOG_LEVEL": "INFO"},
    )
    with open("test_model_download.log", "w") as f:
        f.write(result.stdout)
        f.write(result.stderr)
    assert result.returncode == 0
    assert f"{model_tag} successfully downloaded and saved" in result.stdout


def test_hf_model_download():
    # Let's try downloading the smallest possible model to stop the test from taking too long
    org_repo_model = "unsloth/gemma-3-270m-it-GGUF:Q4_K_M"
    # Typer's CliRunner is unable to handle to cleanup of temp directories properly.
    # Hence, we will invoke the CLI via subprocess instead.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ollama_downloader.cli",
            "hf-model-download",
            org_repo_model,
        ],
        capture_output=True,
        text=True,
        env={"LOG_LEVEL": "INFO"},
    )
    with open("test_hf_model_download.log", "w") as f:
        f.write(result.stdout)
        f.write(result.stderr)
    assert result.returncode == 0
    assert f"{org_repo_model} successfully downloaded and saved" in result.stdout
