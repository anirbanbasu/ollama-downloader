from typer.testing import CliRunner

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
