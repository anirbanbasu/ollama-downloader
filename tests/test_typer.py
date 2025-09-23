from typer.testing import CliRunner

import pytest

from ollama_downloader.cli import app
from ollama_downloader.common import EnvVar
from ollama_downloader.data.data_models import AppSettings


class TestTyperCalls:
    """
    Class to group tests related to Typer CLI commands.
    """

    @pytest.fixture(autouse=True)
    def runner(self):
        """
        Fixture to provide a Typer CLI runner for testing.
        """
        runner = CliRunner(env={EnvVar.LOG_LEVEL: EnvVar.DEFAULT__LOG_LEVEL})
        return runner

    def test_show_config(self, runner):
        """
        Test the 'show-config' command of the CLI.
        """
        result = runner.invoke(app=app, args=["show-config"])
        assert result.exit_code == 0
        settings = AppSettings.load_settings()
        # Assert that we can read the local settings -- after all, this is what the show-config command does
        assert settings is not None
        # This is a bit fragile, as the indentation matching depends on the print_json implementation of Rich, which defaults to 2.
        assert settings.model_dump_json(indent=2) == result.output.strip()

    def test_auto_config(self, runner):
        """
        Test the 'auto-config' command of the CLI.
        """
        result = runner.invoke(app=app, args=["auto-config"])
        assert result.exit_code == 0
        if result.output.strip() != "":
            assert AppSettings.model_validate_json(result.output.strip()) is not None

    def test_list_models(self, runner):
        """
        Test the 'list-models' command of the CLI.
        """
        result = runner.invoke(app=app, args=["list-models"])
        assert result.exit_code == 0
        # Expect at least few known models to be listed
        assert "gpt-oss" in result.output.lower()
        assert "llama" in result.output.lower()
        assert "granite" in result.output.lower()
        assert "gemma" in result.output.lower()
        assert "deepseek" in result.output.lower()
        assert "made-up-model-that-should-not-exist" not in result.output.lower()

    def test_list_tags(self, runner):
        """
        Test the 'list-tags' command of the CLI.
        """
        result = runner.invoke(app, ["list-tags", model_identifier := "gpt-oss"])
        assert result.exit_code == 0
        # Expect at least two known tags and a cloud tag to be listed for the gpt-oss model
        assert f"{model_identifier}:latest" in result.output
        assert f"{model_identifier}:20b" in result.output
        assert f"{model_identifier}:20b-cloud" in result.output
        result = runner.invoke(
            app=app,
            args=["list-tags", "made-up-model-that-should-not-exist"],
        )
        # Should be an empty output while the error will be logged but exit code will still be 0
        assert result.output == ""

    def test_model_download(self, runner):
        """
        Test the 'model-download' command of the CLI.
        """
        # Let's try downloading the smallest possible model to stop the test from taking too long
        model_tag = "all-minilm:22m"
        result = runner.invoke(app=app, args=["model-download", model_tag])
        assert result.exit_code == 0
        assert f"{model_tag} successfully downloaded and saved" in result.output

        # Let's try downloading a cloud model
        model_tag = "gpt-oss:20b-cloud"
        result = runner.invoke(app=app, args=["model-download", model_tag])
        assert result.exit_code == 0
        assert f"{model_tag} successfully downloaded and saved" in result.output

        model_tag = "made-up:should-fail"
        result = runner.invoke(app=app, args=["model-download", model_tag])
        assert result.exit_code == 0
        assert f"{model_tag} successfully downloaded and saved" not in result.output

    def test_hf_list_models(self, runner):
        """
        Test the hf-list-models' command of the CLI.
        """
        result = runner.invoke(app=app, args=["hf-list-models", "--page", "4"])
        assert result.exit_code == 0
        # Expect the output to contain at least 25 models on page 4
        # Models change often on Hugging Face, so we cannot check for specific models
        assert "25, page 4" in result.output.lower()
        assert "made-up-model-that-should-not-exist" not in result.output.lower()

    def test_hf_list_tags(self, runner):
        """
        Test the 'list-tags' command of the CLI.
        """
        result = runner.invoke(
            app,
            ["hf-list-tags", model_identifier := "unsloth/SmolLM2-135M-Instruct-GGUF"],
        )
        assert result.exit_code == 0
        # Expect at least two known tags to be listed for the gpt-oss model
        assert f"{model_identifier}:F16" in result.output
        assert f"{model_identifier}:Q4_K_M" in result.output
        result = runner.invoke(
            app=app,
            args=["hf-list-tags", "made-up-model-that-should-not-exist"],
        )
        # Should be an empty output while the error will be logged but exit code will still be 0
        assert result.output == ""

    def test_hf_model_download(self, runner):
        """
        Test the 'hf-model-download' command of the CLI.
        """
        # Let's try downloading the smallest possible model to stop the test from taking too long
        user_repo_quant = "unsloth/SmolLM2-135M-Instruct-GGUF:Q4_K_M"
        result = runner.invoke(
            app=app,
            args=["hf-model-download", user_repo_quant],
        )
        assert result.exit_code == 0
        assert f"{user_repo_quant} successfully downloaded and saved" in result.output

        user_repo_quant = "made-up/should-fail:Q0_0_X"
        result = runner.invoke(
            app=app,
            args=["hf-model-download", user_repo_quant],
        )
        assert result.exit_code == 0
        assert (
            f"{user_repo_quant} successfully downloaded and saved" not in result.output
        )
