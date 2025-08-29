from typer.testing import CliRunner

# import regex
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

    # @pytest.fixture(autouse=True)
    # def setup_rich_markup_pattern(self):
    #     # Pattern to remove Rich markup tags from the output for correct assertions
    #     self.rich_markup_pattern = regex.compile(
    #         r"""
    #         \[(?P<tag>[^\[\]/]+)\]                 # [tag]
    #         (?P<body>                              # capture the inner body
    #             (?:                                # repeatedly match...
    #                 (?!\[/\g<tag>\])               # ...anything that's not the closing [/tag]
    #                 .                              # any char (DOTALL enabled below)
    #                 | (?R)                         # ...or a nested [x]...[/x]
    #             )*
    #         )
    #         \[/\g<tag>\]                           # [/tag] that matches the opener
    #         """,
    #         regex.VERBOSE | regex.DOTALL,
    #     )

    # def strip_nested_rich_markup(self, text: str) -> str:
    #     """Recursively remove all [tag]...[/tag] patterns from text."""
    #     while True:
    #         new = self.rich_markup_pattern.sub(r"\g<body>", text)
    #         if new == text:
    #             return new
    #         text = new

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
        # Pass the --update flag to ensure we have the latest tags in case the cache is stale or non-existent
        result = runner.invoke(app, ["list-tags", "gpt-oss", "--update"])
        assert result.exit_code == 0
        # Expect at least two known tags to be listed for the gpt-oss model
        assert ":latest" in result.output.lower()
        assert ":20b" in result.output.lower()
        result = runner.invoke(
            app=app,
            args=["list-tags", "made-up-model-that-should-not-exist", "--update"],
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

        model_tag = "made-up:should-fail"
        result = runner.invoke(app=app, args=["model-download", model_tag])
        assert result.exit_code == 0
        assert f"{model_tag} successfully downloaded and saved" not in result.output

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
