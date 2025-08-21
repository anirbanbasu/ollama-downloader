import signal
import sys
from typing_extensions import Annotated
import typer
from rich import print as print
from rich import print_json as printj


from ollama_downloader.common import logger
from ollama_downloader.model_downloader import OllamaModelDownloader

# Initialize the Typer application
app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="A command-line interface for the Ollama downloader.",
)

model_downloader: OllamaModelDownloader = None


@app.command()
def show_config():
    """Shows the application configuration as JSON."""
    printj(model_downloader.settings.model_dump_json())


@app.command()
def list_models():
    """Lists all available models."""
    models = model_downloader.update_models_list()
    print(models)


@app.command()
def list_tags(
    model: Annotated[
        str | None,
        typer.Argument(
            help="The name of the model to list tags for, e.g., llama3.1. If not provided, all models and their tags will be listed."
        ),
    ] = None,
    update: Annotated[
        bool,
        typer.Option(
            is_flag=True,
            help="Force update the model list and its tags before listing.",
        ),
    ] = False,
):
    """Lists all tags for a specific model."""
    tags = model_downloader.list_models_tags(model=model, update=update)
    print(tags)


@app.command()
def model_download(
    model: Annotated[
        str, typer.Argument(help="The name of the model to download, e.g., llama3.1")
    ],
    tag: Annotated[
        str, typer.Argument(help="The tag of the model to download, e.g., latest")
    ] = "latest",
):
    model_downloader.download_model(
        model=model,
        tag=tag,
    )


def main():
    """Main entry point for the CLI application."""

    def sigint_handler(signum, frame):
        typer.echo("Ctrl+C detected. Performing graceful shutdown...")
        # Add your cleanup logic here, e.g., closing files, releasing resources
        if model_downloader:
            model_downloader._cleanup_unnecessary_files()
        # Exit the application gracefully
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    # All good so far, let's start the Typer app
    try:
        global model_downloader
        # Initialize the model downloader with the application settings
        model_downloader = OllamaModelDownloader()
        app()
    except Exception as e:
        logger.error(f"[bold red]{e}[/bold red]")
    finally:
        # Perform any necessary cleanup here
        if model_downloader:
            # Ensure we clean up temporary files
            model_downloader._cleanup_unnecessary_files()


if __name__ == "__main__":
    main()
