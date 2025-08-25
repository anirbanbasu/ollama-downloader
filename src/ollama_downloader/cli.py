import signal
import sys
from typing_extensions import Annotated
import typer
from rich import print as print
from rich import print_json as printj


from ollama_downloader.common import logger
from ollama_downloader.model_downloader import OllamaModelDownloader
from ollama_downloader.hf_model_downloader import HuggingFaceModelDownloader

# Initialize the Typer application
app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="A command-line interface for the Ollama downloader.",
)

model_downloader: OllamaModelDownloader = OllamaModelDownloader()
hf_model_downloader: HuggingFaceModelDownloader = HuggingFaceModelDownloader()


@app.command()
def show_config():
    """Shows the application configuration as JSON."""
    printj(model_downloader.settings.model_dump_json())


@app.command()
def list_models():
    """Lists all available models in the Ollama library."""
    models = model_downloader.update_models_list()
    print(models)


@app.command()
def list_tags(
    model: Annotated[
        str | None,
        typer.Argument(
            help="The name of the model to list tags for, e.g., llama3.1. If not provided, tags of all models will be listed."
        ),
    ] = None,
    update: Annotated[
        bool,
        typer.Option(
            help="Force update the model list and its tags before listing.",
        ),
    ] = False,
):
    """Lists all tags for a specific model."""
    models_tags = model_downloader.list_models_tags(model=model, update=update)
    for model_name, tags in models_tags.items():
        print(f"[bold]Model[/bold]: {model_name}")
        if tags:
            print(f"\t[bold]Tags[/bold] ({len(tags)}): {tags}")


@app.command()
def model_download(
    model_tag: Annotated[
        str,
        typer.Argument(
            help="The name of the model and a specific to download, specified as <model>:<tag>, e.g., llama3.1:8b. If no tag is specified, 'latest' will be assumed.",
        ),
    ],
):
    """Downloads a specific Ollama model with the given tag."""
    model, tag = model_tag.split(":") if ":" in model_tag else (model_tag, "latest")
    model_downloader.download_model(
        model=model,
        tag=tag,
    )


@app.command()
def hf_model_download(
    org_repo_model: Annotated[
        str,
        typer.Argument(
            help="The name of the specific Hugging Face model to download, specified as <org>/<repo>:<model>, e.g., bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M.",
        ),
    ],
):
    """Downloads a specified Hugging Face model."""
    hf_model_downloader.download_model(org_repo_model=org_repo_model)


def main():
    """Main entry point for the CLI application."""

    def interrupt_handler(signum, frame):
        typer.echo("Program interrupt detected. Performing graceful shutdown...")
        # Add your cleanup logic here, e.g., closing files, releasing resources
        if model_downloader:
            model_downloader._cleanup_unnecessary_files()
        if hf_model_downloader:
            hf_model_downloader._cleanup_unnecessary_files()
        # Exit the application gracefully
        sys.exit(0)

    signal.signal(signal.SIGINT, interrupt_handler)
    signal.signal(signal.SIGTERM, interrupt_handler)
    # All good so far, let's start the Typer app
    try:
        app()
    except Exception as e:
        logger.error(f"[bold red]{e}[/bold red]")
    finally:
        # Perform any necessary cleanup here
        if model_downloader:
            # Ensure we clean up temporary files
            model_downloader._cleanup_unnecessary_files()
        if hf_model_downloader:
            hf_model_downloader._cleanup_unnecessary_files()


if __name__ == "__main__":
    main()
